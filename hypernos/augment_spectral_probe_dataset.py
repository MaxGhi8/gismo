#!/usr/bin/env python3
"""Augment a Schur-action CSV with model-adaptive spectral probe mixtures.

Canonical rows (``probe_family == 2``) reconstruct the exact local Schur
matrix of every patch.  The selected checkpoint supplies the current learned
SPD matrix.  We solve the small generalized eigenproblem

    S_k v = lambda M_theta,k v

and add normalized mixtures of the modes with largest ``abs(log(lambda))``.
These are precisely the local directions for which the learned and exact
operators are least spectrally equivalent, rather than merely new random
right-hand sides.  ``probe_family == 3`` identifies the added active probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch
from torch import Tensor, nn

from model import (
    GeometryConditionedLinearOperator,
    PatchInvariantGeometryNormalizer,
)


class _CheckpointNormalizer(nn.Module):
    """Shape placeholder whose buffers are replaced by the checkpoint."""

    def __init__(self, shape: tuple[int, ...], preserve_padding: bool = False) -> None:
        super().__init__()
        self.preserve_padding = preserve_padding
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("std", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(1.0e-5, dtype=torch.float32))

    def forward(self, values: Tensor) -> Tensor:
        encoded = (values - self.mean) / (self.std + self.eps)
        if not self.preserve_padding:
            return encoded
        valid = values[..., 3:4].ne(0)
        return torch.where(valid, encoded, torch.zeros_like(encoded))


def _load_model(
    checkpoint_path: Path, config_path: Path
) -> GeometryConditionedLinearOperator:
    document = json.loads(config_path.read_text(encoding="utf-8"))
    config = {
        **document["training_properties"],
        **document["iganet_transformer_architecture"],
    }
    normalization = config.get("geometry_normalization", "pointwise")
    channel_normalization = normalization == "channel"
    if normalization == "patch":
        input_normalizer: nn.Module = PatchInvariantGeometryNormalizer()
    else:
        input_shape = (
            (1, 4) if channel_normalization
            else (config["n_control_points"], 4)
        )
        input_normalizer = _CheckpointNormalizer(
            input_shape, preserve_padding=channel_normalization
        )
    model = GeometryConditionedLinearOperator(
        n_dofs=config["n_dofs"],
        n_control_points=config["n_control_points"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        n_heads_A=config["n_heads_A"],
        n_layers_geo=config["n_layers_geo"],
        dropout_rate=config["dropout_rate"],
        activation_str=config["activation_str"],
        zero_mean=config["zero_mean"],
        example_input_normalizer=input_normalizer,
        example_output_normalizer=_CheckpointNormalizer((config["n_dofs"],)),
        device="cpu",
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def _exact_matrix(
    patch_rows: pd.DataFrame,
    rhs_columns: list[str],
    output_columns: list[str],
    size: int,
) -> np.ndarray:
    canonical = patch_rows[patch_rows["probe_family"] == 2]
    probes = canonical[rhs_columns[:size]].to_numpy(dtype=np.float64)
    actions = canonical[output_columns[:size]].to_numpy(dtype=np.float64)
    if len(canonical) != size or np.linalg.matrix_rank(probes) != size:
        raise ValueError(
            f"Patch {int(patch_rows['patch_id'].iloc[0])}: need {size} "
            "linearly independent canonical probes"
        )
    matrix = np.linalg.solve(probes, actions).T
    return 0.5 * (matrix + matrix.T)


@torch.no_grad()
def _learned_matrix(
    model: GeometryConditionedLinearOperator,
    representative: pd.Series,
    geometry_columns: list[str],
    size: int,
) -> np.ndarray:
    geometry = torch.tensor(
        representative[geometry_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    ).reshape(1, -1, 4)
    mask = geometry[..., 3].eq(0)
    normalized = model.input_normalizer(geometry)
    query, scaled_key, epsilon = model.compute_operator_components(
        normalized, mask
    )
    matrix = query[0] @ scaled_key[0].T
    matrix = matrix + epsilon * torch.eye(matrix.shape[0])
    result = matrix[:size, :size].cpu().numpy().astype(np.float64)
    return 0.5 * (result + result.T)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes-per-patch", type=int, default=8)
    parser.add_argument("--probes-per-patch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.modes_per_patch <= 0 or args.probes_per_patch <= 0:
        raise ValueError("Mode and probe counts must be positive")

    frame = pd.read_csv(args.data)
    geometry_columns = [name for name in frame if name.startswith("geom_")]
    rhs_columns = [name for name in frame if name.startswith("dirichlet_")]
    output_columns = [name for name in frame if name.startswith("output_")]
    if not geometry_columns or len(geometry_columns) % 4:
        raise ValueError("Invalid geometry columns")
    if not rhs_columns or len(rhs_columns) != len(output_columns):
        raise ValueError("Invalid Schur action columns")

    model = _load_model(args.checkpoint, args.config)
    if model.n_dofs != len(rhs_columns):
        raise ValueError(
            f"Checkpoint width {model.n_dofs} != CSV width {len(rhs_columns)}"
        )

    rng = np.random.default_rng(args.seed)
    augmented_rows: list[pd.Series] = []
    group_columns = ["patch_id"]
    if "geometry_id" in frame.columns:
        group_columns.insert(0, "geometry_id")
    for patch_key, patch_rows in frame.groupby(group_columns, sort=True):
        representative = patch_rows.iloc[0].copy()
        size = int(representative["n_skeleton"])
        exact = _exact_matrix(patch_rows, rhs_columns, output_columns, size)
        learned = _learned_matrix(
            model, representative, geometry_columns, size
        )
        eigenvalues, eigenvectors = scipy.linalg.eigh(exact, learned)
        positive = eigenvalues > max(1.0e-12, 1.0e-10 * eigenvalues.max())
        indices = np.flatnonzero(positive)
        indices = indices[
            np.argsort(np.abs(np.log(eigenvalues[indices])))[::-1]
        ]
        selected = indices[: min(args.modes_per_patch, len(indices))]
        modes = eigenvectors[:, selected]
        if modes.shape[1] == 0:
            raise ValueError(f"Patch {patch_key}: no positive spectral modes")

        for probe_index in range(args.probes_per_patch):
            if probe_index < modes.shape[1]:
                probe = modes[:, probe_index].copy()
            else:
                probe = modes @ rng.standard_normal(modes.shape[1])
            probe /= np.linalg.norm(probe)
            action = exact @ probe

            row = representative.copy()
            row["sample_id"] = int(patch_rows["sample_id"].max()) + 1 + probe_index
            row["probe_family"] = 3
            row[rhs_columns] = 0.0
            row[output_columns] = 0.0
            row[rhs_columns[:size]] = probe
            row[output_columns[:size]] = action
            augmented_rows.append(row)

        selected_values = eigenvalues[selected]
        print(
            f"patch {patch_key!s}: n={size:2d}, "
            f"equivalence=[{eigenvalues[positive].min():.4g},"
            f"{eigenvalues[positive].max():.4g}], "
            f"selected={np.array2string(selected_values, precision=3)}"
        )

    augmented = pd.DataFrame(augmented_rows, columns=frame.columns)
    result = pd.concat([frame, augmented], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"Wrote {len(frame)} original + {len(augmented)} active rows "
        f"= {len(result)} rows to {args.output}"
    )


if __name__ == "__main__":
    main()
