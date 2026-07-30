#!/usr/bin/env python3
"""Train one geometry-conditioned SPD matrix per patch.

The action datasets contain a complete canonical-basis sweep for every local
patch.  This script reconstructs the corresponding exact Schur matrix and
trains on the whole operator, rather than treating repeated probes from the
same patch as independent samples.  Train/validation/test partitions are made
at patch level, preventing the same geometry-conditioned operator from leaking
across splits.

The objective combines relative Frobenius error with generalized-eigenvalue
errors on the positive eigenspace of the (possibly semidefinite) exact Schur
matrix.  The latter directly targets spectral equivalence, which governs PCG
convergence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from model import (
    GeometryConditionedLinearOperator,
    PatchInvariantGeometryNormalizer,
)


@dataclass(frozen=True)
class PatchOperator:
    key: str
    geometry: np.ndarray
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    skeleton_size: int
    positive_size: int


class StaticNormalizer(nn.Module):
    """Checkpoint-compatible unused output normalizer."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float32))
        self.register_buffer("std", torch.ones(size, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(1.0e-5, dtype=torch.float32))

    def forward(self, values: Tensor) -> Tensor:
        return (values - self.mean) / (self.std + self.eps)


class PatchOperatorDataset(Dataset):
    def __init__(self, records: list[PatchOperator]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str]:
        record = self.records[index]
        return (
            torch.from_numpy(record.geometry),
            torch.from_numpy(record.matrix),
            torch.from_numpy(record.eigenvalues),
            torch.from_numpy(record.eigenvectors),
            torch.tensor(record.skeleton_size, dtype=torch.int64),
            torch.tensor(record.positive_size, dtype=torch.int64),
            record.key,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_matrix(
    rows: pd.DataFrame,
    rhs_columns: list[str],
    output_columns: list[str],
    size: int,
) -> np.ndarray:
    canonical = rows[rows["probe_family"] == 2]
    probes = canonical[rhs_columns[:size]].to_numpy(dtype=np.float64)
    actions = canonical[output_columns[:size]].to_numpy(dtype=np.float64)
    if len(canonical) != size or np.linalg.matrix_rank(probes) != size:
        raise ValueError(
            f"Need exactly {size} independent canonical probes, found "
            f"{len(canonical)}"
        )
    matrix = np.linalg.solve(probes, actions).T
    matrix = 0.5 * (matrix + matrix.T)
    return matrix.astype(np.float32)


def load_patch_operators(paths: list[Path]) -> tuple[list[PatchOperator], int, int]:
    records: list[PatchOperator] = []
    expected_local = 0
    expected_skeleton = 0
    for source_index, path in enumerate(paths):
        frame = pd.read_csv(path, dtype=np.float32)
        geometry_columns = [
            name for name in frame if name.startswith("geom_")
        ]
        rhs_columns = [
            name for name in frame if name.startswith("dirichlet_")
        ]
        output_columns = [
            name for name in frame if name.startswith("output_")
        ]
        if not geometry_columns or len(geometry_columns) % 4:
            raise ValueError(f"{path}: invalid geometry columns")
        if not rhs_columns or len(rhs_columns) != len(output_columns):
            raise ValueError(f"{path}: invalid action columns")
        n_local = len(geometry_columns) // 4
        n_skeleton = len(rhs_columns)
        if expected_local and (
            n_local != expected_local or n_skeleton != expected_skeleton
        ):
            raise ValueError(
                "Every input CSV must use identical padded operator shapes"
            )
        expected_local, expected_skeleton = n_local, n_skeleton

        group_columns = ["patch_id"]
        if "geometry_id" in frame:
            group_columns.insert(0, "geometry_id")
        for group_key, rows in frame.groupby(group_columns, sort=True):
            representative = rows.iloc[0]
            size = int(representative["n_skeleton"])
            matrix = exact_matrix(
                rows, rhs_columns, output_columns, size
            )
            padded_matrix = np.zeros(
                (n_skeleton, n_skeleton), dtype=np.float32
            )
            padded_matrix[:size, :size] = matrix

            eigenvalues, eigenvectors = np.linalg.eigh(
                matrix.astype(np.float64)
            )
            threshold = max(
                1.0e-12,
                1.0e-9 * float(np.max(np.abs(eigenvalues))),
            )
            positive = eigenvalues > threshold
            if not np.any(positive):
                raise ValueError(f"{path}, patch {group_key}: no positive modes")
            padded_values = np.zeros(n_skeleton, dtype=np.float32)
            padded_vectors = np.zeros(
                (n_skeleton, n_skeleton), dtype=np.float32
            )
            positive_values = eigenvalues[positive].astype(np.float32)
            positive_vectors = eigenvectors[:, positive].astype(np.float32)
            positive_count = len(positive_values)
            padded_values[:positive_count] = positive_values
            padded_vectors[:size, :positive_count] = positive_vectors

            geometry = representative[geometry_columns].to_numpy(
                dtype=np.float32
            ).reshape(n_local, 4)
            key = (
                f"source{source_index}:"
                f"{path.stem}:geometry{int(representative.get('geometry_id', 0))}:"
                f"patch{int(representative['patch_id'])}"
            )
            records.append(
                PatchOperator(
                    key=key,
                    geometry=geometry,
                    matrix=padded_matrix,
                    eigenvalues=padded_values,
                    eigenvectors=padded_vectors,
                    skeleton_size=size,
                    positive_size=positive_count,
                )
            )
        del frame
    return records, expected_local, expected_skeleton


def split_records(
    records: list[PatchOperator], seed: int
) -> tuple[list[PatchOperator], list[PatchOperator], list[PatchOperator]]:
    if len(records) < 10:
        raise ValueError("At least ten patch operators are required")
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(0.1 * len(shuffled)))
    test_size = max(1, round(0.1 * len(shuffled)))
    training_size = len(shuffled) - validation_size - test_size
    return (
        shuffled[:training_size],
        shuffled[training_size : training_size + validation_size],
        shuffled[training_size + validation_size :],
    )


def learned_matrix(
    model: GeometryConditionedLinearOperator, geometry: Tensor
) -> Tensor:
    padding_mask = geometry[..., 3].eq(0)
    normalized = model.input_normalizer(geometry)
    query, scaled_key, epsilon = model.compute_operator_components(
        normalized, padding_mask
    )
    matrix = torch.bmm(query, scaled_key.transpose(-2, -1))
    identity = torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    ).unsqueeze(0)
    return matrix + epsilon * identity


def operator_loss(
    prediction: Tensor,
    target: Tensor,
    eigenvalues: Tensor,
    eigenvectors: Tensor,
    skeleton_sizes: Tensor,
    positive_sizes: Tensor,
    spectral_weight: float,
    worst_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    frobenius_losses: list[Tensor] = []
    spectral_losses: list[Tensor] = []
    worst_losses: list[Tensor] = []
    for batch_index, count_tensor in enumerate(positive_sizes):
        count = int(count_tensor.item())
        skeleton_size = int(skeleton_sizes[batch_index].item())
        exact = target[batch_index, :skeleton_size, :skeleton_size]
        learned = prediction[
            batch_index, :skeleton_size, :skeleton_size
        ]
        relative_frobenius = torch.linalg.matrix_norm(
            learned - exact
        ) / torch.linalg.matrix_norm(exact).clamp_min(1.0e-12)
        frobenius_losses.append(relative_frobenius)

        basis = eigenvectors[
            batch_index, :skeleton_size, :count
        ]
        positive_exact = torch.diag(eigenvalues[batch_index, :count])
        projected_learned = basis.transpose(0, 1) @ learned @ basis
        projected_learned = 0.5 * (
            projected_learned + projected_learned.transpose(0, 1)
        )
        cholesky = torch.linalg.cholesky(projected_learned)
        left = torch.linalg.solve_triangular(
            cholesky, positive_exact, upper=False
        )
        whitened = torch.linalg.solve_triangular(
            cholesky, left.transpose(0, 1), upper=False
        ).transpose(0, 1)
        whitened = 0.5 * (whitened + whitened.transpose(0, 1))
        generalized = torch.linalg.eigvalsh(whitened).clamp_min(1.0e-8)
        logarithms = torch.log(generalized)
        spectral_losses.append(logarithms.square().mean())
        worst_losses.append(logarithms.abs().amax().square())

    frobenius = torch.stack(frobenius_losses).mean()
    spectral = torch.stack(spectral_losses).mean()
    worst = torch.stack(worst_losses).mean()
    total = (
        frobenius
        + spectral_weight * spectral
        + worst_weight * worst
    )
    return total, {
        "frobenius": float(frobenius.detach()),
        "spectral": float(spectral.detach()),
        "worst": float(worst.detach()),
    }


def run_epoch(
    model: GeometryConditionedLinearOperator,
    loader: DataLoader,
    device: torch.device,
    spectral_weight: float,
    worst_weight: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "frobenius": 0.0, "spectral": 0.0, "worst": 0.0}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for (
            geometry, target, values, vectors,
            skeleton_sizes, positive_sizes, _keys,
        ) in loader:
            geometry = geometry.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            vectors = vectors.to(device, non_blocking=True)
            skeleton_sizes = skeleton_sizes.to(device, non_blocking=True)
            positive_sizes = positive_sizes.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = learned_matrix(model, geometry)
            loss, components = operator_loss(
                prediction, target, values, vectors,
                skeleton_sizes, positive_sizes,
                spectral_weight, worst_weight,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )
                optimizer.step()
            batch_size = geometry.shape[0]
            samples += batch_size
            totals["loss"] += float(loss.detach()) * batch_size
            for name, value in components.items():
                totals[name] += value * batch_size
    return {name: value / samples for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--spectral-weight", type=float, default=0.25)
    parser.add_argument("--worst-weight", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Initialize from a compatible operator checkpoint",
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help=(
            "Fit every supplied patch operator for a deployment model. "
            "Validation/test metrics then describe the fitted distribution "
            "and are not held-out estimates."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.layers, args.heads) <= 0:
        raise ValueError("Epochs, batch size, layers, and heads must be positive")
    if min(
        args.learning_rate, args.spectral_weight,
        args.worst_weight, args.gradient_clip,
    ) < 0:
        raise ValueError("Optimization weights must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    paths = [path.expanduser().resolve() for path in args.data]
    records, n_control_points, n_dofs = load_patch_operators(paths)
    if args.fit_all:
        training = records.copy()
        validation = records.copy()
        test = records.copy()
    else:
        training, validation, test = split_records(records, args.seed)
    hidden_dim = args.hidden_dim or 2 * n_dofs
    if hidden_dim % args.heads:
        raise ValueError("hidden_dim must be divisible by heads")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryConditionedLinearOperator(
        n_dofs=n_dofs,
        n_control_points=n_control_points,
        hidden_dim=hidden_dim,
        n_heads=args.heads,
        n_heads_A=1,
        n_layers_geo=args.layers,
        dropout_rate=0.0,
        activation_str="gelu",
        zero_mean=False,
        example_input_normalizer=PatchInvariantGeometryNormalizer(),
        example_output_normalizer=StaticNormalizer(n_dofs),
        device=device,
    )
    if args.checkpoint is not None:
        initial = torch.load(
            args.checkpoint.expanduser().resolve(),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(initial["state_dict"], strict=True)
    loaders = {
        "training": DataLoader(
            PatchOperatorDataset(training), batch_size=args.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed),
        ),
        "validation": DataLoader(
            PatchOperatorDataset(validation), batch_size=args.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        ),
        "test": DataLoader(
            PatchOperatorDataset(test), batch_size=args.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        ),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        args.output
        / "model_GeometryConditionedLinearOperator_default_YetiSchurTransformer.tar"
    )
    history_path = args.output / "history.csv"
    config_path = args.output / "config.json"
    config = {
        "training_properties": {
            "training_samples": len(training),
            "val_samples": len(validation),
            "test_samples": len(test),
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "scheduler_step": args.epochs,
            "scheduler_gamma": 1.0,
            "p": 2,
            "relative_loss": True,
            "operator_matrix_loss": True,
            "spectral_weight": args.spectral_weight,
            "worst_weight": args.worst_weight,
            "patch_grouped_split": not args.fit_all,
            "deployment_fit_all": args.fit_all,
        },
        "iganet_transformer_architecture": {
            "problem_dim": 1,
            "n_dofs": n_dofs,
            "n_control_points": n_control_points,
            "hidden_dim": hidden_dim,
            "n_heads_A": 1,
            "n_heads": args.heads,
            "head_dim": hidden_dim // args.heads,
            "n_layers_geo": args.layers,
            "dropout_rate": 0.0,
            "activation_str": "gelu",
            "zero_mean": False,
            "internal_normalization": True,
            "retrain": args.seed,
            "geometry_normalization": "patch",
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    best_validation = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch", "learning_rate",
                "train_loss", "train_frobenius", "train_spectral", "train_worst",
                "validation_loss", "validation_frobenius",
                "validation_spectral", "validation_worst",
            ],
        )
        writer.writeheader()
        for epoch in range(args.epochs):
            train_metrics = run_epoch(
                model, loaders["training"], device,
                args.spectral_weight, args.worst_weight,
                optimizer, args.gradient_clip,
            )
            validation_metrics = run_epoch(
                model, loaders["validation"], device,
                args.spectral_weight, args.worst_weight,
                None, args.gradient_clip,
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            writer.writerow(row)
            history_file.flush()

            if validation_metrics["loss"] < best_validation:
                best_validation = validation_metrics["loss"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": best_epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": best_validation,
                        "scheduler": scheduler.state_dict(),
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
            scheduler.step()
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"epoch {epoch + 1:4d}: "
                    f"train={train_metrics['loss']:.6g}, "
                    f"validation={validation_metrics['loss']:.6g}, "
                    f"best={best_validation:.6g}@{best_epoch}",
                    flush=True,
                )
            if (
                args.patience > 0
                and epochs_without_improvement >= args.patience
            ):
                print(f"Early stopping after epoch {epoch + 1}", flush=True)
                break

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test_metrics = run_epoch(
        model, loaders["test"], device,
        args.spectral_weight, args.worst_weight,
        None, args.gradient_clip,
    )
    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in paths
        ],
        "patch_operators": len(records),
        "split_sizes": {
            "training": len(training),
            "validation": len(validation),
            "test": len(test),
        },
        "deployment_fit_all": args.fit_all,
        "split_keys": {
            "training": [record.key for record in training],
            "validation": [record.key for record in validation],
            "test": [record.key for record in test],
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "seed": args.seed,
        "device": str(device),
        "initial_checkpoint": (
            str(args.checkpoint.expanduser().resolve())
            if args.checkpoint is not None else None
        ),
        "initial_checkpoint_sha256": (
            sha256(args.checkpoint.expanduser().resolve())
            if args.checkpoint is not None else None
        ),
        "command_options": vars(args) | {
            "data": [str(path) for path in args.data],
            "output": str(args.output),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"best epoch {best_epoch}, validation={best_validation:.6g}, "
        f"test={test_metrics['loss']:.6g}; wrote {checkpoint_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
