"""Export a trained SPD Schur-operator checkpoint with cacheable ONNX outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

from model import (
    GeometryConditionedLinearOperator,
    PatchInvariantGeometryNormalizer,
)


class _CheckpointNormalizer(nn.Module):
    """Shape-compatible placeholder whose buffers are replaced by a checkpoint."""

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


class _CacheableExport(nn.Module):
    """Expose the factorization used by the C++ cached preconditioner."""

    def __init__(self, model: GeometryConditionedLinearOperator) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, rhs: Tensor, geometry: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        padding_mask = geometry[..., 3].eq(0)
        normalized_geometry = self.model.input_normalizer(geometry)
        query, scaled_key, epsilon = self.model.compute_operator_components(
            normalized_geometry, padding_mask
        )
        output = self.model.apply_operator(rhs, query, scaled_key, epsilon)
        return output, query, scaled_key, epsilon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("default_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.config.read_text(encoding="utf-8"))
    config = {
        **document["training_properties"],
        **document["iganet_transformer_architecture"],
    }

    normalization = config.get("geometry_normalization", "pointwise")
    channel_normalization = normalization == "channel"
    if normalization == "patch":
        input_normalizer = PatchInvariantGeometryNormalizer()
    else:
        input_shape = (
            (1, 4) if channel_normalization
            else (config["n_control_points"], 4)
        )
        input_normalizer = _CheckpointNormalizer(
            input_shape, preserve_padding=channel_normalization
        )
    output_normalizer = _CheckpointNormalizer((config["n_dofs"],))
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
        example_output_normalizer=output_normalizer,
        device="cpu",
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    export_model = _CacheableExport(model.eval()).eval()

    torch.manual_seed(0)
    rhs = torch.randn(1, config["n_dofs"], dtype=torch.float32)
    geometry = torch.randn(
        1, config["n_control_points"], 4, dtype=torch.float32
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        (rhs, geometry),
        args.output,
        input_names=["input", "obj.1"],
        output_names=["u", "Q", "K_scaled", "epsilon"],
        dynamic_axes={
            "input": {0: "patch_batch"},
            "obj.1": {0: "patch_batch"},
            "u": {0: "patch_batch"},
            "Q": {0: "patch_batch"},
            "K_scaled": {0: "patch_batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print(
        f"Exported epoch {checkpoint.get('epoch', 'unknown')} to {args.output} "
        f"with inputs {[node.name for node in onnx_model.graph.input]} and "
        f"outputs {[node.name for node in onnx_model.graph.output]}"
    )


if __name__ == "__main__":
    main()
