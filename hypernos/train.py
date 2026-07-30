"""Train the local multi-patch AFIETI transformer with installed HyperNOs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from hypernos.loss_fun import lpLoss
from hypernos.train import train_fixed_model
from torch import nn

if __package__:
    from .dataset import YetiSchurTransformer, resolve_dataset_path
    from .model import GeometryConditionedLinearOperator
else:  # Direct execution: ``python train.py``.
    from dataset import YetiSchurTransformer, resolve_dataset_path
    from model import GeometryConditionedLinearOperator


HERE = Path(__file__).resolve().parent
config_mode = "default"
DEFAULT_CONFIG = HERE / f"{config_mode}_config.json"
DEFAULT_DATA = HERE / "data" / "mp_afieti" / "yeti_dataset.csv"
DEFAULT_OUTPUT = HERE / "runs" / f"loss_l2_mode_{config_mode}"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and flatten the local training/architecture configuration."""

    with Path(path).expanduser().open(encoding="utf-8") as stream:
        document = json.load(stream)
    try:
        training = document["training_properties"]
        architecture = document["iganet_transformer_architecture"]
    except KeyError as error:
        raise ValueError(f"Missing configuration section: {error.args[0]}") from error
    return {**training, **architecture}


def train_iganet_transformer(
    filename: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    output_folder: str | Path = DEFAULT_OUTPUT,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    training_samples: int | None = None,
    validation_samples: int | None = None,
    test_samples: int | None = None,
    auto_shapes: bool = False,
    full_rank: bool = False,
    checkpoint: str | Path | None = None,
) -> None:
    """Run the same fixed-model experiment as the original HyperNOs example."""

    config = load_config(config_path)
    if epochs is not None:
        config["epochs"] = epochs
    if batch_size is not None:
        config["batch_size"] = batch_size
    if training_samples is not None:
        config["training_samples"] = training_samples
    if validation_samples is not None:
        config["val_samples"] = validation_samples
    if test_samples is not None:
        config["test_samples"] = test_samples

    data_path = resolve_dataset_path(filename)
    if auto_shapes:
        with data_path.open(newline="", encoding="utf-8") as stream:
            columns = next(csv.reader(stream))
        n_geometry_values = sum(name.startswith("geom_") for name in columns)
        n_dofs = sum(name.startswith("dirichlet_") for name in columns)
        if n_geometry_values == 0 or n_geometry_values % 4 or n_dofs == 0:
            raise ValueError(f"Cannot infer valid model shapes from {data_path}")
        config["n_control_points"] = n_geometry_values // 4
        config["n_dofs"] = n_dofs
        if full_rank:
            config["hidden_dim"] = n_dofs
            requested_heads = int(config["n_heads"])
            config["n_heads"] = next(
                heads
                for heads in range(min(requested_heads, n_dofs), 0, -1)
                if n_dofs % heads == 0
            )
        print(
            "Inferred dataset shapes: "
            f"n_control_points={config['n_control_points']}, "
            f"n_dofs={config['n_dofs']}, hidden_dim={config['hidden_dim']}"
        )

    positive_parameters = (
        "epochs",
        "batch_size",
        "training_samples",
        "val_samples",
        "test_samples",
    )
    for name in positive_parameters:
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_fixed_model builds the dataset before the model.  Retaining that
    # object here lets the model reuse its fitted normalizers without reading
    # the wide CSV a second time.
    build_context: dict[str, YetiSchurTransformer] = {}

    def dataset_builder(current_config: dict[str, Any]) -> YetiSchurTransformer:
        dataset = YetiSchurTransformer(
            filename=data_path,
            network_properties={"retrain": current_config["retrain"]},
            batch_size=current_config["batch_size"],
            training_samples=current_config["training_samples"],
            validation_samples=current_config["val_samples"],
            test_samples=current_config["test_samples"],
            geometry_normalization=current_config.get(
                "geometry_normalization", "pointwise"
            ),
        )
        build_context["dataset"] = dataset
        return dataset

    def model_builder(current_config: dict[str, Any]) -> nn.Module:
        try:
            dataset = build_context["dataset"]
        except KeyError as error:
            raise RuntimeError("The dataset must be built before the model") from error

        if dataset.s_rhs != current_config["n_dofs"]:
            raise ValueError(
                f"Config n_dofs={current_config['n_dofs']} does not match "
                f"the dataset RHS width {dataset.s_rhs}"
            )
        if dataset.s_geo != current_config["n_control_points"]:
            raise ValueError(
                f"Config n_control_points={current_config['n_control_points']} "
                f"does not match the dataset geometry width {dataset.s_geo}"
            )

        model = GeometryConditionedLinearOperator(
            n_dofs=current_config["n_dofs"],
            n_control_points=current_config["n_control_points"],
            hidden_dim=current_config["hidden_dim"],
            n_heads=current_config["n_heads"],
            n_heads_A=current_config["n_heads_A"],
            n_layers_geo=current_config["n_layers_geo"],
            dropout_rate=current_config["dropout_rate"],
            activation_str=current_config["activation_str"],
            zero_mean=current_config["zero_mean"],
            example_input_normalizer=(
                dataset.input_normalizer
                if current_config["internal_normalization"]
                else None
            ),
            example_output_normalizer=(
                dataset.output_normalizer
                if current_config["internal_normalization"]
                else None
            ),
            device=device,
        )
        if checkpoint is not None:
            saved = torch.load(
                Path(checkpoint).expanduser(), map_location=device, weights_only=True
            )
            model.load_state_dict(saved["state_dict"], strict=True)
        return model

    if bool(config.get("relative_loss", False)):
        loss_name = "relative_l2"

        def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            numerator = torch.linalg.vector_norm(
                prediction - target, ord=config["p"], dim=1
            )
            denominator = torch.linalg.vector_norm(target, ord=config["p"], dim=1)
            return torch.mean(numerator / (denominator + 1.0e-8))

    else:
        loss_name = "l2"
        loss_fn = lpLoss(config["p"], True)

    experiment_name = (
        f"IgaNet_transformer/mp_afieti/loss_{loss_name}_mode_{config_mode}"
    )

    output_path = Path(output_folder).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "norm_info.txt").write_text(
        f"Norm used during the training:\n{loss_name}\n", encoding="utf-8"
    )

    train_fixed_model(
        config,
        model_builder,
        dataset_builder,
        loss_fn,
        experiment_name,
        full_validation=False,
        output_folder=str(output_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to yeti_dataset.csv",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, help="Override config epochs")
    parser.add_argument("--batch-size", type=int, help="Override config batch size")
    parser.add_argument(
        "--training-samples", type=int, help="Override config training split size"
    )
    parser.add_argument(
        "--validation-samples", type=int, help="Override validation split size"
    )
    parser.add_argument("--test-samples", type=int, help="Override test split size")
    parser.add_argument(
        "--auto-shapes",
        action="store_true",
        help="Infer n_control_points and n_dofs from the CSV header",
    )
    parser.add_argument(
        "--full-rank",
        action="store_true",
        help="With --auto-shapes, set hidden_dim equal to n_dofs",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Continue from a saved .tar checkpoint (optimizer is restarted)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_iganet_transformer(
        filename=arguments.data,
        config_path=arguments.config,
        output_folder=arguments.output,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        training_samples=arguments.training_samples,
        validation_samples=arguments.validation_samples,
        test_samples=arguments.test_samples,
        auto_shapes=arguments.auto_shapes,
        full_rank=arguments.full_rank,
        checkpoint=arguments.checkpoint,
    )
