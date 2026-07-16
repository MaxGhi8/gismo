"""Ray Tune optimization for the standalone multi-patch AFIETI model.

The model and dataset come from this directory.  Only the generic loss and
Ray Tune orchestration are imported from the separately installed HyperNOs
package.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from ray import cloudpickle, tune

from hypernos.loss_fun import lpLoss
from hypernos.tune import tune_hyperparameters

if __package__:
    from .dataset import YetiSchurTransformer, resolve_dataset_path
    from .model import GeometryConditionedLinearOperator
    from .train import DEFAULT_CONFIG, DEFAULT_DATA, load_config
else:  # Direct execution: ``python ray_tune.py``.
    from dataset import YetiSchurTransformer, resolve_dataset_path
    from model import GeometryConditionedLinearOperator
    from train import DEFAULT_CONFIG, DEFAULT_DATA, load_config


# The layout is sampled jointly so every Ray trial satisfies the hard
# MultiheadAttention constraint hidden_dim % n_heads == 0.  Encoding it as a
# string also keeps the space compatible with HyperOpt's categorical sampler.
ATTENTION_LAYOUTS = (
    "16x1",
    "16x2",
    "20x1",
    "20x2",
    "20x4",
    "24x2",
    "24x3",
    "32x2",
    "32x4",
    "40x4",
    "40x5",
    "48x4",
    "48x6",
    "64x4",
    "64x8",
)


def decode_attention_layout(layout: str) -> tuple[int, int]:
    """Return ``(hidden_dim, n_heads)`` from a validated ``WIDTHxHEADS`` value."""

    try:
        hidden_dim, n_heads = (int(value) for value in layout.split("x", 1))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid attention layout: {layout!r}") from error
    if hidden_dim <= 0 or n_heads <= 0 or hidden_dim % n_heads != 0:
        raise ValueError(
            f"Attention layout {layout!r} must have positive values and a "
            "hidden dimension divisible by the head count"
        )
    return hidden_dim, n_heads


def default_attention_layout(config: dict[str, Any]) -> str:
    layout = f"{config['hidden_dim']}x{config['n_heads']}"
    # A custom config is allowed as long as its default is structurally valid.
    decode_attention_layout(layout)
    return layout


def build_search_space(
    base_config: dict[str, Any], max_epochs: int
) -> dict[str, Any]:
    """Combine meaningful search domains with all non-tuned fixed parameters."""

    scheduler_steps = sorted(
        {
            int(base_config["scheduler_step"]),
            max(1, max_epochs // 4),
            max(1, max_epochs // 2),
            max_epochs,
        }
    )
    layouts = list(ATTENTION_LAYOUTS)
    default_layout = default_attention_layout(base_config)
    if default_layout not in layouts:
        layouts.append(default_layout)

    search_domains = {
        # Log scales are appropriate for optimizer magnitudes.
        "learning_rate": tune.loguniform(5.0e-5, 3.0e-3),
        "weight_decay": tune.loguniform(1.0e-7, 1.0e-3),
        "scheduler_step": tune.choice(scheduler_steps),
        "scheduler_gamma": tune.choice([0.90, 0.95, 0.98, 1.0]),
        # Network-capacity and regularization parameters.
        "attention_layout": tune.choice(layouts),
        "n_layers_geo": tune.choice([1, 2, 3, 4]),
        "dropout_rate": tune.choice([0.0, 0.05, 0.10, 0.15]),
        "activation_str": tune.choice(["gelu", "relu"]),
    }

    fixed = dict(base_config)
    for name in search_domains:
        fixed.pop(name, None)
    # These values are represented jointly by attention_layout.  head_dim was
    # present in the historical JSON but is not consumed by the architecture.
    for name in ("hidden_dim", "n_heads", "head_dim"):
        fixed.pop(name, None)

    # Keep batch_size fixed. HyperNOs' Ray validation currently divides an
    # already batch-averaged lpLoss by the sample count, so comparing trials
    # with different batch sizes would bias the reported metric.
    # zero_mean is also fixed because the reproduced SPD forward path does not
    # currently apply its post_processing member.

    return {**search_domains, **fixed}


def build_default_trial(base_config: dict[str, Any]) -> dict[str, Any]:
    """Translate the original best configuration into the joint search schema."""

    default_trial = dict(base_config)
    for name in ("hidden_dim", "n_heads", "head_dim"):
        default_trial.pop(name, None)
    default_trial["attention_layout"] = default_attention_layout(base_config)
    return default_trial


def run_hyperparameter_optimization(
    filename: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    num_samples: int = 40,
    max_epochs: int = 300,
    grace_period: int = 50,
    reduction_factor: int = 4,
    cpus_per_trial: float = 2.0,
    gpus_per_trial: float | None = None,
):
    """Optimize the local architecture and return Ray's best trial result."""

    for name, value in {
        "num_samples": num_samples,
        "max_epochs": max_epochs,
        "grace_period": grace_period,
        "reduction_factor": reduction_factor,
        "cpus_per_trial": cpus_per_trial,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if grace_period > max_epochs:
        raise ValueError("grace_period cannot exceed max_epochs")

    data_path = resolve_dataset_path(filename)
    base_config = load_config(config_path)
    base_config["epochs"] = max_epochs
    config_space = build_search_space(base_config, max_epochs)
    default_trial = build_default_trial(base_config)
    if set(default_trial) != set(config_space):
        raise RuntimeError("Default trial and Ray search-space keys do not agree")

    # Fit the normalizers once on the driver, retain only their small statistics,
    # and let every trial load its own shuffled tensors.  Capturing the complete
    # 277 MB dataset inside model_builder would place a very large object in
    # Ray's object store.
    normalization_dataset = YetiSchurTransformer(
        filename=data_path,
        network_properties={"retrain": base_config["retrain"]},
        batch_size=base_config["batch_size"],
        training_samples=base_config["training_samples"],
        validation_samples=base_config["val_samples"],
        test_samples=base_config["test_samples"],
    )
    input_normalizer = copy.deepcopy(normalization_dataset.input_normalizer).cpu()
    output_normalizer = copy.deepcopy(normalization_dataset.output_normalizer).cpu()
    del normalization_dataset

    def dataset_builder(config: dict[str, Any]) -> YetiSchurTransformer:
        return YetiSchurTransformer(
            filename=data_path,
            network_properties={"retrain": config["retrain"]},
            batch_size=config["batch_size"],
            training_samples=config["training_samples"],
            validation_samples=config["val_samples"],
            test_samples=config["test_samples"],
        )

    def model_builder(config: dict[str, Any]) -> GeometryConditionedLinearOperator:
        hidden_dim, n_heads = decode_attention_layout(config["attention_layout"])
        # Evaluate CUDA visibility inside the Ray worker. Ray masks devices on a
        # per-trial basis according to the requested GPU resources.
        trial_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return GeometryConditionedLinearOperator(
            n_dofs=config["n_dofs"],
            n_control_points=config["n_control_points"],
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            # n_heads_A remains fixed at one: in this SPD implementation the
            # projected heads are averaged, so increasing it only introduces a
            # redundant linear parameterization rather than a new attention head.
            n_heads_A=config["n_heads_A"],
            n_layers_geo=config["n_layers_geo"],
            dropout_rate=config["dropout_rate"],
            activation_str=config["activation_str"],
            zero_mean=config["zero_mean"],
            example_input_normalizer=(
                copy.deepcopy(input_normalizer)
                if config["internal_normalization"]
                else None
            ),
            example_output_normalizer=(
                copy.deepcopy(output_normalizer)
                if config["internal_normalization"]
                else None
            ),
            device=trial_device,
        )

    loss_fn = lpLoss(base_config["p"], True)
    if gpus_per_trial is None:
        gpus_per_trial = 1.0 if torch.cuda.is_available() else 0.0
    if gpus_per_trial < 0:
        raise ValueError("gpus_per_trial cannot be negative")

    # A copied standalone folder is not an installed Python package on every
    # Ray worker. Serialize its two local modules by value so the closures do
    # not depend on a worker-side import path.
    local_modules = {
        sys.modules[YetiSchurTransformer.__module__],
        sys.modules[GeometryConditionedLinearOperator.__module__],
    }
    for module in local_modules:
        cloudpickle.register_pickle_by_value(module)
    try:
        best_result = tune_hyperparameters(
            config_space,
            model_builder,
            dataset_builder,
            loss_fn,
            default_hyper_params=[default_trial],
            num_samples=num_samples,
            max_epochs=max_epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            # The HyperNOs API calls these runs_per_*, but they are Ray resources
            # reserved by each individual trial.
            runs_per_cpu=cpus_per_trial,
            runs_per_gpu=gpus_per_trial,
            checkpoint_freq=max_epochs + 1,
        )
    finally:
        for module in local_modules:
            cloudpickle.unregister_pickle_by_value(module)

    best_config = dict(best_result.config)
    hidden_dim, n_heads = decode_attention_layout(
        best_config.pop("attention_layout")
    )
    best_config.update(
        {
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "head_dim": hidden_dim // n_heads,
        }
    )
    print("Best expanded hyperparameters:", best_config)
    return best_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--grace-period", type=int, default=50)
    parser.add_argument("--reduction-factor", type=int, default=4)
    parser.add_argument("--cpus-per-trial", type=float, default=2.0)
    parser.add_argument(
        "--gpus-per-trial",
        type=float,
        default=None,
        help="Defaults to 1 with CUDA and 0 without CUDA",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_hyperparameter_optimization(
        filename=arguments.data,
        config_path=arguments.config,
        num_samples=arguments.num_samples,
        max_epochs=arguments.max_epochs,
        grace_period=arguments.grace_period,
        reduction_factor=arguments.reduction_factor,
        cpus_per_trial=arguments.cpus_per_trial,
        gpus_per_trial=arguments.gpus_per_trial,
    )
