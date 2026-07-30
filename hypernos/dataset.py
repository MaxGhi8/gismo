"""Local data pipeline for the multi-patch AFIETI transformer example.

This module deliberately does not import ``hypernos.datasets``.  It can live
next to the training script in a project that consumes HyperNOs as an installed
dependency.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .model import PatchInvariantGeometryNormalizer
else:
    from model import PatchInvariantGeometryNormalizer


class UnitGaussianNormalizer(nn.Module):
    """Point-wise Gaussian normalization fitted along the sample dimension.

    The statistics are registered as buffers.  Consequently, when this object
    is attached to a model they move with the model and are included in its
    ``state_dict``.
    """

    def __init__(self, values: Tensor, eps: float = 1.0e-5) -> None:
        super().__init__()
        if not torch.isfinite(values).all():
            raise ValueError("Cannot fit a normalizer to NaN or infinite values")

        mean = values.mean(dim=0)
        std = values.std(dim=0)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("The fitted normalization statistics are not finite")

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("eps", torch.tensor(eps, dtype=values.dtype))

    def encode(self, values: Tensor) -> Tensor:
        return (values - self.mean) / (self.std + self.eps)

    def decode(self, values: Tensor) -> Tensor:
        return values * (self.std + self.eps) + self.mean

    def forward(self, values: Tensor) -> Tensor:
        return self.encode(values)


class MaskedChannelGaussianNormalizer(nn.Module):
    """Normalize geometry features without assuming point-index alignment.

    Multi-geometry datasets pad every patch to a common number of control
    points.  A control point at array position ``i`` does not, in general,
    represent the same geometric location in two unrelated patches.  We
    therefore fit one mean and standard deviation per feature channel over all
    *valid* control points.  Rows whose NURBS weight is zero are padding and
    remain zero after normalization.
    """

    def __init__(self, values: Tensor, eps: float = 1.0e-5) -> None:
        super().__init__()
        if values.ndim != 3 or values.shape[-1] != 4:
            raise ValueError("Geometry must have shape (samples, points, 4)")
        valid = values[..., 3].ne(0)
        samples = values[valid]
        if samples.numel() == 0 or not torch.isfinite(samples).all():
            raise ValueError("Cannot fit a channel normalizer to empty/non-finite data")
        mean = samples.mean(dim=0, keepdim=True)
        std = samples.std(dim=0, keepdim=True)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("The fitted channel statistics are not finite")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("eps", torch.tensor(eps, dtype=values.dtype))

    def forward(self, values: Tensor) -> Tensor:
        valid = values[..., 3:4].ne(0)
        encoded = (values - self.mean) / (self.std + self.eps)
        return torch.where(valid, encoded, torch.zeros_like(encoded))


class _YetiSchurTorchDataset(Dataset):
    """Return the nested input structure expected by HyperNOs training code."""

    def __init__(self, geometry: Tensor, rhs: Tensor, output: Tensor) -> None:
        if not (len(geometry) == len(rhs) == len(output)):
            raise ValueError("Geometry, right-hand side, and output sizes must agree")
        self.geometry = geometry
        self.rhs = rhs
        self.output = output

    def __len__(self) -> int:
        return self.output.shape[0]

    def __getitem__(self, index: int) -> tuple[tuple[Tensor, Tensor], Tensor]:
        return (self.rhs[index], self.geometry[index]), self.output[index]


def _seed_training(generator: torch.Generator, retrain: int) -> None:
    """Match the deterministic seeding convention used by HyperNOs examples."""

    if retrain <= 0:
        return

    os.environ["PYTHONHASHSEED"] = str(retrain)
    random.seed(retrain)
    np.random.seed(retrain)
    torch.manual_seed(retrain)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(retrain)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator.manual_seed(retrain)


def resolve_dataset_path(filename: str | Path) -> Path:
    """Resolve an explicit path or the repository's conventional data path."""

    requested = Path(filename).expanduser()
    candidates = [requested]

    if not requested.is_absolute():
        here = Path(__file__).resolve().parent
        candidates.extend(
            [
                Path.cwd() / requested,
                here / requested,
                here.parents[1] / "data" / "mp_afieti" / requested.name,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename!s}. Paths checked:\n  - {tried}\n"
        "Pass the CSV explicitly with --data /path/to/yeti_dataset.csv."
    )


class YetiSchurTransformer:
    """Load and split ``yeti_dataset.csv`` for the GCLO transformer.

    Expected columns are ``geom_*`` (four values per control point),
    ``dirichlet_*`` (right-hand side), and ``output_*`` (target solution).
    Rows are shuffled before splitting because the source CSV is grouped by
    patch.
    """

    def __init__(
        self,
        filename: str | Path,
        network_properties: dict,
        batch_size: int,
        training_samples: int,
        validation_samples: int = 2_000,
        test_samples: int = 2_000,
        shuffle_seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        geometry_normalization: str = "pointwise",
    ) -> None:
        if min(training_samples, validation_samples, test_samples) <= 0:
            raise ValueError("All split sizes must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        generator = torch.Generator(device="cpu")
        _seed_training(generator, int(network_properties["retrain"]))

        self.data_path = resolve_dataset_path(filename)
        print(f"Loading AFIETI data from {self.data_path}")

        # Reading directly as float32 substantially lowers the peak memory use
        # of this wide (roughly 1,300-column) CSV.
        frame = pd.read_csv(self.data_path, dtype=np.float32)
        geometry_columns = [name for name in frame if name.startswith("geom_")]
        rhs_columns = [name for name in frame if name.startswith("dirichlet_")]
        output_columns = [name for name in frame if name.startswith("output_")]
        self._validate_columns(geometry_columns, rhs_columns, output_columns)

        required_samples = training_samples + validation_samples + test_samples
        if len(frame) < required_samples:
            raise ValueError(
                f"Dataset has {len(frame)} rows, but the requested splits need "
                f"{required_samples}"
            )

        n_control_points = len(geometry_columns) // 4
        geometry = torch.from_numpy(
            frame[geometry_columns].to_numpy(dtype=np.float32, copy=True)
        ).reshape(len(frame), n_control_points, 4)
        rhs = torch.from_numpy(
            frame[rhs_columns].to_numpy(dtype=np.float32, copy=True)
        )
        output = torch.from_numpy(
            frame[output_columns].to_numpy(dtype=np.float32, copy=True)
        )

        permutation = torch.randperm(
            len(frame), generator=torch.Generator().manual_seed(shuffle_seed)
        )
        geometry = geometry[permutation]
        rhs = rhs[permutation]
        output = output[permutation]

        train_end = training_samples
        validation_end = train_end + validation_samples
        test_end = validation_end + test_samples

        geometry_train = geometry[:train_end]
        rhs_train = rhs[:train_end]
        output_train = output[:train_end]
        geometry_validation = geometry[train_end:validation_end]
        rhs_validation = rhs[train_end:validation_end]
        output_validation = output[train_end:validation_end]
        geometry_test = geometry[validation_end:test_end]
        rhs_test = rhs[validation_end:test_end]
        output_test = output[validation_end:test_end]

        if geometry_normalization == "pointwise":
            self.input_normalizer = UnitGaussianNormalizer(geometry_train)
        elif geometry_normalization == "channel":
            self.input_normalizer = MaskedChannelGaussianNormalizer(geometry_train)
        elif geometry_normalization == "patch":
            self.input_normalizer = PatchInvariantGeometryNormalizer()
        else:
            raise ValueError(
                "geometry_normalization must be pointwise, channel, or patch; got "
                f"{geometry_normalization!r}"
            )
        self.output_normalizer = UnitGaussianNormalizer(output_train)

        loader_options = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "generator": generator,
            "persistent_workers": num_workers > 0,
        }
        self.train_loader = DataLoader(
            _YetiSchurTorchDataset(geometry_train, rhs_train, output_train),
            shuffle=True,
            **loader_options,
        )
        self.val_loader = DataLoader(
            _YetiSchurTorchDataset(
                geometry_validation, rhs_validation, output_validation
            ),
            shuffle=False,
            **loader_options,
        )
        self.test_loader = DataLoader(
            _YetiSchurTorchDataset(geometry_test, rhs_test, output_test),
            shuffle=False,
            **loader_options,
        )

        self.s_geo = geometry.shape[1]
        self.s_rhs = rhs.shape[1]
        self.s_out = output.shape[1]

    @staticmethod
    def _validate_columns(
        geometry_columns: list[str],
        rhs_columns: list[str],
        output_columns: list[str],
    ) -> None:
        if not geometry_columns or len(geometry_columns) % 4 != 0:
            raise ValueError(
                "The CSV must contain a non-empty multiple of four geom_* columns"
            )
        if not rhs_columns or not output_columns:
            raise ValueError("The CSV must contain dirichlet_* and output_* columns")
        if len(rhs_columns) != len(output_columns):
            raise ValueError("dirichlet_* and output_* column counts must match")
