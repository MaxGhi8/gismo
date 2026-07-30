"""Local geometry-conditioned linear operator used by the AFIETI example."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def zero_mean_imposition(values: Tensor) -> Tensor:
    """Project every vector in a batch onto the zero-mean subspace."""

    return values - values.mean(dim=1, keepdim=True)


class PatchInvariantGeometryNormalizer(nn.Module):
    """Remove irrelevant patch translation and uniform physical scale.

    For the scalar two-dimensional Laplacian, translating a patch does not
    alter its stiffness operator, and a uniform physical dilation cancels
    between the Jacobian determinant and the two inverse Jacobians.  Encoding
    absolute position and diameter therefore makes the learning problem harder,
    especially after patch subdivision.  This normalizer centers the valid
    physical control points, divides all spatial coordinates by one isotropic
    RMS radius (so anisotropy is retained), and normalizes rational weights by
    their valid-patch mean.  Zero-weight padding remains exactly zero.
    """

    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != 4:
            raise ValueError("Geometry must have shape (batch, points, 4)")

        valid = values[..., 3:4].ne(0)
        valid_float = valid.to(values.dtype)
        count = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)

        coordinates = values[..., :3]
        centroid = (coordinates * valid_float).sum(dim=1, keepdim=True) / count
        centered = (coordinates - centroid) * valid_float
        radius = torch.sqrt(
            centered.square().sum(dim=-1, keepdim=True).sum(
                dim=1, keepdim=True
            )
            / count
        ).clamp_min(self.eps.to(values.dtype))
        normalized_coordinates = centered / radius

        weights = values[..., 3:4]
        mean_weight = (
            weights.abs().sum(dim=1, keepdim=True) / count
        ).clamp_min(self.eps.to(values.dtype))
        normalized_weights = torch.where(
            valid, weights / mean_weight, torch.zeros_like(weights)
        )
        return torch.cat((normalized_coordinates, normalized_weights), dim=-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the control-point sequence."""

    def __init__(
        self, hidden_dim: int, max_len: int = 5_000, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden_dim, 2)
            * (-math.log(10_000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_len, 1, hidden_dim)
        encoding[:, 0, 0::2] = torch.sin(position * divisor)
        cosine = torch.cos(position * divisor)
        encoding[:, 0, 1::2] = cosine[:, : encoding[:, 0, 1::2].shape[1]]
        self.register_buffer("pe", encoding)

    def forward(self, values: Tensor) -> Tensor:
        return self.dropout(values + self.pe[: values.shape[0]])


class GeometryEncoderResampler(nn.Module):
    """Encode control points and resample them at the solution DOFs."""

    def __init__(
        self,
        n_control_points_input: int,
        n_dofs_output: int,
        hidden_dim: int,
        n_layers_encoder: int,
        n_heads: int,
        dropout: float = 0.0,
        activation_str: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")

        self.input_projection = nn.Linear(4, hidden_dim)
        self.pos_encoder = PositionalEncoding(
            hidden_dim,
            max_len=n_control_points_input + 100,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation=activation_str,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers_encoder
        )

        self.target_queries = nn.Parameter(
            torch.randn(n_dofs_output, 1, hidden_dim)
        )
        self.resampling_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.target_queries, mean=0.0, std=0.02)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self, geometry: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        # Transformer modules in this model use (sequence, batch, feature).
        source = self.input_projection(geometry).permute(1, 0, 2)
        memory = self.transformer_encoder(
            self.pos_encoder(source), src_key_padding_mask=padding_mask
        )
        queries = self.target_queries.repeat(1, geometry.shape[0], 1)
        latent_geometry, _ = self.resampling_attn(
            queries, memory, memory, key_padding_mask=padding_mask
        )
        return latent_geometry.permute(1, 0, 2)


class GeometryConditionedLinearOperator(nn.Module):
    """Approximate ``u = A(g) f`` using a geometry-dependent SPD operator.

    ``A(g)`` has the efficient representation
    ``Q(g) @ K(g).T + epsilon * I`` and is never materialized as a dense matrix.
    The forward input is ``(rhs, geometry)`` so it matches the nested batches
    produced by :class:`dataset.YetiSchurTransformer`.
    """

    def __init__(
        self,
        n_dofs: int,
        n_control_points: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_heads_A: int = 1,
        n_layers_geo: int = 2,
        dropout_rate: float = 0.0,
        activation_str: str = "gelu",
        zero_mean: bool = True,
        example_input_normalizer: nn.Module | None = None,
        example_output_normalizer: nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if n_heads_A <= 0:
            raise ValueError("n_heads_A must be positive")

        self.n_dofs = n_dofs
        self.hidden_dim = hidden_dim
        self.n_heads_A = n_heads_A
        self.device = torch.device(device)
        self.input_normalizer = (
            nn.Identity()
            if example_input_normalizer is None
            else example_input_normalizer
        )
        self.geo_branch = GeometryEncoderResampler(
            n_control_points_input=n_control_points,
            n_dofs_output=n_dofs,
            hidden_dim=hidden_dim,
            n_layers_encoder=n_layers_geo,
            n_heads=n_heads,
            dropout=dropout_rate,
            activation_str=activation_str,
        )

        self.W_Q = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_heads_A)]
        )
        # Kept in the definition for checkpoint compatibility with the original
        # architecture.  The positive/SPD variant intentionally uses W_Q for K.
        self.W_K = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_heads_A)]
        )
        self.scale = 1.0 / math.sqrt(hidden_dim)
        self.output_denormalizer = (
            nn.Identity()
            if example_output_normalizer is None
            else example_output_normalizer
        )
        self.post_processing = zero_mean_imposition if zero_mean else nn.Identity()

        initial_epsilon = math.log(math.exp(0.1) - 1.0)
        self.raw_epsilon = nn.Parameter(torch.tensor(initial_epsilon))
        self.to(device)

    @property
    def epsilon(self) -> Tensor:
        return F.softplus(self.raw_epsilon)

    def compute_operator_components(
        self, geometry: Tensor, padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        latent = self.geo_branch(geometry, padding_mask)
        query = torch.zeros_like(latent)
        key = torch.zeros_like(latent)
        for index in range(self.n_heads_A):
            projected = self.W_Q[index](latent)
            query += projected
            key += projected

        query = query / self.n_heads_A
        key = key / self.n_heads_A
        return query, key * self.scale, self.epsilon

    def apply_operator(
        self,
        rhs: Tensor,
        query: Tensor,
        scaled_key: Tensor,
        epsilon: Tensor,
    ) -> Tensor:
        key_times_rhs = torch.bmm(
            scaled_key.transpose(-2, -1), rhs.unsqueeze(-1)
        )
        low_rank_result = torch.bmm(query, key_times_rhs).squeeze(-1)
        return low_rank_result + epsilon * rhs

    def forward(self, inputs: tuple[Tensor, Tensor] | list[Tensor]) -> Tensor:
        rhs, geometry = inputs
        padding_mask = geometry[..., 3].eq(0)
        geometry = self.input_normalizer(geometry)
        query, scaled_key, epsilon = self.compute_operator_components(
            geometry, padding_mask
        )
        return self.apply_operator(rhs, query, scaled_key, epsilon)
