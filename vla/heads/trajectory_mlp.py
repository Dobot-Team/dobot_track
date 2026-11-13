from typing import List, Sequence

import torch
import torch.nn as nn


_HEAD_DEPTHS = {
    "mlp_2": 2,
    "mlp_4": 4,
    "mlp_6": 6,
}
_HEAD_TYPE_ALIASES = {
    "default": "legacy",
    "legacy_mlp": "legacy",
    "mlp2": "mlp_2",
    "mlp4": "mlp_4",
    "mlp6": "mlp_6",
    "mlp_2_layer": "mlp_2",
    "mlp_4_layer": "mlp_4",
    "mlp_6_layer": "mlp_6",
    "mlp_2layer": "mlp_2",
    "mlp_4layer": "mlp_4",
    "mlp_6layer": "mlp_6",
    "res_mlp_2": "mlp_2",
    "res_mlp_4": "mlp_4",
    "res_mlp_6": "mlp_6",
    "residual_mlp_2": "mlp_2",
    "residual_mlp_4": "mlp_4",
    "residual_mlp_6": "mlp_6",
}


def canonicalize_action_head_type(action_head_type: str) -> str:
    """Normalize public action-head aliases to their checkpoint-stable names."""
    if not isinstance(action_head_type, str):
        raise TypeError(
            "action_head_type must be a string, got "
            f"{type(action_head_type).__name__}"
        )

    normalized = action_head_type.strip().lower().replace("-", "_")
    normalized = "_".join(
        part for part in normalized.replace("_", " ").split() if part
    )
    canonical = _HEAD_TYPE_ALIASES.get(normalized, normalized)
    if canonical not in {"legacy", *_HEAD_DEPTHS}:
        supported = ", ".join(("legacy", *_HEAD_DEPTHS))
        raise ValueError(
            f"Unsupported action head type {action_head_type!r}. "
            f"Supported values: {supported}."
        )
    return canonical


class ResidualMLPBlock(nn.Module):
    """Pre-normalized two-layer MLP block with an identity skip connection."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class TrajectoryMLP(nn.Module):
    """Decode trajectory query states into future waypoints.

    Args:
        hidden_size: Dimensionality of the VLM hidden states.
        num_queries: Number of learnable query tokens / input query states.
        hidden_dims: For preset heads, ``[hidden_dim, output_dim]``. The output
            dimension is per waypoint (2 for x, y).
        n_future: Number of future waypoints to predict.
        head_type: Head architecture. ``mlp_2``, ``mlp_4``, and ``mlp_6``
            create the corresponding number of Linear layers. Four- and six-layer
            heads contain one and two pre-normalized residual MLP blocks,
            respectively. ``legacy`` preserves the activation-free stack used by
            checkpoints created before head presets were introduced.
    """

    _HEAD_DEPTHS = _HEAD_DEPTHS
    _HEAD_TYPE_ALIASES = _HEAD_TYPE_ALIASES

    @staticmethod
    def canonicalize_action_head_type(action_head_type: str) -> str:
        """Return the canonical public name for an action-head architecture."""
        return canonicalize_action_head_type(action_head_type)

    canonicalize_head_type = canonicalize_action_head_type

    def __init__(
        self,
        hidden_size: int,
        num_queries: int,
        hidden_dims: Sequence[int],
        n_future: int,
        head_type: str = "legacy",
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_queries = int(num_queries)
        self.n_future = int(n_future)
        self.is_single_query = self.num_queries == 1
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {self.num_queries}")
        if self.n_future <= 0:
            raise ValueError(f"n_future must be positive, got {self.n_future}")

        self.head_type = self.canonicalize_action_head_type(head_type)
        self.action_head_type = self.head_type
        if not hidden_dims:
            raise ValueError("hidden_dims must include at least an output dimension")
        head_dims = [int(dim) for dim in hidden_dims]
        if any(dim <= 0 for dim in head_dims):
            raise ValueError(f"hidden_dims must be positive, got {list(hidden_dims)}")

        if self.head_type == "legacy":
            self._build_legacy_head(head_dims)
        else:
            if len(head_dims) != 2:
                raise ValueError(
                    f"{self.head_type} expects action_head_dims=[hidden_dim, output_dim], "
                    f"got {list(hidden_dims)}"
                )
            hidden_dim, output_dim = head_dims
            self._build_residual_head(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_residual_blocks=(self._HEAD_DEPTHS[self.head_type] - 2) // 2,
            )

    def _build_legacy_head(self, head_dims: List[int]) -> None:
        """Keep historical parameter names and forward behavior for old checkpoints."""
        self.output_dim = head_dims[-1]
        if self.is_single_query:
            head_dims[-1] *= self.n_future

        layers: List[nn.Module] = []
        in_dim = self.hidden_size
        for out_dim in head_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.num_linear_layers = len(layers)

    def _build_residual_head(
        self,
        hidden_dim: int,
        output_dim: int,
        num_residual_blocks: int,
    ) -> None:
        self.output_dim = output_dim
        expanded_output_dim = output_dim * self.n_future if self.is_single_query else output_dim

        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_projection = nn.Linear(self.hidden_size, hidden_dim)
        self.input_activation = nn.GELU()
        self.residual_blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim)
            for _ in range(num_residual_blocks)
        )
        self.output_projection = nn.Linear(hidden_dim, expanded_output_dim)
        self.num_linear_layers = 2 + 2 * num_residual_blocks

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, num_queries, hidden_size)
        Returns:
            traj: (B, n_future, output_dim)
        """
        if self.head_type == "legacy":
            traj = self.mlp(hidden_states)
        else:
            traj = self.input_norm(hidden_states)
            traj = self.input_projection(traj)
            traj = self.input_activation(traj)
            for block in self.residual_blocks:
                traj = block(traj)
            traj = self.output_projection(traj)
        if self.is_single_query:
            return traj.reshape(hidden_states.shape[0], self.n_future, self.output_dim)
        return traj
