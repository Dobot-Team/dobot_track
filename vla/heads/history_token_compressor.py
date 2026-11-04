"""Per-frame compression for historical visual tokens.

The compressor deliberately leaves temporal modelling to the language model:
each history frame is compressed independently, and the caller remains
responsible for preserving the frame order when assembling the language-model
sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.config.history_compressor import HistoryCompressorConfig
from vla.utils.history_boxes import augment_boxes, clip_boxes, masks_to_boxes


def _add_first_query_residual(
    queries: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> torch.Tensor:
    """Add an optional ``[B,1,D]`` residual without mutating an expanded view."""
    if residual is None:
        return queries
    if (
        residual.ndim != 3
        or residual.shape[0] != queries.shape[0]
        or residual.shape[1] != 1
        or residual.shape[2] != queries.shape[2]
    ):
        raise ValueError(
            "box residual must have shape [B,1,D] matching queries, got "
            f"{tuple(residual.shape)} for queries {tuple(queries.shape)}"
        )
    return torch.cat((queries[:, :1] + residual, queries[:, 1:]), dim=1)


@dataclass(frozen=True)
class HistoryCompressionOutput:
    """Named result shared by the LLM and dense current-frame consumers.

    ``llm_tokens`` follows oldest-to-newest frame order. ``current_tokens`` is
    always a view of the untouched final frame, so auxiliary heads such as
    SAM2 never consume a compressed or position-encoded representation.
    """

    llm_tokens: torch.Tensor
    current_tokens: torch.Tensor
    frame_token_counts: tuple[int, ...]
    llm_token_counts: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        for field_name in ("llm_tokens", "current_tokens"):
            value = getattr(self, field_name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{field_name} must be a torch.Tensor")
            if value.ndim != 3:
                raise ValueError(
                    f"{field_name} must have shape [B,N,D], got "
                    f"{tuple(value.shape)}"
                )
        if (
            self.llm_tokens.shape[0] != self.current_tokens.shape[0]
            or self.llm_tokens.shape[2] != self.current_tokens.shape[2]
        ):
            raise ValueError(
                "llm_tokens and current_tokens must share batch and hidden dimensions"
            )
        if (
            self.llm_tokens.device != self.current_tokens.device
            or self.llm_tokens.dtype != self.current_tokens.dtype
        ):
            raise ValueError(
                "llm_tokens and current_tokens must share device and dtype"
            )
        if (
            not isinstance(self.frame_token_counts, tuple)
            or not self.frame_token_counts
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in self.frame_token_counts
            )
        ):
            raise ValueError(
                "frame_token_counts must be a non-empty tuple of positive integers"
            )
        if self.llm_tokens.shape[1] != sum(self.frame_token_counts):
            raise ValueError(
                "llm_tokens length does not match frame_token_counts"
            )
        if self.llm_token_counts is not None:
            counts = self.llm_token_counts
            if (
                not isinstance(counts, tuple)
                or len(counts) != self.llm_tokens.shape[0]
                or any(
                    not isinstance(count, int)
                    or isinstance(count, bool)
                    or count <= 0
                    or count > self.llm_tokens.shape[1]
                    for count in counts
                )
                or max(counts) != self.llm_tokens.shape[1]
            ):
                raise ValueError(
                    "llm_token_counts must contain one valid length per sample"
                )
        if self.current_tokens.shape[1] <= 0:
            raise ValueError("current_tokens must contain dense visual tokens")

    @property
    def total_tokens(self) -> int:
        return sum(self.frame_token_counts)


class _QIdentityLayer(nn.Module):
    """Joint query/target self-attention followed by full-frame cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        *,
        dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()
        factory_kwargs = {} if dtype is None else {"dtype": dtype}
        attention_kwargs = {
            "embed_dim": hidden_size,
            "num_heads": num_heads,
            "dropout": dropout,
            "batch_first": True,
            **factory_kwargs,
        }

        self.self_attention_norm = nn.LayerNorm(hidden_size, **factory_kwargs)
        self.self_attention = nn.MultiheadAttention(**attention_kwargs)
        self.full_attention_norm = nn.LayerNorm(hidden_size, **factory_kwargs)
        self.full_attention = nn.MultiheadAttention(**attention_kwargs)
        self.ffn_norm = nn.LayerNorm(hidden_size, **factory_kwargs)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, **factory_kwargs),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size, **factory_kwargs),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        target_context: Optional[torch.Tensor],
        target_padding_mask: Optional[torch.Tensor],
        full_context: torch.Tensor,
        box_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized_queries = self.self_attention_norm(
            torch.cat((queries, target_context), dim=1)
        )
        self_padding_mask = torch.cat((
            torch.zeros(queries.shape[:2], dtype=torch.bool, device=queries.device),
            target_padding_mask,
        ), dim=1)
        update, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            key_padding_mask=self_padding_mask,
            need_weights=False,
        )
        queries = queries + update[:, :queries.shape[1]]

        if box_residual is not None:
            queries = _add_first_query_residual(queries, box_residual)

        update, _ = self.full_attention(
            self.full_attention_norm(queries),
            full_context,
            full_context,
            need_weights=False,
        )
        queries = queries + update
        return queries + self.ffn(self.ffn_norm(queries))


class HistoryTokenCompressor(nn.Module):
    """Compress every history frame without mixing information across frames.

    Args:
        config: Compressor architecture and mask-selection settings.
        hidden_size: Internal Q-Former hidden size.
        llm_hidden_size: Dimensionality of the projected visual/LLM tokens.
        dtype: Optional construction dtype for learnable parameters.

    Inputs:
        frame_tokens: ``[B, H, T, D]`` projected tokens. ``T`` must form a
            square visual grid.
        history_masks: Optional ``[B, H, S, S]`` mask probabilities. Required
            only for target-aware selection; missing masks use ``no_target``.
        history_mask_valid: Optional ``[B, H]`` validity flags. When omitted,
            supplied masks are considered valid.

    Returns:
        Compressed tokens with shape ``[B, H, K, D]``.
    """

    def __init__(
        self,
        config: HistoryCompressorConfig,
        hidden_size: int,
        llm_hidden_size: int,
        expected_tokens_per_frame: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, HistoryCompressorConfig):
            raise TypeError(
                "config must be a HistoryCompressorConfig, got "
                f"{type(config).__name__}"
            )
        config.validate()
        for name, value in (
            ("hidden_size", hidden_size),
            ("llm_hidden_size", llm_hidden_size),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype or None, got {dtype!r}")
        if expected_tokens_per_frame is not None:
            if (
                not isinstance(expected_tokens_per_frame, int)
                or isinstance(expected_tokens_per_frame, bool)
            ):
                raise TypeError("expected_tokens_per_frame must be an integer or None")
            if expected_tokens_per_frame <= 0:
                raise ValueError("expected_tokens_per_frame must be positive")

        self.config = config
        self.qformer_hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.expected_tokens_per_frame = expected_tokens_per_frame
        self.num_output_tokens = config.num_output_tokens
        self.mode = config.mode

        # A disabled compressor is a parameter-free identity plugin. Keeping an
        # actual module object lets callers use one stable interface without
        # branching on Optional[HistoryTokenCompressor].
        if not config.enabled:
            return

        if self.mode == "adaptive_pool":
            self.pool_grid_size = math.isqrt(self.num_output_tokens)
            self.pool = nn.AdaptiveAvgPool2d(
                (self.pool_grid_size, self.pool_grid_size)
            )
            return

        if hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({config.num_heads})"
            )
        factory_kwargs = {} if dtype is None else {"dtype": dtype}
        self.query_tokens = nn.Parameter(
            torch.empty(1, self.num_output_tokens, hidden_size, **factory_kwargs)
        )
        self.no_target_token = nn.Parameter(
            torch.empty(1, 1, hidden_size, **factory_kwargs)
        )
        self.position_projection = nn.Linear(
            2, hidden_size, bias=False, **factory_kwargs
        )
        if config.box_embedding:
            # Keep the legacy 2D projection above for disabled configurations
            # and checkpoints.  The enabled path shares this MLP between patch
            # positions and mask-derived box geometry.
            self.box_projection = nn.Sequential(
                nn.Linear(4, hidden_size, **factory_kwargs),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size, **factory_kwargs),
            )

        # Project visual tokens into the QFormer's hidden space, then back to the LLM's hidden space.
        self.llm_proj_in = nn.Linear(
            llm_hidden_size,
            hidden_size,
            **factory_kwargs,
        )

        self.output_norm = nn.LayerNorm(
            hidden_size,
            **factory_kwargs,
        )

        self.llm_proj_out = nn.Linear(
            hidden_size,
            llm_hidden_size,
            **factory_kwargs,
        )

        self.layers = nn.ModuleList(
            _QIdentityLayer(
                hidden_size,
                config.num_heads,
                config.dropout,
                dtype=dtype,
            )
            for _ in range(config.num_layers)
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.no_target_token, mean=0.0, std=0.02)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Loading with strict=False must not hide a different attention topology.
        if any(key.startswith(prefix + "layers.") and ".target_attention" in key
               for key in state_dict):
            error_msgs.append(
                "alternating_qformer weights are not compatible with q_identity; "
                "use the original training revision for this checkpoint."
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _validate_frame_tokens(
        frame_tokens: torch.Tensor,
        hidden_size: int,
        *,
        require_square_grid: bool,
    ) -> tuple[int, int, int, int, int]:
        if not isinstance(frame_tokens, torch.Tensor):
            raise TypeError(
                f"frame_tokens must be a torch.Tensor, got {type(frame_tokens).__name__}"
            )
        if frame_tokens.ndim != 4:
            raise ValueError(
                "frame_tokens must have shape [B,H,T,D], got "
                f"{tuple(frame_tokens.shape)}"
            )
        batch_size, num_frames, num_tokens, token_dim = frame_tokens.shape
        if batch_size <= 0:
            raise ValueError("frame_tokens batch dimension must be positive")
        if num_frames <= 0:
            raise ValueError("frame_tokens must contain at least one frame")
        if num_tokens <= 0:
            raise ValueError("frame_tokens must contain at least one token")
        if token_dim != hidden_size:
            raise ValueError(
                f"frame token dimension must equal hidden_size ({hidden_size}), "
                f"got {token_dim}"
            )
        grid_size = math.isqrt(num_tokens)
        if require_square_grid and grid_size * grid_size != num_tokens:
            raise ValueError(
                f"frame token count must form a square visual grid, got T={num_tokens}"
            )
        if not frame_tokens.is_floating_point():
            raise TypeError(
                f"frame_tokens must be floating point, got {frame_tokens.dtype}"
            )
        return batch_size, num_frames, num_tokens, token_dim, grid_size

    def compress_sequence(
        self,
        frame_tokens: torch.Tensor,
        history_masks: Optional[torch.Tensor] = None,
        history_mask_valid: Optional[torch.Tensor] = None,
        current_frame_qformer: Optional[torch.Tensor] = None,
        current_masks: Optional[torch.Tensor] = None,
        history_boxes: Optional[torch.Tensor] = None,
        history_box_valid: Optional[torch.Tensor] = None,
    ) -> HistoryCompressionOutput:
        """Build one per-sample LLM sequence and retain the dense current frame.

        Args:
            frame_tokens: Projected visual tokens shaped ``[B,H,T,D]`` in
                oldest-to-newest order; the final frame is current.
            history_masks: Optional masks for only the past frames,
                ``[B,H-1,S,S]``. Missing masks use the QFormer's no-target
                fallback.
            history_mask_valid: Optional validity flags ``[B,H-1]``.
            current_frame_qformer: Optional per-sample ``[B]`` choices for
                whether the LLM receives a compressed current frame.
            current_masks: Optional ``[B,S,S]`` current-frame ground-truth
                masks, used only for selected training samples when enabled.

        Adaptive pooling deliberately ignores masks. With
        ``config.enabled=False`` this method preserves the legacy ``H*T``
        layout and does not impose the v1 256-token grid contract.
        """
        batch_size, num_frames, num_tokens, token_dim, _ = (
            self._validate_frame_tokens(
                frame_tokens,
                self.llm_hidden_size,
                require_square_grid=False,
            )
        )
        current_tokens = frame_tokens[:, -1]
        if current_frame_qformer is not None:
            if not isinstance(current_frame_qformer, torch.Tensor):
                raise TypeError("current_frame_qformer must be a bool tensor")
            if current_frame_qformer.dtype != torch.bool:
                raise TypeError("current_frame_qformer must use bool dtype")
            if tuple(current_frame_qformer.shape) != (batch_size,):
                raise ValueError("current_frame_qformer must have shape [B]")
            current_frame_qformer = current_frame_qformer.to(frame_tokens.device)
            if not self.config.is_qformer and not bool(
                (~current_frame_qformer).all()
            ):
                raise ValueError(
                    "current_frame_qformer choices require an enabled QFormer mode"
                )
        frame_token_counts = self.config.frame_token_counts(
            num_frames,
            num_tokens,
            current_frame_qformer=(
                bool(current_frame_qformer.all())
                if current_frame_qformer is not None
                else False
            ),
        )
        if not self.config.enabled:
            return HistoryCompressionOutput(
                llm_tokens=frame_tokens.reshape(
                    frame_tokens.shape[0],
                    num_frames * num_tokens,
                    token_dim,
                ),
                current_tokens=current_tokens,
                frame_token_counts=frame_token_counts,
            )

        if (
            self.expected_tokens_per_frame is not None
            and num_tokens != self.expected_tokens_per_frame
        ):
            raise ValueError(
                "History compression requires exactly "
                f"{self.expected_tokens_per_frame} projected visual tokens per "
                f"frame, got {num_tokens}"
            )
        grid_size = math.isqrt(num_tokens)
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                "History compression requires a square visual-token grid, "
                f"got {num_tokens} tokens per frame"
            )
        if num_frames == 1 and self.config.requires_history_masks:
            self._validate_masks(
                history_masks,
                history_mask_valid,
                batch_size=batch_size,
                num_frames=0,
                device=frame_tokens.device,
            )
        compressed_history = (
            self(
                frame_tokens[:, :-1],
                history_masks=history_masks,
                history_mask_valid=history_mask_valid,
                history_boxes=history_boxes,
                history_box_valid=history_box_valid,
            ).reshape(batch_size, (num_frames - 1) * self.num_output_tokens, token_dim)
            if num_frames > 1
            else frame_tokens.new_empty((batch_size, 0, token_dim))
        )
        if current_frame_qformer is None or not bool(current_frame_qformer.any()):
            return HistoryCompressionOutput(
                llm_tokens=torch.cat((compressed_history, current_tokens), dim=1),
                current_tokens=current_tokens,
                frame_token_counts=frame_token_counts,
            )

        selected = current_frame_qformer.nonzero(as_tuple=False).flatten()
        selected_masks = None
        if self.training and self.config.current_frame_qformer_use_gt_mask:
            if current_masks is None:
                raise ValueError(
                    "current_masks is required when training with "
                    "current_frame_qformer_use_gt_mask=true"
                )
            if not isinstance(current_masks, torch.Tensor):
                raise TypeError("current_masks must be a torch.Tensor")
            if (
                current_masks.ndim != 3
                or current_masks.shape[0] != batch_size
                or current_masks.shape[-2] != current_masks.shape[-1]
                or current_masks.shape[-1] <= 0
            ):
                raise ValueError("current_masks must have shape [B,S,S]")
            selected_masks = current_masks[selected.to(current_masks.device)].unsqueeze(1)
        current_boxes = current_valid = None
        if self.config.requires_history_boxes:
            if selected_masks is not None:
                current_boxes, current_valid = masks_to_boxes(selected_masks, torch.ones(selected_masks.shape[:2], dtype=torch.bool, device=selected_masks.device))
            else:
                current_boxes = frame_tokens.new_zeros((selected.shape[0], 1, 4))
                current_valid = torch.zeros((selected.shape[0], 1), dtype=torch.bool, device=frame_tokens.device)
        compressed_current = self(
            frame_tokens[selected, -1:], history_masks=selected_masks,
            history_boxes=current_boxes, history_box_valid=current_valid,
        )[:, 0]
        if bool(current_frame_qformer.all()):
            return HistoryCompressionOutput(
                llm_tokens=torch.cat((compressed_history, compressed_current), dim=1),
                current_tokens=current_tokens,
                frame_token_counts=frame_token_counts,
            )

        selected_rows = {int(row): index for index, row in enumerate(selected.tolist())}
        sequences = [
            torch.cat(
                (
                    compressed_history[row],
                    compressed_current[selected_rows[row]]
                    if row in selected_rows
                    else current_tokens[row],
                ),
                dim=0,
            )
            for row in range(batch_size)
        ]
        return HistoryCompressionOutput(
            llm_tokens=nn.utils.rnn.pad_sequence(sequences, batch_first=True),
            current_tokens=current_tokens,
            frame_token_counts=self.config.frame_token_counts(
                num_frames, num_tokens, current_frame_qformer=False
            ),
            llm_token_counts=tuple(sequence.shape[0] for sequence in sequences),
        )

    def _adaptive_pool(
        self,
        frame_tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        token_dim: int,
        grid_size: int,
    ) -> torch.Tensor:
        feature_maps = (
            frame_tokens.reshape(
                batch_size, num_frames, grid_size, grid_size, token_dim
            )
            .permute(0, 1, 4, 2, 3)
            .reshape(
                batch_size * num_frames,
                token_dim,
                grid_size,
                grid_size,
            )
        )
        pooled = self.pool(feature_maps)
        return (
            pooled.reshape(
                batch_size,
                num_frames,
                token_dim,
                self.pool_grid_size,
                self.pool_grid_size,
            )
            .permute(0, 1, 3, 4, 2)
            .reshape(
                batch_size,
                num_frames,
                self.num_output_tokens,
                token_dim,
            )
        )

    @staticmethod
    def _validate_masks(
        history_masks: Optional[torch.Tensor],
        history_mask_valid: Optional[torch.Tensor],
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        expected_prefix = (batch_size, num_frames)
        if history_masks is not None:
            if not isinstance(history_masks, torch.Tensor):
                raise TypeError(
                    "history_masks must be a torch.Tensor or None, got "
                    f"{type(history_masks).__name__}"
                )
            if (
                history_masks.ndim != 4
                or tuple(history_masks.shape[:2]) != expected_prefix
                or history_masks.shape[-2] <= 0
                or history_masks.shape[-1] <= 0
            ):
                raise ValueError(
                    "history_masks must have shape [B,H,S,S] matching frame_tokens, "
                    f"got {tuple(history_masks.shape)}"
                )
            if history_masks.shape[-2] != history_masks.shape[-1]:
                raise ValueError(
                    "history_masks must have square spatial dimensions, got "
                    f"{tuple(history_masks.shape[-2:])}"
                )

        if history_mask_valid is None:
            valid = torch.full(
                expected_prefix,
                history_masks is not None,
                dtype=torch.bool,
                device=device,
            )
        else:
            if not isinstance(history_mask_valid, torch.Tensor):
                raise TypeError(
                    "history_mask_valid must be a torch.Tensor or None, got "
                    f"{type(history_mask_valid).__name__}"
                )
            if tuple(history_mask_valid.shape) != expected_prefix:
                raise ValueError(
                    "history_mask_valid must have shape [B,H] matching frame_tokens, "
                    f"got {tuple(history_mask_valid.shape)}"
                )
            if history_mask_valid.dtype != torch.bool:
                raise TypeError(
                    "history_mask_valid must use torch.bool dtype, got "
                    f"{history_mask_valid.dtype}"
                )
            valid = history_mask_valid.to(device=device)

        if history_masks is None and bool(valid.any()):
            raise ValueError(
                "history_masks is required when history_mask_valid contains true"
            )
        return history_masks, valid

    def _prepare_history_masks(
        self,
        history_masks: Optional[torch.Tensor],
        history_mask_valid: Optional[torch.Tensor],
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Validate masks once and return sanitized per-frame values."""
        masks, valid = self._validate_masks(
            history_masks,
            history_mask_valid,
            batch_size=batch_size,
            num_frames=num_frames,
            device=device,
        )
        valid = valid.reshape(batch_size * num_frames)
        if masks is None:
            return None, valid
        mask_values = torch.nan_to_num(
            masks.detach().to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        return mask_values.reshape(
            batch_size * num_frames,
            mask_values.shape[-2],
            mask_values.shape[-1],
        ), valid

    @staticmethod
    def _box_geometry_from_masks(
        mask_values: Optional[torch.Tensor],
        mask_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[cx, cy, 2w/W-1, 2h/H-1]`` boxes and validity.

        Boxes use the largest 8-connected foreground component with exclusive
        upper bounds.  Empty, unavailable, and invalid masks deliberately
        produce zero geometry and a false validity flag.
        """
        if not isinstance(mask_valid, torch.Tensor):
            raise TypeError(
                "mask_valid must be a torch.Tensor with shape [B*H], got "
                f"{type(mask_valid).__name__}"
            )
        if mask_valid.dtype != torch.bool:
            raise TypeError(
                "mask_valid must use torch.bool dtype, got "
                f"{mask_valid.dtype}"
            )
        if mask_valid.ndim != 1:
            raise ValueError(
                "mask_valid must have shape [B*H], got "
                f"{tuple(mask_valid.shape)}"
            )
        batch_frames = mask_valid.shape[0]
        if mask_values is None:
            return (
                torch.zeros(
                    batch_frames,
                    4,
                    device=mask_valid.device,
                    dtype=torch.float32,
                ),
                torch.zeros_like(mask_valid),
            )
        if not isinstance(mask_values, torch.Tensor):
            raise TypeError(
                "mask_values must be a torch.Tensor or None, got "
                f"{type(mask_values).__name__}"
            )
        if (
            mask_values.ndim != 3
            or mask_values.shape[0] != batch_frames
            or mask_values.shape[-2] <= 0
            or mask_values.shape[-1] <= 0
        ):
            raise ValueError(
                "mask_values must have shape [B*H,S,S] matching mask_valid, got "
                f"{tuple(mask_values.shape)}"
            )
        if mask_values.device != mask_valid.device:
            raise ValueError(
                "mask_values and mask_valid must be on the same device, got "
                f"{mask_values.device} and {mask_valid.device}"
            )

        height, width = mask_values.shape[-2:]
        foreground = mask_values > 0.5
        non_empty = foreground.flatten(1).any(dim=1)
        box_valid = mask_valid & non_empty
        geometry = torch.zeros(
            batch_frames,
            4,
            device=mask_values.device,
            dtype=torch.float32,
        )
        if not bool(box_valid.any()):
            return geometry, box_valid

        largest_component = HistoryTokenCompressor._largest_connected_component(
            foreground[box_valid]
        )
        row_has_foreground = largest_component.any(dim=-1)
        column_has_foreground = largest_component.any(dim=-2)
        top = row_has_foreground.to(torch.int64).argmax(dim=-1)
        bottom_exclusive = height - row_has_foreground.flip(-1).to(
            torch.int64
        ).argmax(dim=-1)
        left = column_has_foreground.to(torch.int64).argmax(dim=-1)
        right_exclusive = width - column_has_foreground.flip(-1).to(
            torch.int64
        ).argmax(dim=-1)
        valid_geometry = torch.stack(
            (
                (left.to(torch.float32) + right_exclusive.to(torch.float32))
                / float(width)
                - 1.0,
                (top.to(torch.float32) + bottom_exclusive.to(torch.float32))
                / float(height)
                - 1.0,
                2.0
                * (right_exclusive - left).to(torch.float32)
                / float(width)
                - 1.0,
                2.0
                * (bottom_exclusive - top).to(torch.float32)
                / float(height)
                - 1.0,
            ),
            dim=-1,
        )
        valid_geometry = torch.nan_to_num(
            valid_geometry,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        return geometry.index_copy(
            0,
            box_valid.nonzero(as_tuple=False).flatten(),
            valid_geometry,
        ), box_valid

    @staticmethod
    def _largest_connected_component(foreground: torch.Tensor) -> torch.Tensor:
        """Select each mask's largest 8-connected foreground component."""
        if not isinstance(foreground, torch.Tensor):
            raise TypeError(
                "foreground must be a torch.Tensor with shape [N,H,W], got "
                f"{type(foreground).__name__}"
            )
        if foreground.dtype != torch.bool:
            raise TypeError(
                "foreground must use torch.bool dtype, got "
                f"{foreground.dtype}"
            )
        if (
            foreground.ndim != 3
            or foreground.shape[0] <= 0
            or foreground.shape[-2] <= 0
            or foreground.shape[-1] <= 0
        ):
            raise ValueError(
                "foreground must have non-empty shape [N,H,W], got "
                f"{tuple(foreground.shape)}"
            )

        batch_size, height, width = foreground.shape
        num_pixels = height * width
        label_dtype = (
            torch.float64 if num_pixels > (1 << 24) else torch.float32
        )
        labels = torch.arange(
            1,
            num_pixels + 1,
            device=foreground.device,
            dtype=label_dtype,
        ).reshape(1, height, width)
        labels = torch.where(
            foreground,
            labels.expand(batch_size, -1, -1),
            torch.zeros((), device=foreground.device, dtype=label_dtype),
        )
        background_label = float(num_pixels + 1)
        foreground_flat = foreground.reshape(batch_size, num_pixels)

        # Neighbor labels only decrease.  Repeated pointer jumping reaches the
        # exact 8-connected components without moving masks off-device.
        for _ in range(num_pixels):
            neighbor_labels = torch.where(
                foreground,
                labels,
                torch.full_like(labels, background_label),
            )
            neighbor_min = -F.max_pool2d(
                -neighbor_labels.unsqueeze(1),
                kernel_size=3,
                stride=1,
                padding=1,
            ).squeeze(1)
            hooked = torch.where(
                foreground,
                torch.minimum(labels, neighbor_min),
                torch.zeros_like(labels),
            )
            hooked_flat = hooked.reshape(batch_size, num_pixels)
            parent_indices = hooked_flat.to(torch.long).sub(1).clamp_min(0)
            compressed = torch.where(
                foreground_flat,
                hooked_flat.gather(1, parent_indices),
                torch.zeros_like(hooked_flat),
            ).reshape(batch_size, height, width)
            if torch.equal(compressed, labels):
                break
            labels = compressed
        else:
            raise RuntimeError(
                "largest-component label propagation did not converge within "
                f"{num_pixels} iterations"
            )

        flat_labels = labels.reshape(batch_size, num_pixels).to(torch.long)
        offsets = (
            torch.arange(batch_size, device=foreground.device, dtype=torch.long)
            .unsqueeze(1)
            * (num_pixels + 1)
        )
        component_counts = torch.bincount(
            (flat_labels + offsets).flatten(),
            minlength=batch_size * (num_pixels + 1),
        ).reshape(batch_size, num_pixels + 1)
        largest_labels = component_counts[:, 1:].argmax(dim=1).add(1)
        return flat_labels.eq(largest_labels.unsqueeze(1)).reshape(
            batch_size,
            height,
            width,
        )

    def _encode_geometry(
        self,
        geometry: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project finite normalized geometry into the Q-Former space."""
        if not isinstance(geometry, torch.Tensor):
            raise TypeError(
                "geometry must be a torch.Tensor with last dimension 4, got "
                f"{type(geometry).__name__}"
            )
        if geometry.ndim < 1 or geometry.shape[-1] != 4:
            raise ValueError(
                "geometry must have four coordinates [cx,cy,w,h], got "
                f"{tuple(geometry.shape)}"
            )
        geometry = torch.nan_to_num(
            geometry.to(device=device, dtype=dtype),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        encoded = self.box_projection(geometry)
        return torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_target_context_from_prepared_masks(
        self,
        visual_context: torch.Tensor,
        mask_values: Optional[torch.Tensor],
        mask_valid: torch.Tensor,
        *,
        num_tokens: int,
        grid_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build target attention inputs from validated flattened masks."""
        flattened_batch = visual_context.shape[0]
        if mask_values is None:
            occupancy = torch.zeros(
                flattened_batch,
                num_tokens,
                dtype=torch.float32,
                device=visual_context.device,
            )
        else:
            occupancy = F.adaptive_avg_pool2d(
                mask_values.reshape(
                    flattened_batch,
                    1,
                    mask_values.shape[-2],
                    mask_values.shape[-1],
                ),
                (grid_size, grid_size),
            ).reshape(flattened_batch, num_tokens)

        if tuple(mask_valid.shape) != (flattened_batch,):
            raise ValueError(
                "mask_valid must have shape [B*H] matching visual_context, got "
                f"{tuple(mask_valid.shape)}"
            )
        valid = mask_valid
        non_empty = occupancy.amax(dim=1) > 0
        selected = (
            occupancy > self.config.mask_patch_threshold
        ) & valid.unsqueeze(1) & non_empty.unsqueeze(1)

        needs_top_patch = valid & non_empty & ~selected.any(dim=1)
        if bool(needs_top_patch.any()):
            rows = torch.arange(
                flattened_batch, device=visual_context.device
            )[needs_top_patch]
            columns = occupancy.argmax(dim=1)[needs_top_patch]
            selected[rows, columns] = True

        use_no_target = ~(valid & non_empty)
        no_target = self.no_target_token.expand(flattened_batch, -1, -1)
        target_context = torch.cat((visual_context, no_target), dim=1)
        target_padding_mask = torch.ones(
            flattened_batch,
            num_tokens + 1,
            dtype=torch.bool,
            device=visual_context.device,
        )
        target_padding_mask[:, :num_tokens] = ~selected
        target_padding_mask[:, num_tokens] = ~use_no_target
        return target_context, target_padding_mask

    def _build_target_context(
        self,
        visual_context: torch.Tensor,
        history_masks: Optional[torch.Tensor],
        history_mask_valid: Optional[torch.Tensor],
        *,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        grid_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks, valid = self._validate_masks(
            history_masks,
            history_mask_valid,
            batch_size=batch_size,
            num_frames=num_frames,
            device=visual_context.device,
        )
        flattened_batch = batch_size * num_frames
        if masks is None:
            occupancy = torch.zeros(
                flattened_batch,
                num_tokens,
                dtype=torch.float32,
                device=visual_context.device,
            )
        else:
            mask_values = torch.nan_to_num(
                masks.detach().to(device=visual_context.device, dtype=torch.float32),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp_(0.0, 1.0)
            occupancy = F.adaptive_avg_pool2d(
                mask_values.reshape(
                    flattened_batch,
                    1,
                    mask_values.shape[-2],
                    mask_values.shape[-1],
                ),
                (grid_size, grid_size),
            ).reshape(flattened_batch, num_tokens)

        valid = valid.reshape(flattened_batch)
        non_empty = occupancy.amax(dim=1) > 0
        selected = (
            occupancy > self.config.mask_patch_threshold
        ) & valid.unsqueeze(1) & non_empty.unsqueeze(1)

        needs_top_patch = valid & non_empty & ~selected.any(dim=1)
        if bool(needs_top_patch.any()):
            rows = torch.arange(
                flattened_batch, device=visual_context.device
            )[needs_top_patch]
            columns = occupancy.argmax(dim=1)[needs_top_patch]
            selected[rows, columns] = True

        use_no_target = ~(valid & non_empty)
        no_target = self.no_target_token.expand(flattened_batch, -1, -1)
        target_context = torch.cat((visual_context, no_target), dim=1)
        target_padding_mask = torch.ones(
            flattened_batch,
            num_tokens + 1,
            dtype=torch.bool,
            device=visual_context.device,
        )
        target_padding_mask[:, :num_tokens] = ~selected
        target_padding_mask[:, num_tokens] = ~use_no_target
        return target_context, target_padding_mask

    def _prepare_boxes(self, boxes, valid, batch_size, num_frames, device):
        if boxes is None:
            raise ValueError("bbox target_selection requires history_boxes")
        if valid is None:
            raise ValueError("bbox target_selection requires history_box_valid")
        if not isinstance(boxes, torch.Tensor) or tuple(boxes.shape) != (batch_size, num_frames, 4):
            raise ValueError("history_boxes must have shape [B,H,4]")
        if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != (batch_size, num_frames) or valid.dtype != torch.bool:
            raise ValueError("history_box_valid must be a bool tensor of shape [B,H]")
        boxes = boxes.detach().to(device=device, dtype=torch.float32).reshape(-1, 4)
        valid = valid.to(device=device).reshape(-1)
        if self.training:
            return augment_boxes(boxes, valid,
                jitter_probability=self.config.bbox_jitter_probability,
                jitter_scale=self.config.bbox_jitter_scale,
                dropout_probability=self.config.bbox_dropout_probability)
        return clip_boxes(boxes, valid)

    def _build_bbox_context(self, visual_context, boxes, valid, grid_size):
        # Patch centers use normalized image coordinates, not the [-1,1]
        # geometry embedding. A valid subpatch box gets its center patch.
        coordinates = (torch.arange(grid_size, device=boxes.device) + 0.5) / grid_size
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        centers = torch.stack((xx, yy), -1).reshape(1, -1, 2)
        lo, hi = boxes[:, None, :2] - boxes[:, None, 2:] / 2, boxes[:, None, :2] + boxes[:, None, 2:] / 2
        selected = ((centers >= lo) & (centers < hi)).all(-1) & valid[:, None]
        center = (boxes[:, :2] * grid_size).long().clamp(0, grid_size-1)
        fallback_index = center[:, 1] * grid_size + center[:, 0]
        fallback = F.one_hot(fallback_index, grid_size * grid_size).bool()
        selected = selected | (fallback & (valid & ~selected.any(-1))[:, None])
        context = torch.cat((visual_context, self.no_target_token.expand(boxes.shape[0], -1, -1)), 1)
        padding = torch.cat((~selected, valid[:, None]), 1)
        return context, padding

    def _qformer(
        self,
        frame_tokens: torch.Tensor,
        history_masks: Optional[torch.Tensor],
        history_mask_valid: Optional[torch.Tensor],
        *,
        history_boxes: Optional[torch.Tensor] = None,
        history_box_valid: Optional[torch.Tensor] = None,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        grid_size: int,
    ) -> torch.Tensor:
        flattened_batch = batch_size * num_frames

        if frame_tokens.shape[-1] != self.qformer_hidden_size:
            raise ValueError(
                f"Q-Former input dimension must be {self.qformer_hidden_size}, "
                f"got {frame_tokens.shape[-1]}"
            )

        visual_context = frame_tokens.reshape(
            flattened_batch,
            num_tokens,
            self.qformer_hidden_size,
        )

        coordinates = torch.linspace(
            -1.0,
            1.0,
            grid_size,
            device=frame_tokens.device,
            dtype=frame_tokens.dtype,
        )
        grid_y, grid_x = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        if self.config.requires_history_boxes:
            boxes, box_valid = self._prepare_boxes(
                history_boxes, history_box_valid, batch_size, num_frames, frame_tokens.device)
            if self.config.box_embedding:
                position_input = torch.stack((grid_x, grid_y, torch.full_like(grid_x, -1), torch.full_like(grid_y, -1)), -1).reshape(1, num_tokens, 4)
                visual_context = visual_context + self._encode_geometry(
                    position_input, device=frame_tokens.device, dtype=frame_tokens.dtype)
            else:
                position_input = torch.stack((grid_x, grid_y), -1).reshape(1, num_tokens, 2)
                visual_context = visual_context + self.position_projection(position_input)
            target_context, target_padding_mask = self._build_bbox_context(
                visual_context, boxes, box_valid, grid_size)
        elif self.config.box_embedding:
            position_input = torch.stack(
                (
                    grid_x,
                    grid_y,
                    torch.full_like(grid_x, -1.0),
                    torch.full_like(grid_y, -1.0),
                ),
                dim=-1,
            ).reshape(1, num_tokens, 4)
            visual_context = visual_context + self._encode_geometry(
                position_input,
                device=frame_tokens.device,
                dtype=frame_tokens.dtype,
            )
            mask_values, mask_valid = self._prepare_history_masks(
                history_masks,
                history_mask_valid,
                batch_size=batch_size,
                num_frames=num_frames,
                device=frame_tokens.device,
            )
            target_context, target_padding_mask = (
                self._build_target_context_from_prepared_masks(
                    visual_context,
                    mask_values,
                    mask_valid,
                    num_tokens=num_tokens,
                    grid_size=grid_size,
                )
            )
        else:
            # Keep this legacy path byte-for-byte equivalent in its learned
            # parameters and computations for existing checkpoints.
            position_input = torch.stack(
                (grid_x, grid_y),
                dim=-1,
            ).reshape(1, num_tokens, 2)
            visual_context = visual_context + self.position_projection(
                position_input
            )
            target_context, target_padding_mask = self._build_target_context(
                visual_context,
                history_masks,
                history_mask_valid,
                batch_size=batch_size,
                num_frames=num_frames,
                num_tokens=num_tokens,
                grid_size=grid_size,
            )

        queries = self.query_tokens.expand(
            flattened_batch,
            -1,
            -1,
        )

        if self.config.box_embedding:
            if self.config.requires_history_boxes:
                box_geometry = boxes * 2 - 1
            else:
                box_geometry, box_valid = self._box_geometry_from_masks(
                    mask_values, mask_valid,
                )
            box_embedding = self._encode_geometry(
                box_geometry,
                device=frame_tokens.device,
                dtype=frame_tokens.dtype,
            )
            box_embedding = box_embedding * box_valid.to(
                dtype=box_embedding.dtype,
            ).unsqueeze(-1)
            box_residual = (
                box_embedding / float(len(self.layers) + 1)
            ).unsqueeze(1)
            queries = _add_first_query_residual(queries, box_residual)
            for layer in self.layers:
                queries = layer(
                    queries,
                    target_context,
                    target_padding_mask,
                    visual_context,
                    box_residual,
                )
        else:
            for layer in self.layers:
                queries = layer(
                    queries,
                    target_context,
                    target_padding_mask,
                    visual_context,
                )

        return queries.reshape(
            batch_size,
            num_frames,
            self.num_output_tokens,
            self.qformer_hidden_size,
        )

    def forward(
        self,
        frame_tokens: torch.Tensor,
        history_masks: Optional[torch.Tensor] = None,
        history_mask_valid: Optional[torch.Tensor] = None,
        history_boxes: Optional[torch.Tensor] = None,
        history_box_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_frames, num_tokens, input_dim, grid_size = (
            self._validate_frame_tokens(
                frame_tokens,
                self.llm_hidden_size,
                require_square_grid=self.config.enabled,
            )
        )

        if not self.config.enabled:
            return frame_tokens

        if self.mode == "adaptive_pool":
            return self._adaptive_pool(
                frame_tokens,
                batch_size=batch_size,
                num_frames=num_frames,
                token_dim=input_dim,
                grid_size=grid_size,
            )

        qformer_tokens = self.llm_proj_in(frame_tokens)

        out = self._qformer(
            qformer_tokens,
            history_masks,
            history_mask_valid,
            history_boxes=history_boxes,
            history_box_valid=history_box_valid,
            batch_size=batch_size,
            num_frames=num_frames,
            num_tokens=num_tokens,
            grid_size=grid_size,
        )

        out = self.output_norm(out)
        return self.llm_proj_out(out)


def compress_flattened_history_sequence(
    compressor: nn.Module,
    image_features: torch.Tensor,
    *,
    batch_size: int,
    history_masks: Optional[torch.Tensor] = None,
    history_mask_valid: Optional[torch.Tensor] = None,
    history_boxes: Optional[torch.Tensor] = None,
    history_box_valid: Optional[torch.Tensor] = None,
    current_frame_qformer: Optional[torch.Tensor] = None,
    current_masks: Optional[torch.Tensor] = None,
) -> HistoryCompressionOutput:
    """Adapt a VLM's ``[B*H,T,D]`` output to the canonical plugin contract.

    This stateless adapter is reusable by any vision backbone that flattens
    frames into its image batch. It owns only layout validation, so compressor
    parameters remain directly under the caller's module and retain stable
    checkpoint keys.
    """
    compress_sequence = getattr(compressor, "compress_sequence", None)
    if not isinstance(compressor, nn.Module) or not callable(
        compress_sequence
    ):
        raise TypeError(
            "compressor must be an nn.Module implementing compress_sequence, got "
            f"{type(compressor).__name__}"
        )
    if not isinstance(image_features, torch.Tensor):
        raise TypeError(
            "image_features must be a torch.Tensor, got "
            f"{type(image_features).__name__}"
        )
    if image_features.ndim != 3:
        raise ValueError(
            "flattened image_features must have shape [B*H,T,D], got "
            f"{tuple(image_features.shape)}"
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if image_features.shape[0] <= 0:
        raise ValueError("image_features must contain at least one frame")
    if image_features.shape[0] % batch_size != 0:
        raise ValueError(
            "Flattened visual batch size must be divisible by text batch size"
        )

    num_frames = image_features.shape[0] // batch_size
    grouped_features = image_features.reshape(
        batch_size,
        num_frames,
        image_features.shape[1],
        image_features.shape[2],
    )
    kwargs = dict(history_masks=history_masks, history_mask_valid=history_mask_valid)
    if history_boxes is not None:
        kwargs['history_boxes'] = history_boxes
    if history_box_valid is not None:
        kwargs['history_box_valid'] = history_box_valid
    if current_frame_qformer is not None:
        kwargs['current_frame_qformer'] = current_frame_qformer
    if current_masks is not None:
        kwargs['current_masks'] = current_masks
    output = compress_sequence(grouped_features, **kwargs)
    if not isinstance(output, HistoryCompressionOutput):
        raise TypeError(
            "compress_sequence must return HistoryCompressionOutput, got "
            f"{type(output).__name__}"
        )
    return output
