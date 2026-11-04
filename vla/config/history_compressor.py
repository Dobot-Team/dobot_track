"""Serializable configuration for historical visual-token compression."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Union


_QFORMER_MODES = frozenset({"q_identity"})
_SUPPORTED_MODES = _QFORMER_MODES | {"adaptive_pool"}


@dataclass(frozen=True)
class HistoryCompressorConfig:
    """Configuration for the per-frame history-token compressor."""

    enabled: bool = False
    mode: str = "q_identity"
    num_output_tokens: int = 16
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    mask_patch_threshold: float = 0.25
    hidden_size: Optional[int] = 512
    box_embedding: bool = False
    current_frame_qformer_probability: float = 0.0
    current_frame_qformer_use_gt_mask: bool = False
    target_selection: str = "mask"
    bbox_visibility_threshold: float = 0.5
    bbox_jitter_probability: float = 0.0
    bbox_jitter_scale: float = 0.1
    bbox_dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        # Alias only: the existing Joint architecture and state-dict keys are unchanged.
        if self.mode == "qformer_joint_self_attention":
            object.__setattr__(self, "mode", "q_identity")
        self.validate()

    @classmethod
    def from_mapping(
        cls,
        values: Optional[Mapping[str, Any]],
    ) -> "HistoryCompressorConfig":
        """Build a typed configuration from a YAML-compatible mapping."""
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise TypeError(
                "model.history_compressor_config must be a mapping or null, "
                f"got {type(values).__name__}"
            )
        if not values:
            return cls()
        # ``box_injection`` existed on the experimental training branch.  A
        # serialized ``false`` is a no-op and therefore remains compatible
        # with this topology.  ``true`` adds both parameters and one visual
        # token per history frame, so accepting it here would silently build a
        # different model from the checkpoint.
        values = dict(values)
        if "box_injection" in values:
            box_injection = values.pop("box_injection")
            if not isinstance(box_injection, bool):
                raise TypeError(
                    "model.history_compressor_config.box_injection must be "
                    "a boolean"
                )
            if box_injection:
                raise ValueError(
                    "model.history_compressor_config.box_injection=true is "
                    "not supported by this checkout. It changes the history "
                    "compressor topology; use the revision that trained the "
                    "checkpoint."
                )
        unknown_keys = set(values) - {
            "enabled",
            "mode",
            "num_output_tokens",
            "num_layers",
            "num_heads",
            "dropout",
            "mask_patch_threshold",
            "box_embedding",
            "hidden_size",
            "current_frame_qformer_probability",
            "current_frame_qformer_use_gt_mask",
            "target_selection",
            "bbox_visibility_threshold",
            "bbox_jitter_probability",
            "bbox_jitter_scale",
            "bbox_dropout_probability",
        }
        if unknown_keys:
            raise ValueError(
                "Unsupported model.history_compressor_config option(s): "
                + ", ".join(sorted(map(str, unknown_keys)))
            )

        enabled = values.get("enabled", cls.enabled)
        if not isinstance(enabled, bool):
            raise TypeError(
                "model.history_compressor_config.enabled must be a boolean"
            )
        mode = values.get("mode", cls.mode)
        if not isinstance(mode, str):
            raise TypeError(
                "model.history_compressor_config.mode must be a string"
            )
        box_embedding = values.get("box_embedding", cls.box_embedding)
        if not isinstance(box_embedding, bool):
            raise TypeError(
                "model.history_compressor_config.box_embedding must be a boolean"
            )

        integer_values = {}
        for key, default in (
            ("num_output_tokens", cls.num_output_tokens),
            ("num_layers", cls.num_layers),
            ("num_heads", cls.num_heads),
        ):
            value = values.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"model.history_compressor_config.{key} must be an integer"
                )
            integer_values[key] = value

        float_values = {}
        for key, default in (
            ("dropout", cls.dropout),
            ("mask_patch_threshold", cls.mask_patch_threshold),
        ):
            value = values.get(key, default)
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError(
                    f"model.history_compressor_config.{key} must be numeric"
                )
            float_values[key] = float(value)

        hidden_size = values.get("hidden_size", cls.hidden_size)
        if hidden_size is not None and (
            not isinstance(hidden_size, int) or isinstance(hidden_size, bool)
        ):
            raise TypeError(
                "model.history_compressor_config.hidden_size must be a positive "
                "integer or null"
            )

        return cls(
            enabled=enabled,
            mode=mode,
            num_output_tokens=integer_values["num_output_tokens"],
            num_layers=integer_values["num_layers"],
            num_heads=integer_values["num_heads"],
            dropout=float_values["dropout"],
            mask_patch_threshold=float_values["mask_patch_threshold"],
            box_embedding=box_embedding,
            hidden_size=hidden_size,
            current_frame_qformer_probability=values.get("current_frame_qformer_probability", cls.current_frame_qformer_probability),
            current_frame_qformer_use_gt_mask=values.get("current_frame_qformer_use_gt_mask", cls.current_frame_qformer_use_gt_mask),
            target_selection=values.get("target_selection", cls.target_selection),
            bbox_visibility_threshold=values.get("bbox_visibility_threshold", cls.bbox_visibility_threshold),
            bbox_jitter_probability=values.get("bbox_jitter_probability", cls.bbox_jitter_probability),
            bbox_jitter_scale=values.get("bbox_jitter_scale", cls.bbox_jitter_scale),
            bbox_dropout_probability=values.get("bbox_dropout_probability", cls.bbox_dropout_probability),
        )

    @classmethod
    def from_value(
        cls,
        value: Optional[
            Union["HistoryCompressorConfig", Mapping[str, Any]]
        ],
    ) -> "HistoryCompressorConfig":
        """Normalize a config object, YAML mapping, or ``None``."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            value.validate()
            return value
        return cls.from_mapping(value)

    def to_mapping(self) -> Dict[str, Any]:
        """Return a checkpoint-stable representation containing primitives."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "num_output_tokens": self.num_output_tokens,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "mask_patch_threshold": self.mask_patch_threshold,
            "box_embedding": self.box_embedding,
            "hidden_size": self.hidden_size,
            "current_frame_qformer_probability": self.current_frame_qformer_probability,
            "current_frame_qformer_use_gt_mask": self.current_frame_qformer_use_gt_mask,
            "target_selection": self.target_selection,
            "bbox_visibility_threshold": self.bbox_visibility_threshold,
            "bbox_jitter_probability": self.bbox_jitter_probability,
            "bbox_jitter_scale": self.bbox_jitter_scale,
            "bbox_dropout_probability": self.bbox_dropout_probability,
        }

    def validate(self) -> None:
        """Validate architectural values before parameters are constructed."""
        if not isinstance(self.target_selection, str) or self.target_selection not in {"mask", "bbox"}:
            raise ValueError("target_selection must be 'mask' or 'bbox'")
        if self.target_selection == "bbox" and (not self.enabled or self.mode not in _QFORMER_MODES):
            raise ValueError("bbox target_selection requires an enabled QFormer mode")
        for key in ("bbox_visibility_threshold", "bbox_jitter_probability", "bbox_jitter_scale", "bbox_dropout_probability"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{key} must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be in [0, 1]")
        if not isinstance(self.current_frame_qformer_use_gt_mask, bool):
            raise TypeError("current_frame_qformer_use_gt_mask must be boolean")
        probability = self.current_frame_qformer_probability
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise TypeError("current_frame_qformer_probability must be numeric")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("current_frame_qformer_probability must be in [0,1]")
        if (probability != 0 or self.current_frame_qformer_use_gt_mask) and not self.is_qformer:
            raise ValueError("current-frame compression requires an enabled QFormer")
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "model.history_compressor_config.enabled must be a boolean"
            )
        if not isinstance(self.mode, str):
            raise TypeError(
                "model.history_compressor_config.mode must be a string"
            )
        if self.mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported history compressor mode {self.mode!r}. "
                f"Supported values: {', '.join(sorted(_SUPPORTED_MODES))}."
            )
        if not isinstance(self.box_embedding, bool):
            raise TypeError(
                "model.history_compressor_config.box_embedding must be a boolean"
            )
        if self.box_embedding and (
            not self.enabled or self.mode not in _QFORMER_MODES
        ):
            raise ValueError(
                "model.history_compressor_config.box_embedding requires "
                "enabled=true and a QFormer mode"
            )
        if (
            not isinstance(self.num_output_tokens, int)
            or isinstance(self.num_output_tokens, bool)
        ):
            raise TypeError(
                "model.history_compressor_config.num_output_tokens must be an "
                "integer"
            )
        if self.num_output_tokens <= 0:
            raise ValueError(
                "model.history_compressor_config.num_output_tokens must be a "
                f"positive integer, got {self.num_output_tokens!r}"
            )
        if (
            not isinstance(self.num_layers, int)
            or isinstance(self.num_layers, bool)
        ):
            raise TypeError(
                "model.history_compressor_config.num_layers must be an integer"
            )
        if self.num_layers <= 0:
            raise ValueError(
                "model.history_compressor_config.num_layers must be a positive "
                f"integer, got {self.num_layers!r}"
            )
        if (
            not isinstance(self.num_heads, int)
            or isinstance(self.num_heads, bool)
        ):
            raise TypeError(
                "model.history_compressor_config.num_heads must be an integer"
            )
        if self.num_heads <= 0:
            raise ValueError(
                "model.history_compressor_config.num_heads must be a positive "
                f"integer, got {self.num_heads!r}"
            )
        if self.hidden_size is not None and (
            not isinstance(self.hidden_size, int)
            or isinstance(self.hidden_size, bool)
        ):
            raise TypeError(
                "model.history_compressor_config.hidden_size must be a positive "
                "integer or null"
            )
        if self.hidden_size is not None and self.hidden_size <= 0:
            raise ValueError(
                "model.history_compressor_config.hidden_size must be a positive "
                f"integer, got {self.hidden_size!r}"
            )
        if not isinstance(self.dropout, Real) or isinstance(self.dropout, bool):
            raise TypeError(
                "model.history_compressor_config.dropout must be numeric"
            )
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "model.history_compressor_config.dropout must be in [0, 1), got "
                f"{self.dropout}"
            )
        if not isinstance(self.mask_patch_threshold, Real) or isinstance(
            self.mask_patch_threshold, bool
        ):
            raise TypeError(
                "model.history_compressor_config.mask_patch_threshold must be "
                "numeric"
            )
        if (
            not math.isfinite(float(self.mask_patch_threshold))
            or not 0.0 <= self.mask_patch_threshold <= 1.0
        ):
            raise ValueError(
                "model.history_compressor_config.mask_patch_threshold must be "
                f"in [0, 1], got {self.mask_patch_threshold}"
            )
        if self.mode == "adaptive_pool":
            pool_size = math.isqrt(self.num_output_tokens)
            if pool_size * pool_size != self.num_output_tokens:
                raise ValueError(
                    "adaptive_pool requires num_output_tokens to be a perfect "
                    f"square, got {self.num_output_tokens}"
                )

    @property
    def requires_history_masks(self) -> bool:
        """Whether this mode consumes target masks for past frames."""
        return self.enabled and self.mode in _QFORMER_MODES and self.target_selection == "mask"

    @property
    def requires_history_boxes(self) -> bool:
        return self.enabled and self.mode in _QFORMER_MODES and self.target_selection == "bbox"

    @property
    def is_qformer(self):
        return self.enabled and self.mode in _QFORMER_MODES

    def validate_grounding(self, grounding_enabled: bool) -> None:
        if self.requires_history_boxes and not grounding_enabled:
            raise ValueError("bbox target_selection requires auxiliary_tasks_config.grounding_enabled=true for online bbox/visibility predictions")

    def frame_token_counts(
        self,
        num_frames: int,
        dense_tokens_per_frame: int = 256,
        *,
        current_frame_qformer: Optional[bool] = None,
    ) -> tuple[int, ...]:
        """Plan oldest-to-current visual placeholders; default to dense current."""
        if (
            not isinstance(num_frames, int)
            or isinstance(num_frames, bool)
            or num_frames <= 0
        ):
            raise ValueError(
                f"num_frames must be a positive integer, got {num_frames!r}"
            )
        if (
            not isinstance(dense_tokens_per_frame, int)
            or isinstance(dense_tokens_per_frame, bool)
            or dense_tokens_per_frame <= 0
        ):
            raise ValueError(
                "dense_tokens_per_frame must be a positive integer, got "
                f"{dense_tokens_per_frame!r}"
            )
        if not self.enabled:
            return (dense_tokens_per_frame,) * num_frames
        if current_frame_qformer is None:
            current_frame_qformer = False
        if not isinstance(current_frame_qformer, bool):
            raise TypeError("current_frame_qformer must be a bool")
        if self.is_qformer:
            current_count = (
                self.num_output_tokens
                if current_frame_qformer
                else dense_tokens_per_frame
            )
            return (
                *((self.num_output_tokens,) * (num_frames - 1)),
                current_count,
            )
        return (
            (self.num_output_tokens,) * (num_frames - 1)
            + (dense_tokens_per_frame,)
        )

    def total_sequence_tokens(
        self,
        num_frames: int,
        dense_tokens_per_frame: int = 256,
    ) -> int:
        """Return the total visual-token budget for one sample."""
        return sum(
            self.frame_token_counts(
                num_frames,
                dense_tokens_per_frame,
            )
        )

    def checkpoint_compatible_with(
        self,
        checkpoint_config: "HistoryCompressorConfig",
    ) -> bool:
        """Compare only architecture that is active in the checkpoint."""
        if not isinstance(checkpoint_config, HistoryCompressorConfig):
            return False
        if self.enabled != checkpoint_config.enabled:
            return False
        if not self.enabled:
            return True
        current, checkpoint = self.to_mapping(), checkpoint_config.to_mapping()
        for key in ("bbox_visibility_threshold", "bbox_jitter_probability", "bbox_jitter_scale", "bbox_dropout_probability", "current_frame_qformer_probability", "current_frame_qformer_use_gt_mask"):
            current.pop(key)
            checkpoint.pop(key)
        return current == checkpoint
