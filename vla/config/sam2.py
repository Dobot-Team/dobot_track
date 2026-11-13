"""Serializable configuration for the optional SAM2 mask decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Union


@dataclass(frozen=True)
class SAM2CurrentMaskConfig:
    """Gitee/MH4 current-frame mask supervision settings."""

    enabled: bool = True
    loss_weight: float = 0.001


@dataclass(frozen=True)
class SAM2FutureMaskConfig:
    """Gitee/MH4 future-mask settings.

    The current TrackVLA decoder is single-frame.  ``num_frames=0`` is the
    supported MH4 setting and explicitly disables future-frame decoding.
    """

    num_frames: int = 0
    frame_stride: int = 1
    loss_weight: float = 0.001


@dataclass(frozen=True)
class SAM2Config:
    """Configuration for the bundled SAM2 mask decoder."""

    enabled: bool = False
    checkpoint: Optional[str] = None
    config_file: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    feature_size: int = 16
    train_decoder: bool = True
    loss_weight: float = 0.1
    use_high_res_features: Optional[bool] = None
    current_mask: SAM2CurrentMaskConfig = SAM2CurrentMaskConfig()
    future_masks: SAM2FutureMaskConfig = SAM2FutureMaskConfig()

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "SAM2Config":
        """Build a typed configuration from ``model.sam2_config`` YAML."""
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise TypeError(
                "model.sam2_config must be a mapping or null, got "
                f"{type(values).__name__}"
            )
        unknown_keys = set(values) - {
            "enabled",
            "checkpoint",
            "config_file",
            "feature_size",
            "train_decoder",
            "loss_weight",
            "use_high_res_features",
            "current_mask",
            "future_masks",
        }
        if unknown_keys:
            raise ValueError(
                "Unsupported model.sam2_config option(s): "
                + ", ".join(sorted(map(str, unknown_keys)))
            )
        current = values.get("current_mask")
        future = values.get("future_masks")
        if current is not None and not isinstance(current, Mapping):
            raise TypeError("model.sam2_config.current_mask must be a mapping")
        if future is not None and not isinstance(future, Mapping):
            raise TypeError("model.sam2_config.future_masks must be a mapping")
        current_cfg = SAM2CurrentMaskConfig(**dict(current or {}))
        future_cfg = SAM2FutureMaskConfig(**dict(future or {}))
        # Keep the legacy flat fields as aliases while accepting the MH4 schema.
        enabled = values.get("enabled", False)
        flat_weight = values.get("loss_weight", None)
        if current is not None:
            enabled = bool(enabled and current_cfg.enabled)
            if flat_weight is None:
                flat_weight = current_cfg.loss_weight
        if flat_weight is None:
            flat_weight = cls.loss_weight
        return cls(
            enabled=enabled,
            checkpoint=values.get("checkpoint"),
            config_file=values.get("config_file", cls.config_file),
            feature_size=values.get("feature_size", cls.feature_size),
            train_decoder=values.get("train_decoder", cls.train_decoder),
            loss_weight=flat_weight,
            use_high_res_features=values.get("use_high_res_features"),
            current_mask=current_cfg,
            future_masks=future_cfg,
        )

    @classmethod
    def from_value(
        cls,
        value: Optional[Union["SAM2Config", Mapping[str, Any]]],
    ) -> "SAM2Config":
        """Normalize an existing typed config, YAML mapping, or ``None``."""
        if isinstance(value, cls):
            return value
        return cls.from_mapping(value)

    @classmethod
    def from_checkpoint_value(
        cls,
        value: Optional[Union["SAM2Config", Mapping[str, Any]]],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SAM2Config":
        """Restore SAM2 config through the common extension checkpoint API."""
        del metadata
        return cls.from_value(value)

    def to_mapping(self) -> Dict[str, Any]:
        """Return the complete serializable configuration."""
        return {
            "enabled": self.enabled,
            "checkpoint": self.checkpoint,
            "config_file": self.config_file,
            "feature_size": self.feature_size,
            "train_decoder": self.train_decoder,
            "loss_weight": self.loss_weight,
            "use_high_res_features": self.use_high_res_features,
            "current_mask": {
                "enabled": self.current_mask.enabled,
                "loss_weight": self.current_mask.loss_weight,
            },
            "future_masks": {
                "num_frames": self.future_masks.num_frames,
                "frame_stride": self.future_masks.frame_stride,
                "loss_weight": self.future_masks.loss_weight,
            },
        }

    def architecture_mapping(self) -> Dict[str, Any]:
        """Return fields that determine module/state compatibility."""
        return {
            "enabled": self.enabled,
            "config_file": self.config_file,
            "feature_size": self.feature_size,
            "train_decoder": self.train_decoder,
            "use_high_res_features": self.use_high_res_features,
        }

    def validate(self) -> None:
        for field_name in ("enabled", "train_decoder"):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"model.sam2_config.{field_name} must be a boolean, got "
                    f"{type(value).__name__}"
                )
        if self.use_high_res_features is not None and not isinstance(
            self.use_high_res_features, bool
        ):
            raise TypeError(
                "model.sam2_config.use_high_res_features must be true, false, "
                "or null"
            )
        if self.checkpoint is not None and not isinstance(self.checkpoint, str):
            raise TypeError(
                "model.sam2_config.checkpoint must be a string or null, got "
                f"{type(self.checkpoint).__name__}"
            )
        if not isinstance(self.config_file, str):
            raise TypeError(
                "model.sam2_config.config_file must be a non-empty string, got "
                f"{type(self.config_file).__name__}"
            )
        if not isinstance(self.feature_size, int) or isinstance(
            self.feature_size, bool
        ):
            raise TypeError(
                "model.sam2_config.feature_size must be an integer, got "
                f"{type(self.feature_size).__name__}"
            )
        if isinstance(self.loss_weight, bool) or not isinstance(
            self.loss_weight, Real
        ):
            raise TypeError(
                "model.sam2_config.loss_weight must be a real number, got "
                f"{type(self.loss_weight).__name__}"
            )
        if self.enabled and (
            self.checkpoint is None or not self.checkpoint.strip()
        ):
            raise ValueError(
                "model.sam2_config.checkpoint is required when "
                "model.sam2_config.enabled=True"
            )
        if self.checkpoint is not None and not self.checkpoint.strip():
            raise ValueError("model.sam2_config.checkpoint must not be empty")
        if not self.config_file.strip():
            raise ValueError("model.sam2_config.config_file must not be empty")
        if self.feature_size <= 0:
            raise ValueError(
                "model.sam2_config.feature_size must be positive, got "
                f"{self.feature_size}"
            )
        if not math.isfinite(self.loss_weight):
            raise ValueError(
                "model.sam2_config.loss_weight must be finite, got "
                f"{self.loss_weight}"
            )
        if self.loss_weight < 0:
            raise ValueError(
                "model.sam2_config.loss_weight must be non-negative, got "
                f"{self.loss_weight}"
            )
        if not isinstance(self.current_mask.enabled, bool):
            raise TypeError("model.sam2_config.current_mask.enabled must be boolean")
        if self.current_mask.loss_weight < 0:
            raise ValueError("model.sam2_config.current_mask.loss_weight must be non-negative")
        if self.future_masks.num_frames < 0:
            raise ValueError("model.sam2_config.future_masks.num_frames must be non-negative")
        if self.future_masks.frame_stride <= 0:
            raise ValueError("model.sam2_config.future_masks.frame_stride must be positive")
        if self.future_masks.loss_weight < 0:
            raise ValueError("model.sam2_config.future_masks.loss_weight must be non-negative")
        if self.future_masks.num_frames > 0:
            raise NotImplementedError(
                "Future-mask decoding requires temporal SAM2 memory; set "
                "model.sam2_config.future_masks.num_frames=0 for this decoder."
            )

    def checkpoint_compatible_with(self, other: "SAM2Config") -> bool:
        """Whether ``other`` describes the same checkpoint-side architecture."""
        if not isinstance(other, SAM2Config):
            return False
        if self.enabled != other.enabled:
            return False
        if not self.enabled:
            return True
        return self.architecture_mapping() == other.architecture_mapping()
