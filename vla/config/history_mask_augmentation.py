"""Configuration for training-time historical-mask corruption."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Union


@dataclass(frozen=True)
class HistoryMaskAugmentationConfig:
    """Configuration for balanced historical-mask corruption."""

    enabled: bool = True
    clean_probability: float = 0.20

    @classmethod
    def from_value(
        cls,
        value: Union[
            bool,
            "HistoryMaskAugmentationConfig",
            Mapping[str, object],
            None,
        ],
    ) -> "HistoryMaskAugmentationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, Mapping):
            unknown = set(value) - {"enabled", "clean_probability"}
            if unknown:
                raise ValueError(
                    "Unsupported history_mask_augmentation option(s): "
                    + ", ".join(sorted(str(key) for key in unknown))
                )
            enabled = value.get("enabled", True)
            if not isinstance(enabled, bool):
                raise TypeError(
                    "history_mask_augmentation.enabled must be a boolean"
                )
            clean_probability = value.get("clean_probability", 0.20)
            if isinstance(clean_probability, bool) or not isinstance(
                clean_probability, Real
            ):
                raise TypeError(
                    "history_mask_augmentation.clean_probability must be numeric"
                )
            return cls(
                enabled=enabled,
                clean_probability=float(clean_probability),
            )
        raise TypeError(
            "history_mask_augmentation must be a boolean, mapping, "
            "HistoryMaskAugmentationConfig, or None; got "
            f"{type(value).__name__}"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "history_mask_augmentation.enabled must be a boolean"
            )
        if isinstance(self.clean_probability, bool) or not isinstance(
            self.clean_probability, Real
        ):
            raise TypeError(
                "history_mask_augmentation.clean_probability must be numeric"
            )
        if not 0.0 <= self.clean_probability <= 1.0:
            raise ValueError(
                "history_mask_augmentation.clean_probability must be in [0, 1], "
                f"got {self.clean_probability}"
            )
