"""Configuration for grounding and structured VQA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Union


@dataclass(frozen=True)
class AuxiliaryTasksConfig:
    """Loss weights and label thresholds for optional auxiliary tasks."""

    grounding_enabled: bool = False
    vqa_enabled: bool = False
    grounding_loss_weight: float = 0.1
    vqa_loss_weight: float = 0.1
    bbox_l1_weight: float = 5.0
    bbox_giou_weight: float = 2.0
    visibility_weight: float = 1.0
    small_area_threshold: float = 0.05
    large_area_threshold: float = 0.20
    near_distance_threshold: float = 1.5
    far_distance_threshold: float = 3.0

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(
        cls, values: Optional[Mapping[str, Any]]
    ) -> "AuxiliaryTasksConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise TypeError("model.auxiliary_tasks_config must be a mapping or null")
        values = dict(values)
        if "grounding_before_action" in values:
            legacy_order = values.pop("grounding_before_action")
            if not isinstance(legacy_order, bool):
                raise TypeError("grounding_before_action must be boolean in legacy configs")
            if not legacy_order:
                raise ValueError(
                    "Box-before-Action is fixed on this branch; use the original "
                    "training revision for grounding_before_action=false checkpoints."
                )
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "Unsupported model.auxiliary_tasks_config option(s): "
                + ", ".join(sorted(map(str, unknown)))
            )
        return cls(**dict(values))

    @classmethod
    def from_value(
        cls,
        value: Optional[Union["AuxiliaryTasksConfig", Mapping[str, Any]]],
    ) -> "AuxiliaryTasksConfig":
        return value if isinstance(value, cls) else cls.from_mapping(value)

    @property
    def enabled(self) -> bool:
        return self.grounding_enabled or self.vqa_enabled

    def to_mapping(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    def validate(self) -> None:
        for name in ("grounding_enabled", "vqa_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"model.auxiliary_tasks_config.{name} must be boolean")
        if self.vqa_enabled and not self.grounding_enabled:
            raise ValueError("VQA requires grounding_enabled=true")
        numeric_names = (
            "grounding_loss_weight", "vqa_loss_weight", "bbox_l1_weight",
            "bbox_giou_weight", "visibility_weight", "small_area_threshold",
            "large_area_threshold", "near_distance_threshold",
            "far_distance_threshold",
        )
        for name in numeric_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"model.auxiliary_tasks_config.{name} must be real")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"model.auxiliary_tasks_config.{name} must be finite and non-negative")
        if not self.small_area_threshold < self.large_area_threshold <= 1:
            raise ValueError("VQA area thresholds must satisfy 0 <= small < large <= 1")
        if not self.near_distance_threshold < self.far_distance_threshold:
            raise ValueError("VQA distance thresholds must satisfy near < far")
