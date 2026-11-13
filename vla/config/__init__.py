"""Typed configuration and policy for optional VLA model extensions."""

from .auxiliary_tasks import AuxiliaryTasksConfig
from .custom_model import CustomVLAConfig
from .history_compressor import HistoryCompressorConfig
from .history_mask_augmentation import HistoryMaskAugmentationConfig
from .model_extensions import (
    ExecutionStage,
    ExtensionInputRequirements,
    ModelExtensionContext,
    ModelExtensionsConfig,
    VisualTokenLayout,
)
from .sam2 import SAM2Config

__all__ = [
    "AuxiliaryTasksConfig",
    "CustomVLAConfig",
    "ExecutionStage",
    "ExtensionInputRequirements",
    "HistoryCompressorConfig",
    "HistoryMaskAugmentationConfig",
    "ModelExtensionContext",
    "ModelExtensionsConfig",
    "SAM2Config",
    "VisualTokenLayout",
]
