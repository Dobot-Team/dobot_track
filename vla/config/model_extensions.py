"""Single source of truth for optional VLA model extensions.

Entry points should parse and validate SAM2/history extensions here
instead of reproducing support matrices and checkpoint precedence rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeVar

from .history_compressor import HistoryCompressorConfig
from .sam2 import SAM2Config


class ExecutionStage(str, Enum):
    """Execution stages whose mask requirements differ."""

    MODEL = "model"
    TRAIN = "train"
    OFFLINE_EVAL = "offline_eval"
    ONLINE_EVAL = "online_eval"
    EXPORT = "export"


@dataclass(frozen=True)
class ModelExtensionContext:
    """Environment used to validate the model-extension support matrix."""

    stage: ExecutionStage = ExecutionStage.MODEL
    backend: str = "pytorch"
    dataset_type: Optional[str] = None
    model_type: Optional[str] = None
    dense_tokens_per_frame: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ExecutionStage):
            raise TypeError("stage must be an ExecutionStage")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise TypeError("backend must be a non-empty string")
        for field_name in ("dataset_type", "model_type"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise TypeError(f"{field_name} must be a non-empty string or None")
        if self.dense_tokens_per_frame is not None:
            value = self.dense_tokens_per_frame
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("dense_tokens_per_frame must be an integer or None")
            if value <= 0:
                raise ValueError("dense_tokens_per_frame must be positive")


@dataclass(frozen=True)
class ExtensionInputRequirements:
    """Inputs required by a validated model-extension combination."""

    current_gt_masks: bool = False
    history_masks: bool = False
    online_mask_cache: bool = False
    history_boxes: bool = False
    online_box_cache: bool = False


@dataclass(frozen=True)
class VisualTokenLayout:
    """Oldest-to-current visual placeholder layout for one sample."""

    per_frame_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.per_frame_counts:
            raise ValueError("VisualTokenLayout requires at least one frame")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in self.per_frame_counts
        ):
            raise ValueError(
                "VisualTokenLayout per-frame counts must be positive integers"
            )

    @property
    def total_tokens(self) -> int:
        return sum(self.per_frame_counts)

    @property
    def current_tokens(self) -> int:
        return self.per_frame_counts[-1]

    @property
    def history_tokens(self) -> int:
        return sum(self.per_frame_counts[:-1])


_ConfigT = TypeVar("_ConfigT", SAM2Config, HistoryCompressorConfig)

# Private checkpoint metadata. Never copy this mapping into YAML/model config.
_CHECKPOINT_METADATA_KEY = "_model_extensions_checkpoint_metadata"


@dataclass(frozen=True)
class ModelExtensionsConfig:
    """Configuration and lifecycle policy for optional model extensions."""

    sam2: SAM2Config = field(default_factory=SAM2Config)
    history_compressor: HistoryCompressorConfig = field(
        default_factory=HistoryCompressorConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.sam2, SAM2Config):
            raise TypeError("sam2 must be a SAM2Config")
        if not isinstance(
            self.history_compressor,
            HistoryCompressorConfig,
        ):
            raise TypeError(
                "history_compressor must be a HistoryCompressorConfig"
            )
        self.sam2.validate()
        self.history_compressor.validate()

    @classmethod
    def from_values(
        cls,
        *,
        sam2: Any = None,
        history_compressor: Any = None,
    ) -> "ModelExtensionsConfig":
        """Normalize typed objects, YAML mappings, or ``None``."""
        return cls(
            sam2=SAM2Config.from_value(sam2),
            history_compressor=HistoryCompressorConfig.from_value(
                history_compressor
            ),
        )

    @classmethod
    def from_model_mapping(
        cls,
        model_config: Mapping[str, Any],
    ) -> "ModelExtensionsConfig":
        """Parse optional extension sections from a model configuration."""
        if not isinstance(model_config, Mapping):
            raise TypeError(
                "model config must be a mapping, got "
                f"{type(model_config).__name__}"
            )
        return cls.from_values(
            sam2=model_config.get("sam2_config"),
            history_compressor=model_config.get(
                "history_compressor_config"
            ),
        )

    def apply_to_model_mapping(
        self,
        model_config: MutableMapping[str, Any],
    ) -> None:
        """Write normalized primitive mappings back to a YAML model section."""
        if not isinstance(model_config, MutableMapping):
            raise TypeError(
                "model config must be mutable, got "
                f"{type(model_config).__name__}"
            )
        model_config.update(self.architecture_mapping())

    def architecture_mapping(self) -> dict[str, dict[str, Any]]:
        """Return normalized public extension configuration."""
        return {
            "sam2_config": self.sam2.to_mapping(),
            "history_compressor_config": (
                self.history_compressor.to_mapping()
            ),
        }

    def checkpoint_hparams_mapping(self) -> dict[str, Any]:
        """Return public extension config plus private checkpoint layout metadata."""
        hparams: dict[str, Any] = self.architecture_mapping()
        history_metadata = self.history_compressor.checkpoint_metadata()
        if history_metadata:
            hparams[_CHECKPOINT_METADATA_KEY] = {
                "history_compressor_config": history_metadata,
            }
        return hparams

    def token_layout(
        self,
        num_frames: int,
        dense_tokens_per_frame: int = 256,
    ) -> VisualTokenLayout:
        """Plan prompt placeholders using the compressor's public contract."""
        return VisualTokenLayout(
            self.history_compressor.frame_token_counts(
                num_frames,
                dense_tokens_per_frame,
            )
        )

    def validate_for(
        self,
        context: ModelExtensionContext,
    ) -> ExtensionInputRequirements:
        """Validate support and return stage-specific mask requirements."""
        if not isinstance(context, ModelExtensionContext):
            raise TypeError("context must be a ModelExtensionContext")

        history = self.history_compressor
        if history.requires_history_boxes and context.model_type is not None and context.model_type.lower() not in {"custom", "internvl3"}:
            raise ValueError("bbox target_selection currently requires model_type='custom' or 'internvl3' with grounding heads")
        any_extension_enabled = self.sam2.enabled or history.enabled
        backend = context.backend.lower()
        if any_extension_enabled and backend != "pytorch":
            raise ValueError(
                "SAM2/history model extensions currently support only the "
                f"PyTorch backend, got {context.backend!r}"
            )

        if context.dataset_type is not None:
            dataset_type = context.dataset_type.lower()
            if self.sam2.enabled and dataset_type != "trackvla":
                raise ValueError(
                    "model.sam2_config.enabled currently requires "
                    "data.dataset_type='trackvla'"
                )
            if history.enabled and dataset_type != "trackvla":
                raise ValueError(
                    "model.history_compressor_config.enabled currently "
                    "requires data.dataset_type='trackvla'"
                )

        supported_history_models = {"custom", "internvl3", "qwen3vl"}
        if (
            history.enabled
            and context.model_type is not None
            and context.model_type.lower() not in supported_history_models
        ):
            raise ValueError(
                "History token compression currently supports only model_type "
                "'custom', 'internvl3', or 'qwen3vl'"
            )
        if (
            history.enabled
            and context.dense_tokens_per_frame is not None
            and context.model_type is not None
            and context.model_type.lower() == "internvl3"
            and context.dense_tokens_per_frame != 256
        ):
            raise ValueError(
                "InternVL3 history compression requires exactly 256 dense "
                "visual tokens per frame, got "
                f"{context.dense_tokens_per_frame}"
            )
        if (
            history.enabled
            and context.dense_tokens_per_frame is not None
            and (
                context.model_type is None
                or context.model_type.lower() in {"custom", "qwen3vl"}
            )
            and math.isqrt(context.dense_tokens_per_frame) ** 2
            != context.dense_tokens_per_frame
        ):
            raise ValueError(
                "History compression requires a square dense visual-token "
                "grid per frame, got "
                f"{context.dense_tokens_per_frame}"
            )
        if (
            history.enabled
            and context.model_type is not None
            and context.model_type.lower() == "qwen3vl"
            and math.isqrt(history.num_output_tokens) ** 2
            != history.num_output_tokens
        ):
            raise ValueError(
                "Qwen3-VL history compression requires num_output_tokens to "
                "be a perfect square so compressed image placeholders retain "
                "a valid two-dimensional MRoPE grid, got "
                f"{history.num_output_tokens}"
            )

        qformer_enabled = history.requires_history_masks
        if (
            context.stage in {ExecutionStage.TRAIN, ExecutionStage.ONLINE_EVAL}
            and qformer_enabled
            and not self.sam2.enabled
        ):
            stage_name = (
                "training"
                if context.stage is ExecutionStage.TRAIN
                else "online evaluation"
            )
            raise ValueError(
                f"{history.mode} {stage_name} requires "
                "model.sam2_config.enabled=true so current predictions can "
                "supply future history masks"
            )
        if context.stage is ExecutionStage.EXPORT and history.enabled:
            raise ValueError(
                "History token compression is not supported by ONNX/TensorRT "
                "export"
            )

        return ExtensionInputRequirements(
            history_boxes=(history.requires_history_boxes and context.stage in {ExecutionStage.TRAIN, ExecutionStage.OFFLINE_EVAL}),
            online_box_cache=(history.requires_history_boxes and context.stage is ExecutionStage.ONLINE_EVAL),
            current_gt_masks=(
                self.sam2.enabled
                and context.stage
                in {ExecutionStage.TRAIN, ExecutionStage.OFFLINE_EVAL}
            ),
            history_masks=(
                qformer_enabled
                and context.stage
                in {ExecutionStage.TRAIN, ExecutionStage.OFFLINE_EVAL}
            ),
            online_mask_cache=(
                qformer_enabled
                and context.stage is ExecutionStage.ONLINE_EVAL
            ),
        )

    @staticmethod
    def _checkpoint_component_metadata(
        checkpoint_hparams: Mapping[str, Any],
        field_name: str,
    ) -> Optional[Mapping[str, Any]]:
        """Read one extension's private checkpoint metadata, if present."""
        metadata = checkpoint_hparams.get(_CHECKPOINT_METADATA_KEY)
        if metadata is None:
            return None
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"Checkpoint {_CHECKPOINT_METADATA_KEY} must be a mapping"
            )
        component_metadata = metadata.get(field_name)
        if component_metadata is None:
            return None
        if not isinstance(component_metadata, Mapping):
            raise TypeError(
                "Checkpoint metadata for "
                f"{field_name} must be a mapping"
            )
        return component_metadata

    @staticmethod
    def _checkpoint_compatible(
        configured: _ConfigT,
        checkpoint: _ConfigT,
    ) -> bool:
        checker = getattr(configured, "checkpoint_compatible_with", None)
        if callable(checker):
            return bool(checker(checkpoint))
        if configured.enabled != checkpoint.enabled:
            return False
        if not configured.enabled:
            return True
        return configured.to_mapping() == checkpoint.to_mapping()

    @classmethod
    def _resolve_component(
        cls,
        *,
        field_name: str,
        configured: _ConfigT,
        configured_explicitly: bool,
        checkpoint_hparams: Mapping[str, Any],
        checkpoint_metadata: Optional[Mapping[str, Any]],
        config_type: type[_ConfigT],
    ) -> _ConfigT:
        checkpoint_value = checkpoint_hparams.get(field_name)
        if checkpoint_value is None:
            if configured.enabled:
                raise ValueError(
                    f"The config enables model.{field_name}, but the "
                    f"checkpoint has no {field_name} architecture metadata"
                )
            return configured

        checkpoint_loader = getattr(
            config_type,
            "from_checkpoint_value",
            getattr(config_type, "from_value"),
        )
        if not hasattr(config_type, "from_checkpoint_value"):
            checkpoint_config = config_type.from_value(checkpoint_value)
        else:
            checkpoint_config = checkpoint_loader(
            checkpoint_value,
            metadata=checkpoint_metadata,
        )
        if configured_explicitly and not cls._checkpoint_compatible(
            configured,
            checkpoint_config,
        ):
            raise ValueError(
                f"Config model.{field_name} differs from the checkpoint "
                "architecture"
            )
        # Explicit config keeps runtime/resource overrides after its active
        # architecture has been proven compatible. When absent, the checkpoint
        # remains authoritative and supplies the full extension configuration.
        return configured if configured_explicitly else checkpoint_config

    @classmethod
    def resolve_checkpoint(
        cls,
        *,
        model_config: Mapping[str, Any],
        checkpoint_hparams: Optional[Mapping[str, Any]],
        context: Optional[ModelExtensionContext] = None,
    ) -> "ModelExtensionsConfig":
        """Resolve extension architecture with checkpoint metadata authoritative.

        Explicit YAML values are allowed only when compatible. Missing
        metadata remains compatible with legacy checkpoints only while the
        corresponding extension is disabled.
        """
        if not isinstance(model_config, Mapping):
            raise TypeError("model_config must be a mapping")
        if checkpoint_hparams is None:
            checkpoint_hparams = {}
        if not isinstance(checkpoint_hparams, Mapping):
            raise TypeError("checkpoint_hparams must be a mapping or None")

        configured = cls.from_model_mapping(model_config)
        sam2_metadata = cls._checkpoint_component_metadata(
            checkpoint_hparams,
            "sam2_config",
        )
        history_metadata = cls._checkpoint_component_metadata(
            checkpoint_hparams,
            "history_compressor_config",
        )
        sam2 = cls._resolve_component(
            field_name="sam2_config",
            configured=configured.sam2,
            configured_explicitly=(
                "sam2_config" in model_config
                and model_config["sam2_config"] is not None
            ),
            checkpoint_hparams=checkpoint_hparams,
            checkpoint_metadata=sam2_metadata,
            config_type=SAM2Config,
        )
        history = cls._resolve_component(
            field_name="history_compressor_config",
            configured=configured.history_compressor,
            configured_explicitly=(
                "history_compressor_config" in model_config
                and model_config["history_compressor_config"] is not None
            ),
            checkpoint_hparams=checkpoint_hparams,
            checkpoint_metadata=history_metadata,
            config_type=HistoryCompressorConfig,
        )
        resolved = cls(sam2=sam2, history_compressor=history)
        if context is not None:
            resolved.validate_for(context)
        return resolved

    def assert_checkpoint_compatible(
        self,
        checkpoint_hparams: Optional[Mapping[str, Any]],
    ) -> None:
        """Validate checkpoint metadata against an already-built model."""
        self.resolve_checkpoint(
            model_config=self.architecture_mapping(),
            checkpoint_hparams=checkpoint_hparams,
        )
