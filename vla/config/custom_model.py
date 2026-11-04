"""Serializable configuration for independently selected vision and LLM backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union


_LLM_LOADERS = frozenset({"nested_vlm", "causal_lm"})
_PROJECTOR_TYPES = frozenset({"mlp2x_gelu"})
_PROJECTOR_INITS = frozenset({"random", "pretrained"})
_VISION_TOKEN_REDUCTIONS = frozenset({"none", "pool", "pixel_shuffle"})
_VISION_ADAPTERS = frozenset({"auto", "standard", "qwen3vl"})


@dataclass(frozen=True)
class CustomVLAConfig:
    """Configuration for the generic ViT + decoder-only LLM adapter."""

    image_processor_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    vision_feature_layer: int = -1
    vision_prefix_tokens: Union[str, int] = "auto"
    vision_trust_remote_code: bool = False
    # ``auto`` keeps ordinary ViTs unchanged and recognises common VLM vision
    # towers.  ``qwen3vl`` uses Qwen's patchified visual-input contract.
    vision_adapter: str = "auto"
    # Relative path inside a VLM checkpoint (for example ``vision_tower``).
    # ``auto`` resolves InternVL2/3/3.5 and Qwen3-VL conventions.
    vision_submodule: str = "auto"
    vision_token_reduction: str = "none"
    vision_token_reduction_factor: int = 1
    llm_loader: str = "causal_lm"
    llm_submodule: str = "language_model"
    llm_decoder_submodule: str = "model"
    llm_trust_remote_code: bool = False
    projector_type: str = "mlp2x_gelu"
    projector_hidden_size: Optional[int] = None
    # ``pretrained`` copies a compatible projector from a full VLM supplied as
    # vision_model_path. Standalone ViTs keep the default random policy.
    projector_init: str = "random"
    image_token_override: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(
        cls,
        values: Optional[Mapping[str, Any]],
    ) -> "CustomVLAConfig":
        if values is None:
            return cls()
        if isinstance(values, cls):
            values.validate()
            return values
        if not isinstance(values, Mapping):
            raise TypeError(
                "model.custom must be a mapping or null, got "
                f"{type(values).__name__}"
            )
        unknown = set(values) - {
            "image_processor_path",
            "tokenizer_path",
            "vision_feature_layer",
            "vision_prefix_tokens",
            "vision_trust_remote_code",
            "vision_adapter",
            "vision_submodule",
            "vision_token_reduction",
            "vision_token_reduction_factor",
            "llm_loader",
            "llm_submodule",
            "llm_decoder_submodule",
            "llm_trust_remote_code",
            "projector_type",
            "projector_hidden_size",
            "projector_init",
            "image_token_override",
        }
        if unknown:
            raise ValueError(
                "Unsupported model.custom option(s): "
                + ", ".join(sorted(map(str, unknown)))
            )
        return cls(**dict(values))

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "image_processor_path": self.image_processor_path,
            "tokenizer_path": self.tokenizer_path,
            "vision_feature_layer": self.vision_feature_layer,
            "vision_prefix_tokens": self.vision_prefix_tokens,
            "vision_trust_remote_code": self.vision_trust_remote_code,
            "vision_adapter": self.vision_adapter,
            "vision_submodule": self.vision_submodule,
            "vision_token_reduction": self.vision_token_reduction,
            "vision_token_reduction_factor": self.vision_token_reduction_factor,
            "llm_loader": self.llm_loader,
            "llm_submodule": self.llm_submodule,
            "llm_decoder_submodule": self.llm_decoder_submodule,
            "llm_trust_remote_code": self.llm_trust_remote_code,
            "projector_type": self.projector_type,
            "projector_hidden_size": self.projector_hidden_size,
            "projector_init": self.projector_init,
            "image_token_override": self.image_token_override,
        }

    def validate(self) -> None:
        for name in ("vision_trust_remote_code", "llm_trust_remote_code"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"model.custom.{name} must be a boolean")
        if not isinstance(self.vision_feature_layer, int) or isinstance(
            self.vision_feature_layer, bool
        ):
            raise TypeError("model.custom.vision_feature_layer must be an integer")
        if self.vision_prefix_tokens != "auto" and (
            not isinstance(self.vision_prefix_tokens, int)
            or isinstance(self.vision_prefix_tokens, bool)
            or self.vision_prefix_tokens < 0
        ):
            raise ValueError(
                "model.custom.vision_prefix_tokens must be 'auto' or a "
                "non-negative integer"
            )
        if self.vision_adapter not in _VISION_ADAPTERS:
            raise ValueError(
                "Unsupported model.custom.vision_adapter "
                f"{self.vision_adapter!r}; expected one of "
                + ", ".join(sorted(_VISION_ADAPTERS))
            )
        if not isinstance(self.vision_submodule, str) or not self.vision_submodule.strip():
            raise ValueError(
                "model.custom.vision_submodule must be a non-empty string"
            )
        if self.vision_token_reduction not in _VISION_TOKEN_REDUCTIONS:
            raise ValueError(
                "Unsupported model.custom.vision_token_reduction "
                f"{self.vision_token_reduction!r}; expected one of "
                + ", ".join(sorted(_VISION_TOKEN_REDUCTIONS))
            )
        if (
            not isinstance(self.vision_token_reduction_factor, int)
            or isinstance(self.vision_token_reduction_factor, bool)
            or self.vision_token_reduction_factor <= 0
        ):
            raise ValueError(
                "model.custom.vision_token_reduction_factor must be a positive integer"
            )
        if (
            self.vision_token_reduction == "none"
            and self.vision_token_reduction_factor != 1
        ):
            raise ValueError(
                "model.custom.vision_token_reduction_factor must be 1 when "
                "vision_token_reduction='none'"
            )
        if (
            self.vision_token_reduction != "none"
            and self.vision_token_reduction_factor <= 1
        ):
            raise ValueError(
                "model.custom.vision_token_reduction_factor must be greater "
                "than 1 when vision_token_reduction is 'pool' or 'pixel_shuffle'"
            )
        if self.llm_loader not in _LLM_LOADERS:
            raise ValueError(
                f"Unsupported model.custom.llm_loader {self.llm_loader!r}; "
                f"expected one of {', '.join(sorted(_LLM_LOADERS))}"
            )
        if self.projector_type not in _PROJECTOR_TYPES:
            raise ValueError(
                f"Unsupported model.custom.projector_type {self.projector_type!r}; "
                f"expected one of {', '.join(sorted(_PROJECTOR_TYPES))}"
            )
        if self.projector_init not in _PROJECTOR_INITS:
            raise ValueError(
                f"Unsupported model.custom.projector_init {self.projector_init!r}; "
                f"expected one of {', '.join(sorted(_PROJECTOR_INITS))}"
            )
        for name in ("llm_submodule", "llm_decoder_submodule"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"model.custom.{name} must be a non-empty string")
        for name in (
            "image_processor_path",
            "tokenizer_path",
            "image_token_override",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"model.custom.{name} must be null or a non-empty string")
        if self.projector_hidden_size is not None and (
            not isinstance(self.projector_hidden_size, int)
            or isinstance(self.projector_hidden_size, bool)
            or self.projector_hidden_size <= 0
        ):
            raise ValueError(
                "model.custom.projector_hidden_size must be null or a positive integer"
            )


__all__ = ["CustomVLAConfig"]
