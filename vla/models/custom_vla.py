"""Composable VLA adapter for an independent Hugging Face ViT and LLM."""

from __future__ import annotations

import gc
import inspect
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from vla.config import (
    AuxiliaryTasksConfig,
    CustomVLAConfig,
    HistoryCompressorConfig,
    ModelExtensionContext,
    ModelExtensionsConfig,
    SAM2Config,
)
from vla.heads import (
    GroundingVQAHeads,
    compress_flattened_history_sequence,
    HistoryTokenCompressor,
    SAM2MaskAuxiliaryHead,
    TrajectoryMLP,
)


_KNOWN_IMAGE_TOKENS = ("<IMG_CONTEXT>", "<|image_pad|>", "<image>")
_CONFIG_IMAGE_TOKEN_IDS = ("image_token_id", "img_context_token_id")
_INTERNAL_IMAGE_TOKEN = "<|vision_token|>"


def _resolve_submodule(module: Any, dotted_path: str) -> Any:
    value = module
    for part in dotted_path.split("."):
        if not part or not hasattr(value, part):
            raise AttributeError(
                f"{type(module).__name__} has no submodule path {dotted_path!r}"
            )
        value = getattr(value, part)
    return value


def _config_hidden_size(config: Any, *, component: str) -> int:
    for candidate in (
        getattr(config, "out_hidden_size", None),
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "llm_config", None), "hidden_size", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
    raise ValueError(f"Could not determine {component} hidden_size from its config")


def _uses_legacy_internvl_remote_code(config: Any) -> bool:
    """Whether an InternVL checkpoint uses the legacy ``InternVLChatModel`` API."""
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type.startswith("internvl_chat"):
        return True
    architectures = tuple(getattr(config, "architectures", ()) or ())
    auto_map = getattr(config, "auto_map", {}) or {}
    remote_targets = tuple(auto_map.values()) if isinstance(auto_map, Mapping) else ()
    identifiers = (model_type, *architectures, *remote_targets)
    return any(
        "internvlchatmodel" in str(identifier).replace("_", "").lower()
        or "modeling_internvl_chat" in str(identifier).lower()
        for identifier in identifiers
    )


class CustomVLA(nn.Module):
    """Trajectory model assembled from independently selected ViT and LLM weights."""

    def __init__(
        self,
        vision_model_path: str,
        llm_model_path: str,
        n_future: int,
        input_size: int = 448,
        num_queries: int = 24,
        freeze_vision: bool = True,
        freeze_llm: bool = False,
        freeze_action_head: bool = False,
        action_head_dims: Tuple[int, ...] = (512, 2),
        action_head_type: str = "legacy",
        dtype: Optional[torch.dtype] = None,
        local_files_only: bool = True,
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = False,
        custom_config: Optional[Union[CustomVLAConfig, Mapping[str, Any]]] = None,
        sam2_config: Optional[Union[SAM2Config, Mapping[str, Any]]] = None,
        history_compressor_config: Optional[
            Union[HistoryCompressorConfig, Mapping[str, Any]]
        ] = None,
        auxiliary_tasks_config: Optional[
            Union[AuxiliaryTasksConfig, Mapping[str, Any]]
        ] = None,
    ) -> None:
        super().__init__()
        if not isinstance(vision_model_path, str) or not vision_model_path.strip():
            raise ValueError("vision_model_path must be a non-empty string")
        if not isinstance(llm_model_path, str) or not llm_model_path.strip():
            raise ValueError("llm_model_path must be a non-empty string")
        if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")

        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.input_size = int(input_size)
        self.num_queries = int(num_queries)
        self.n_future = int(n_future)
        self.custom_config = CustomVLAConfig.from_mapping(custom_config)
        self.action_head_type = TrajectoryMLP.canonicalize_action_head_type(
            action_head_type
        )
        self.model_extensions = ModelExtensionsConfig.from_values(
            sam2=sam2_config,
            history_compressor=history_compressor_config,
        )
        self.sam2_config = self.model_extensions.sam2
        self.history_compressor_config = self.model_extensions.history_compressor
        self.mask_aux_enabled = self.sam2_config.enabled
        self.auxiliary_tasks_config = AuxiliaryTasksConfig.from_value(
            auxiliary_tasks_config
        )
        self.history_compressor_config.validate_grounding(
            self.auxiliary_tasks_config.grounding_enabled)

        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            )
        self.dtype = dtype

        processor_path = self.custom_config.image_processor_path or vision_model_path
        tokenizer_path = self.custom_config.tokenizer_path or llm_model_path
        print(f"[CustomVLA] Loading image processor from {processor_path} ...")
        self.image_processor = AutoImageProcessor.from_pretrained(
            processor_path,
            trust_remote_code=self.custom_config.vision_trust_remote_code,
            local_files_only=local_files_only,
        )
        print(f"[CustomVLA] Loading vision model from {vision_model_path} ...")
        vision_kwargs = {
            "dtype": dtype,
            "trust_remote_code": self.custom_config.vision_trust_remote_code,
            "local_files_only": local_files_only,
            "device_map": None,
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation,
        }
        # Legacy InternVL3.5 uses remote ``InternVLChatModel`` code.  Like the
        # nested LLM loader below, its top-level constructor does not advertise
        # SDPA support even though only its vision child will be retained.
        # Resolve its own FlashAttention switch before instantiating it.
        if self.custom_config.vision_trust_remote_code:
            vision_config_hint = AutoConfig.from_pretrained(
                vision_model_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            if _uses_legacy_internvl_remote_code(vision_config_hint):
                vision_kwargs["config"] = vision_config_hint
                vision_kwargs["use_flash_attn"] = (
                    attn_implementation == "flash_attention_2"
                )
                if attn_implementation != "flash_attention_2":
                    vision_kwargs["attn_implementation"] = "eager"
        loaded_vision_model = AutoModel.from_pretrained(
            vision_model_path,
            **vision_kwargs,
        )
        self.vision_adapter = self._resolve_vision_adapter(loaded_vision_model)
        if (
            self.vision_adapter == "qwen3vl"
            and self.custom_config.vision_feature_layer != -1
        ):
            raise ValueError(
                "Qwen3-VL custom vision exposes only final merged features; "
                "set model.custom.vision_feature_layer=-1"
            )
        self.vision_model = self._extract_vision_backbone(
            loaded_vision_model,
            submodule=self.custom_config.vision_submodule,
            adapter=self.vision_adapter,
        )
        self.vision_hidden_size = _config_hidden_size(
            self.vision_model.config,
            component="vision model",
        )
        self.vision_prefix_tokens = self._resolve_vision_prefix_tokens(
            self.vision_model.config,
            0 if self.vision_adapter == "qwen3vl" else self.custom_config.vision_prefix_tokens,
        )
        self.patch_size = self._resolve_patch_size(self.vision_model.config)
        if self.vision_adapter == "qwen3vl":
            merge_size = getattr(self.vision_model.config, "spatial_merge_size", None)
            if (
                not isinstance(merge_size, int)
                or isinstance(merge_size, bool)
                or merge_size <= 0
            ):
                raise ValueError(
                    "Qwen3-VL vision config must expose a positive spatial_merge_size"
                )
            self.patch_size = tuple(value * merge_size for value in self.patch_size)
        self.vision_grid_size = self._resolve_vision_grid_size(
            self.input_size,
            self.patch_size,
        )
        self.image_grid_size, self.projector_input_size = (
            self._resolve_vision_token_reduction(
                raw_grid_size=self.vision_grid_size,
                vision_hidden_size=self.vision_hidden_size,
                mode=self.custom_config.vision_token_reduction,
                reduction_factor=self.custom_config.vision_token_reduction_factor,
            )
        )
        self.image_seq_length = self.image_grid_size[0] * self.image_grid_size[1]

        print(f"[CustomVLA] Loading tokenizer from {tokenizer_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.custom_config.llm_trust_remote_code,
            local_files_only=local_files_only,
            # InternVL3.5-1B ships a tokenizer derived from the affected
            # Mistral/Tekken regex. Transformers ignores this compatibility
            # flag for unrelated tokenizers, while affected checkpoints need
            # it to avoid silently splitting text with the legacy expression.
            fix_mistral_regex=True,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token

        print(f"[CustomVLA] Loading LLM source from {llm_model_path} ...")
        llm_kwargs = {
            "dtype": dtype,
            "trust_remote_code": self.custom_config.llm_trust_remote_code,
            "local_files_only": local_files_only,
            "device_map": None,
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation,
        }
        parent_model = None
        if self.custom_config.llm_loader == "nested_vlm":
            parent_config_hint = AutoConfig.from_pretrained(
                llm_model_path,
                trust_remote_code=self.custom_config.llm_trust_remote_code,
                local_files_only=local_files_only,
            )
            # InternVL remote code has its own flash-attention switch whose
            # default is True. Other nested VLM constructors may reject this
            # private kwarg, so gate it by the resolved parent architecture.
            if _uses_legacy_internvl_remote_code(parent_config_hint):
                llm_kwargs["use_flash_attn"] = (
                    attn_implementation == "flash_attention_2"
                )
                # The remote InternVLChatModel wrapper does not advertise SDPA
                # support to Transformers. It also configures its nested Qwen
                # attention from use_flash_attn, so use eager unless FlashAttention
                # was explicitly requested. The wrapper is discarded after its
                # language_model is extracted.
                if attn_implementation != "flash_attention_2":
                    llm_kwargs["attn_implementation"] = "eager"
            llm_kwargs["config"] = parent_config_hint
            parent_model = AutoModel.from_pretrained(llm_model_path, **llm_kwargs)
            language_model = _resolve_submodule(
                parent_model,
                self.custom_config.llm_submodule,
            )
            if not isinstance(language_model, nn.Module):
                raise TypeError(
                    "model.custom.llm_submodule must resolve to a torch.nn.Module"
                )
            self.llm = language_model
            parent_config = parent_model.config
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                **llm_kwargs,
            )
            parent_config = self.llm.config

        self.llm_hidden_size = _config_hidden_size(
            self.llm.config,
            component="language model",
        )
        self.image_token, self.img_context_token_id, token_was_added = (
            self._resolve_image_token(
                tokenizer=self.tokenizer,
                parent_model=parent_model,
                parent_config=parent_config,
                loader=self.custom_config.llm_loader,
                override=self.custom_config.image_token_override,
            )
        )
        if token_was_added:
            resize_embeddings = getattr(self.llm, "resize_token_embeddings", None)
            if not callable(resize_embeddings):
                raise RuntimeError(
                    "The selected standalone LLM cannot resize token embeddings "
                    "for the internal visual placeholder token"
                )
            resize_embeddings(len(self.tokenizer))

        # Keep only the extracted language model for nested VLM sources.
        parent_model = None
        gc.collect()

        projector_hidden_size = (
            self.llm_hidden_size
            if self.custom_config.projector_hidden_size is None
            else self.custom_config.projector_hidden_size
        )
        self.projector_hidden_size = int(projector_hidden_size)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.projector_input_size),
            nn.Linear(self.projector_input_size, self.projector_hidden_size),
            nn.GELU(),
            nn.Linear(self.projector_hidden_size, self.llm_hidden_size),
        ).to(dtype)
        if self.custom_config.projector_init == "pretrained":
            projector_source = self._initialize_projector_from_vision_vlm(
                loaded_vision_model
            )
            print(
                "[CustomVLA] Initialized projector from pretrained vision VLM "
                f"submodule {projector_source}."
            )
        # Only the extracted tower and copied bridge parameters are needed now;
        # release unused language/head modules owned by a full vision-side VLM.
        loaded_vision_model = None
        gc.collect()
        self.query_tokens = nn.Embedding(
            self.num_queries,
            self.llm_hidden_size,
        ).to(dtype)
        nn.init.normal_(self.query_tokens.weight, std=0.02)
        self.action_head = TrajectoryMLP(
            self.llm_hidden_size,
            self.num_queries,
            list(action_head_dims),
            n_future=self.n_future,
            head_type=self.action_head_type,
        ).to(dtype)

        self.grounding_query_token: Optional[nn.Embedding] = None
        self.vqa_query_token: Optional[nn.Embedding] = None
        if self.auxiliary_tasks_config.grounding_enabled:
            self.grounding_query_token = nn.Embedding(1, self.llm_hidden_size).to(dtype)
            nn.init.normal_(self.grounding_query_token.weight, std=0.02)
        if self.auxiliary_tasks_config.vqa_enabled:
            self.vqa_query_token = nn.Embedding(1, self.llm_hidden_size).to(dtype)
            nn.init.normal_(self.vqa_query_token.weight, std=0.02)
        self.auxiliary_heads = GroundingVQAHeads(
            self.llm_hidden_size, self.auxiliary_tasks_config, dtype=dtype
        )

        compressor_hidden_size = (
            self.llm_hidden_size
            if self.history_compressor_config.hidden_size is None
            else self.history_compressor_config.hidden_size
        )
        self.history_compressor = HistoryTokenCompressor(
            config=self.history_compressor_config,
            hidden_size=compressor_hidden_size,
            llm_hidden_size=self.llm_hidden_size,
            expected_tokens_per_frame=self.image_seq_length,
            dtype=dtype,
        )
        self.mask_query_token: Optional[nn.Embedding] = None
        self.mask_auxiliary_head: Optional[SAM2MaskAuxiliaryHead] = None
        if self.mask_aux_enabled:
            self.mask_query_token = nn.Embedding(1, self.llm_hidden_size).to(dtype)
            nn.init.normal_(self.mask_query_token.weight, std=0.02)
            self.mask_auxiliary_head = SAM2MaskAuxiliaryHead(
                sam2_config=self.sam2_config,
                image_feature_dim=self.llm_hidden_size,
                language_feature_dim=self.llm_hidden_size,
            )

        self.model_extensions.validate_for(
            ModelExtensionContext(
                model_type="custom",
                dense_tokens_per_frame=self.image_seq_length,
            )
        )
        self._set_requires_grad(self.vision_model, not freeze_vision)
        self._set_requires_grad(self.llm, not freeze_llm)
        self._set_requires_grad(self.action_head, not freeze_action_head)
        # The bridge remains trainable after either random or pretrained
        # initialization and is therefore always present in lightweight checkpoints.
        self._set_requires_grad(self.projector, True)
        if gradient_checkpointing:
            enable_checkpointing = getattr(
                self.llm,
                "gradient_checkpointing_enable",
                None,
            )
            if not callable(enable_checkpointing):
                raise RuntimeError(
                    "The selected LLM does not support gradient checkpointing"
                )
            try:
                enable_checkpointing(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                enable_checkpointing()
        self._print_param_stats()

    def structure_mapping(self) -> dict:
        """Return checkpoint-critical structure resolved from both backbones."""
        return {
            "image_token": self.image_token,
            "image_token_id": int(self.img_context_token_id),
            "vision_hidden_size": int(self.vision_hidden_size),
            "projector_input_size": int(self.projector_input_size),
            "llm_hidden_size": int(self.llm_hidden_size),
            "vision_prefix_tokens": int(self.vision_prefix_tokens),
            "patch_size": [int(value) for value in self.patch_size],
            "vision_grid_size": [int(value) for value in self.vision_grid_size],
            "image_grid_size": [int(value) for value in self.image_grid_size],
            "image_seq_length": int(self.image_seq_length),
            "vision_token_reduction": self.custom_config.vision_token_reduction,
            "vision_adapter": getattr(self, "vision_adapter", "standard"),
            "vision_submodule": self.custom_config.vision_submodule,
            "projector_type": self.custom_config.projector_type,
            "projector_hidden_size": int(self.projector_hidden_size),
            "auxiliary_tasks": {
                "grounding_enabled": self.auxiliary_tasks_config.grounding_enabled,
                "vqa_enabled": self.auxiliary_tasks_config.vqa_enabled,
            },
        }

    def assert_checkpoint_structure(self, saved: Any) -> None:
        if not isinstance(saved, Mapping):
            raise TypeError("custom checkpoint structure must be a mapping")
        current = self.structure_mapping()
        mismatches = []
        for key, current_value in current.items():
            if key not in saved:
                # Before VLM vision towers were supported, every custom
                # checkpoint necessarily used the standard ViT protocol.
                if key == "vision_adapter" and current_value == "standard":
                    continue
                if key == "vision_submodule" and current_value == "auto":
                    continue
                if key == "auxiliary_tasks" and current_value == {
                    "grounding_enabled": False,
                    "vqa_enabled": False,
                }:
                    continue
                mismatches.append(f"{key}: missing (current {current_value!r})")
                continue
            saved_value = saved[key]
            if key in {"patch_size", "vision_grid_size", "image_grid_size"}:
                saved_value = list(saved_value)
            if saved_value != current_value:
                mismatches.append(
                    f"{key}: checkpoint {saved_value!r}, current {current_value!r}"
                )
        if mismatches:
            raise ValueError(
                "Custom VLA checkpoint structure mismatch: " + "; ".join(mismatches)
            )

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    @staticmethod
    def _projector_components(
        projector: nn.Module,
    ) -> Tuple[nn.LayerNorm, nn.Linear, nn.Linear]:
        """Resolve the parameterized layers of a supported InternVL projector."""
        if isinstance(projector, nn.Sequential):
            layer_norms = [
                module for module in projector if isinstance(module, nn.LayerNorm)
            ]
            linears = [
                module for module in projector if isinstance(module, nn.Linear)
            ]
            if len(layer_norms) == 1 and len(linears) == 2:
                return layer_norms[0], linears[0], linears[1]

        layer_norm = getattr(projector, "layer_norm", None)
        linear_1 = getattr(projector, "linear_1", None)
        linear_2 = getattr(projector, "linear_2", None)
        if (
            isinstance(layer_norm, nn.LayerNorm)
            and isinstance(linear_1, nn.Linear)
            and isinstance(linear_2, nn.Linear)
        ):
            return layer_norm, linear_1, linear_2

        raise TypeError(
            "Pretrained projector must have InternVL's LayerNorm -> Linear -> "
            "activation -> Linear topology"
        )

    def _initialize_projector_from_vision_vlm(self, vision_parent: nn.Module) -> str:
        """Copy a compatible pretrained InternVL bridge into the custom layout."""
        source_name = None
        source_projector = None
        for candidate in ("multi_modal_projector", "mlp1"):
            value = getattr(vision_parent, candidate, None)
            if isinstance(value, nn.Module):
                source_name = candidate
                source_projector = value
                break
        if source_projector is None or source_name is None:
            raise ValueError(
                "model.custom.projector_init='pretrained' requires "
                "vision_model_path to point to a full InternVL VLM exposing "
                "multi_modal_projector (native InternVL3) or mlp1 (legacy "
                "InternVL/InternVL3.5); a standalone ViT has no compatible "
                "pretrained vision-to-LLM bridge"
            )

        source_components = self._projector_components(source_projector)
        target_components = self._projector_components(self.projector)
        source_state = tuple(
            parameter
            for component in source_components
            for parameter in (component.weight, component.bias)
        )
        target_state = tuple(
            parameter
            for component in target_components
            for parameter in (component.weight, component.bias)
        )
        mismatches = [
            f"parameter {index}: source {tuple(source.shape)} != target {tuple(target.shape)}"
            for index, (source, target) in enumerate(zip(source_state, target_state))
            if tuple(source.shape) != tuple(target.shape)
        ]
        if mismatches:
            raise ValueError(
                "The pretrained vision VLM projector is incompatible with the "
                "selected CustomVLA vision reduction/LLM dimensions: "
                + "; ".join(mismatches)
            )
        with torch.no_grad():
            for source, target in zip(source_state, target_state):
                target.copy_(source.to(device=target.device, dtype=target.dtype))
        return source_name

    def _resolve_vision_adapter(self, model: nn.Module) -> str:
        configured = self.custom_config.vision_adapter
        if configured != "auto":
            return configured
        model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
        return "qwen3vl" if model_type.startswith("qwen3_vl") else "standard"

    @staticmethod
    def _extract_vision_backbone(
        model: nn.Module,
        *,
        submodule: str = "auto",
        adapter: str = "standard",
    ) -> nn.Module:
        """Select the vision tower when a vision-language checkpoint is supplied.

        ``AutoModel`` resolves ``google/siglip-*`` to ``SiglipModel``, whose
        top-level forward returns contrastive-model outputs rather than the
        patch-level ``last_hidden_state`` required by CustomVLA.  Its
        ``vision_model`` child is the corresponding vision-only transformer.
        Keep ordinary ViTs untouched.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                "AutoModel.from_pretrained() must return a torch.nn.Module, got "
                f"{type(model).__name__}"
            )
        model_type = str(
            getattr(getattr(model, "config", None), "model_type", "")
        ).lower()
        if submodule != "auto":
            tower = _resolve_submodule(model, submodule)
            if not isinstance(tower, nn.Module):
                raise TypeError("model.custom.vision_submodule must resolve to a torch.nn.Module")
            return tower
        if adapter == "qwen3vl":
            for candidate in ("visual", "model.visual"):
                try:
                    tower = _resolve_submodule(model, candidate)
                except AttributeError:
                    continue
                if isinstance(tower, nn.Module):
                    return tower
            raise ValueError("Qwen3-VL model does not expose visual or model.visual")
        if model_type.startswith("internvl"):
            for candidate in ("vision_tower", "vision_model"):
                try:
                    tower = _resolve_submodule(model, candidate)
                except AttributeError:
                    continue
                if isinstance(tower, nn.Module):
                    return tower
        vision_tower = getattr(model, "vision_model", None)
        if model_type.startswith("siglip") and isinstance(vision_tower, nn.Module):
            return vision_tower
        return model

    @staticmethod
    def _resolve_patch_size(config: Any) -> Tuple[int, int]:
        patch_size = getattr(config, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(getattr(config, "vision_config", None), "patch_size", None)
        if isinstance(patch_size, int) and not isinstance(patch_size, bool):
            patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, Sequence) and len(patch_size) == 2:
            patch_size = tuple(int(value) for value in patch_size)
        else:
            raise ValueError("Could not determine a two-dimensional ViT patch_size")
        if any(value <= 0 for value in patch_size):
            raise ValueError(f"ViT patch_size must be positive, got {patch_size}")
        return patch_size

    @staticmethod
    def _resolve_vision_grid_size(
        input_size: int,
        patch_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Return the valid-convolution patch grid for a square model input.

        Some pretrained ViTs, notably SigLIP SO400M patch14-384, deliberately
        use an image size that is not divisible by the patch stride.  Their
        patch embedding is a valid convolution, so the emitted grid uses floor
        division (384 // 14 == 27), not an exact-divisibility constraint.
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        height = input_size // patch_size[0]
        width = input_size // patch_size[1]
        if height <= 0 or width <= 0:
            raise ValueError(
                f"input_size={input_size} is smaller than vision patch size "
                f"{patch_size}"
            )
        return height, width

    @staticmethod
    def _resolve_vision_prefix_tokens(config: Any, configured: Union[str, int]) -> int:
        if configured != "auto":
            return int(configured)
        model_type = str(getattr(config, "model_type", "")).lower()
        # SigLIP has no CLS token: all returned sequence positions are patch
        # tokens.  Treating its first position as a generic ViT CLS token would
        # silently discard a real patch and make the grid validation fail.
        if model_type.startswith("siglip"):
            return 0
        register_tokens = getattr(config, "num_register_tokens", 0)
        if not isinstance(register_tokens, int) or isinstance(register_tokens, bool):
            raise TypeError("vision config num_register_tokens must be an integer")
        if register_tokens < 0:
            raise ValueError("vision config num_register_tokens must be non-negative")
        return 1 + register_tokens

    @staticmethod
    def _resolve_vision_token_reduction(
        *,
        raw_grid_size: Tuple[int, int],
        vision_hidden_size: int,
        mode: str,
        reduction_factor: int,
    ) -> Tuple[Tuple[int, int], int]:
        """Resolve output geometry and projector input width before loading the LLM."""
        raw_height, raw_width = raw_grid_size
        if mode == "none":
            return raw_grid_size, vision_hidden_size
        if reduction_factor <= 1:
            raise ValueError(
                f"vision_token_reduction_factor must be greater than 1 for {mode}"
            )
        target_grid = (
            raw_height // reduction_factor,
            raw_width // reduction_factor,
        )
        if target_grid[0] <= 0 or target_grid[1] <= 0:
            raise ValueError(
                "vision_token_reduction_factor must not exceed either dimension "
                f"of the raw ViT grid, got grid {raw_grid_size} and factor "
                f"{reduction_factor}"
            )
        if mode == "pool":
            # AdaptiveAvgPool2d supports arbitrary input/output sizes. In
            # particular, SigLIP patch14-384 emits a 27x27 grid which can be
            # reduced by an approximate factor of two to floor(27 / 2)=13.
            return target_grid, vision_hidden_size
        if mode != "pixel_shuffle":
            raise ValueError(f"Unsupported vision token reduction mode: {mode!r}")
        if raw_height % reduction_factor or raw_width % reduction_factor:
            raise ValueError(
                "vision_token_reduction_factor must divide both dimensions of "
                "the raw ViT grid for pixel_shuffle, got grid "
                f"{raw_grid_size} and factor {reduction_factor}"
            )
        return target_grid, vision_hidden_size * reduction_factor**2

    @staticmethod
    def _internvl_pixel_shuffle(
        feature_map: torch.Tensor,
        reduction_factor: int,
    ) -> torch.Tensor:
        """InternVL v2 pixel shuffle (space-to-depth) for channel-last features."""
        if feature_map.ndim != 4:
            raise ValueError("pixel-shuffle input must have shape [B,H,W,C]")
        if reduction_factor <= 0:
            raise ValueError("pixel-shuffle reduction_factor must be positive")
        batch_size, height, width, channels = feature_map.shape
        if height % reduction_factor or width % reduction_factor:
            raise ValueError(
                "pixel-shuffle spatial dimensions must be divisible by the "
                f"reduction factor, got {(height, width)} and {reduction_factor}"
            )
        scale_factor = 1.0 / reduction_factor
        shuffled = feature_map.reshape(
            batch_size,
            height,
            int(width * scale_factor),
            int(channels / scale_factor),
        )
        shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
        shuffled = shuffled.reshape(
            batch_size,
            int(width * scale_factor),
            int(height * scale_factor),
            int(channels / (scale_factor * scale_factor)),
        )
        return shuffled.permute(0, 2, 1, 3).contiguous()

    def _reduce_vision_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        mode = self.custom_config.vision_token_reduction
        if mode == "none":
            return patch_tokens
        batch_size = patch_tokens.shape[0]
        raw_height, raw_width = self.vision_grid_size
        feature_map = patch_tokens.reshape(
            batch_size,
            raw_height,
            raw_width,
            self.vision_hidden_size,
        )
        if mode == "pool":
            target_height, target_width = self.image_grid_size
            return F.adaptive_avg_pool2d(
                feature_map.permute(0, 3, 1, 2),
                (target_height, target_width),
            ).permute(0, 2, 3, 1).reshape(
                batch_size,
                target_height * target_width,
                self.vision_hidden_size,
            )
        reduction_factor = raw_height // self.image_grid_size[0]
        shuffled = self._internvl_pixel_shuffle(feature_map, reduction_factor)
        return shuffled.reshape(
            batch_size,
            self.image_seq_length,
            self.projector_input_size,
        )

    @staticmethod
    def _single_token_id(tokenizer: Any, token: str) -> Optional[int]:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            return None
        token_id = int(encoded[0])
        if tokenizer.convert_ids_to_tokens(token_id) != token:
            return None
        return token_id

    @classmethod
    def _resolve_image_token(
        cls,
        *,
        tokenizer: Any,
        parent_model: Optional[nn.Module],
        parent_config: Any,
        loader: str,
        override: Optional[str],
    ) -> Tuple[str, int, bool]:
        """Resolve the checkpoint-owned visual token or add one for a text LLM."""
        if override is not None:
            token_id = cls._single_token_id(tokenizer, override)
            if token_id is None:
                added = tokenizer.add_special_tokens(
                    {"additional_special_tokens": [override]}
                )
                token_id = cls._single_token_id(tokenizer, override)
                if added != 1 or token_id is None:
                    raise ValueError(
                        "model.custom.image_token_override must resolve to one "
                        "special tokenizer token"
                    )
                return override, token_id, True
            return override, token_id, False

        configured_ids = set()
        for source in (parent_model, parent_config, getattr(parent_model, "config", None)):
            if source is None:
                continue
            for field_name in _CONFIG_IMAGE_TOKEN_IDS:
                value = getattr(source, field_name, None)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    configured_ids.add(value)
        if len(configured_ids) > 1:
            raise ValueError(
                "The selected LLM/VLM exposes conflicting image token IDs: "
                + ", ".join(map(str, sorted(configured_ids)))
            )

        special_tokens = set(getattr(tokenizer, "all_special_tokens", ()) or ())
        discovered = []
        for token in _KNOWN_IMAGE_TOKENS:
            if token not in special_tokens:
                continue
            token_id = cls._single_token_id(tokenizer, token)
            if token_id is not None:
                discovered.append((token, token_id))

        if configured_ids:
            configured_id = next(iter(configured_ids))
            matching = [item for item in discovered if item[1] == configured_id]
            if len(matching) != 1:
                token = tokenizer.convert_ids_to_tokens(configured_id)
                if not isinstance(token, str) or cls._single_token_id(tokenizer, token) != configured_id:
                    raise ValueError(
                        "The VLM image_token_id does not map to one tokenizer token"
                    )
                return token, configured_id, False
            return matching[0][0], configured_id, False

        # Tokenizers reused inside VLM checkpoints may contain tokens from
        # several multimodal protocols. Let the parent VLM architecture choose
        # its native contract before treating multiple candidates as damage.
        model_types = {
            str(getattr(source, "model_type", "")).lower()
            for source in (
                parent_config,
                getattr(parent_model, "config", None),
            )
            if source is not None
        }
        contract_token = None
        if any(model_type.startswith("internvl") for model_type in model_types):
            contract_token = "<IMG_CONTEXT>"
        elif any(
            model_type.startswith(("qwen2_vl", "qwen3_vl"))
            for model_type in model_types
        ):
            contract_token = "<|image_pad|>"
        if contract_token is not None:
            token_id = cls._single_token_id(tokenizer, contract_token)
            if token_id is None or contract_token not in special_tokens:
                raise ValueError(
                    f"The {sorted(model_types)!r} VLM contract requires "
                    f"{contract_token!r} as one special tokenizer token"
                )
            return contract_token, token_id, False

        unique_discovered = list(dict.fromkeys(discovered))
        if len(unique_discovered) == 1:
            return unique_discovered[0][0], unique_discovered[0][1], False
        if len(unique_discovered) > 1:
            raise ValueError(
                "The selected tokenizer contains multiple plausible image tokens: "
                + ", ".join(token for token, _ in unique_discovered)
            )
        if loader == "nested_vlm":
            raise ValueError(
                "Could not resolve an image token from the nested VLM config/tokenizer"
            )

        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [_INTERNAL_IMAGE_TOKEN]}
        )
        token_id = cls._single_token_id(tokenizer, _INTERNAL_IMAGE_TOKEN)
        if added != 1 or token_id is None:
            raise RuntimeError("Could not add the internal visual placeholder token")
        return _INTERNAL_IMAGE_TOKEN, token_id, True

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.vision_adapter == "qwen3vl":
            return self._encode_qwen3vl_images(pixel_values)
        output_hidden_states = self.custom_config.vision_feature_layer != -1
        typed_pixel_values = pixel_values.to(dtype=self.dtype)
        tokens = None
        if output_hidden_states:
            # Some Transformers 4.57 releases store a bare
            # SiglipVisionTransformer under SiglipModel.vision_model. That
            # child accepts output_hidden_states but lacks the PreTrainedModel
            # output-capture wrapper, so its returned hidden_states is always
            # None. Read the requested encoder state directly for that exact
            # legacy layout. This also avoids changing state-dict paths by
            # wrapping or replacing the loaded vision tower.
            tokens = self._encode_legacy_siglip_intermediate(
                typed_pixel_values,
                self.custom_config.vision_feature_layer,
            )
        if tokens is None:
            forward_parameters = inspect.signature(self.vision_model.forward).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in forward_parameters.values()
            )
            vision_kwargs: Dict[str, Any] = {}
            if "output_hidden_states" in forward_parameters or accepts_kwargs:
                vision_kwargs["output_hidden_states"] = output_hidden_states
            elif output_hidden_states:
                raise ValueError(
                    "The selected vision tower cannot return an intermediate layer; "
                    "set model.custom.vision_feature_layer=-1"
                )
            if "return_dict" in forward_parameters or accepts_kwargs:
                vision_kwargs["return_dict"] = True
            outputs = self.vision_model(
                typed_pixel_values,
                **vision_kwargs,
            )
            if output_hidden_states:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError("The selected ViT did not return hidden_states")
                try:
                    tokens = hidden_states[self.custom_config.vision_feature_layer]
                except IndexError as exc:
                    raise ValueError(
                        "model.custom.vision_feature_layer is outside the ViT "
                        f"hidden_states range (received "
                        f"{self.custom_config.vision_feature_layer}, available "
                        f"states: {len(hidden_states)})"
                    ) from exc
            else:
                tokens = getattr(outputs, "last_hidden_state", None)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            actual = type(tokens).__name__ if tokens is not None else "None"
            raise ValueError(
                "The selected ViT must return [B,T,D] hidden states, got " + actual
            )
        if tokens.shape[-1] != self.vision_hidden_size:
            raise ValueError(
                f"ViT hidden dimension changed from {self.vision_hidden_size} "
                f"to {tokens.shape[-1]}"
            )
        if tokens.shape[1] <= self.vision_prefix_tokens:
            raise ValueError("ViT output contains no patch tokens after prefix removal")
        patch_tokens = tokens[:, self.vision_prefix_tokens :, :]
        raw_token_count = self.vision_grid_size[0] * self.vision_grid_size[1]
        if patch_tokens.shape[1] != raw_token_count:
            raise ValueError(
                "ViT patch-token count does not match configured input geometry: "
                f"expected {raw_token_count}, got {patch_tokens.shape[1]}"
            )
        reduced_tokens = self._reduce_vision_tokens(
            patch_tokens.to(dtype=self.dtype)
        )
        if (
            reduced_tokens.shape[1] != self.image_seq_length
            or reduced_tokens.shape[2] != self.projector_input_size
        ):
            raise RuntimeError(
                "Vision token reduction returned an invalid shape: expected "
                f"[B,{self.image_seq_length},{self.projector_input_size}], got "
                f"{tuple(reduced_tokens.shape)}"
            )
        return self.projector(reduced_tokens)

    def _encode_legacy_siglip_intermediate(
        self,
        pixel_values: torch.Tensor,
        layer_index: int,
    ) -> Optional[torch.Tensor]:
        """Read a layer from the bare SigLIP transformer used by HF 4.57.x.

        Hugging Face hidden-state indexing consists of the patch embeddings at
        index zero followed by one state per encoder layer.  Reproducing that
        convention here keeps values such as ``-2`` identical across affected
        and fixed Transformers releases.
        """
        config = getattr(self.vision_model, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        if model_type not in {"siglip", "siglip_vision_model"}:
            return None
        embeddings = getattr(self.vision_model, "embeddings", None)
        encoder = getattr(self.vision_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if not isinstance(embeddings, nn.Module) or not isinstance(
            layers,
            nn.ModuleList,
        ):
            return None

        state_count = len(layers) + 1
        resolved_index = layer_index if layer_index >= 0 else state_count + layer_index
        if resolved_index < 0 or resolved_index >= state_count:
            raise ValueError(
                "model.custom.vision_feature_layer is outside the SigLIP "
                f"hidden_states range (received {layer_index}, available "
                f"states: {state_count})"
            )

        hidden_states = embeddings(pixel_values)
        if resolved_index == 0:
            return hidden_states
        for state_index, encoder_layer in enumerate(layers, start=1):
            layer_outputs = encoder_layer(hidden_states, None)
            hidden_states = (
                layer_outputs[0]
                if isinstance(layer_outputs, (tuple, list))
                else layer_outputs
            )
            if state_index == resolved_index:
                return hidden_states
        raise RuntimeError("Could not obtain the selected SigLIP encoder state")

    def _encode_qwen3vl_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode Qwen3-VL's patchified image inputs with a fixed square grid.

        Qwen processors return one flattened patch matrix per image instead of
        ``[C,H,W]`` pixels.  Dataset collation preserves it as ``[N,P,D]``;
        infer the common grid from the fixed ``input_size`` and pass it to the
        visual tower exactly as the native Qwen3-VL model does.
        """
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 3:
            raise ValueError(
                "Qwen3-VL custom vision expects processor pixel_values [N,P,D], got "
                f"{getattr(pixel_values, 'shape', None)}"
            )
        raw_height = self.input_size // (self.patch_size[0] // int(self.vision_model.config.spatial_merge_size))
        raw_width = self.input_size // (self.patch_size[1] // int(self.vision_model.config.spatial_merge_size))
        if pixel_values.shape[1] != raw_height * raw_width:
            raise ValueError(
                "Qwen3-VL processor patch count does not match input_size: "
                f"expected {raw_height * raw_width}, got {pixel_values.shape[1]}. "
                "Use an input_size compatible with the Qwen image processor."
            )
        grid_thw = torch.tensor(
            [1, raw_height, raw_width], device=pixel_values.device, dtype=torch.long
        ).expand(pixel_values.shape[0], -1)
        outputs = self.vision_model(
            pixel_values.reshape(-1, pixel_values.shape[-1]).to(dtype=self.dtype),
            grid_thw=grid_thw,
        )
        if isinstance(outputs, (tuple, list)):
            tokens = outputs[0]
        else:
            tokens = getattr(outputs, "last_hidden_state", None)
            if tokens is None:
                tokens = getattr(outputs, "image_embeds", outputs)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("Qwen3-VL visual tower must return merged tokens [N*T,D]")
        expected = self.vision_grid_size[0] * self.vision_grid_size[1]
        if tokens.shape[0] != pixel_values.shape[0] * expected:
            raise ValueError(
                "Qwen3-VL visual output token count does not match its merged grid: "
                f"expected {pixel_values.shape[0] * expected}, got {tokens.shape[0]}"
            )
        tokens = tokens.reshape(pixel_values.shape[0], expected, tokens.shape[-1])
        if tokens.shape[-1] != self.vision_hidden_size:
            raise ValueError(
                f"Qwen3-VL visual hidden dimension changed from {self.vision_hidden_size} "
                f"to {tokens.shape[-1]}"
            )
        reduced_tokens = self._reduce_vision_tokens(tokens.to(dtype=self.dtype))
        return self.projector(reduced_tokens)

    def _llm_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        get_embeddings = getattr(self.llm, "get_input_embeddings", None)
        if not callable(get_embeddings):
            raise RuntimeError("The selected LLM does not expose input embeddings")
        embeddings = get_embeddings()
        if not isinstance(embeddings, nn.Module):
            raise RuntimeError("LLM get_input_embeddings() did not return a module")
        return embeddings(input_ids)

    def _decoder(self) -> nn.Module:
        decoder_path = self.custom_config.llm_decoder_submodule
        # Legacy InternVL checkpoints expose a causal-LM wrapper whose decoder
        # lives at ``language_model.model``. Native HF InternVL loaded through
        # AutoModel already exposes the decoder-only base model at
        # ``language_model``. Preserve the legacy default while accepting that
        # native base model directly when it has no redundant ``model`` child.
        if decoder_path == "model" and not hasattr(self.llm, "model"):
            decoder = self.llm
        else:
            decoder = _resolve_submodule(self.llm, decoder_path)
        if not isinstance(decoder, nn.Module):
            raise TypeError("model.custom.llm_decoder_submodule must resolve to a module")
        return decoder

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        trajectories: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
        history_masks: Optional[torch.FloatTensor] = None,
        history_mask_valid: Optional[torch.BoolTensor] = None,
        history_boxes: Optional[torch.FloatTensor] = None,
        history_box_valid: Optional[torch.BoolTensor] = None,
        current_frame_qformer: Optional[torch.BoolTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,N], got {tuple(input_ids.shape)}")
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError("attention_mask must match input_ids")
        batch_size = input_ids.shape[0]
        if batch_size <= 0:
            raise ValueError("input_ids batch dimension must be positive")
        device = input_ids.device

        projected_features = self._encode_images(pixel_values)
        visual_sequence = compress_flattened_history_sequence(
            self.history_compressor,
            projected_features,
            batch_size=batch_size,
            history_masks=history_masks,
            history_mask_valid=history_mask_valid,
            history_boxes=history_boxes,
            history_box_valid=history_box_valid,
            current_frame_qformer=current_frame_qformer,
            current_masks=masks if self.training and self.history_compressor_config.current_frame_qformer_use_gt_mask else None,
        )
        input_embeds = self._llm_input_embeddings(input_ids)
        input_embeds = self._inject_visual_features(
            input_embeds,
            input_ids,
            visual_sequence.llm_tokens,
            visual_token_counts=getattr(visual_sequence, "llm_token_counts", None),
        )

        query_ids = torch.arange(self.num_queries, device=device).unsqueeze(0)
        query_embeds = self.query_tokens(
            query_ids.expand(batch_size, -1)
        ).to(input_embeds.dtype)
        query_parts = []
        if self.mask_query_token is not None:
            query_parts.append(
                self.mask_query_token.weight.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .to(input_embeds.dtype)
            )
        if self.grounding_query_token is not None:
            query_parts.append(
                self.grounding_query_token.weight.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .to(input_embeds.dtype)
            )
        query_parts.append(query_embeds)
        if self.vqa_query_token is not None:
            query_parts.append(
                self.vqa_query_token.weight.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .to(input_embeds.dtype)
            )
        appended_queries = torch.cat(query_parts, dim=1)
        inputs_embeds = torch.cat((input_embeds, appended_queries), dim=1)
        query_attention = torch.ones(
            batch_size,
            appended_queries.shape[1],
            device=device,
            dtype=attention_mask.dtype,
        )
        full_attention = torch.cat((attention_mask, query_attention), dim=1)
        position_ids = self._build_position_ids(
            attention_mask,
            appended_queries.shape[1],
        )
        outputs = self._decoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden_states = outputs[0]
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            raise RuntimeError("The selected LLM decoder did not return last_hidden_state")
        appended_hidden = hidden_states[:, -appended_queries.shape[1] :, :]
        grounding_index = int(self.mask_query_token is not None)
        action_query_start = grounding_index + int(self.grounding_query_token is not None)
        query_hidden = appended_hidden[
            :, action_query_start : action_query_start + self.num_queries, :
        ]
        pred_traj = self.action_head(query_hidden)
        auxiliary_index = action_query_start + self.num_queries
        grounding_hidden = None
        vqa_hidden = None
        if self.grounding_query_token is not None:
            grounding_hidden = appended_hidden[:, grounding_index, :]
        if self.vqa_query_token is not None:
            vqa_hidden = appended_hidden[:, auxiliary_index, :]
        auxiliary_outputs = self.auxiliary_heads(
            grounding_hidden=grounding_hidden,
            vqa_hidden=vqa_hidden,
        )
        loss = None
        if trajectories is not None:
            loss = F.mse_loss(pred_traj, trajectories.to(pred_traj.dtype))

        mask_outputs: Dict[str, Optional[torch.Tensor]] = {
            "mask_loss": None,
            "mask_iou": None,
        }
        if self.mask_auxiliary_head is not None:
            decoded_masks = self.mask_auxiliary_head.decode(
                self._tokens_to_feature_map(visual_sequence.current_tokens),
                appended_hidden[:, 0, :],
            )
            mask_outputs = {"mask_loss": None, "mask_iou": None, **decoded_masks}
            if masks is not None:
                mask_outputs.update(
                    self.mask_auxiliary_head.compute_loss(decoded_masks, masks)
                )
        return {
            "loss": loss,
            "pred_traj": pred_traj,
            **mask_outputs,
            **auxiliary_outputs,
        }

    def _inject_visual_features(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        visual_token_counts: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        if input_embeds.ndim != 3:
            raise ValueError("input_embeds must have shape [B,N,D]")
        batch_size, sequence_length, hidden_size = input_embeds.shape
        if tuple(input_ids.shape) != (batch_size, sequence_length):
            raise ValueError("input_ids must match input_embeds [B,N]")
        if (
            image_features.ndim != 3
            or image_features.shape[0] != batch_size
            or image_features.shape[2] != hidden_size
        ):
            raise ValueError("image_features must have shape [B,L,D]")
        selected = input_ids == self.img_context_token_id
        counts = selected.sum(dim=1)
        expected = (
            torch.full_like(counts, image_features.shape[1])
            if visual_token_counts is None
            else torch.as_tensor(
                visual_token_counts, device=counts.device, dtype=counts.dtype
            )
        )
        if tuple(expected.shape) != (batch_size,):
            raise ValueError("visual_token_counts must have one count per sample")
        bad_rows = torch.nonzero(counts != expected, as_tuple=False).flatten()
        if bad_rows.numel():
            details = ", ".join(
                f"sample {row}: {int(counts[row].item())}"
                for row in bad_rows.tolist()
            )
            raise ValueError(
                "Custom VLA image-token mismatch: every prompt must contain "
                f"the configured number of {self.image_token} tokens, got {details}"
            )
        result = input_embeds.clone()
        features = image_features.to(device=result.device, dtype=result.dtype)
        for row in range(batch_size):
            result[row, selected[row]] = features[row, : int(expected[row].item())]
        return result

    @staticmethod
    def _build_position_ids(
        attention_mask: torch.Tensor,
        appended_query_count: int,
    ) -> torch.LongTensor:
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B,N]")
        if appended_query_count < 0:
            raise ValueError("appended_query_count must be non-negative")
        valid = attention_mask != 0
        prompt_positions = valid.long().cumsum(dim=-1) - 1
        prompt_positions = prompt_positions.masked_fill(~valid, 0)
        valid_lengths = valid.long().sum(dim=-1, keepdim=True)
        offsets = torch.arange(
            appended_query_count,
            device=attention_mask.device,
            dtype=torch.long,
        ).unsqueeze(0)
        return torch.cat((prompt_positions, valid_lengths + offsets), dim=-1)

    @staticmethod
    def _tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
        token_count = tokens.shape[1]
        side = math.isqrt(token_count)
        if side * side != token_count:
            raise ValueError(
                "SAM2 requires a square current-frame patch grid, got "
                f"{token_count} tokens"
            )
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], side, side
        )

    def _print_param_stats(self) -> None:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        print(f"[CustomVLA] Total params:    {total / 1e6:.2f} M")
        print(f"[CustomVLA] Trainable params: {trainable / 1e6:.2f} M")


__all__ = ["CustomVLA"]
