"""TrackVLA trajectory head on Hugging Face's native InternVL3 architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from vla.config import (
    AuxiliaryTasksConfig,
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


def encode_internvl3_vit_features(
    internvl: nn.Module,
    pixel_values: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run InternVL3's frozen visual path up to (but excluding) the projector.

    This mirrors Transformers' native ``get_image_features`` implementation:
    feature-layer selection, optional CLS removal, spatial reshape and
    pixel-shuffle are all retained, while the trainable multimodal projector is
    intentionally left for the caller.
    """

    pixel_values = pixel_values.to(dtype=dtype)
    config = internvl.config
    vision_feature_layer = int(getattr(config, "vision_feature_layer", -1))
    if vision_feature_layer != -1:
        raise ValueError(
            "InternVL3 feature caching currently requires vision_feature_layer=-1, "
            f"got {vision_feature_layer}"
        )
    vision_feature_select_strategy = getattr(
        config,
        "vision_feature_select_strategy",
        "default",
    )
    vision_features = internvl.vision_tower(
        pixel_values=pixel_values
    ).last_hidden_state
    if vision_feature_select_strategy == "default":
        vision_features = vision_features[:, 1:, :]
    channels = vision_features.shape[1]
    feature_size = int(channels**0.5)
    if feature_size * feature_size != channels:
        raise ValueError(
            "InternVL3 ViT tokens must form a square grid after feature selection, "
            f"got {channels} tokens"
        )
    vision_features = vision_features.reshape(
        vision_features.shape[0],
        feature_size,
        feature_size,
        vision_features.shape[-1],
    )
    downsample_ratio = float(getattr(config, "downsample_ratio", 0.5))
    pixel_shuffle = getattr(internvl, "pixel_shuffle", None)
    if not callable(pixel_shuffle):
        raise RuntimeError("Loaded InternVL3 model does not expose pixel_shuffle")
    vision_features = pixel_shuffle(
        vision_features,
        scale_factor=downsample_ratio,
    )
    return vision_features.reshape(
        vision_features.shape[0],
        -1,
        vision_features.shape[-1],
    )


def internvl3_vit_feature_cache_metadata(
    internvl: nn.Module,
    *,
    model_path: str,
    input_size: int,
    dtype: torch.dtype,
) -> Dict[str, object]:
    """Build the shared feature-shard compatibility metadata."""

    config = internvl.config
    projector_norm = getattr(internvl.multi_modal_projector, "layer_norm", None)
    normalized_shape = getattr(projector_norm, "normalized_shape", None)
    if isinstance(normalized_shape, tuple):
        projector_input_dim = int(normalized_shape[0])
    elif isinstance(normalized_shape, int):
        projector_input_dim = int(normalized_shape)
    else:
        raise RuntimeError(
            "InternVL3 projector does not expose a usable layer_norm input shape"
        )
    return {
        "model_type": "internvl3",
        "model_path": str(Path(model_path).resolve()),
        "input_size": int(input_size),
        "compute_dtype": str(dtype).replace("torch.", ""),
        "vision_feature_layer": int(getattr(config, "vision_feature_layer", -1)),
        "vision_feature_select_strategy": str(
            getattr(config, "vision_feature_select_strategy", "default")
        ),
        "downsample_ratio": float(getattr(config, "downsample_ratio", 0.5)),
        "image_seq_length": int(getattr(config, "image_seq_length")),
        "projector_input_dim": projector_input_dim,
    }


class InternVL3VLA(nn.Module):
    """InternVL3 VLA model with learnable trajectory query tokens.

    InternVL3's native Hugging Face implementation differs from InternVL2.5's
    remote-code API. It exposes ``vision_tower``, ``multi_modal_projector`` and
    ``get_image_features``; this adapter injects those projected image features
    into the text embeddings before sending them through the Qwen decoder.

    Architecture:
      1. The native vision tower and projector encode every history RGB frame.
      2. Projected image features replace ``<IMG_CONTEXT>`` token embeddings.
      3. Learnable trajectory query tokens are appended after the prompt.
      4. The native Qwen decoder processes the complete causal sequence.
      5. An MLP maps the final query states to future trajectory points.
    """

    def __init__(
        self,
        model_path: str,
        n_future: int,
        num_queries: int = 24,
        freeze_vision: bool = True,
        freeze_llm: bool = False,
        freeze_mlp1: bool = True,
        action_head_dims: Tuple[int, ...] = (512, 2),
        action_head_type: str = "legacy",
        dtype: Optional[torch.dtype] = None,
        local_files_only: bool = True,
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = False,
        sam2_config: Optional[SAM2Config] = None,
        history_compressor_config: Optional[
            Union[HistoryCompressorConfig, Mapping[str, object]]
        ] = None,
        auxiliary_tasks_config: Optional[
            Union[AuxiliaryTasksConfig, Mapping[str, object]]
        ] = None,
    ):
        super().__init__()
        # Resolve a cached Hub ID first so the native fast tokenizer and model
        # load the same local snapshot without probing the network.
        model_path = self._resolve_model_path(model_path, local_files_only)
        self.model_path = model_path
        self.freeze_vision = bool(freeze_vision)
        self.num_queries = int(num_queries)
        self.n_future = int(n_future)
        self.action_head_type = TrajectoryMLP.canonicalize_action_head_type(
            action_head_type
        )
        self.model_extensions = ModelExtensionsConfig.from_values(
            sam2=sam2_config,
            history_compressor=history_compressor_config,
        )
        self.sam2_config = self.model_extensions.sam2
        self.mask_aux_enabled = self.sam2_config.enabled
        self.history_compressor_config = (
            self.model_extensions.history_compressor
        )
        self.auxiliary_tasks_config = AuxiliaryTasksConfig.from_value(
            auxiliary_tasks_config
        )
        self.history_compressor_config.validate_grounding(
            self.auxiliary_tasks_config.grounding_enabled)
        if self.auxiliary_tasks_config.enabled:
            self.auxiliary_tasks_config.validate()

        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            )
        self.dtype = dtype

        print(f"[InternVL3VLA] Loading tokenizer from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )

        try:
            config = AutoConfig.from_pretrained(
                model_path,
                local_files_only=local_files_only,
            )
        except ValueError as exc:
            if "model type `internvl`" in str(exc):
                raise RuntimeError(
                    "InternVL3 requires a Transformers release with native InternVL "
                    "support (transformers>=4.57.0)."
                ) from exc
            raise
        if getattr(config, "model_type", None) != "internvl":
            raise ValueError(
                f"InternVL3VLA expects a native InternVL checkpoint, got "
                f"model_type={getattr(config, 'model_type', None)!r}."
            )
        if int(getattr(config, "vision_feature_layer", -1)) != -1:
            raise ValueError(
                "InternVL3VLA currently requires config.vision_feature_layer=-1; "
                "other layers are not supported by the native Transformers implementation."
            )
        configured_image_tokens = int(getattr(config, "image_seq_length", 0))
        self.model_extensions.validate_for(
            ModelExtensionContext(
                model_type="internvl3",
                dense_tokens_per_frame=configured_image_tokens,
            )
        )
        configured_llm_hidden_size = int(config.text_config.hidden_size)
        configured_qformer_hidden_size = (
            configured_llm_hidden_size
            if self.history_compressor_config.hidden_size is None
            else self.history_compressor_config.hidden_size
        )
        if (
            (self.history_compressor_config.requires_history_masks or self.history_compressor_config.requires_history_boxes)
            and configured_qformer_hidden_size
            % self.history_compressor_config.num_heads
            != 0
        ):
            raise ValueError(
                "InternVL3 history compressor hidden_size "
                f"({configured_qformer_hidden_size}) must be "
                "divisible by history compressor num_heads "
                f"({self.history_compressor_config.num_heads})"
            )

        print(f"[InternVL3VLA] Loading native InternVL3 model from {model_path} (dtype={dtype}) ...")
        model_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
            "device_map": None,
            "low_cpu_mem_usage": True,
        }
        model_kwargs["attn_implementation"] = attn_implementation
        self.internvl = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.config = self.internvl.config

        required_attributes = (
            "vision_tower",
            "multi_modal_projector",
            "language_model",
            "get_image_features",
        )
        missing_attributes = [
            name for name in required_attributes if not hasattr(self.internvl, name)
        ]
        if missing_attributes:
            raise RuntimeError(
                "Loaded InternVL model does not expose the native InternVL3 API: "
                + ", ".join(missing_attributes)
            )

        self.llm_hidden_size = int(self.config.text_config.hidden_size)
        self.history_compressor_hidden_size = (
            self.llm_hidden_size
            if self.history_compressor_config.hidden_size is None
            else self.history_compressor_config.hidden_size
        )
        self.image_seq_length = int(self.config.image_seq_length)
        self.img_context_token_id = int(self.config.image_token_id)
        tokenizer_image_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if tokenizer_image_token_id != self.img_context_token_id:
            raise ValueError(
                "InternVL3 tokenizer/config image token mismatch: "
                f"<IMG_CONTEXT>={tokenizer_image_token_id}, "
                f"config.image_token_id={self.img_context_token_id}."
            )

        # Learnable trajectory query tokens are the only trainable sequence
        # inputs when the visual tower, projector and language model are frozen.
        self.query_tokens = nn.Embedding(self.num_queries, self.llm_hidden_size).to(dtype)
        nn.init.normal_(self.query_tokens.weight, std=0.02)
        # The head decodes query states into the future trajectory.
        self.action_head = TrajectoryMLP(
            self.llm_hidden_size,
            self.num_queries,
            list(action_head_dims),
            n_future=self.n_future,
            head_type=self.action_head_type,
        ).to(dtype)
        self.grounding_query_token: Optional[nn.Embedding] = None
        if self.auxiliary_tasks_config.grounding_enabled:
            self.grounding_query_token = nn.Embedding(1, self.llm_hidden_size).to(dtype)
            nn.init.normal_(self.grounding_query_token.weight, std=0.02)
        self.vqa_query_token: Optional[nn.Embedding] = None
        if self.auxiliary_tasks_config.vqa_enabled:
            self.vqa_query_token = nn.Embedding(1, self.llm_hidden_size).to(dtype)
            nn.init.normal_(self.vqa_query_token.weight, std=0.02)
        self.auxiliary_heads = GroundingVQAHeads(
            self.llm_hidden_size, self.auxiliary_tasks_config, dtype=dtype
        )
        self.history_compressor = HistoryTokenCompressor(
            config=self.history_compressor_config,
            hidden_size=self.history_compressor_hidden_size,
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

        # Freeze native InternVL3 backbone components independently. The legacy
        # config name freeze_mlp1 maps to the native multimodal projector.
        self._set_requires_grad(self.internvl.vision_tower, not freeze_vision)
        self._set_requires_grad(self.internvl.language_model, not freeze_llm)
        # ``freeze_mlp1`` is retained as the public config name for compatibility
        # with InternVL2.5; InternVL3 calls the equivalent module a projector.
        self._set_requires_grad(self.internvl.multi_modal_projector, not freeze_mlp1)
        if gradient_checkpointing:
            try:
                self.internvl.language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.internvl.language_model.gradient_checkpointing_enable()
            print("[InternVL3VLA] Enabled language-model gradient checkpointing.")

        self._print_param_stats()

    def vit_feature_cache_metadata(
        self,
        input_size: int,
    ) -> Dict[str, object]:
        """Return metadata required to validate TrackVLA ViT feature shards."""

        if not self.freeze_vision:
            raise ValueError(
                "ViT feature caching requires model.freeze_vision=true"
            )
        return internvl3_vit_feature_cache_metadata(
            self.internvl,
            model_path=self.model_path,
            input_size=input_size,
            dtype=self.dtype,
        )

    def project_vit_features(self, vit_features: torch.Tensor) -> torch.Tensor:
        """Apply the native multimodal projector to cached ViT tokens."""

        projector = self.internvl.multi_modal_projector
        projector_parameter = next(projector.parameters(), None)
        if projector_parameter is None:
            raise RuntimeError("InternVL3 projector has no parameters")
        expected_dim = int(projector.layer_norm.normalized_shape[0])
        if vit_features.ndim != 3:
            raise ValueError(
                "vit_features must have shape [B*H,T,D], got "
                f"{tuple(vit_features.shape)}"
            )
        if vit_features.shape[1] != self.image_seq_length:
            raise ValueError(
                "vit_features token count does not match InternVL3 image_seq_length: "
                f"{vit_features.shape[1]} != {self.image_seq_length}"
            )
        if vit_features.shape[2] != expected_dim:
            raise ValueError(
                "vit_features channel count does not match projector input: "
                f"{vit_features.shape[2]} != {expected_dim}"
            )
        vit_features = vit_features.to(
            device=projector_parameter.device,
            dtype=self.dtype,
        )
        return projector(vit_features)

    @staticmethod
    def _resolve_model_path(model_path: str, local_files_only: bool) -> str:
        """Resolve cached Hub IDs before loading the fast native tokenizer."""
        path = Path(model_path).expanduser()
        if path.exists() or not local_files_only:
            return str(path)
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(repo_id=model_path, local_files_only=True)
        except Exception as exc:
            raise FileNotFoundError(
                f"InternVL3 model is not available locally: {model_path}. "
                "Download it first or set local_files_only=False."
            ) from exc

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _print_param_stats(self) -> None:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        print(f"[InternVL3VLA] Total params:    {total / 1e6:.2f} M")
        print(f"[InternVL3VLA] Trainable params: {trainable / 1e6:.2f} M")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        vit_features: Optional[torch.FloatTensor] = None,
        trajectories: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
        history_masks: Optional[torch.FloatTensor] = None,
        history_mask_valid: Optional[torch.BoolTensor] = None,
        history_boxes: Optional[torch.FloatTensor] = None,
        history_box_valid: Optional[torch.BoolTensor] = None,
        current_frame_qformer: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Predict trajectory points from flattened history images and a prompt.

        Args:
            pixel_values: ``(B * num_images, C, H, W)`` normalized RGB frames.
            vit_features: Optional cached FP32 ViT tokens before the projector.
            input_ids: ``(B, N)`` prompt IDs with ``<IMG_CONTEXT>`` placeholders.
            attention_mask: ``(B, N)`` prompt mask before query-token append.
            trajectories: Optional ``(B, n_future, action_dim)`` target.
            history_masks: Optional compressor masks with shape
                ``(B, num_images - 1, H, W)``. Missing QFormer masks use the
                no-target fallback and adaptive pooling ignores them.
            history_mask_valid: Optional validity flags with shape
                ``(B, num_images - 1)``.

        Returns:
            ``pred_traj`` with shape ``(B, n_future, action_dim)`` and an
            optional unweighted MSE ``loss`` for callers that need it. When
            the SAM2 auxiliary head is enabled, the result also contains
            ``mask_logits``, ``pred_iou``, and ``pred_obj_scores``; labeled
            batches additionally contain ``mask_loss`` and ``mask_iou``.
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [B,N], got {tuple(input_ids.shape)}"
            )
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "attention_mask must match input_ids [B,N], got "
                f"{tuple(attention_mask.shape)}"
            )
        batch_size = input_ids.shape[0]
        if batch_size <= 0:
            raise ValueError("input_ids batch dimension must be positive")
        device = input_ids.device
        if (pixel_values is None) == (vit_features is None):
            raise ValueError(
                "Exactly one of pixel_values or vit_features must be supplied"
            )

        # 1. The cached path skips the frozen ViT but keeps the projector in the
        # graph. The uncached path keeps the native helper for compatibility
        # with the Transformers model API and existing test doubles.
        if vit_features is not None:
            if not self.freeze_vision:
                raise RuntimeError(
                    "vit_features cannot be used when freeze_vision=false"
                )
            image_features = self.project_vit_features(vit_features)
        else:
            assert pixel_values is not None
            pixel_values = pixel_values.to(device=device, dtype=self.dtype)
            image_features = self.internvl.get_image_features(
                pixel_values=pixel_values
            )
        visual_sequence = compress_flattened_history_sequence(
            self.history_compressor,
            image_features,
            batch_size=batch_size,
            history_masks=history_masks,
            history_mask_valid=history_mask_valid,
            history_boxes=history_boxes,
            history_box_valid=history_box_valid,
            current_frame_qformer=current_frame_qformer,
            current_masks=masks if self.training and self.history_compressor_config.current_frame_qformer_use_gt_mask else None,
        )
        # 2. Replace each image placeholder with its matching projected feature.
        input_embeds = self.internvl.get_input_embeddings()(input_ids)
        input_embeds = self._inject_visual_features(
            input_embeds,
            input_ids,
            visual_sequence.llm_tokens,
            visual_token_counts=getattr(visual_sequence, "llm_token_counts", None),
        )

        # 3. Append a dedicated mask query before planner queries when enabled.
        # All are per-sample sequence tokens; no video-memory state is present.
        # The order lets every causal planner query attend to the mask query.
        query_ids = torch.arange(self.num_queries, device=device).unsqueeze(0)
        query_ids = query_ids.expand(batch_size, -1)
        query_embeds = self.query_tokens(query_ids).to(input_embeds.dtype)
        query_parts = []
        if self.mask_query_token is not None:
            query_parts.append(
                self.mask_query_token.weight.unsqueeze(0).expand(batch_size, -1, -1).to(input_embeds.dtype)
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
        inputs_embeds = torch.cat([input_embeds, appended_queries], dim=1)

        query_attention_mask = torch.ones(
            batch_size,
            appended_queries.shape[1],
            device=device,
            dtype=attention_mask.dtype,
        )
        full_attention_mask = torch.cat([attention_mask, query_attention_mask], dim=1)
        # 4. Native InternVL3 uses Qwen's 1-D RoPE, not Qwen2-VL MRoPE. Build
        # pad-aware positions so queries follow each sample's real prompt.
        position_ids = self._build_position_ids(attention_mask, appended_queries.shape[1])

        # 5. Call the decoder directly to obtain final states for the planner;
        # caching is unnecessary during training and wastes memory.
        outputs = self.internvl.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        appended_hidden = outputs.last_hidden_state[:, -appended_queries.shape[1] :, :]
        grounding_index = int(self.mask_query_token is not None)
        action_query_start = grounding_index + int(self.grounding_query_token is not None)
        query_hidden = appended_hidden[
            :, action_query_start : action_query_start + self.num_queries, :
        ]
        # 6. Decode query states into trajectory points and optional supervision.
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
        mask_outputs: Dict[str, Optional[torch.Tensor]] = {"mask_loss": None, "mask_iou": None}
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

    @staticmethod
    def _tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
        token_count = tokens.shape[1]
        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError(
                "SAM2 auxiliary head requires square InternVL3 visual tokens, got "
                f"{token_count} tokens"
            )
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    def _inject_visual_features(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        visual_token_counts: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Replace each sample's image slots with only that sample's features.

        Per-row checks are intentional: a batch-total check can silently move
        features across samples when one prompt has too many placeholders and
        another has too few.
        """
        if input_embeds.ndim != 3:
            raise ValueError(
                "input_embeds must have shape [B,N,D], got "
                f"{tuple(input_embeds.shape)}"
            )
        batch_size, sequence_length, hidden_size = input_embeds.shape
        if tuple(input_ids.shape) != (batch_size, sequence_length):
            raise ValueError(
                "input_ids must match input_embeds [B,N], got "
                f"{tuple(input_ids.shape)}"
            )
        if (
            not isinstance(image_features, torch.Tensor)
            or image_features.ndim != 3
            or image_features.shape[0] != batch_size
            or image_features.shape[2] != hidden_size
        ):
            actual_shape = (
                tuple(image_features.shape)
                if isinstance(image_features, torch.Tensor)
                else type(image_features).__name__
            )
            raise ValueError(
                "image_features must have shape [B,L,D] matching embeddings, "
                f"got {actual_shape}"
            )

        selected = input_ids == self.img_context_token_id
        placeholder_counts = selected.sum(dim=1)
        expected_counts = (
            torch.full_like(placeholder_counts, image_features.shape[1])
            if visual_token_counts is None
            else torch.as_tensor(
                visual_token_counts,
                device=placeholder_counts.device,
                dtype=placeholder_counts.dtype,
            )
        )
        if tuple(expected_counts.shape) != (batch_size,):
            raise ValueError("visual_token_counts must have one count per sample")
        mismatched_rows = torch.nonzero(
            placeholder_counts != expected_counts,
            as_tuple=False,
        ).flatten()
        if mismatched_rows.numel() > 0:
            details = ", ".join(
                f"sample {row}: {int(placeholder_counts[row].item())}"
                for row in mismatched_rows.tolist()
            )
            raise ValueError(
                "InternVL3 image-token mismatch: every prompt row must contain "
                "the configured number of <IMG_CONTEXT> tokens, got "
                f"{details}. Keep "
                "the current frame at "
                f"data.num_image_tokens={self.image_seq_length}, keep past-frame "
                "placeholder counts aligned with model.history_compressor_config, "
                "and ensure max_length does not truncate image placeholders."
            )

        result = input_embeds.clone()
        image_features = image_features.to(
            device=result.device,
            dtype=result.dtype,
        )
        for row in range(batch_size):
            result[row, selected[row]] = image_features[
                row, : int(expected_counts[row].item())
            ]
        return result

    @staticmethod
    def _build_position_ids(
        attention_mask: torch.Tensor,
        appended_query_count: int,
    ) -> torch.LongTensor:
        """Use native 1-D RoPE positions that ignore padding before queries.

        InternVL3-1B has a Qwen2 text decoder and does not accept Qwen2-VL's
        MRoPE ``image_grid_thw`` protocol. Each planner query instead continues
        from the number of valid prompt tokens in its own batch row.
        """
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape [B,N], got "
                f"{tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] <= 0:
            raise ValueError("attention_mask batch dimension must be positive")
        if (
            not isinstance(appended_query_count, int)
            or isinstance(appended_query_count, bool)
        ):
            raise TypeError("appended_query_count must be an integer")
        if appended_query_count < 0:
            raise ValueError("appended_query_count must be non-negative")

        valid_prompt = attention_mask != 0
        prompt_positions = valid_prompt.long().cumsum(dim=-1) - 1
        prompt_positions = prompt_positions.masked_fill(~valid_prompt, 0)
        valid_lengths = valid_prompt.long().sum(dim=-1, keepdim=True)
        query_offsets = torch.arange(
            appended_query_count,
            device=attention_mask.device,
            dtype=torch.long,
        ).unsqueeze(0)
        query_positions = valid_lengths + query_offsets
        return torch.cat(
            [prompt_positions, query_positions],
            dim=-1,
        )
