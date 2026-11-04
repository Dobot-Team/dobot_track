import inspect
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from vla.config import ModelExtensionsConfig, SAM2Config
from vla.heads import SAM2MaskAuxiliaryHead, TrajectoryMLP


class Qwen2VLVLA(nn.Module):
    """VLA model built on Qwen2-VL with learnable trajectory query tokens.

    Architecture:
      1. Qwen2-VL vision encoder extracts visual tokens from history camera frames.
      2. Text prompt embeddings are built and visual features are injected at
         <|image_pad|> positions.
      3. Learnable query tokens are appended to the input sequence.
      4. The LLM processes the multimodal sequence causally with MRoPE.
      5. A lightweight MLP head maps the query hidden states to future
         trajectory points.
    """

    def __init__(
        self,
        model_path: str,
        n_future: int,
        num_queries: int = 24,
        freeze_vision: bool = True,
        freeze_llm: bool = False,
        freeze_action_head: bool = False,
        action_head_dims: Tuple[int, ...] = (512, 2),
        action_head_type: str = "legacy",
        dtype: Optional[torch.dtype] = None,
        local_files_only: bool = True,
        attn_implementation: str = "sdpa",
        sam2_config: Optional[SAM2Config] = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.num_queries = int(num_queries)
        self.n_future = int(n_future)
        self.action_head_type = TrajectoryMLP.canonicalize_action_head_type(
            action_head_type
        )
        self.model_extensions = ModelExtensionsConfig.from_values(
            sam2=sam2_config
        )
        self.sam2_config = self.model_extensions.sam2
        self.mask_aux_enabled = self.sam2_config.enabled
        self.mask_feature_size = self.sam2_config.feature_size

        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        print(f"[Qwen2VLVLA] Loading processor from {model_path} ...")
        self.tokenizer = Qwen2VLProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        print(f"[Qwen2VLVLA] Loading Qwen2-VL model from {model_path} (dtype={dtype}) ...")
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
            device_map=None,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
        self.config = self.qwen_model.config
        # Compatible with both transformers 4.x and 5.x API.
        text_config = self.config.get_text_config() if hasattr(self.config, "get_text_config") else self.config.text_config
        self.llm_hidden_size = text_config.hidden_size
        self.dtype = dtype
        self.image_token_id = self.config.image_token_id

        # Learnable trajectory query tokens.
        self.query_tokens = nn.Embedding(self.num_queries, self.llm_hidden_size)
        nn.init.normal_(self.query_tokens.weight, std=0.02)

        # Action head supports either one waypoint per query or all waypoints
        # from one query state.
        self.action_head = TrajectoryMLP(
            self.llm_hidden_size,
            self.num_queries,
            list(action_head_dims),
            n_future=self.n_future,
            head_type=self.action_head_type,
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

        # Freeze parts of the backbone if requested.
        # Resolve vision and text modules independently: in Transformers 4.57
        # ``model`` contains both, so freezing it as the LLM would override the
        # requested vision setting.
        self._set_requires_grad(self.qwen_model.visual, not freeze_vision)
        self._set_requires_grad(self._language_backbone(), not freeze_llm)
        self._set_requires_grad(self.action_head, not freeze_action_head)

        self._print_param_stats()

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _language_backbone(self) -> nn.Module:
        """Resolve the text decoder across supported Transformers layouts."""
        language_model = getattr(self.qwen_model, "language_model", None)
        if language_model is None:
            base_model = getattr(self.qwen_model, "model", None)
            language_model = getattr(base_model, "language_model", base_model)
        if not isinstance(language_model, nn.Module):
            raise RuntimeError(
                "Qwen2-VL model does not expose a language-model module"
            )
        return language_model

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Qwen2VLVLA] Total params:    {total / 1e6:.2f} M")
        print(f"[Qwen2VLVLA] Trainable params: {trainable / 1e6:.2f} M")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        trajectories: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (total_image_patches, 1176) preprocessed patches from Qwen2VLProcessor.
            input_ids: (B, N) tokenized prompt with <|image_pad|> placeholders.
            attention_mask: (B, N) attention mask for the prompt.
            image_grid_thw: (total_images, 3) grid sizes for each image.
            mm_token_type_ids: (B, N) 0=text, 1=image; used for MRoPE.
            trajectories: (B, n_future, 2) ground-truth future trajectory.
        Returns:
            dict with keys:
                - loss: scalar training loss (None if trajectories not provided)
                - pred_traj: (B, n_future, 2) predicted trajectory
                - mask_logits, pred_iou, pred_obj_scores: decoder outputs when
                  the SAM2 auxiliary head is enabled
                - mask_loss, mask_iou: mask supervision metrics when masks are
                  provided
        """
        B = input_ids.shape[0]
        device = input_ids.device

        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(input_ids)

        # 1. Visual features.
        # In transformers 4.50+ visual() returns the merged vision feature tensor
        # directly (num_image_tokens, hidden_size).
        vit_embeds = self.qwen_model.visual(
            pixel_values.to(self.dtype),
            grid_thw=image_grid_thw,
        )  # (total_image_tokens, C)
        _, tokens_per_image, _, images_per_sample = (
            self._visual_token_layout(
                image_grid_thw,
                B,
                total_visual_tokens=vit_embeds.shape[0],
            )
        )
        tokens_per_sample = tuple(
            sum(
                tokens_per_image[
                    batch_index
                    * images_per_sample : (batch_index + 1)
                    * images_per_sample
                ]
            )
            for batch_index in range(B)
        )

        # 2. Text embeddings + visual injection.
        input_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        input_embeds = self._inject_visual_features(
            input_embeds,
            input_ids,
            vit_embeds,
            tokens_per_sample,
        )

        # 3. Append the optional mask query before trajectory queries so the
        # causal action-query states can attend to it.
        query_ids = torch.arange(self.num_queries, device=device).unsqueeze(0).expand(B, -1)
        query_embeds = self.query_tokens(query_ids).to(input_embeds.dtype)
        query_parts = []
        if self.mask_query_token is not None:
            query_parts.append(
                self.mask_query_token.weight.unsqueeze(0)
                .expand(B, -1, -1)
                .to(input_embeds.dtype)
            )
        query_parts.append(query_embeds)
        appended_queries = torch.cat(query_parts, dim=1)
        inputs_embeds = torch.cat([input_embeds, appended_queries], dim=1)

        query_attention_mask = torch.ones(
            B, appended_queries.shape[1], device=device, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([attention_mask, query_attention_mask], dim=1)

        query_mm_token_type = torch.zeros(
            B, appended_queries.shape[1], device=device, dtype=mm_token_type_ids.dtype
        )
        full_mm_token_type_ids = torch.cat([mm_token_type_ids, query_mm_token_type], dim=1)

        # 4. MRoPE position_ids.
        position_ids = self._get_mrope_position_ids(
            input_ids=input_ids,
            query_ids=torch.arange(appended_queries.shape[1], device=device).unsqueeze(0).expand(B, -1),
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=full_mm_token_type_ids,
            attention_mask=full_attention_mask,
        )

        # 5. LLM forward.
        outputs = self.qwen_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, N+num_queries, C)
        appended_hidden = hidden_states[:, -appended_queries.shape[1] :, :]
        action_query_start = int(self.mask_query_token is not None)
        query_hidden = appended_hidden[
            :, action_query_start : action_query_start + self.num_queries, :
        ]

        # 6. Action head.
        pred_traj = self.action_head(query_hidden)  # (B, n_future, 2)

        # 7. Loss.
        loss = None
        if trajectories is not None:
            loss = F.mse_loss(pred_traj, trajectories.to(pred_traj.dtype))

        mask_outputs: Dict[str, Optional[torch.Tensor]] = {"mask_loss": None, "mask_iou": None}
        if self.mask_auxiliary_head is not None:
            current_maps = self._current_image_feature_maps(vit_embeds, image_grid_thw, B)
            decoded_masks = self.mask_auxiliary_head.decode(
                current_maps,
                appended_hidden[:, 0, :],
            )
            mask_outputs = {"mask_loss": None, "mask_iou": None, **decoded_masks}
            if masks is not None:
                mask_outputs.update(
                    self.mask_auxiliary_head.compute_loss(decoded_masks, masks)
                )
        return {"loss": loss, "pred_traj": pred_traj, **mask_outputs}

    def _current_image_feature_maps(
        self,
        vit_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Split Qwen's flattened vision output and retain each sample's last image.

        Qwen2-VL allows different image grids in one batch. Each current image
        is independently resized to the SAM auxiliary grid before stacking.
        """
        grids, tokens_per_image, merge_size, images_per_sample = (
            self._visual_token_layout(
                image_grid_thw,
                batch_size,
                total_visual_tokens=vit_embeds.shape[0],
            )
        )
        offsets = [0]
        for count in tokens_per_image:
            offsets.append(offsets[-1] + int(count))

        maps = []
        for batch_index in range(batch_size):
            image_index = batch_index * images_per_sample + images_per_sample - 1
            temporal, grid_height, grid_width = [int(value) for value in grids[image_index].tolist()]
            if temporal != 1:
                raise ValueError("SAM2 mask auxiliary task supports image inputs, not Qwen video grids")
            height = grid_height // merge_size
            width = grid_width // merge_size
            start, end = offsets[image_index], offsets[image_index + 1]
            tokens = vit_embeds[start:end]
            if tokens.shape[0] != height * width:
                raise ValueError("Qwen2-VL current-image token grid is inconsistent with image_grid_thw")
            feature_map = tokens.transpose(0, 1).reshape(1, tokens.shape[-1], height, width)
            maps.append(
                F.interpolate(
                    feature_map,
                    size=(self.mask_feature_size, self.mask_feature_size),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(maps, dim=0)

    def _visual_token_layout(
        self,
        image_grid_thw: torch.Tensor,
        batch_size: int,
        *,
        total_visual_tokens: int,
    ) -> tuple[torch.Tensor, list[int], int, int]:
        """Validate Qwen's flattened per-image/per-sample token layout."""
        if (
            not isinstance(image_grid_thw, torch.Tensor)
            or image_grid_thw.ndim != 2
            or image_grid_thw.shape[1] != 3
        ):
            actual_shape = (
                tuple(image_grid_thw.shape)
                if isinstance(image_grid_thw, torch.Tensor)
                else type(image_grid_thw).__name__
            )
            raise ValueError(
                "image_grid_thw must have shape [num_images,3], got "
                f"{actual_shape}"
            )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            image_grid_thw.shape[0] == 0
            or image_grid_thw.shape[0] % batch_size != 0
        ):
            raise ValueError(
                "image_grid_thw must contain the same positive number of "
                "images for every batch row"
            )
        vision_config = getattr(self.qwen_model.config, "vision_config", None)
        merge_size = int(
            getattr(
                self.qwen_model.visual,
                "spatial_merge_size",
                getattr(vision_config, "spatial_merge_size", 2),
            )
        )
        if merge_size <= 0:
            raise ValueError("Qwen2-VL spatial_merge_size must be positive")
        grids = image_grid_thw.to(dtype=torch.long)
        if bool((grids <= 0).any()):
            raise ValueError("image_grid_thw values must be positive")
        if bool(
            (
                (grids[:, 1] % merge_size != 0)
                | (grids[:, 2] % merge_size != 0)
            ).any()
        ):
            raise ValueError(
                "Qwen2-VL image grids must be divisible by spatial_merge_size"
            )
        tokens_per_image = (
            grids.prod(dim=1) // (merge_size * merge_size)
        ).tolist()
        if sum(tokens_per_image) != total_visual_tokens:
            raise ValueError(
                "Qwen2-VL visual token count does not match image_grid_thw"
            )
        return (
            grids,
            [int(count) for count in tokens_per_image],
            merge_size,
            image_grid_thw.shape[0] // batch_size,
        )

    def _inject_visual_features(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        vit_embeds: torch.Tensor,
        visual_tokens_per_sample: Sequence[int],
    ) -> torch.Tensor:
        """Replace image slots without moving vision features across samples."""
        if input_embeds.ndim != 3:
            raise ValueError("input_embeds must have shape [B,N,D]")
        batch_size, sequence_length, hidden_size = input_embeds.shape
        if batch_size <= 0:
            raise ValueError("input_embeds batch dimension must be positive")
        if tuple(input_ids.shape) != (batch_size, sequence_length):
            raise ValueError("input_ids must match input_embeds [B,N]")
        if (
            not isinstance(vit_embeds, torch.Tensor)
            or vit_embeds.ndim != 2
            or vit_embeds.shape[1] != hidden_size
        ):
            actual_shape = (
                tuple(vit_embeds.shape)
                if isinstance(vit_embeds, torch.Tensor)
                else type(vit_embeds).__name__
            )
            raise ValueError(
                "vit_embeds must have shape [total_visual_tokens,D], got "
                f"{actual_shape}"
            )
        if (
            len(visual_tokens_per_sample) != batch_size
            or any(
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for count in visual_tokens_per_sample
            )
            or sum(visual_tokens_per_sample) != vit_embeds.shape[0]
        ):
            raise ValueError(
                "visual_tokens_per_sample must contain one non-negative "
                "count per batch row and sum to vit_embeds.shape[0]"
            )

        selected = input_ids == self.image_token_id
        placeholder_counts = selected.sum(dim=1)
        mismatched_rows = [
            row
            for row, expected_count in enumerate(visual_tokens_per_sample)
            if int(placeholder_counts[row].item()) != expected_count
        ]
        if mismatched_rows:
            details = ", ".join(
                f"sample {row}: placeholders="
                f"{int(placeholder_counts[row].item())}, "
                f"visual_tokens={visual_tokens_per_sample[row]}"
                for row in mismatched_rows
            )
            raise ValueError(
                "Qwen2-VL image-token mismatch: " + details
            )

        result = input_embeds.clone()
        vit_embeds = vit_embeds.to(
            device=result.device,
            dtype=result.dtype,
        )
        offset = 0
        for row, count in enumerate(visual_tokens_per_sample):
            result[row, selected[row]] = vit_embeds[offset : offset + count]
            offset += count
        return result

    def _get_mrope_position_ids(
        self,
        input_ids: torch.Tensor,
        query_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Qwen2-VL MRoPE 3-D position_ids (3, batch, seq_len).

        Query tokens are treated as text tokens (token type 0) for position
        computation; their actual embeddings come from the learnable query_tokens.
        """
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError(
                "Qwen2-VL MRoPE query construction requires a pad_token_id "
                "or eos_token_id"
            )
        full_input_ids = torch.cat(
            [
                input_ids,
                torch.full_like(query_ids, int(pad_token_id)),
            ],
            dim=1,
        )
        expected_shape = (
            3,
            input_ids.shape[0],
            full_input_ids.shape[1],
        )

        helper_candidates = []
        for owner_name, owner in (
            ("qwen_model.model", getattr(self.qwen_model, "model", None)),
            ("qwen_model", self.qwen_model),
        ):
            helper = getattr(owner, "get_rope_index", None)
            if callable(helper) and all(
                candidate[1] is not helper
                for candidate in helper_candidates
            ):
                helper_candidates.append((owner_name, helper))

        if not helper_candidates:
            raise RuntimeError(
                "The installed Qwen2-VL model exposes no get_rope_index "
                "helper; refusing to replace 3-D MRoPE with plain 1-D "
                "positions"
            )

        failures = []
        for owner_name, helper in helper_candidates:
            try:
                helper_kwargs = {
                    "input_ids": full_input_ids,
                    "image_grid_thw": image_grid_thw,
                    "video_grid_thw": None,
                    "attention_mask": attention_mask,
                }
                try:
                    parameters = inspect.signature(helper).parameters
                except (TypeError, ValueError):
                    parameters = {}
                if (
                    "mm_token_type_ids" in parameters
                    or any(
                        parameter.kind
                        is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                ):
                    helper_kwargs["mm_token_type_ids"] = mm_token_type_ids
                out = helper(**helper_kwargs)
                position_ids = out[0] if isinstance(out, tuple) else out
                if (
                    not isinstance(position_ids, torch.Tensor)
                    or tuple(position_ids.shape) != expected_shape
                    or position_ids.dtype != torch.long
                    or position_ids.device != input_ids.device
                ):
                    actual_shape = (
                        tuple(position_ids.shape)
                        if isinstance(position_ids, torch.Tensor)
                        else type(position_ids).__name__
                    )
                    raise ValueError(
                        "returned incompatible position_ids "
                        f"(shape={actual_shape}, "
                        f"dtype={getattr(position_ids, 'dtype', None)}, "
                        f"device={getattr(position_ids, 'device', None)}); "
                        f"expected shape={expected_shape}, dtype=torch.long, "
                        f"device={input_ids.device}"
                    )
                return position_ids
            except Exception as exc:
                failures.append(f"{owner_name}: {exc}")

        raise RuntimeError(
            "Qwen2-VL get_rope_index failed; refusing an invalid plain "
            "position fallback. " + " | ".join(failures)
        )

    def save_pretrained(self, save_directory: str):
        """Save trainable parts; backbone weights are kept at the original path."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        state = {k: v for k, v in self.state_dict().items() if v.requires_grad}
        torch.save(state, os.path.join(save_directory, "trainable_state_dict.pt"))

    def load_pretrained(self, save_directory: str):
        """Load trainable parts."""
        import os
        path = os.path.join(save_directory, "trainable_state_dict.pt")
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state, strict=False)
