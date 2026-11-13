from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from vla.config import ModelExtensionsConfig, SAM2Config
from vla.heads import SAM2MaskAuxiliaryHead, TrajectoryMLP


class InternVLVLA(nn.Module):
    """VLA model built on InternVL 2.5 / 3 with learnable trajectory query tokens.

    Architecture:
      1. InternVL vision encoder extracts visual tokens from history camera frames.
      2. Text prompt embeddings are built and visual features are injected at
         <IMG_CONTEXT> positions.
      3. Learnable query tokens are appended to the input sequence.
      4. The LLM processes the multimodal sequence causally.
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
        freeze_mlp1: bool = True,
        action_head_dims: Tuple[int, ...] = (512, 2),
        action_head_type: str = "legacy",
        dtype: Optional[torch.dtype] = None,
        local_files_only: bool = True,
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = False,
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

        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        print(f"[InternVLVLA] Loading tokenizer from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

        print(f"[InternVLVLA] Loading InternVL model from {model_path} (dtype={dtype}) ...")
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "device_map": None,
            "low_cpu_mem_usage": True,
        }
        model_kwargs["attn_implementation"] = attn_implementation
        # InternVL's remote model constructor independently controls Flash
        # Attention through ``use_flash_attn``. Keep it aligned with the
        # Transformers attention backend requested by the YAML configuration.
        model_kwargs["use_flash_attn"] = attn_implementation == "flash_attention_2"
        self.internvl = AutoModel.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self.config = self.internvl.config
        self.llm_hidden_size = self.config.llm_config.hidden_size
        self.dtype = dtype

        # Learnable trajectory query tokens.
        self.query_tokens = nn.Embedding(self.num_queries, self.llm_hidden_size).to(dtype)
        nn.init.normal_(self.query_tokens.weight, std=0.02)

        # Action head supports either one waypoint per query or all waypoints
        # from one query state.
        self.action_head = TrajectoryMLP(
            self.llm_hidden_size,
            self.num_queries,
            list(action_head_dims),
            n_future=self.n_future,
            head_type=self.action_head_type,
        ).to(dtype)
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
        self._set_requires_grad(self.internvl.vision_model, not freeze_vision)
        self._set_requires_grad(self.internvl.language_model, not freeze_llm)
        self._set_requires_grad(self.internvl.mlp1, not freeze_mlp1)
        if gradient_checkpointing:
            self.internvl.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[InternVLVLA] Enabled language-model gradient checkpointing.")

        self._print_param_stats()

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[InternVLVLA] Total params:    {total / 1e6:.2f} M")
        print(f"[InternVLVLA] Trainable params: {trainable / 1e6:.2f} M")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        trajectories: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B * num_images, C, H, W) preprocessed images.
            input_ids: (B, N) tokenized prompt with <IMG_CONTEXT> placeholders.
            attention_mask: (B, N) attention mask for the prompt.
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

        # Ensure visual inputs match model dtype (e.g. bfloat16).
        pixel_values = pixel_values.to(self.dtype)

        # 1. Visual features.
        vit_embeds = self.internvl.extract_feature(pixel_values)  # (B*num_imgs, T, C)

        # 2. Text embeddings + visual injection.
        input_embeds = self.internvl.get_input_embeddings()(input_ids)  # (B, N, C)
        input_embeds = self._inject_visual_features(input_embeds, input_ids, vit_embeds)

        # 3. Append the optional mask query before trajectory queries. This
        # lets every causal action query attend to the mask-query state.
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
        position_ids = self._build_position_ids(
            attention_mask,
            appended_queries.shape[1],
        )

        # 4. LLM forward.
        outputs = self.internvl.language_model(
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

        # 5. Action head.
        pred_traj = self.action_head(query_hidden)  # (B, n_future, 2)

        # 6. Loss.
        loss = None
        if trajectories is not None:
            loss = F.mse_loss(pred_traj, trajectories.to(pred_traj.dtype))

        mask_outputs: Dict[str, Optional[torch.Tensor]] = {"mask_loss": None, "mask_iou": None}
        if self.mask_auxiliary_head is not None:
            if vit_embeds.shape[0] % B != 0:
                raise ValueError(
                    f"Visual batch size {vit_embeds.shape[0]} is not divisible by text batch size {B}"
                )
            history_size = vit_embeds.shape[0] // B
            current_tokens = vit_embeds.reshape(B, history_size, *vit_embeds.shape[1:])[:, -1]
            mask_hidden = appended_hidden[:, 0, :]
            decoded_masks = self.mask_auxiliary_head.decode(
                self._tokens_to_feature_map(current_tokens),
                mask_hidden,
            )
            mask_outputs = {"mask_loss": None, "mask_iou": None, **decoded_masks}
            if masks is not None:
                mask_outputs.update(
                    self.mask_auxiliary_head.compute_loss(decoded_masks, masks)
                )
        return {"loss": loss, "pred_traj": pred_traj, **mask_outputs}

    @staticmethod
    def _build_position_ids(
        attention_mask: torch.Tensor,
        appended_query_count: int,
    ) -> torch.LongTensor:
        """Keep query RoPE positions independent of batch-side right padding."""
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

    @staticmethod
    def _tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
        """Restore current-frame square visual tokens to a spatial feature map."""
        token_count = tokens.shape[1]
        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError(
                "SAM2 auxiliary head requires square InternVL visual tokens, got "
                f"{token_count} tokens"
            )
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    def _inject_visual_features(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        vit_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Replace image slots without allowing features to cross batch rows."""
        if input_embeds.ndim != 3:
            raise ValueError(
                "input_embeds must have shape [B,N,D], got "
                f"{tuple(input_embeds.shape)}"
            )
        batch_size, sequence_length, hidden_size = input_embeds.shape
        if batch_size <= 0:
            raise ValueError("input_embeds batch dimension must be positive")
        if tuple(input_ids.shape) != (batch_size, sequence_length):
            raise ValueError("input_ids must match input_embeds [B,N]")
        if (
            not isinstance(vit_embeds, torch.Tensor)
            or vit_embeds.ndim != 3
            or vit_embeds.shape[0] % batch_size != 0
            or vit_embeds.shape[2] != hidden_size
        ):
            actual_shape = (
                tuple(vit_embeds.shape)
                if isinstance(vit_embeds, torch.Tensor)
                else type(vit_embeds).__name__
            )
            raise ValueError(
                "vit_embeds must have shape [B*H,T,D] matching text "
                f"embeddings, got {actual_shape}"
            )

        grouped_features = vit_embeds.reshape(
            batch_size,
            -1,
            hidden_size,
        )
        selected = input_ids == self.img_context_token_id
        placeholder_counts = selected.sum(dim=1)
        expected_count = grouped_features.shape[1]
        mismatched_rows = torch.nonzero(
            placeholder_counts != expected_count,
            as_tuple=False,
        ).flatten()
        if mismatched_rows.numel() > 0:
            details = ", ".join(
                f"sample {row}: {int(placeholder_counts[row].item())}"
                for row in mismatched_rows.tolist()
            )
            raise ValueError(
                "InternVL image-token mismatch: every prompt row must contain "
                f"{expected_count} <IMG_CONTEXT> tokens, got {details}"
            )

        result = input_embeds.clone()
        grouped_features = grouped_features.to(
            device=result.device,
            dtype=result.dtype,
        )
        for row in range(batch_size):
            result[row, selected[row]] = grouped_features[row]
        return result

    def save_pretrained(self, save_directory: str):
        """Save trainable parts; backbone weights are kept at the original path."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        state = {k: v for k, v in self.state_dict().items() if v.requires_grad}
        torch.save(state, os.path.join(save_directory, "trainable_state_dict.pt"))

    def load_pretrained(self, save_directory: str):
        import os
        path = os.path.join(save_directory, "trainable_state_dict.pt")
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state, strict=False)
