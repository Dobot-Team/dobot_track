"""TrackVLA trajectory and mask heads on dense Qwen3-VL checkpoints.

Qwen3-VL exposes additional DeepStack features from its vision tower.  This
adapter intentionally discards them: only the final merged visual embeddings
are injected into the decoder and compressed for historical frames.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

try:
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )
except ImportError:  # Keep existing InternVL/Qwen2 imports usable on older envs.
    Qwen3VLForConditionalGeneration = None
    Qwen3VLProcessor = None

from vla.config import (
    HistoryCompressorConfig,
    ModelExtensionContext,
    ModelExtensionsConfig,
    SAM2Config,
)
from vla.heads import (
    HistoryCompressionOutput,
    HistoryTokenCompressor,
    SAM2MaskAuxiliaryHead,
    TrajectoryMLP,
    compress_flattened_history_sequence,
)


class Qwen3VLVLA(nn.Module):
    """Qwen3-VL VLA adapter with optional SAM2 and history compression.

    The processor supplies dense image placeholders for every frame.  After
    vision encoding, historical features are compressed and the corresponding
    dense placeholder runs are compacted before the decoder forward.  The
    current frame remains dense for planning and mask decoding.
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
        gradient_checkpointing: bool = False,
        sam2_config: Optional[SAM2Config] = None,
        history_compressor_config: Optional[
            Union[HistoryCompressorConfig, Mapping[str, object]]
        ] = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path
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
        self.mask_feature_size = self.sam2_config.feature_size
        self.history_compressor_config = self.model_extensions.history_compressor
        self.model_extensions.validate_for(
            ModelExtensionContext(model_type="qwen3vl")
        )

        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            )
        self.dtype = dtype

        if Qwen3VLForConditionalGeneration is None or Qwen3VLProcessor is None:
            raise ImportError(
                "Qwen3-VL TrackVLA requires transformers>=4.57.0,<5.0. "
                "Install this repository's requirements.txt in the active environment."
            )

        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        architectures = tuple(getattr(config, "architectures", ()) or ())
        model_type = str(getattr(config, "model_type", ""))
        if "moe" in model_type.lower() or any(
            "moe" in architecture.lower() for architecture in architectures
        ):
            raise ValueError(
                "Qwen3VLVLA supports dense Qwen3-VL checkpoints only; MoE "
                f"checkpoint detected ({model_type!r}, {architectures!r})"
            )
        if model_type != "qwen3_vl" or (
            architectures
            and "Qwen3VLForConditionalGeneration" not in architectures
        ):
            raise ValueError(
                "Qwen3VLVLA expects a dense Qwen3-VL conditional-generation "
                f"checkpoint, got model_type={model_type!r}, "
                f"architectures={architectures!r}"
            )

        print(f"[Qwen3VLVLA] Loading processor from {model_path} ...")
        self.tokenizer = Qwen3VLProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        print(
            f"[Qwen3VLVLA] Loading dense Qwen3-VL model from {model_path} "
            f"(dtype={dtype}) ..."
        )
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=local_files_only,
            device_map=None,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
        self.config = self.qwen_model.config
        self.llm_hidden_size = int(self.config.text_config.hidden_size)
        self.image_token_id = int(self.config.image_token_id)
        self.spatial_merge_size = int(self.config.vision_config.spatial_merge_size)
        if self.spatial_merge_size <= 0:
            raise ValueError("Qwen3-VL spatial_merge_size must be positive")

        compressor_hidden_size = (
            self.llm_hidden_size
            if self.history_compressor_config.hidden_size is None
            else self.history_compressor_config.hidden_size
        )
        self.history_compressor = HistoryTokenCompressor(
            config=self.history_compressor_config,
            hidden_size=compressor_hidden_size,
            llm_hidden_size=self.llm_hidden_size,
            dtype=dtype,
        )

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

        self._set_requires_grad(self.qwen_model.visual, not freeze_vision)
        deepstack_mergers = getattr(
            self.qwen_model.visual,
            "deepstack_merger_list",
            None,
        )
        if isinstance(deepstack_mergers, nn.Module):
            # These projections only feed the discarded DeepStack outputs and
            # must not enter the optimizer even when the main vision tower is
            # fine-tuned.
            self._set_requires_grad(deepstack_mergers, False)
        self._set_requires_grad(self.qwen_model.language_model, not freeze_llm)
        self._set_requires_grad(self.action_head, not freeze_action_head)
        if gradient_checkpointing:
            try:
                self.qwen_model.language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.qwen_model.language_model.gradient_checkpointing_enable()
            print("[Qwen3VLVLA] Enabled language-model gradient checkpointing.")

        self._print_param_stats()

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _print_param_stats(self) -> None:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        print(f"[Qwen3VLVLA] Total params:    {total / 1e6:.2f} M")
        print(f"[Qwen3VLVLA] Trainable params: {trainable / 1e6:.2f} M")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
        trajectories: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
        history_masks: Optional[torch.FloatTensor] = None,
        history_mask_valid: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [B,N], got {tuple(input_ids.shape)}"
            )
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "attention_mask must match input_ids, got "
                f"{tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
            )
        batch_size = input_ids.shape[0]
        if batch_size <= 0:
            raise ValueError("input_ids batch dimension must be positive")
        device = input_ids.device

        # Qwen3-VL returns (final image embeddings, DeepStack embeddings).
        # DeepStack is deliberately discarded and never reaches the decoder or
        # history compressor.
        grids, tokens_per_image, images_per_sample = self._visual_token_layout(
            image_grid_thw,
            batch_size,
        )
        dense_token_count = tokens_per_image[0]
        if any(count != dense_token_count for count in tokens_per_image):
            raise ValueError(
                "Every Qwen3-VL frame in a TrackVLA batch must use the same "
                f"visual-token count, got {tokens_per_image}"
            )
        image_features = self._encode_final_image_features(
            pixel_values.to(device=device),
            image_grid_thw,
        )
        if image_features.shape[1] != dense_token_count:
            raise ValueError(
                "Qwen3-VL final visual embeddings disagree with image_grid_thw: "
                f"expected {dense_token_count} tokens per image, got "
                f"{image_features.shape[1]}"
            )
        if image_features.shape[0] != batch_size * images_per_sample:
            raise ValueError(
                "Qwen3-VL image-feature count is inconsistent with image_grid_thw"
            )

        visual_sequence = compress_flattened_history_sequence(
            self.history_compressor,
            image_features,
            batch_size=batch_size,
            history_masks=history_masks,
            history_mask_valid=history_mask_valid,
        )

        token_embeddings = self.qwen_model.get_input_embeddings()(input_ids)
        compact_ids, compact_attention, compact_embeds = (
            self._compact_multimodal_prompt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_embeddings=token_embeddings,
                visual_sequence=visual_sequence,
                images_per_sample=images_per_sample,
                dense_token_count=dense_token_count,
            )
        )

        query_ids = torch.arange(self.num_queries, device=device).unsqueeze(0)
        query_ids = query_ids.expand(batch_size, -1)
        query_embeds = self.query_tokens(query_ids).to(compact_embeds.dtype)
        query_parts = []
        if self.mask_query_token is not None:
            query_parts.append(
                self.mask_query_token.weight.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .to(compact_embeds.dtype)
            )
        query_parts.append(query_embeds)
        appended_queries = torch.cat(query_parts, dim=1)
        inputs_embeds = torch.cat((compact_embeds, appended_queries), dim=1)
        query_attention = torch.ones(
            batch_size,
            appended_queries.shape[1],
            device=device,
            dtype=compact_attention.dtype,
        )
        full_attention = torch.cat((compact_attention, query_attention), dim=1)

        pad_token_id = self._pad_token_id()
        query_token_ids = torch.full(
            (batch_size, appended_queries.shape[1]),
            pad_token_id,
            device=device,
            dtype=compact_ids.dtype,
        )
        full_ids = torch.cat((compact_ids, query_token_ids), dim=1)
        logical_grids = self._logical_image_grids(
            grids,
            images_per_sample=images_per_sample,
            frame_token_counts=visual_sequence.frame_token_counts,
        )
        position_ids, _rope_deltas = self.qwen_model.model.get_rope_index(
            input_ids=full_ids,
            image_grid_thw=logical_grids,
            video_grid_thw=None,
            attention_mask=full_attention,
        )

        decoder_outputs = self.qwen_model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        appended_hidden = decoder_outputs.last_hidden_state[
            :, -appended_queries.shape[1] :, :
        ]
        action_query_start = int(self.mask_query_token is not None)
        query_hidden = appended_hidden[
            :, action_query_start : action_query_start + self.num_queries, :
        ]
        pred_traj = self.action_head(query_hidden)

        loss = None
        if trajectories is not None:
            loss = F.mse_loss(pred_traj, trajectories.to(pred_traj.dtype))

        mask_outputs: Dict[str, Optional[torch.Tensor]] = {
            "mask_loss": None,
            "mask_iou": None,
        }
        if self.mask_auxiliary_head is not None:
            current_maps = self._current_image_feature_maps(
                visual_sequence.current_tokens,
                grids,
                images_per_sample=images_per_sample,
            )
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

    def _encode_final_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Return only final merged image embeddings and release DeepStack."""
        image_feature_parts, deepstack_features = self.qwen_model.get_image_features(
            pixel_values,
            image_grid_thw=image_grid_thw,
        )
        del deepstack_features
        if not isinstance(image_feature_parts, (tuple, list)) or not image_feature_parts:
            raise RuntimeError(
                "Qwen3-VL get_image_features must return non-empty per-image "
                "feature tensors"
            )
        shapes = {tuple(part.shape) for part in image_feature_parts}
        if len(shapes) != 1:
            raise ValueError(
                "Qwen3-VL TrackVLA requires a fixed token grid for every frame, "
                f"got per-image feature shapes {sorted(shapes)}"
            )
        return torch.stack(tuple(image_feature_parts), dim=0)

    def _pad_token_id(self) -> int:
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError("Qwen3-VL tokenizer requires a pad_token_id or eos_token_id")
        return int(pad_token_id)

    def _visual_token_layout(
        self,
        image_grid_thw: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, list[int], int]:
        if (
            not isinstance(image_grid_thw, torch.Tensor)
            or image_grid_thw.ndim != 2
            or image_grid_thw.shape[1] != 3
        ):
            raise ValueError(
                "image_grid_thw must have shape [num_images,3], got "
                f"{getattr(image_grid_thw, 'shape', None)}"
            )
        if image_grid_thw.shape[0] == 0 or image_grid_thw.shape[0] % batch_size:
            raise ValueError(
                "image_grid_thw must contain the same positive number of "
                "images for every batch row"
            )
        grids = image_grid_thw.to(dtype=torch.long)
        if bool((grids <= 0).any()):
            raise ValueError("image_grid_thw values must be positive")
        if bool(
            (
                (grids[:, 1] % self.spatial_merge_size != 0)
                | (grids[:, 2] % self.spatial_merge_size != 0)
            ).any()
        ):
            raise ValueError(
                "Qwen3-VL image grids must be divisible by spatial_merge_size"
            )
        tokens_per_image = (
            grids.prod(dim=1) // (self.spatial_merge_size**2)
        ).tolist()
        return (
            grids,
            [int(count) for count in tokens_per_image],
            image_grid_thw.shape[0] // batch_size,
        )

    def _compact_multimodal_prompt(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_embeddings: torch.Tensor,
        visual_sequence: HistoryCompressionOutput,
        images_per_sample: int,
        dense_token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace dense image runs with the canonical compressed sequence."""
        batch_size = input_ids.shape[0]
        frame_counts = visual_sequence.frame_token_counts
        if len(frame_counts) != images_per_sample:
            raise ValueError(
                "history compressor frame layout does not match Qwen3-VL image count"
            )
        visual_offsets = [0]
        for count in frame_counts:
            visual_offsets.append(visual_offsets[-1] + count)

        compact_ids_rows = []
        compact_embed_rows = []
        for row in range(batch_size):
            row_mask = attention_mask[row] != 0
            valid_length = int(row_mask.sum().item())
            if valid_length <= 0 or not bool(row_mask[:valid_length].all()) or bool(
                row_mask[valid_length:].any()
            ):
                raise ValueError(
                    "Qwen3-VL prompt attention_mask must describe non-empty "
                    "right-padded sequences"
                )
            row_ids = input_ids[row, :valid_length]
            row_embeds = token_embeddings[row, :valid_length]
            selected = torch.nonzero(
                row_ids == self.image_token_id,
                as_tuple=False,
            ).flatten()
            expected_dense = images_per_sample * dense_token_count
            if selected.numel() != expected_dense:
                raise ValueError(
                    f"Qwen3-VL sample {row} contains {selected.numel()} image "
                    f"placeholders, expected {expected_dense}"
                )

            spans = []
            for frame_index in range(images_per_sample):
                frame_positions = selected[
                    frame_index * dense_token_count : (frame_index + 1)
                    * dense_token_count
                ]
                if frame_positions.numel() != dense_token_count or (
                    frame_positions.numel() > 1
                    and not bool((frame_positions[1:] - frame_positions[:-1] == 1).all())
                ):
                    raise ValueError(
                        "Each Qwen3-VL image must occupy one contiguous dense "
                        "placeholder run"
                    )
                spans.append((int(frame_positions[0]), int(frame_positions[-1]) + 1))

            id_parts = []
            embed_parts = []
            cursor = 0
            for frame_index, (start, end) in enumerate(spans):
                id_parts.append(row_ids[cursor:start])
                embed_parts.append(row_embeds[cursor:start])
                count = frame_counts[frame_index]
                id_parts.append(
                    torch.full(
                        (count,),
                        self.image_token_id,
                        device=row_ids.device,
                        dtype=row_ids.dtype,
                    )
                )
                embed_parts.append(
                    visual_sequence.llm_tokens[
                        row,
                        visual_offsets[frame_index] : visual_offsets[frame_index + 1],
                    ].to(device=row_embeds.device, dtype=row_embeds.dtype)
                )
                cursor = end
            id_parts.append(row_ids[cursor:])
            embed_parts.append(row_embeds[cursor:])
            compact_ids_rows.append(torch.cat(id_parts, dim=0))
            compact_embed_rows.append(torch.cat(embed_parts, dim=0))

        max_length = max(row.shape[0] for row in compact_ids_rows)
        compact_ids = torch.full(
            (batch_size, max_length),
            self._pad_token_id(),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        compact_attention = torch.zeros(
            (batch_size, max_length),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        compact_embeds = token_embeddings.new_zeros(
            batch_size,
            max_length,
            token_embeddings.shape[-1],
        )
        for row, (row_ids, row_embeds) in enumerate(
            zip(compact_ids_rows, compact_embed_rows)
        ):
            length = row_ids.shape[0]
            compact_ids[row, :length] = row_ids
            compact_attention[row, :length] = 1
            compact_embeds[row, :length] = row_embeds
        return compact_ids, compact_attention, compact_embeds

    def _logical_image_grids(
        self,
        grids: torch.Tensor,
        *,
        images_per_sample: int,
        frame_token_counts: tuple[int, ...],
    ) -> torch.Tensor:
        """Describe compressed history spans as virtual square MRoPE grids."""
        if len(frame_token_counts) != images_per_sample:
            raise ValueError("frame_token_counts must match images_per_sample")
        logical = grids.clone()
        for image_index in range(logical.shape[0]):
            frame_index = image_index % images_per_sample
            token_count = frame_token_counts[frame_index]
            native_token_count = int(
                logical[image_index].prod().item()
                // (self.spatial_merge_size**2)
            )
            if token_count == native_token_count:
                continue
            side = math.isqrt(token_count)
            if side * side != token_count:
                raise ValueError(
                    "Qwen3-VL compressed image token count must be a perfect "
                    f"square, got {token_count}"
                )
            logical[image_index] = torch.tensor(
                [1, side * self.spatial_merge_size, side * self.spatial_merge_size],
                device=logical.device,
                dtype=logical.dtype,
            )
        return logical

    def _current_image_feature_maps(
        self,
        current_tokens: torch.Tensor,
        grids: torch.Tensor,
        *,
        images_per_sample: int,
    ) -> torch.Tensor:
        maps = []
        for row in range(current_tokens.shape[0]):
            grid = grids[row * images_per_sample + images_per_sample - 1]
            temporal, grid_height, grid_width = [int(value) for value in grid.tolist()]
            if temporal != 1:
                raise ValueError("SAM2 auxiliary masks support Qwen3-VL images, not video grids")
            height = grid_height // self.spatial_merge_size
            width = grid_width // self.spatial_merge_size
            tokens = current_tokens[row]
            if tokens.shape[0] != height * width:
                raise ValueError(
                    "Current Qwen3-VL token count is inconsistent with image_grid_thw"
                )
            feature_map = tokens.transpose(0, 1).reshape(
                1,
                tokens.shape[-1],
                height,
                width,
            )
            maps.append(
                F.interpolate(
                    feature_map,
                    size=(self.mask_feature_size, self.mask_feature_size),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(maps, dim=0)

    def save_pretrained(self, save_directory: str) -> None:
        import os

        os.makedirs(save_directory, exist_ok=True)
        state = {
            key: value
            for key, value in self.state_dict().items()
            if value.requires_grad
        }
        torch.save(state, os.path.join(save_directory, "trainable_state_dict.pt"))

    def load_pretrained(self, save_directory: str) -> None:
        import os

        state = torch.load(
            os.path.join(save_directory, "trainable_state_dict.pt"),
            map_location="cpu",
        )
        self.load_state_dict(state, strict=False)
