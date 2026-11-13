"""Export components using the same query ordering as InternVL3VLA."""

import torch
from torch import nn


def query_embeddings(model):
    """Mask (optional), Box, Action; never reuse Box-after-Action templates."""
    if model.vqa_query_token is not None:
        raise ValueError(
            "This deployment supports Box + Action with optional Mask, not VQA"
        )
    if model.grounding_query_token is None:
        raise ValueError("A trained grounding query/head is required")
    parts = []
    if model.mask_query_token is not None:
        parts.append(model.mask_query_token.weight)
    parts.extend((model.grounding_query_token.weight, model.query_tokens.weight))
    return torch.cat(parts, dim=0).unsqueeze(0)


class Vision(nn.Module):
    def __init__(self, internvl):
        super().__init__()
        self.vision = internvl.vision_tower
        self.projector = internvl.multi_modal_projector
        self.pixel_shuffle = internvl.pixel_shuffle
        self.strategy = getattr(
            internvl.config, "vision_feature_select_strategy", "default"
        )
        self.ratio = float(getattr(internvl.config, "downsample_ratio", 0.5))
        if int(getattr(internvl.config, "vision_feature_layer", -1)) != -1:
            raise ValueError("Only the final ViT feature layer is supported")

    def forward(self, pixel_values):
        x = self.vision(pixel_values=pixel_values).last_hidden_state
        if self.strategy == "default":
            x = x[:, 1:]
        n, tokens, channels = x.shape
        side = int(tokens**0.5)
        x = self.pixel_shuffle(
            x.reshape(n, side, side, channels), scale_factor=self.ratio
        )
        return self.projector(x.reshape(n, -1, x.shape[-1]))


class Heads(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.action = model.action_head
        self.ground = model.auxiliary_heads.grounding_head
        self.box_index = int(model.mask_query_token is not None)
        self.action_start = self.box_index + 1
        self.action_count = model.num_queries

    def forward(self, hidden):
        trajectory = self.action(
            hidden[:, self.action_start : self.action_start + self.action_count]
        )
        raw = self.ground(hidden[:, self.box_index])
        return trajectory, raw[:, :4].sigmoid(), raw[:, 4]


class Mask(nn.Module):
    def __init__(self, model, grid=16):
        super().__init__()
        self.head = model.mask_auxiliary_head
        self.grid = grid

    def forward(self, features, query):
        x = features.transpose(1, 2).reshape(
            features.shape[0], features.shape[-1], self.grid, self.grid
        )
        return self.head.decode(x, query)["mask_logits"]
