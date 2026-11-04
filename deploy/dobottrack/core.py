"""Exportable checkpoint-faithful bbox Joint QFormer; no target GT or perturbations."""

import torch
from torch import nn
from torch.nn import functional as F


class ExportAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.num_heads = source.num_heads
        self.hidden = source.embed_dim
        self.q = nn.Linear(self.hidden, self.hidden)
        self.k = nn.Linear(self.hidden, self.hidden)
        self.v = nn.Linear(self.hidden, self.hidden)
        for i, m in enumerate([self.q, self.k, self.v]):
            m.weight = nn.Parameter(
                source.in_proj_weight[i * self.hidden : (i + 1) * self.hidden]
                .detach()
                .clone()
            )
            m.bias = nn.Parameter(
                source.in_proj_bias[i * self.hidden : (i + 1) * self.hidden]
                .detach()
                .clone()
            )
        self.out = source.out_proj

    def forward(self, q, k, v, padding=None):
        batch = q.shape[0]
        d = self.hidden // self.num_heads
        q = self.q(q).reshape(batch, -1, self.num_heads, d).transpose(1, 2)
        k = self.k(k).reshape(batch, -1, self.num_heads, d).transpose(1, 2)
        v = self.v(v).reshape(batch, -1, self.num_heads, d).transpose(1, 2)
        # Standard attention math is explicit so the ONNX/TRT graph can fuse it.
        score = torch.matmul(q, k.transpose(-1, -2)) * (d**-0.5)
        if padding is not None:
            score = score.masked_fill(padding[:, None, None, :], float("-inf"))
        out = (
            torch.matmul(score.softmax(-1), v)
            .transpose(1, 2)
            .reshape(batch, -1, self.hidden)
        )
        return self.out(out)


class ExportLayer(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.self_norm = source.self_attention_norm
        self.full_norm = source.full_attention_norm
        self.ffn_norm = source.ffn_norm
        self.self_attention = ExportAttention(source.self_attention)
        self.full_attention = ExportAttention(source.full_attention)
        self.ffn = source.ffn

    def forward(self, queries, target, padding, visual):
        norm = self.self_norm(torch.cat((queries, target), 1))
        qpad = torch.zeros_like(queries[:, :, 0], dtype=torch.bool)
        update = self.self_attention(norm, norm, norm, torch.cat((qpad, padding), 1))
        queries = queries + update[:, : queries.shape[1]]
        queries = queries + self.full_attention(self.full_norm(queries), visual, visual)
        return queries + self.ffn(self.ffn_norm(queries))


class BoxJointCore(nn.Module):
    def __init__(self, source, grid_size=16):
        super().__init__()
        if (
            source.mode not in ("q_identity", "qformer_joint_self_attention")
            or not source.config.requires_history_boxes
            or source.config.box_embedding
        ):
            raise ValueError("Expected Joint bbox selection without geometry embedding")
        self.project_in = source.llm_proj_in
        self.project_out = source.llm_proj_out
        self.position_projection = source.position_projection
        self.query_tokens = source.query_tokens
        self.no_target_token = source.no_target_token
        self.output_norm = source.output_norm
        self.layers = nn.ModuleList(ExportLayer(x) for x in source.layers)
        self.grid = grid_size
        coords = torch.linspace(-1, 1, grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer(
            "positions", torch.stack((xx, yy), -1).reshape(1, grid_size**2, 2)
        )
        coords = (torch.arange(grid_size) + 0.5) / grid_size
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer(
            "centers", torch.stack((xx, yy), -1).reshape(1, grid_size**2, 2)
        )

    def forward(self, features, boxes, valid):
        finite = torch.isfinite(boxes).all(-1)
        safe = torch.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
        lo = (safe[:, :2] - safe[:, 2:] / 2).clamp(0, 1)
        hi = (safe[:, :2] + safe[:, 2:] / 2).clamp(0, 1)
        valid = valid & finite & (safe[:, 2:] > 0).all(-1) & ((hi - lo) > 0).all(-1)
        boxes = torch.where(
            valid[:, None],
            torch.cat(((lo + hi) / 2, hi - lo), -1),
            torch.zeros_like(boxes),
        )
        visual = self.project_in(features)
        visual = visual + self.position_projection(self.positions.to(visual))
        lo = boxes[:, None, :2] - boxes[:, None, 2:] / 2
        hi = boxes[:, None, :2] + boxes[:, None, 2:] / 2
        selected = ((self.centers >= lo) & (self.centers < hi)).all(-1) & valid[:, None]
        center = (boxes[:, :2] * self.grid).long().clamp(0, self.grid - 1)
        index = center[:, 1] * self.grid + center[:, 0]
        fallback = F.one_hot(index, self.grid**2).bool()
        selected = selected | (fallback & (valid & ~selected.any(-1))[:, None])
        target = torch.cat(
            (visual, self.no_target_token.expand(features.shape[0], -1, -1)), 1
        )
        padding = torch.cat((~selected, valid[:, None]), 1)
        queries = self.query_tokens.expand(features.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, target, padding, visual)
        return self.project_out(self.output_norm(queries))
