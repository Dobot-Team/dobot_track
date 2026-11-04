"""Prediction heads and losses for grounding and structured VQA."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.config import AuxiliaryTasksConfig


class GroundingVQAHeads(nn.Module):
    """Small heads driven by causal query tokens appended after action queries."""

    def __init__(self, hidden_size: int, config: AuxiliaryTasksConfig, dtype=None):
        super().__init__()
        self.config = config
        self.grounding_head = None
        self.vqa_heads = nn.ModuleDict()
        if config.grounding_enabled:
            self.grounding_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 5),
            )
        if config.vqa_enabled:
            self.vqa_heads = nn.ModuleDict({
                "position": nn.Linear(hidden_size, 3),
                "scale": nn.Linear(hidden_size, 3),
                "distance": nn.Linear(hidden_size, 3),
                "facing": nn.Linear(hidden_size, 2),
            })
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(
        self,
        grounding_hidden: Optional[torch.Tensor],
        vqa_hidden: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if self.grounding_head is not None:
            if grounding_hidden is None:
                raise ValueError("grounding hidden state is required")
            raw = self.grounding_head(grounding_hidden)
            outputs["grounding_bbox"] = raw[..., :4].sigmoid()
            outputs["grounding_visibility_logits"] = raw[..., 4]
        if self.vqa_heads:
            if vqa_hidden is None:
                raise ValueError("VQA hidden state is required")
            for name, head in self.vqa_heads.items():
                outputs[f"vqa_{name}_logits"] = head(vqa_hidden)
        return outputs


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:].clamp_min(0)
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def _generalized_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_xyxy = _cxcywh_to_xyxy(pred)
    target_xyxy = _cxcywh_to_xyxy(target)
    intersection_min = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    intersection_max = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    intersection_size = (intersection_max - intersection_min).clamp_min(0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    pred_size = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp_min(0)
    pred_area = pred_size[..., 0] * pred_size[..., 1]
    target_size = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp_min(0)
    target_area = target_size[..., 0] * target_size[..., 1]
    union = pred_area + target_area - intersection
    enclosing_min = torch.minimum(pred_xyxy[..., :2], target_xyxy[..., :2])
    enclosing_max = torch.maximum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    enclosing_size = (enclosing_max - enclosing_min).clamp_min(0)
    enclosing = enclosing_size[..., 0] * enclosing_size[..., 1]
    iou = intersection / union.clamp_min(1e-7)
    giou = iou - (enclosing - union) / enclosing.clamp_min(1e-7)
    return (1 - giou).mean()


def compute_auxiliary_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: AuxiliaryTasksConfig,
) -> Dict[str, torch.Tensor]:
    """Compute only losses backed by valid labels in the current batch."""
    losses: Dict[str, torch.Tensor] = {}
    if config.grounding_enabled:
        pred_bbox = outputs["grounding_bbox"]
        target_bbox = batch["grounding_bbox_target"].to(pred_bbox)
        visibility = batch["grounding_visibility_target"].to(pred_bbox)
        valid = visibility > 0.5
        zero = pred_bbox.sum() * 0
        if valid.any():
            bbox_l1 = F.smooth_l1_loss(pred_bbox[valid], target_bbox[valid])
            bbox_giou = _generalized_iou_loss(pred_bbox[valid], target_bbox[valid])
        else:
            bbox_l1 = zero
            bbox_giou = zero
        visibility_loss = F.binary_cross_entropy_with_logits(
            outputs["grounding_visibility_logits"], visibility
        )
        losses.update({
            "grounding_bbox_l1": bbox_l1,
            "grounding_bbox_giou": bbox_giou,
            "grounding_visibility": visibility_loss,
            "grounding": (
                config.bbox_l1_weight * bbox_l1
                + config.bbox_giou_weight * bbox_giou
                + config.visibility_weight * visibility_loss
            ),
        })
    if config.vqa_enabled:
        task_losses = []
        task_loss_sums = []
        total_valid = 0
        for index, name in enumerate(("position", "scale", "distance", "facing")):
            logits = outputs[f"vqa_{name}_logits"]
            labels = batch["vqa_labels"][:, index].to(device=logits.device)
            valid = labels >= 0
            if valid.any():
                # Keep the per-task metric, while aggregating the joint loss
                # over only labels that exist.  Real samples have valid
                # position/scale labels and -100 distance/facing labels;
                # simulation samples have all four labels.
                task_loss_sum = F.cross_entropy(
                    logits[valid], labels[valid], reduction="sum"
                )
                task_loss = task_loss_sum / valid.sum().to(task_loss_sum.dtype)
                task_loss_sums.append(task_loss_sum)
                total_valid += int(valid.sum().item())
            else:
                task_loss = logits.sum() * 0
            losses[f"vqa_{name}"] = task_loss
            task_losses.append(task_loss)
        if task_loss_sums:
            losses["vqa"] = torch.stack(task_loss_sums).sum() / float(total_valid)
        else:
            losses["vqa"] = outputs["vqa_position_logits"].sum() * 0
    return losses
