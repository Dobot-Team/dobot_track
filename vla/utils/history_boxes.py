"""Normalized cxcywh boxes in the padded visual-token coordinate system."""
from collections import deque
from typing import Optional
import torch


def clip_boxes(boxes: torch.Tensor, valid: torch.Tensor):
    """Validate and clip to the image, without changing caller-owned tensors."""
    finite = torch.isfinite(boxes).all(-1)
    safe = torch.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
    lo = (safe[..., :2] - safe[..., 2:] / 2).clamp(0, 1)
    hi = (safe[..., :2] + safe[..., 2:] / 2).clamp(0, 1)
    size = hi - lo
    good = valid & finite & (safe[..., 2:] > 0).all(-1) & (size > 0).all(-1)
    clipped = torch.cat(((lo + hi) / 2, size), -1)
    return torch.where(good[..., None], clipped, torch.zeros_like(clipped)), good


def masks_to_boxes(masks: torch.Tensor, valid: torch.Tensor):
    """Tight GT bbox, including both edge pixels; empty/missing masks are invalid."""
    if masks.ndim < 3 or masks.shape[:-2] != valid.shape:
        raise ValueError('masks/valid must have shapes [...,H,W] / [...]')
    h, w = masks.shape[-2:]
    fg = masks > 0.5
    rows, cols = fg.any(-1), fg.any(-2)
    top = rows.to(torch.int64).argmax(-1)
    bottom = h - rows.flip(-1).to(torch.int64).argmax(-1)
    left = cols.to(torch.int64).argmax(-1)
    right = w - cols.flip(-1).to(torch.int64).argmax(-1)
    boxes = torch.stack(((left + right) / (2*w), (top + bottom) / (2*h),
                         (right-left) / w, (bottom-top) / h), -1).float()
    return clip_boxes(boxes, valid & fg.flatten(-2).any(-1))


def augment_boxes(boxes, valid, *, jitter_probability, jitter_scale, dropout_probability):
    """Input-only augmentation driven by PyTorch's seeded worker/rank RNG."""
    boxes, valid = clip_boxes(boxes, valid)
    if jitter_probability > 0:
        apply = torch.rand(valid.shape, device=boxes.device) < jitter_probability
        noise = (torch.rand_like(boxes) * 2 - 1) * jitter_scale
        center = boxes[..., :2] + noise[..., :2] * boxes[..., 2:]
        size = boxes[..., 2:] * noise[..., 2:].exp()
        boxes = torch.where(apply[..., None], torch.cat((center, size), -1), boxes)
    if dropout_probability > 0:
        valid = valid & (torch.rand(valid.shape, device=boxes.device) >= dropout_probability)
    return clip_boxes(boxes, valid)


class HistoryBoxCache:
    """One slot per observed RGB frame; no carry-forward of stale detections."""
    def __init__(self, capacity: int):
        self.entries = deque(maxlen=capacity)

    def clear(self):
        self.entries.clear()

    def observe(self):
        self.entries.append(None)

    def update(self, bbox: Optional[list], visibility: Optional[float]):
        if not self.entries:
            raise RuntimeError('observe must precede bbox update')
        self.entries[-1] = None if bbox is None or visibility is None else (list(bbox), float(visibility))

    def batch(self, slots, *, threshold: float, device):
        boxes = torch.zeros(1, len(slots)-1, 4, dtype=torch.float32, device=device)
        valid = torch.zeros(1, len(slots)-1, dtype=torch.bool, device=device)
        entries = list(self.entries)
        for out, (index, real) in enumerate(slots[:-1]):
            entry = entries[index] if real else None
            if entry is not None:
                box, visibility = entry
                if len(box) == 4 and visibility >= threshold:
                    boxes[0, out] = torch.tensor(box, device=device)
                    valid[0, out] = True
        return clip_boxes(boxes, valid)


def resize_pad_boxes(boxes, geometry, input_size, *, inverse=False):
    """Map normalized original-image cxcywh to/from the exact padded image."""
    height, width, top, _, left, _ = geometry
    scale = boxes.new_tensor([width / input_size, height / input_size])
    offset = boxes.new_tensor([left / input_size, top / input_size])
    if inverse:
        return torch.cat(((boxes[..., :2] - offset) / scale, boxes[..., 2:] / scale), -1)
    return torch.cat((boxes[..., :2] * scale + offset, boxes[..., 2:] * scale), -1)
