"""Physically gather selected patch indices; no ground-truth or bbox perturbations."""

import numpy as np


def pack_box_indices(boxes, valid, grid=16):
    boxes = np.asarray(boxes, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).copy()
    finite = np.isfinite(boxes).all(-1)
    safe = np.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.clip(safe[:, :2] - safe[:, 2:] / 2, 0, 1)
    hi = np.clip(safe[:, :2] + safe[:, 2:] / 2, 0, 1)
    valid &= finite & (safe[:, 2:] > 0).all(-1) & ((hi - lo) > 0).all(-1)
    normalized = np.where(
        valid[:, None],
        np.concatenate(((lo + hi) / 2, hi - lo), -1),
        np.zeros_like(boxes),
    )
    coords = (np.arange(grid, dtype=np.float32) + np.float32(0.5)) / np.float32(grid)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    centers = np.stack((xx, yy), -1).reshape(-1, 2)
    # Match the original clip -> normalized cxcywh -> corner arithmetic.
    lo = normalized[:, :2] - normalized[:, 2:] / 2
    hi = normalized[:, :2] + normalized[:, 2:] / 2
    rows = []
    for b in range(len(boxes)):
        if not valid[b]:
            rows.append(np.array([grid * grid], dtype=np.int32))
            continue
        indices = np.flatnonzero(
            ((centers >= lo[b]) & (centers < hi[b])).all(-1)
        ).astype(np.int32)
        if not len(indices):
            xy = np.clip((normalized[b, :2] * grid).astype(np.int64), 0, grid - 1)
            indices = np.array([xy[1] * grid + xy[0]], np.int32)
        rows.append(indices)
    k = max(map(len, rows))
    indices = np.full((len(rows), k), grid * grid, np.int32)
    padding = np.ones((len(rows), k), bool)
    for b, row in enumerate(rows):
        indices[b, : len(row)] = row
        padding[b, : len(row)] = False
    return indices, padding


def pad_to_bucket(indices, padding):
    k = indices.shape[1]
    bucket = next(b for b in [32, 64, 128, 256] if b >= k)
    if k == bucket:
        return indices, padding
    idx = np.full((len(indices), bucket), 256, np.int32)
    pad = np.ones((len(indices), bucket), bool)
    idx[:, :k] = indices
    pad[:, :k] = padding
    return idx, pad


def attach_cpu_box_cache(agent):
    original = agent.history_box_cache.batch

    def cpu_batch(slots, *, threshold, device):
        return original(slots, threshold=threshold, device="cpu")

    agent.history_box_cache.batch = cpu_batch
