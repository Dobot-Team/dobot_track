"""Training-time corruptions for oracle historical tracking masks.

TrackVLA provides clean ground-truth masks, while online inference consumes
predicted masks.  This module applies one correlated corruption family to a
whole history clip so the compressor sees realistic boundary, localization,
false-positive, missing-mask, and temporal-lag errors during training.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from vla.config.history_mask_augmentation import (
    HistoryMaskAugmentationConfig,
)


class HistoryMaskAugmenter:
    """Apply one balanced, clip-correlated corruption to historical masks.

    Randomness is local to each DataLoader worker.  Worker processes use
    ``WorkerInfo.seed``; single-process loading uses the explicit constructor
    seed.  Consequently, two equally seeded loaders produce the same
    augmentation stream without sharing NumPy's global RNG.
    """

    _CORRUPTIONS = (
        "morphology",
        "affine",
        "holes",
        "false_positive",
        "dropout",
        "lag",
    )
    _CORRUPTION_PROBABILITIES = np.asarray(
        [0.35, 0.25, 0.15, 0.10, 0.075, 0.075],
        dtype=np.float64,
    )

    def __init__(
        self,
        config: Union[
            bool,
            HistoryMaskAugmentationConfig,
            Mapping[str, object],
            None,
        ] = None,
        *,
        seed: int = 0,
    ) -> None:
        self.config = HistoryMaskAugmentationConfig.from_value(config)
        self.seed = int(seed)
        self._rng_key: Optional[tuple[int, int, int]] = None
        self._rng: Optional[np.random.Generator] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never pickle a partially consumed parent-process stream into workers.
        state["_rng_key"] = None
        state["_rng"] = None
        return state

    def _worker_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = -1 if worker is None else int(worker.id)
        worker_seed = self.seed if worker is None else int(worker.seed)
        key = (os.getpid(), worker_id, worker_seed)
        if self._rng is None or self._rng_key != key:
            self._rng = np.random.default_rng(worker_seed)
            self._rng_key = key
        return self._rng

    @staticmethod
    def _validate_inputs(
        masks: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
            raise ValueError(
                "history masks must be a tensor with shape [H,S,S], got "
                f"{getattr(masks, 'shape', None)}"
            )
        if masks.shape[-2] != masks.shape[-1] or masks.shape[-1] <= 0:
            raise ValueError(
                "history masks must have positive square spatial dimensions, got "
                f"{tuple(masks.shape[-2:])}"
            )
        if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != (masks.shape[0],):
            raise ValueError(
                "history mask validity must have shape [H], got "
                f"{getattr(valid, 'shape', None)}"
            )
        return masks.float().clamp(0.0, 1.0), valid.to(dtype=torch.bool)

    @staticmethod
    def _bbox_short_side(mask: torch.Tensor) -> int:
        coordinates = torch.nonzero(mask > 0.5, as_tuple=False)
        if coordinates.numel() == 0:
            return 0
        height = int(coordinates[:, 0].amax() - coordinates[:, 0].amin() + 1)
        width = int(coordinates[:, 1].amax() - coordinates[:, 1].amin() + 1)
        return min(height, width)

    @staticmethod
    def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        kernel = 2 * radius + 1
        padded = F.pad(mask[None, None], (radius,) * 4, value=0.0)
        return F.max_pool2d(padded, kernel_size=kernel, stride=1)[0, 0]

    @staticmethod
    def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        kernel = 2 * radius + 1
        # Outside-of-image pixels are background for binary erosion.
        padded = F.pad(mask[None, None], (radius,) * 4, value=0.0)
        return -F.max_pool2d(-padded, kernel_size=kernel, stride=1)[0, 0]

    def _morphology(
        self,
        masks: torch.Tensor,
        valid: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        output = masks.clone()
        mode = str(rng.choice(("erode", "dilate", "opening", "closing")))
        base_ratio = float(rng.uniform(0.01, 0.10))
        for index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            short_side = self._bbox_short_side(masks[index])
            if short_side <= 0:
                continue
            ratio = float(np.clip(base_ratio * rng.uniform(0.85, 1.15), 0.01, 0.10))
            radius = max(1, int(round(short_side * ratio)))
            current = masks[index]
            if mode == "erode":
                current = self._erode(current, radius)
            elif mode == "dilate":
                current = self._dilate(current, radius)
            elif mode == "opening":
                current = self._dilate(self._erode(current, radius), radius)
            else:
                current = self._erode(self._dilate(current, radius), radius)
            output[index] = current
        return output

    def _affine(
        self,
        masks: torch.Tensor,
        valid: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        output = masks.clone()
        image_size = masks.shape[-1]
        use_translation = bool(rng.integers(0, 2))
        base_dx = float(rng.uniform(-0.05, 0.05) * image_size)
        base_dy = float(rng.uniform(-0.05, 0.05) * image_size)
        base_scale = float(rng.uniform(0.85, 1.15))
        for index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            if use_translation:
                dx = int(
                    round(
                        np.clip(
                            base_dx + rng.uniform(-0.005, 0.005) * image_size,
                            -0.05 * image_size,
                            0.05 * image_size,
                        )
                    )
                )
                dy = int(
                    round(
                        np.clip(
                            base_dy + rng.uniform(-0.005, 0.005) * image_size,
                            -0.05 * image_size,
                            0.05 * image_size,
                        )
                    )
                )
                scale = 1.0
            else:
                dx = dy = 0
                scale = float(
                    np.clip(base_scale + rng.uniform(-0.01, 0.01), 0.85, 1.15)
                )
            output[index] = transform_functional.affine(
                masks[index].unsqueeze(0),
                angle=0.0,
                translate=[dx, dy],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )[0]
        return output

    @staticmethod
    def _nearest_coordinates(
        candidates: torch.Tensor,
        count: int,
        center_y: float,
        center_x: float,
    ) -> torch.Tensor:
        if count <= 0 or candidates.numel() == 0:
            return candidates[:0]
        distances = (
            (candidates[:, 0].float() - center_y).square()
            + (candidates[:, 1].float() - center_x).square()
        )
        count = min(count, candidates.shape[0])
        selected = torch.topk(distances, k=count, largest=False).indices
        return candidates[selected]

    def _holes(
        self,
        masks: torch.Tensor,
        valid: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        output = masks.clone()
        base_fraction = float(rng.uniform(0.05, 0.25))
        relative_y = float(rng.uniform(0.0, 1.0))
        relative_x = float(rng.uniform(0.0, 1.0))
        for index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            foreground = torch.nonzero(masks[index] > 0.5, as_tuple=False)
            if foreground.numel() == 0:
                continue
            fraction = float(
                np.clip(base_fraction * rng.uniform(0.9, 1.1), 0.01, 0.25)
            )
            count = max(1, int(np.floor(foreground.shape[0] * fraction)))
            y_min, x_min = foreground.amin(dim=0).tolist()
            y_max, x_max = foreground.amax(dim=0).tolist()
            center_y = y_min + relative_y * (y_max - y_min)
            center_x = x_min + relative_x * (x_max - x_min)
            removed = self._nearest_coordinates(
                foreground,
                count,
                center_y,
                center_x,
            )
            output[index, removed[:, 0], removed[:, 1]] = 0.0
        return output

    def _false_positive(
        self,
        masks: torch.Tensor,
        valid: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        output = masks.clone()
        image_size = masks.shape[-1]
        base_fraction = float(rng.uniform(0.05, 0.30))
        relative_y = float(rng.uniform(0.0, 1.0))
        relative_x = float(rng.uniform(0.0, 1.0))
        for index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            foreground_count = int((masks[index] > 0.5).sum())
            background = torch.nonzero(masks[index] <= 0.5, as_tuple=False)
            if foreground_count == 0 or background.numel() == 0:
                continue
            fraction = float(
                np.clip(base_fraction * rng.uniform(0.9, 1.1), 0.05, 0.30)
            )
            count = max(1, int(round(foreground_count * fraction)))
            center_y = relative_y * (image_size - 1)
            center_x = relative_x * (image_size - 1)
            added = self._nearest_coordinates(
                background,
                count,
                center_y,
                center_x,
            )
            output[index, added[:, 0], added[:, 1]] = 1.0
        return output

    @staticmethod
    def _lag(masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        output = masks.clone()
        previous_valid: Optional[int] = None
        for index in range(masks.shape[0]):
            if not bool(valid[index]):
                continue
            if previous_valid is not None:
                output[index] = masks[previous_valid]
            previous_valid = index
        return output

    def __call__(
        self,
        masks: torch.Tensor,
        valid: torch.Tensor,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        masks, valid = self._validate_inputs(masks, valid)
        if not self.config.enabled or masks.shape[0] == 0:
            return masks.clone()
        if rng is None:
            rng = self._worker_rng()
        if float(rng.random()) < self.config.clean_probability:
            return masks.clone()

        corruption = str(
            rng.choice(self._CORRUPTIONS, p=self._CORRUPTION_PROBABILITIES)
        )
        if corruption == "morphology":
            output = self._morphology(masks, valid, rng)
        elif corruption == "affine":
            output = self._affine(masks, valid, rng)
        elif corruption == "holes":
            output = self._holes(masks, valid, rng)
        elif corruption == "false_positive":
            output = self._false_positive(masks, valid, rng)
        elif corruption == "dropout":
            output = masks.clone()
            output[valid] = 0.0
        else:
            output = self._lag(masks, valid)

        # Missing/padded masks remain exact zero, and all model inputs stay
        # binary after interpolation/morphology.
        output[~valid] = 0.0
        return (output > 0.5).to(dtype=masks.dtype)
