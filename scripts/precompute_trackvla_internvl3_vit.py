#!/usr/bin/env python3
"""Precompute TrackVLA InternVL3 ViT feature shards.

The command adds ``features/<episode>.vit.pt`` files next to the existing
TrackVLA ``frames/``, ``jsonl/`` and ``masks/`` assets.  It never modifies the
source manifests or RGB/mask archives.

Example:
    python scripts/precompute_trackvla_internvl3_vit.py \
        --config configs/dobottrack_internvl3_1b_q_identity_box_mask.yaml \
        --device cuda --batch-size 16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence
import zipfile

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from vla.data.trackvla import (
    TrackVLADataset,
    _find_base_root,
    build_trackvla_image_transform,
    preprocess_trackvla_rgb_batch_on_gpu,
)
from vla.data.trackvla_features import (
    feature_storage_dtype_name,
    feature_path_for_archive,
    load_feature_shard,
    resolve_feature_storage_dtype,
    save_feature_shard,
)
from vla.models.internvl3_vla import (
    InternVL3VLA,
    encode_internvl3_vit_features,
    internvl3_vit_feature_cache_metadata,
)


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class _PendingShard:
    """One feature shard awaiting visual encoding."""

    archive_path: Path
    shard_path: Path
    frame_names: Sequence[str]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class _FrameTask:
    """A single RGB member and its destination row in a feature shard."""

    shard_index: int
    frame_index: int
    archive_path: str
    frame_name: str


class _PrecomputeFrameDataset(Dataset):
    """CPU-only JPEG decoding for GPU-side TrackVLA preprocessing."""

    def __init__(self, tasks: Sequence[_FrameTask]) -> None:
        self.tasks = list(tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> tuple[int, int, torch.Tensor]:
        task = self.tasks[index]
        with zipfile.ZipFile(task.archive_path, "r") as archive:
            with Image.open(io.BytesIO(archive.read(task.frame_name))) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return task.shard_index, task.frame_index, torch.from_numpy(rgb)


def _shard_batch_sampler(
    shards: Sequence[_PendingShard],
    *,
    batch_size: int,
) -> tuple[list[_FrameTask], list[list[int]]]:
    """Keep every GPU batch within one shard while workers prefetch globally."""

    tasks: list[_FrameTask] = []
    batches: list[list[int]] = []
    for shard_index, shard in enumerate(shards):
        start = len(tasks)
        tasks.extend(
            _FrameTask(
                shard_index=shard_index,
                frame_index=frame_index,
                archive_path=str(shard.archive_path),
                frame_name=frame_name,
            )
            for frame_index, frame_name in enumerate(shard.frame_names)
        )
        for batch_start in range(start, len(tasks), batch_size):
            batches.append(list(range(batch_start, min(batch_start + batch_size, len(tasks)))))
    return tasks, batches


def _configure_preprocess_worker(_: int) -> None:
    """Avoid N workers each creating a full CPU kernel thread pool."""

    torch.set_num_threads(1)


def _collate_decoded_frames(
    batch: Sequence[tuple[int, int, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack same-episode RGB frames before pinning and GPU preprocessing."""

    shard_indices, frame_indices, rgb_frames = zip(*batch)
    try:
        stacked_rgb = torch.stack(rgb_frames, dim=0)
    except RuntimeError as exc:
        shapes = [tuple(frame.shape) for frame in rgb_frames]
        raise ValueError(
            "All frames within one TrackVLA .jpgs archive must share an RGB "
            f"shape for GPU preprocessing, got {shapes}"
        ) from exc
    return (
        torch.tensor(shard_indices, dtype=torch.long),
        torch.tensor(frame_indices, dtype=torch.long),
        stacked_rgb,
    )


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return config


def _source_archive_metadata(base_root: Path, archive_path: Path) -> Dict[str, Any]:
    stat = archive_path.stat()
    return {
        "source_archive": archive_path.resolve().relative_to(
            base_root.resolve()
        ).as_posix(),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _existing_shard_is_usable(
    path: Path,
    *,
    expected_metadata: Dict[str, Any],
    source_metadata: Dict[str, Any],
    frame_names: list[str],
) -> bool:
    try:
        payload = load_feature_shard(path)
        metadata = payload["metadata"]
        for key, value in {**expected_metadata, **source_metadata}.items():
            if metadata.get(key) != value:
                return False
        existing_names = set(payload["frame_names"])
        return set(frame_names).issubset(existing_names)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="CPU workers for ZIP reads and JPEG decode; set 0 to disable.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches each preprocessing worker keeps ready for the GPU.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    config = _load_config(args.config)
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    if data_config.get("dataset_type") != "trackvla":
        raise ValueError("The precompute script requires data.dataset_type=trackvla")
    if model_config.get("model_type") not in (None, "internvl3"):
        raise ValueError("The precompute script requires model.model_type=internvl3")
    if not bool(model_config.get("freeze_vision", True)):
        raise ValueError("ViT feature precomputation requires model.freeze_vision=true")
    source_paths = data_config.get("trackvla_sources") or data_config.get("train_json")
    if not source_paths:
        raise ValueError("Config must provide data.trackvla_sources")

    dtype_name = str(model_config.get("dtype", "bf16")).lower()
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported model.dtype {dtype_name!r}")
    compute_dtype = DTYPE_MAP[dtype_name]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if device.type != "cuda":
        raise ValueError(
            "TrackVLA InternVL3 feature precomputation uses GPU preprocessing; "
            "pass --device cuda"
        )
    feature_storage_dtype = resolve_feature_storage_dtype(
        data_config.get("vit_feature_storage_dtype", "bf16")
    )
    if feature_storage_dtype == torch.bfloat16 and compute_dtype != torch.bfloat16:
        raise ValueError(
            "data.vit_feature_storage_dtype=bf16 requires model.dtype=bf16 "
            "so cached ViT tokens remain numerically exact"
        )
    model_path = InternVL3VLA._resolve_model_path(
        str(model_config["model_path"]),
        local_files_only=True,
    )
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "local_files_only": True,
        "device_map": None,
        "low_cpu_mem_usage": True,
    }
    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    print(f"[precompute] Loading InternVL3 vision model from {model_path}")
    native_model = AutoModel.from_pretrained(model_path, **model_kwargs)
    native_model.vision_tower.to(device=device)

    input_size = int(data_config["input_size"])
    expected_metadata = internvl3_vit_feature_cache_metadata(
        native_model,
        model_path=model_path,
        input_size=input_size,
        dtype=compute_dtype,
    )
    expected_metadata["storage_dtype"] = feature_storage_dtype_name(
        feature_storage_dtype
    )

    dataset = TrackVLADataset(
        source_paths=source_paths,
        tokenizer=None,
        transform=build_trackvla_image_transform(input_size),
        num_history_frames=int(data_config["num_history_frames"]),
        history_frame_interval=int(data_config.get("history_frame_interval", 0)),
        n_future=int(data_config["n_future"]),
        trajectory_dim=int(data_config.get("trajectory_dim", 3)),
        prompt=str(data_config.get("prompt", "")),
        num_image_tokens=int(data_config.get("num_image_tokens", 256)),
        max_length=int(data_config.get("max_length", 1200)),
        model_type="internvl3",
        index_cache_dir=data_config.get("index_cache_dir"),
        mask_input_size=input_size,
        history_compressor_config=model_config.get("history_compressor_config"),
        history_mask_augmentation=False,
    )
    required = dataset.required_feature_frames()
    print(
        f"[precompute] Indexed {len(dataset)} samples and "
        f"{sum(len(names) for names in required.values())} unique episode-frame groups"
    )

    total_frames = 0
    skipped_shards = 0
    written_shards = 0
    total_bytes = 0
    pending_shards: list[_PendingShard] = []
    for archive_string, frame_names in sorted(required.items()):
        archive_path = Path(archive_string)
        base_root = _find_base_root(archive_path)
        shard_path = feature_path_for_archive(base_root, archive_path)
        source_metadata = _source_archive_metadata(base_root, archive_path)
        if (
            not args.overwrite
            and _existing_shard_is_usable(
                shard_path,
                expected_metadata=expected_metadata,
                source_metadata=source_metadata,
                frame_names=frame_names,
            )
        ):
            skipped_shards += 1
            total_frames += len(frame_names)
            total_bytes += int(shard_path.stat().st_size)
            continue

        pending_shards.append(
            _PendingShard(
                archive_path=archive_path,
                shard_path=shard_path,
                frame_names=frame_names,
                metadata={**expected_metadata, **source_metadata},
            )
        )

    tasks, batch_sampler = _shard_batch_sampler(
        pending_shards,
        batch_size=args.batch_size,
    )
    if tasks:
        loader_kwargs: Dict[str, Any] = {
            "batch_sampler": batch_sampler,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "collate_fn": _collate_decoded_frames,
        }
        if args.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=args.prefetch_factor,
                worker_init_fn=_configure_preprocess_worker,
            )
        frame_loader = DataLoader(_PrecomputeFrameDataset(tasks), **loader_kwargs)
        print(
            f"[precompute] preprocessing workers={args.num_workers}, "
            f"prefetch_factor={args.prefetch_factor}, batches={len(batch_sampler)}"
        )

        active_shard_index: int | None = None
        active_features: torch.Tensor | None = None

        def write_active_shard() -> None:
            nonlocal active_shard_index, active_features
            nonlocal written_shards, total_frames, total_bytes
            if active_shard_index is None or active_features is None:
                return
            shard = pending_shards[active_shard_index]
            save_feature_shard(
                shard.shard_path,
                features=active_features,
                frame_names=shard.frame_names,
                metadata=shard.metadata,
                storage_dtype=feature_storage_dtype,
            )
            written_shards += 1
            total_frames += len(shard.frame_names)
            total_bytes += int(shard.shard_path.stat().st_size)
            print(
                f"[precompute] wrote {shard.shard_path} "
                f"frames={len(shard.frame_names)} shape={tuple(active_features.shape)}"
            )
            active_shard_index = None
            active_features = None

        for shard_indices, frame_indices, cpu_rgb_frames in frame_loader:
            batch_shard_indices = shard_indices.tolist()
            if len(set(batch_shard_indices)) != 1:
                raise RuntimeError("Precompute batch unexpectedly spans feature shards")
            shard_index = int(batch_shard_indices[0])
            if active_shard_index is not None and shard_index != active_shard_index:
                write_active_shard()
            if active_shard_index is None:
                active_shard_index = shard_index

            pixel_values = preprocess_trackvla_rgb_batch_on_gpu(
                cpu_rgb_frames,
                input_size=input_size,
                device=device,
            )
            with torch.inference_mode():
                encoded_features = encode_internvl3_vit_features(
                    native_model,
                    pixel_values,
                    dtype=compute_dtype,
                )
            if (
                feature_storage_dtype == torch.bfloat16
                and encoded_features.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "InternVL3 ViT returned "
                    f"{encoded_features.dtype} under bf16 computation; refusing "
                    "lossy bf16 feature storage"
                )
            batch_features = encoded_features.to(
                device="cpu",
                dtype=feature_storage_dtype,
            )
            if active_features is None:
                active_features = torch.empty(
                    (
                        len(pending_shards[shard_index].frame_names),
                        *batch_features.shape[1:],
                    ),
                    dtype=feature_storage_dtype,
                )
            active_features.index_copy_(
                0,
                frame_indices.to(dtype=torch.long),
                batch_features,
            )
        write_active_shard()

    print(
        "[precompute] complete: "
        f"frames={total_frames}, written_shards={written_shards}, "
        f"skipped_shards={skipped_shards}, storage={total_bytes / (1024**3):.2f} GiB"
    )


if __name__ == "__main__":
    main()
