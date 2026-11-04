"""TrackVLA episode-level InternVL3 ViT feature shards.

The feature files intentionally live beside the TrackVLA assets instead of in
an opaque global cache.  A frame archive under ``frames/`` maps to a matching
``features/`` shard, which keeps the feature artifact aligned with the source
dataset and makes it possible to move a subset as one unit.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch


FEATURE_SCHEMA_VERSION = 1
FEATURE_STORAGE_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}
FEATURE_STORAGE_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def resolve_feature_storage_dtype(value: str) -> torch.dtype:
    """Resolve the user-facing feature storage dtype name."""

    normalized = str(value).lower()
    try:
        return FEATURE_STORAGE_DTYPES[normalized]
    except KeyError as exc:
        raise ValueError(
            "TrackVLA ViT feature storage dtype must be one of "
            f"{', '.join(FEATURE_STORAGE_DTYPES)}, got {value!r}"
        ) from exc


def feature_storage_dtype_name(dtype: torch.dtype) -> str:
    """Return the stable metadata spelling for an allowed storage dtype."""

    try:
        return FEATURE_STORAGE_DTYPE_NAMES[dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported TrackVLA ViT feature storage dtype: {dtype}"
        ) from exc


def feature_path_for_archive(base_root: Path, archive_path: Path) -> Path:
    """Map ``frames/foo/bar.jpgs`` to ``features/foo/bar.vit.pt``."""

    frames_root = (base_root / "frames").resolve()
    archive_path = archive_path.resolve()
    try:
        relative = archive_path.relative_to(frames_root)
    except ValueError as exc:
        raise ValueError(
            f"TrackVLA frame archive {archive_path} is not under {frames_root}"
        ) from exc
    if relative.suffix.lower() != ".jpgs":
        raise ValueError(
            f"TrackVLA frame archive must use .jpgs, got {archive_path}"
        )
    return (
        base_root
        / "features"
        / relative.parent
        / f"{relative.stem}.vit.pt"
    )


def normalize_frame_name(frame_name: str) -> str:
    """Normalize ZIP member names to the canonical POSIX representation."""

    return PurePosixPath(str(frame_name).replace("\\", "/")).as_posix()


def _metadata_mismatches(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[str]:
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    return mismatches


def save_feature_shard(
    path: Path,
    *,
    features: torch.Tensor,
    frame_names: Sequence[str],
    metadata: Mapping[str, Any],
    storage_dtype: torch.dtype = torch.float32,
) -> None:
    """Atomically save one FP32 or BF16 episode shard."""

    storage_dtype_name = feature_storage_dtype_name(storage_dtype)
    features = features.detach().to(
        device="cpu",
        dtype=storage_dtype,
    ).contiguous()
    if features.ndim != 3:
        raise ValueError(
            f"vit_features must have shape [N,T,D], got {tuple(features.shape)}"
        )
    normalized_names = [normalize_frame_name(name) for name in frame_names]
    if len(normalized_names) != features.shape[0]:
        raise ValueError(
            "frame_names length must match the first feature dimension: "
            f"{len(normalized_names)} != {features.shape[0]}"
        )
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("feature shard frame_names must be unique")

    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "vit_features": features,
        "frame_names": normalized_names,
        "metadata": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            **dict(metadata),
            "storage_dtype": storage_dtype_name,
            "feature_shape": list(features.shape[1:]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_feature_shard(path: Path) -> Dict[str, Any]:
    """Load a locally generated shard using safe primitive-only unpickling."""

    if not path.is_file():
        raise FileNotFoundError(f"TrackVLA ViT feature shard is missing: {path}")
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        payload = torch.load(path, mmap=True, **load_kwargs)
    except (TypeError, RuntimeError):
        # Older torch releases do not support mmap for every serialization
        # layout.  The feature format remains valid without memory mapping.
        payload = torch.load(path, **load_kwargs)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid feature shard payload in {path}")
    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feature shard schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    features = payload.get("vit_features")
    frame_names = payload.get("frame_names")
    metadata = payload.get("metadata")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError(f"Invalid vit_features tensor in {path}")
    if features.dtype not in FEATURE_STORAGE_DTYPE_NAMES:
        raise ValueError(
            f"Feature shard {path} must store fp32 or bf16, got {features.dtype}"
        )
    if not isinstance(frame_names, list) or not all(
        isinstance(name, str) for name in frame_names
    ):
        raise ValueError(f"Invalid frame_names in {path}")
    if features.shape[0] != len(frame_names):
        raise ValueError(
            f"Feature shard {path} has {features.shape[0]} rows but "
            f"{len(frame_names)} frame names"
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata in {path}")
    expected_storage_dtype = feature_storage_dtype_name(features.dtype)
    if metadata.get("storage_dtype") != expected_storage_dtype:
        raise ValueError(
            f"Feature shard {path} storage_dtype metadata must be "
            f"{expected_storage_dtype!r}, got {metadata.get('storage_dtype')!r}"
        )
    return payload


class TrackVLAFeatureStore:
    """Read-only access to dataset-aligned episode feature shards."""

    def __init__(
        self,
        *,
        expected_metadata: Mapping[str, Any],
    ) -> None:
        self.expected_metadata = dict(expected_metadata)

    def _load(self, path: Path) -> tuple[Dict[str, Any], Dict[str, int]]:
        payload = load_feature_shard(path)
        metadata = payload["metadata"]
        mismatches = _metadata_mismatches(metadata, self.expected_metadata)
        if mismatches:
            raise ValueError(
                f"Feature shard metadata mismatch in {path}: "
                + "; ".join(mismatches)
            )
        frame_names = [normalize_frame_name(name) for name in payload["frame_names"]]
        if len(set(frame_names)) != len(frame_names):
            raise ValueError(f"Duplicate frame names in feature shard {path}")
        by_name = {name: index for index, name in enumerate(frame_names)}
        return payload, by_name

    def get(
        self,
        *,
        base_root: Path,
        archive_path: Path,
        frame_names: Iterable[str],
    ) -> torch.Tensor:
        path = feature_path_for_archive(base_root, archive_path)
        payload, by_name = self._load(path)
        metadata = payload["metadata"]
        archive_stat = archive_path.stat()
        relative_archive = archive_path.resolve().relative_to(
            base_root.resolve()
        ).as_posix()
        source_mismatches = []
        if metadata.get("source_archive") != relative_archive:
            source_mismatches.append(
                f"source_archive: expected {relative_archive!r}, "
                f"got {metadata.get('source_archive')!r}"
            )
        if metadata.get("source_size") != int(archive_stat.st_size):
            source_mismatches.append(
                f"source_size: expected {int(archive_stat.st_size)}, "
                f"got {metadata.get('source_size')!r}"
            )
        if metadata.get("source_mtime_ns") != int(archive_stat.st_mtime_ns):
            source_mismatches.append(
                f"source_mtime_ns: expected {int(archive_stat.st_mtime_ns)}, "
                f"got {metadata.get('source_mtime_ns')!r}"
            )
        if source_mismatches:
            raise ValueError(
                f"Feature shard source mismatch in {path}: "
                + "; ".join(source_mismatches)
            )
        requested = [normalize_frame_name(name) for name in frame_names]
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise KeyError(
                f"Feature shard {path} is missing frame(s): "
                + ", ".join(missing[:8])
            )
        indices = torch.tensor([by_name[name] for name in requested], dtype=torch.long)
        return payload["vit_features"].index_select(0, indices)
