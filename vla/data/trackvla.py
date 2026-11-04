"""Dataset adapter for TrackVLA JSON manifests and ZIP frame archives.

TrackVLA samples reference one ``.jpgs`` ZIP archive per episode. Each JSON
record identifies the current frame, carries a natural-language tracking
instruction, and contains a precomputed ego-frame trajectory. Optional mask
loading keeps past masks aligned with the sampled RGB slots and keeps the
current clean mask separate for SAM2 supervision; it never creates an episode
stream or carries state between independently shuffled samples.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import warnings
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from vla.config import (
    AuxiliaryTasksConfig,
    HistoryCompressorConfig,
    HistoryMaskAugmentationConfig,
    ModelExtensionContext,
    ModelExtensionsConfig,
)
from vla.utils.common import (
    ResizeLongestSideAndPad,
    resize_longest_side_and_pad_geometry,
    apply_qwen_vl_chat_template,
    build_custom_multimodal_prompt,
    build_transform,
    preprocess_custom_images,
)

from .history_mask_augmentation import HistoryMaskAugmenter
from vla.utils.history_boxes import masks_to_boxes, resize_pad_boxes


def _optional_finite_number(example: Mapping[str, Any], names: Sequence[str]) -> Optional[float]:
    for name in names:
        value = example.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if np.isfinite(value):
                return value
    return None


def build_auxiliary_targets(
    mask: torch.Tensor,
    example: Mapping[str, Any],
    config: AuxiliaryTasksConfig,
) -> Dict[str, torch.Tensor]:
    """Build normalized box and categorical answers from authoritative labels."""
    if mask.ndim != 2:
        raise ValueError(f"current mask must have shape [H,W], got {tuple(mask.shape)}")
    height, width = mask.shape
    foreground = mask > 0.5
    points = foreground.nonzero(as_tuple=False)
    visible = points.numel() > 0
    bbox = torch.zeros(4, dtype=torch.float32)
    labels = torch.full((4,), -100, dtype=torch.long)
    if visible:
        y0, x0 = points.amin(dim=0).to(torch.float32)
        y1, x1 = (points.amax(dim=0) + 1).to(torch.float32)
        box_width = (x1 - x0) / float(width)
        box_height = (y1 - y0) / float(height)
        center_x = (x0 + x1) / (2.0 * float(width))
        center_y = (y0 + y1) / (2.0 * float(height))
        computed_bbox = torch.tensor(
            [center_x, center_y, box_width, box_height], dtype=torch.float32
        )
        manifest_bbox = example.get("bbox_cxcywh_norm")
        if isinstance(manifest_bbox, (list, tuple)) and len(manifest_bbox) == 4:
            try:
                candidate = torch.tensor(manifest_bbox, dtype=torch.float32)
                if torch.isfinite(candidate).all() and (candidate >= 0).all():
                    bbox = candidate.clamp(0.0, 1.0)
                else:
                    bbox = computed_bbox
            except (TypeError, ValueError):
                bbox = computed_bbox
        else:
            bbox = computed_bbox
        position = example.get("position")
        if position in {"left", "center", "right"}:
            labels[0] = {"left": 0, "center": 1, "right": 2}[position]
        else:
            labels[0] = 0 if float(bbox[0]) < 1 / 3 else (2 if float(bbox[0]) >= 2 / 3 else 1)
        area = float(bbox[2] * bbox[3])
        scale = example.get("scale")
        if scale in {"small", "medium", "large"}:
            labels[1] = {"small": 0, "medium": 1, "large": 2}[scale]
        else:
            labels[1] = (
                0 if area < config.small_area_threshold
                else (2 if area >= config.large_area_threshold else 1)
            )
        distance = _optional_finite_number(
            example, ("target_distance", "target_distance_m", "distance")
        )
        if distance is not None and distance >= 0:
            labels[2] = (
                0 if distance < config.near_distance_threshold
                else (2 if distance >= config.far_distance_threshold else 1)
            )
        facing = _optional_finite_number(
            example, ("target_facing", "facing", "is_facing")
        )
        if facing is not None:
            labels[3] = int(facing >= 0.5)
    return {
        "grounding_bbox_target": bbox,
        "grounding_visibility_target": torch.tensor(float(visible)),
        "vqa_labels": labels,
    }


@dataclass(frozen=True)
class _ExampleRef:
    """Byte range for one JSON object, kept small enough for lazy loading."""

    manifest_path: str
    offset: int
    length: int
    base_root: str


def _split_sources(source_paths: Union[str, Sequence[str]]) -> List[Path]:
    if isinstance(source_paths, str):
        parts = [part.strip() for part in source_paths.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in source_paths if str(part).strip()]
    if not parts:
        raise ValueError("trackvla source_paths must contain at least one path")
    return [Path(part) for part in parts]


def _find_base_root(path: Path) -> Path:
    """Find the TrackVLA subset root containing ``frames/``."""

    candidate = path if path.is_dir() else path.parent
    for _ in range(16):
        if (candidate / "frames").is_dir():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    raise ValueError(f"Could not find TrackVLA base root with frames/ above {path}")


def _manifest_files_for_root(root: Path) -> List[Path]:
    """Resolve the authoritative manifest for one TrackVLA subset.

    ``dataset.json`` is the complete subset manifest produced by TrackVLA and
    remains lazy through our byte-range index. A ``jsonl/`` directory may be a
    partial conversion, so it is only a fallback when the authoritative array
    is absent; otherwise a single leftover JSONL shard can silently discard
    most training examples.
    """

    manifest = root / "dataset.json"
    if manifest.is_file():
        return [manifest]

    jsonl_root = root / "jsonl"
    if jsonl_root.is_dir():
        files = sorted(jsonl_root.rglob("*.jsonl"))
        if files:
            return files

    files = sorted(root.rglob("*.jsonl"))
    if files:
        return files
    raise FileNotFoundError(
        f"No dataset.json or jsonl/ manifests found under TrackVLA root {root}"
    )


def _resolve_manifest_files(source_paths: Union[str, Sequence[str]]) -> List[Path]:
    manifests: List[Path] = []
    for source in _split_sources(source_paths):
        if source.is_file():
            if source.suffix.lower() not in {".json", ".jsonl"}:
                raise ValueError(f"Unsupported TrackVLA manifest: {source}")
            manifests.append(source)
            continue

        if not source.is_dir():
            raise FileNotFoundError(f"TrackVLA source does not exist: {source}")

        if (source / "frames").is_dir():
            manifests.extend(_manifest_files_for_root(source))
            continue

        # A parent such as trackvla-data/ expands to every immediate subset
        # with frames/. This differs from an explicit subset path, which keeps
        # the training mixture under the caller's direct control.
        subset_roots = sorted(
            child for child in source.iterdir() if child.is_dir() and (child / "frames").is_dir()
        )
        if not subset_roots:
            raise FileNotFoundError(
                f"No TrackVLA subset containing frames/ found under {source}"
            )
        for subset_root in subset_roots:
            manifests.extend(_manifest_files_for_root(subset_root))

    return manifests


def _iter_json_array_objects(path: Path) -> Iterable[Tuple[int, int]]:
    """Yield byte ranges for top-level objects in a JSON array.

    ``dataset.json`` files in TrackVLA can be several hundred megabytes.  This
    scanner tracks JSON string escaping and brace depth so examples can be read
    later on demand instead of materializing the complete array in memory.
    """

    with path.open("rb") as f:
        offset = 0
        while True:
            first = f.read(1)
            if not first:
                raise ValueError(f"Empty JSON manifest: {path}")
            offset += 1
            if not first.isspace():
                break
        if first != b"[":
            raise ValueError(f"TrackVLA JSON manifest must be a top-level array: {path}")

        depth = 0
        object_start: Optional[int] = None
        in_string = False
        escaped = False

        while chunk := f.read(1024 * 1024):
            for byte in chunk:
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                else:
                    if byte == ord('"'):
                        in_string = True
                    elif byte == ord("{"):
                        if depth == 0:
                            object_start = offset
                        depth += 1
                    elif byte == ord("}"):
                        depth -= 1
                        if depth < 0:
                            raise ValueError(f"Malformed JSON object nesting in {path}")
                        if depth == 0:
                            if object_start is None:
                                raise ValueError(f"Missing object start in {path}")
                            yield object_start, offset - object_start + 1
                            object_start = None
                offset += 1

        if in_string or escaped or depth != 0:
            raise ValueError(f"Incomplete JSON array in {path}")


def _index_jsonl(path: Path, base_root: Path) -> List[_ExampleRef]:
    refs: List[_ExampleRef] = []
    offset = 0
    with path.open("rb") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    # Validate once while indexing so malformed records cannot
                    # fail a later DataLoader worker after a long startup.
                    example = json.loads(line)
                    if not isinstance(example, dict):
                        raise ValueError("record is not a JSON object")
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                    warnings.warn(
                        f"[TrackVLADataset] Skipping malformed JSONL record "
                        f"{path}:{line_number} at byte {offset}: {exc}"
                    )
                else:
                    refs.append(
                        _ExampleRef(
                            manifest_path=str(path),
                            offset=offset,
                            length=len(line),
                            base_root=str(base_root),
                        )
                    )
            offset += len(line)
    return refs


def _index_json_array(path: Path, base_root: Path) -> List[_ExampleRef]:
    return [
        _ExampleRef(
            manifest_path=str(path),
            offset=offset,
            length=length,
            base_root=str(base_root),
        )
        for offset, length in _iter_json_array_objects(path)
    ]


def _index_cache_key(
    manifests: Sequence[Path],
    *,
    base_root_override: Optional[Path] = None,
) -> str:
    digest = hashlib.sha256()
    # Bump when index validation behavior changes so stale ref caches are ignored.
    digest.update(b"trackvla-index-v2")
    for manifest in manifests:
        stat = manifest.stat()
        digest.update(str(manifest.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    if base_root_override is not None:
        digest.update(b"base-root-override")
        digest.update(str(base_root_override.resolve()).encode("utf-8"))
    return digest.hexdigest()[:16]


def _write_pickle_atomically(cache_path: Path, value: object) -> None:
    """Publish a pickle only after it has been written completely."""

    temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _valid_episode_indices(value: object, expected_samples: int) -> bool:
    """Validate the small amount of structure needed for an episode cache."""

    if not isinstance(value, dict):
        return False

    sample_count = 0
    for episode_id, indices in value.items():
        if not isinstance(episode_id, str) or not episode_id:
            return False
        if not isinstance(indices, list) or not indices:
            return False
        if not all(
            isinstance(index, int) and 0 <= index < expected_samples
            for index in indices
        ):
            return False
        sample_count += len(indices)
    return sample_count == expected_samples


class TrackVLADataset(Dataset):
    """Load TrackVLA RGB history frames and precomputed future trajectories.

    ``trajectory`` is preferred over ``actions`` because TrackVLA already stores
    the target waypoints in the intended coordinate system. When ``load_masks``
    is enabled, ``mask_file`` supplies a clean current-frame target. QFormer
    mode additionally loads aligned past masks and may corrupt only those
    history inputs during training.
    """

    def __init__(
        self,
        source_paths: Union[str, Sequence[str]],
        tokenizer,
        transform: transforms.Compose,
        num_history_frames: int = 4,
        history_frame_interval: int = 0,
        n_future: int = 9,
        trajectory_dim: int = 3,
        prompt: str = "Predict the follower's future trajectory in the ego coordinate frame.",
        num_image_tokens: int = 256,
        max_length: int = 1200,
        model_type: str = "internvl",
        processor=None,
        image_token: Optional[str] = None,
        qwen_min_pixels: int = 3136,
        qwen_max_pixels: int = 100_000,
        index_cache_dir: Optional[str] = None,
        max_open_archives: int = 32,
        load_masks: bool = False,
        auxiliary_tasks_config: Optional[
            Union[AuxiliaryTasksConfig, Mapping[str, Any]]
        ] = None,
        mask_input_size: Optional[int] = None,
        history_compressor_config: Optional[
            Union[HistoryCompressorConfig, Mapping[str, Any]]
        ] = None,
        history_mask_augmentation: Union[
            bool,
            HistoryMaskAugmentationConfig,
            Mapping[str, object],
            None,
        ] = True,
        augmentation_seed: int = 0,
        base_root_override: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        if num_history_frames <= 0:
            raise ValueError("num_history_frames must be positive")
        if history_frame_interval < 0:
            raise ValueError("history_frame_interval must be non-negative")
        if n_future <= 0:
            raise ValueError("n_future must be positive")
        if trajectory_dim <= 0:
            raise ValueError("trajectory_dim must be positive")
        if max_open_archives <= 0:
            raise ValueError("max_open_archives must be positive")
        if model_type in {"qwen2vl", "qwen3vl"} and processor is None:
            raise ValueError(f"processor is required for model_type={model_type}")
        if model_type == "custom" and processor is None:
            raise ValueError("processor is required for model_type=custom")
        if model_type == "custom" and (
            not isinstance(image_token, str) or not image_token
        ):
            raise ValueError("image_token is required for model_type=custom")
        if model_type == "custom" and (
            mask_input_size is None or int(mask_input_size) <= 0
        ):
            raise ValueError("mask_input_size is required for custom fixed-square preprocessing")
        if model_type == "qwen3vl" and (
            mask_input_size is None or int(mask_input_size) <= 0
        ):
            raise ValueError(
                "mask_input_size is required for qwen3vl fixed-square RGB preprocessing"
            )

        history_compressor_config = HistoryCompressorConfig.from_value(
            history_compressor_config
        )
        ModelExtensionsConfig(
            history_compressor=history_compressor_config
        ).validate_for(
            ModelExtensionContext(
                model_type=model_type,
                dense_tokens_per_frame=int(num_image_tokens),
            )
        )
        history_mask_augmentation_config = (
            HistoryMaskAugmentationConfig.from_value(history_mask_augmentation)
        )

        self.tokenizer = tokenizer
        self.transform = transform
        self.num_history_frames = int(num_history_frames)
        self.history_frame_interval = int(history_frame_interval)
        self.n_future = int(n_future)
        self.trajectory_dim = int(trajectory_dim)
        self.prompt = prompt.strip()
        self.num_image_tokens = int(num_image_tokens)
        self.max_length = int(max_length)
        self.model_type = model_type
        self.processor = processor
        self.image_token = image_token
        self.qwen_min_pixels = int(qwen_min_pixels)
        self.qwen_max_pixels = int(qwen_max_pixels)
        self.index_cache_dir = index_cache_dir
        self.max_open_archives = int(max_open_archives)
        self.base_root_override = (
            None
            if base_root_override is None
            else Path(base_root_override).expanduser().resolve()
        )
        if self.base_root_override is not None and not (
            self.base_root_override / "frames"
        ).is_dir():
            raise ValueError(
                "base_root_override must name a TrackVLA subset containing frames/, got "
                f"{self.base_root_override}"
            )
        self.history_compressor_config = history_compressor_config
        self.load_history_masks = (
            history_compressor_config.requires_history_masks or history_compressor_config.requires_history_boxes
        )
        self.auxiliary_tasks_config = AuxiliaryTasksConfig.from_value(
            auxiliary_tasks_config
        )
        self.load_masks = bool(load_masks or self.auxiliary_tasks_config.enabled or self.history_compressor_config.current_frame_qformer_use_gt_mask)
        self.mask_input_size = int(mask_input_size) if mask_input_size is not None else None
        if (self.load_masks or self.load_history_masks) and (
            self.mask_input_size is None or self.mask_input_size <= 0
        ):
            raise ValueError(
                "mask_input_size must be positive when any masks are requested"
            )
        self.history_mask_augmentation_config = history_mask_augmentation_config
        self.history_mask_augmentation = bool(
            self.history_compressor_config.requires_history_masks and history_mask_augmentation_config.enabled
        )
        self.history_mask_augmenter = HistoryMaskAugmenter(
            history_mask_augmentation_config,
            seed=int(augmentation_seed),
        )
        self._archives: OrderedDict[str, zipfile.ZipFile] = OrderedDict()
        # Missing/corrupt masks are valid for real-data clips where the
        # person is absent.  They are replaced by background masks instead of
        # aborting a worker; keep warnings bounded so logs remain usable.
        self._mask_warning_count = 0
        self._mask_warning_limit = 20

        self.manifests = _resolve_manifest_files(source_paths)
        self.refs = self._load_or_build_refs(index_cache_dir)
        if not self.refs:
            raise RuntimeError(f"No examples indexed from TrackVLA sources: {source_paths}")
        print(
            f"[TrackVLADataset] Indexed {len(self.refs)} examples from "
            f"{len(self.manifests)} manifest file(s)."
        )

    def _load_or_build_refs(self, index_cache_dir: Optional[str]) -> List[_ExampleRef]:
        cache_path: Optional[Path] = None
        if index_cache_dir:
            # The cache contains only manifest byte ranges, never decoded
            # images, so it makes repeated startup fast without copying data.
            cache_dir = Path(index_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / (
                "trackvla_index_"
                f"{_index_cache_key(self.manifests, base_root_override=self.base_root_override)}.pkl"
            )
            if cache_path.is_file():
                try:
                    with cache_path.open("rb") as f:
                        refs = pickle.load(f)
                    if isinstance(refs, list) and all(isinstance(ref, _ExampleRef) for ref in refs):
                        print(f"[TrackVLADataset] Loading index cache from {cache_path}")
                        return refs
                except Exception as exc:
                    print(f"[TrackVLADataset] Ignoring invalid index cache {cache_path}: {exc}")

        refs: List[_ExampleRef] = []
        for manifest in self.manifests:
            base_root = self.base_root_override or _find_base_root(manifest)
            if manifest.suffix.lower() == ".jsonl":
                refs.extend(_index_jsonl(manifest, base_root))
            else:
                refs.extend(_index_json_array(manifest, base_root))

        if cache_path is not None:
            # Publish atomically so concurrent launchers never observe a
            # partially written pickle.
            _write_pickle_atomically(cache_path, refs)
        return refs

    def __getstate__(self):
        state = self.__dict__.copy()
        # ZipFile objects cannot be pickled for spawn-based DataLoader workers.
        state["_archives"] = OrderedDict()
        return state

    def __del__(self):
        try:
            for archive in self._archives.values():
                archive.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.refs)

    def _read_example(self, idx: int) -> Tuple[Dict, Path]:
        ref = self.refs[idx]
        # Read just one indexed object instead of materializing large
        # dataset.json manifests in every training process.
        with open(ref.manifest_path, "rb") as f:
            f.seek(ref.offset)
            payload = f.read(ref.length)
        example = json.loads(payload)
        if not isinstance(example, dict):
            raise ValueError(f"TrackVLA example {idx} in {ref.manifest_path} is not an object")
        return example, Path(ref.base_root)

    def episode_id_for_index(self, idx: int) -> str:
        example, base_root = self._read_example(idx)
        frames_file = example.get("frames_file")
        if not isinstance(frames_file, str) or not frames_file:
            raise KeyError(f"TrackVLA example {idx} is missing frames_file")
        return str((base_root / Path(frames_file)).resolve())

    def load_or_build_episode_indices(self) -> Dict[str, List[int]]:
        """Return sample indices grouped by episode, reusing a persistent cache.

        The regular index cache only stores JSON byte ranges.  Reconstructing
        episode groups from it used to reopen and decode a manifest once per
        sample during every launch.  This cache stores the completed grouping;
        on a cache hit, setup needs no manifest reads or JSON decoding.
        """

        cache_path: Optional[Path] = None
        if self.index_cache_dir:
            cache_dir = Path(self.index_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / (
                "trackvla_episode_groups_v1_"
                f"{_index_cache_key(self.manifests, base_root_override=self.base_root_override)}.pkl"
            )
            if cache_path.is_file():
                try:
                    with cache_path.open("rb") as f:
                        episode_indices = pickle.load(f)
                    if _valid_episode_indices(episode_indices, len(self.refs)):
                        print(
                            "[TrackVLADataset] Loading episode grouping cache "
                            f"from {cache_path}"
                        )
                        return episode_indices
                except Exception as exc:
                    print(
                        "[TrackVLADataset] Ignoring invalid episode grouping "
                        f"cache {cache_path}: {exc}"
                    )

        episode_indices: Dict[str, List[int]] = defaultdict(list)
        current_manifest_path: Optional[str] = None
        manifest_file = None
        episode_id_cache: Dict[Tuple[str, str], str] = {}
        try:
            for idx, ref in enumerate(self.refs):
                # Refs are constructed manifest-by-manifest, so this opens each
                # manifest once while building a cache instead of once per
                # sample.  It also keeps reads in manifest order.
                if ref.manifest_path != current_manifest_path:
                    if manifest_file is not None:
                        manifest_file.close()
                    manifest_file = open(ref.manifest_path, "rb")
                    current_manifest_path = ref.manifest_path

                manifest_file.seek(ref.offset)
                payload = manifest_file.read(ref.length)
                example = json.loads(payload)
                if not isinstance(example, dict):
                    raise ValueError(
                        f"TrackVLA example {idx} in {ref.manifest_path} is not an object"
                    )

                frames_file = example.get("frames_file")
                if not isinstance(frames_file, str) or not frames_file:
                    raise KeyError(f"TrackVLA example {idx} is missing frames_file")
                cache_key = (ref.base_root, frames_file)
                episode_id = episode_id_cache.get(cache_key)
                if episode_id is None:
                    episode_id = str(
                        (Path(ref.base_root) / Path(frames_file)).resolve()
                    )
                    episode_id_cache[cache_key] = episode_id
                episode_indices[episode_id].append(idx)
        finally:
            if manifest_file is not None:
                manifest_file.close()

        result = dict(episode_indices)
        if cache_path is not None:
            _write_pickle_atomically(cache_path, result)
        return result

    def _get_archive(self, archive_path: Path) -> zipfile.ZipFile:
        key = str(archive_path)
        # Keep a per-worker LRU of episode ZIPs. Reopening a ZIP for every
        # frame is expensive, while max_open_archives bounds file descriptors.
        archive = self._archives.pop(key, None)
        if archive is None:
            archive = zipfile.ZipFile(archive_path, "r")
        self._archives[key] = archive

        while len(self._archives) > self.max_open_archives:
            _, stale_archive = self._archives.popitem(last=False)
            stale_archive.close()
        return archive

    @staticmethod
    def _frame_names(
        frame_name: str,
        num_history_frames: int,
        history_frame_interval: int,
    ) -> List[str]:
        """Return an oldest-to-newest sequence ending with ``frame_name``."""

        # ZIP members always use POSIX separators, including on Windows.
        path = PurePosixPath(frame_name.replace("\\", "/"))
        match = re.search(r"(\d+)(?!.*\d)", path.stem)
        if match is None:
            return [path.as_posix()]

        current_index = int(match.group(1))
        prefix = path.stem[: match.start(1)]
        suffix = path.stem[match.end(1) :]
        width = len(match.group(1))
        stride = history_frame_interval + 1
        # Count backward from the current frame rather than creating an
        # absolute range. The latter can miss ``current_index`` whenever the
        # earliest valid index is not aligned with ``stride`` (for example
        # frame 1 with interval 1 would incorrectly yield only frame 0).
        indices = [
            current_index - offset * stride
            for offset in range(num_history_frames - 1, -1, -1)
            if current_index - offset * stride >= 0
        ]
        return [
            path.with_name(
                f"{prefix}{frame_idx:0{width}d}{suffix}{path.suffix}"
            ).as_posix()
            for frame_idx in indices
        ]

    @staticmethod
    def _decode_rgb(image_bytes: bytes) -> Image.Image:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB").copy()

    @staticmethod
    def _decode_mask(mask_bytes: bytes) -> Image.Image:
        with Image.open(io.BytesIO(mask_bytes)) as mask:
            return mask.convert("L").copy()

    @staticmethod
    def _mask_name_for_frame(frame_name: str, mask_file: str) -> str:
        # TrackVLA's SAV mask ZIPs store PNG files while their frame manifests
        # retain the RGB filename. Other subsets use the original frame name.
        path = PurePosixPath(frame_name.replace("\\", "/"))
        if "sav_" in mask_file.lower():
            return path.with_suffix(".png").as_posix()
        return path.as_posix()

    def _decode_and_transform_mask(self, mask_bytes: bytes) -> torch.Tensor:
        if self.mask_input_size is None:
            raise RuntimeError("mask_input_size is unavailable while loading a mask")
        mask = self._decode_mask(mask_bytes)
        # Keep mask geometry aligned with the RGB longest-side resize + pad.
        # Nearest-neighbor interpolation preserves binary labels.
        mask = ResizeLongestSideAndPad(
            self.mask_input_size,
            interpolation=InterpolationMode.NEAREST,
        )(mask)
        mask_array = np.asarray(mask, dtype=np.uint8)
        return torch.from_numpy(mask_array.copy()).float().div_(255.0).gt_(0.5).float()

    def _background_mask(self) -> torch.Tensor:
        if self.mask_input_size is None:
            raise RuntimeError("mask_input_size is unavailable while loading a mask")
        return torch.zeros(
            (self.mask_input_size, self.mask_input_size), dtype=torch.float32
        )

    def _warn_and_use_background(self, message: str) -> torch.Tensor:
        self._mask_warning_count += 1
        if self._mask_warning_count <= self._mask_warning_limit:
            warnings.warn(
                f"{message}; using an all-background mask "
                f"(warning {self._mask_warning_count}/{self._mask_warning_limit})",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self._mask_warning_count == self._mask_warning_limit + 1:
            warnings.warn(
                "Further missing/corrupt mask warnings are suppressed",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._background_mask()

    def _load_current_mask(self, mask_archive_path: Path, current_frame: str) -> torch.Tensor:
        try:
            archive = self._get_archive(mask_archive_path)
            mask_name = self._mask_name_for_frame(current_frame, str(mask_archive_path))
            if mask_name not in archive.NameToInfo:
                return self._warn_and_use_background(
                    f"Current mask {mask_name} is absent from {mask_archive_path}"
                )
            return self._decode_and_transform_mask(archive.read(mask_name))
        except (
            FileNotFoundError,
            OSError,
            KeyError,
            RuntimeError,
            ValueError,
            zipfile.BadZipFile,
            EOFError,
        ) as exc:
            return self._warn_and_use_background(
                f"Could not load current mask {mask_archive_path}: {exc}"
            )

    def _load_history_masks(
        self,
        mask_archive_path: Path,
        frame_names: Sequence[str],
        frame_valid: Sequence[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load past masks aligned exactly with the padded RGB history slots."""

        if len(frame_names) != self.num_history_frames:
            raise ValueError(
                "frame_names must contain exactly num_history_frames entries, "
                f"got {len(frame_names)}"
            )
        if len(frame_valid) != self.num_history_frames:
            raise ValueError(
                "frame_valid must contain exactly num_history_frames entries, "
                f"got {len(frame_valid)}"
            )
        if self.mask_input_size is None:
            raise RuntimeError("mask_input_size is unavailable while loading masks")
        num_past = self.num_history_frames - 1
        if num_past == 0:
            return (
                torch.zeros((0, self.mask_input_size, self.mask_input_size)),
                torch.zeros((0,), dtype=torch.bool),
            )

        archive = self._get_archive(mask_archive_path)
        history_masks: List[torch.Tensor] = []
        history_valid: List[bool] = []
        for frame_name, is_real_frame in zip(
            frame_names[:num_past],
            frame_valid[:num_past],
        ):
            mask_name = self._mask_name_for_frame(
                frame_name,
                str(mask_archive_path),
            )
            is_valid = bool(is_real_frame and mask_name in archive.NameToInfo)
            if is_valid:
                mask = self._decode_and_transform_mask(archive.read(mask_name))
            else:
                mask = torch.zeros(
                    (self.mask_input_size, self.mask_input_size),
                    dtype=torch.float32,
                )
            history_masks.append(mask)
            history_valid.append(is_valid)
        return torch.stack(history_masks), torch.tensor(history_valid, dtype=torch.bool)

    def _load_history_images(
        self,
        archive_path: Path,
        current_frame: str,
        explicit_frame_names: Optional[List[str]] = None,
        explicit_frame_valid: Optional[List[bool]] = None,
    ) -> Tuple[List[str], List[Image.Image], List[bool]]:
        """Return aligned ``(names, images, real_valid)`` oldest to newest.

        Every requested stride slot is preserved. A missing slot copies the
        latest available causal frame (or the earliest later frame during
        episode-start warmup) and is marked invalid, so a middle gap never
        shifts newer RGB/mask pairs into the wrong temporal offset.
        """

        archive = self._get_archive(archive_path)
        current_frame = current_frame.replace("\\", "/")
        if current_frame not in archive.NameToInfo:
            raise FileNotFoundError(f"Current frame {current_frame} is absent from {archive_path}")

        if explicit_frame_names is None:
            requested_names = self._frame_names(
                current_frame,
                self.num_history_frames,
                self.history_frame_interval,
            )
            requested_valid = [True] * len(requested_names)
        else:
            if not explicit_frame_names:
                raise ValueError("history_frames must not be empty")
            if len(explicit_frame_names) != self.num_history_frames:
                raise ValueError(
                    "history_frames must contain exactly "
                    f"{self.num_history_frames} entries, got "
                    f"{len(explicit_frame_names)}"
                )
            requested_names = [
                str(name).replace("\\", "/")
                for name in explicit_frame_names
            ]
            if requested_names[-1] != current_frame:
                raise ValueError(
                    "The last history_frames entry must equal current: "
                    f"{requested_names[-1]!r} != {current_frame!r}"
                )
            if explicit_frame_valid is None:
                requested_valid = [True] * len(requested_names)
            else:
                if len(explicit_frame_valid) != len(requested_names):
                    raise ValueError(
                        "history_frame_valid and history_frames must have "
                        "the same length"
                    )
                requested_valid = [bool(value) for value in explicit_frame_valid]
        decoded_images: Dict[str, Image.Image] = {}
        for frame_name in requested_names:
            if frame_name not in archive.NameToInfo:
                continue
            decoded_images[frame_name] = self._decode_rgb(
                archive.read(frame_name)
            )

        if not decoded_images:
            raise RuntimeError(f"No usable RGB frames for {current_frame} in {archive_path}")

        available_indices = [
            index
            for index, name in enumerate(requested_names)
            if name in decoded_images
        ]
        frame_names: List[str] = []
        images: List[Image.Image] = []
        frame_valid: List[bool] = []
        for index, requested_name in enumerate(requested_names):
            is_valid = (
                requested_name in decoded_images
                and requested_valid[index]
            )
            if is_valid:
                adopted_index = index
            else:
                causal_indices = [
                    candidate
                    for candidate in available_indices
                    if candidate < index
                ]
                adopted_index = (
                    causal_indices[-1]
                    if causal_indices
                    else available_indices[0]
                )
            adopted_name = requested_names[adopted_index]
            adopted_image = decoded_images[adopted_name]
            frame_names.append(adopted_name)
            images.append(
                adopted_image if is_valid else adopted_image.copy()
            )
            frame_valid.append(is_valid)
        while len(frame_names) < self.num_history_frames:
            frame_names.insert(0, frame_names[0])
            images.insert(0, images[0].copy())
            frame_valid.insert(0, False)
        return frame_names, images, frame_valid

    def _trajectory_from_example(self, example: Dict, idx: int) -> torch.Tensor:
        raw_trajectory = example.get("trajectory")
        if raw_trajectory is None:
            raw_trajectory = example.get("waypoints")
        if raw_trajectory is None:
            raise KeyError(
                f"TrackVLA example {idx} has neither trajectory nor waypoints; "
                "actions are not converted because their convention is dataset-specific."
            )

        trajectory = torch.as_tensor(raw_trajectory, dtype=torch.float32)
        if trajectory.ndim != 2:
            raise ValueError(
                f"TrackVLA trajectory for example {idx} must be rank 2, got {tuple(trajectory.shape)}"
            )
        if trajectory.shape[1] != self.trajectory_dim:
            raise ValueError(
                f"TrackVLA trajectory for example {idx} has dimension {trajectory.shape[1]}, "
                f"expected {self.trajectory_dim}"
            )
        if trajectory.shape[0] == 0:
            raise ValueError(f"TrackVLA trajectory for example {idx} is empty")
        if trajectory.shape[0] < self.n_future:
            # Keep supervision shape fixed by repeating the last known target;
            # do not invent motion through interpolation or action integration.
            padding = trajectory[-1:].expand(self.n_future - trajectory.shape[0], -1)
            trajectory = torch.cat([trajectory, padding], dim=0)
        return trajectory[: self.n_future]

    def _effective_prompt(self, example: Dict) -> str:
        instruction = str(example.get("instruction", "")).strip()
        parts = [part for part in (instruction, self.prompt) if part]
        return "\n".join(parts)

    def get_item(
        self,
        idx: int,
        *,
        augment_history_masks: bool,
        sample_current_frame: bool = False,
    ) -> Dict[str, Union[torch.Tensor, str]]:
        example, base_root = self._read_example(idx)
        frames_file = example.get("frames_file")
        current_frame = example.get("current")
        if not isinstance(frames_file, str) or not isinstance(current_frame, str):
            raise KeyError(f"TrackVLA example {idx} requires string frames_file and current fields")

        archive_path = base_root / Path(frames_file)
        explicit_history = example.get("history_frames")
        if explicit_history is not None and not isinstance(explicit_history, list):
            raise TypeError(
                f"TrackVLA example {idx} history_frames must be a list"
            )
        explicit_history_valid = example.get("history_frame_valid")
        if (
            explicit_history_valid is not None
            and not isinstance(explicit_history_valid, list)
        ):
            raise TypeError(
                f"TrackVLA example {idx} history_frame_valid must be a list"
            )
        frame_names, images, frame_valid = self._load_history_images(
            archive_path,
            current_frame,
            explicit_history,
            explicit_history_valid,
        )
        trajectory = self._trajectory_from_example(example, idx)
        prompt = self._effective_prompt(example)
        token = f"{archive_path}:{current_frame}"
        probability = self.history_compressor_config.current_frame_qformer_probability
        current_frame_qformer = sample_current_frame and (probability == 1.0 or (probability > 0 and bool(torch.rand(()).item() < probability)))
        mask: Optional[torch.Tensor] = None
        history_masks: Optional[torch.Tensor] = None
        history_mask_valid: Optional[torch.Tensor] = None
        if self.load_masks or self.load_history_masks:
            mask_file = example.get("mask_file")
            has_mask_file = isinstance(mask_file, str) and bool(
                mask_file.strip()
            )
            if self.load_masks and not has_mask_file:
                mask = self._warn_and_use_background(
                    f"TrackVLA example {idx} has no mask_file"
                )
            mask_archive_path = (
                base_root / Path(mask_file.strip())
                if has_mask_file
                else None
            )
            if self.load_masks and mask is None:
                assert mask_archive_path is not None
                mask = self._load_current_mask(
                    mask_archive_path,
                    current_frame,
                )
            if self.load_history_masks:
                if (
                    mask_archive_path is None
                    or not mask_archive_path.is_file()
                ):
                    history_masks = torch.zeros(
                        (
                            len(frame_names) - 1,
                            self.mask_input_size,
                            self.mask_input_size,
                        ),
                        dtype=torch.float32,
                    )
                    history_mask_valid = torch.zeros(
                        len(frame_names) - 1,
                        dtype=torch.bool,
                    )
                else:
                    try:
                        history_masks, history_mask_valid = (
                            self._load_history_masks(
                                mask_archive_path,
                                frame_names,
                                frame_valid,
                            )
                        )
                    except (
                        FileNotFoundError,
                        OSError,
                        KeyError,
                        RuntimeError,
                        ValueError,
                        zipfile.BadZipFile,
                        EOFError,
                    ) as exc:
                        warnings.warn(
                            f"Could not load history masks for {mask_archive_path}: {exc}; "
                            "using all-background history masks",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        history_masks = torch.zeros(
                            (len(frame_names) - 1, self.mask_input_size, self.mask_input_size),
                            dtype=torch.float32,
                        )
                        history_mask_valid = torch.zeros(
                            len(frame_names) - 1, dtype=torch.bool
                        )
                if augment_history_masks and self.history_mask_augmentation:
                    history_masks = self.history_mask_augmenter(
                        history_masks,
                        history_mask_valid,
                    )

        box_inputs = {}
        if self.history_compressor_config.requires_history_boxes:
            boxes, valid = masks_to_boxes(history_masks, history_mask_valid)
            box_inputs = {"history_boxes": boxes, "history_box_valid": valid}

        auxiliary_targets: Dict[str, torch.Tensor] = {}
        if self.auxiliary_tasks_config.enabled:
            if mask is None:
                raise ValueError("auxiliary supervision requires the current mask")
            target_example = example
            if self.history_compressor_config.requires_history_boxes:
                # Manifest boxes are normalized to the original RGB image;
                # bbox selection and the online cache use the padded model image.
                manifest_box = example.get("bbox_cxcywh_norm")
                if isinstance(manifest_box, (list, tuple)) and len(manifest_box) == 4:
                    try:
                        box = torch.tensor(manifest_box, dtype=torch.float32)
                        if torch.isfinite(box).all() and (box >= 0).all():
                            width, height = images[-1].size
                            geometry = resize_longest_side_and_pad_geometry(height, width, self.mask_input_size)
                            target_example = dict(example)
                            target_example["bbox_cxcywh_norm"] = resize_pad_boxes(box, geometry, self.mask_input_size).tolist()
                    except (TypeError, ValueError):
                        pass  # Existing target builder falls back to the clean mask.
            auxiliary_targets = build_auxiliary_targets(
                mask, target_example, self.auxiliary_tasks_config
            )

        if self.model_type == "custom":
            pixel_values = preprocess_custom_images(
                self.processor,
                images,
                self.mask_input_size,
            )
            image_token_counts = self.history_compressor_config.frame_token_counts(
                len(images),
                self.num_image_tokens,
                current_frame_qformer=current_frame_qformer,
            )
            text = build_custom_multimodal_prompt(
                self.tokenizer,
                self.image_token,
                image_token_counts,
                prompt,
            )
            tokenizer_out = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max(self.max_length, sum(image_token_counts) + 256),
            )
            result = {
                "pixel_values": pixel_values,
                "input_ids": tokenizer_out["input_ids"].squeeze(0),
                "attention_mask": tokenizer_out["attention_mask"].squeeze(0),
                "trajectory": trajectory,
                "token": token,
            }
            if mask is not None:
                result["mask"] = mask
            if self.history_compressor_config.requires_history_masks and history_masks is not None and history_mask_valid is not None:
                result["history_masks"] = history_masks
                result["history_mask_valid"] = history_mask_valid
            if probability > 0:
                result["current_frame_qformer"] = current_frame_qformer
            result.update(box_inputs)
            result.update(auxiliary_targets)
            return result

        if self.model_type in {"qwen2vl", "qwen3vl"}:
            # Qwen2-VL retains its variable-resolution behavior. Qwen3-VL is
            # padded to the same square geometry as masks so every frame has a
            # stable dense token grid for historical compression.
            processor_images = images
            processor_size = {
                "shortest_edge": self.qwen_min_pixels,
                "longest_edge": self.qwen_max_pixels,
            }
            if self.model_type == "qwen3vl":
                assert self.mask_input_size is not None
                square_transform = ResizeLongestSideAndPad(
                    self.mask_input_size,
                    interpolation=InterpolationMode.BICUBIC,
                )
                processor_images = [square_transform(image) for image in images]
                fixed_pixels = self.mask_input_size * self.mask_input_size
                processor_size = {
                    "shortest_edge": fixed_pixels,
                    "longest_edge": fixed_pixels,
                }
            content = [
                {"type": "image", "image": image}
                for image in processor_images
            ]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = apply_qwen_vl_chat_template(
                self.processor,
                messages,
                model_type=self.model_type,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                images=processor_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(self.max_length, 4096),
                size=processor_size,
            )
            result = {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "image_grid_thw": inputs["image_grid_thw"],
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "mm_token_type_ids": inputs.get(
                    "mm_token_type_ids", torch.zeros_like(inputs["input_ids"])
                ).squeeze(0),
                "trajectory": trajectory,
                "token": token,
            }
            if mask is not None:
                result["mask"] = mask
            if self.history_compressor_config.requires_history_masks and history_masks is not None and history_mask_valid is not None:
                result["history_masks"] = history_masks
                result["history_mask_valid"] = history_mask_valid
            if probability > 0:
                result["current_frame_qformer"] = current_frame_qformer
            result.update(box_inputs)
            result.update(auxiliary_targets)
            return result

        # InternVL2.5 and InternVL3 use a fixed visual-token count per image.
        # The H image blocks below must match the flattened visual batch order.
        # Keep num_image_tokens equal to the checkpoint's per-image token count
        # (for InternVL3, config.image_seq_length) to preserve this alignment.
        pixel_values = torch.stack([self.transform(image) for image in images], dim=0)
        image_token_counts = self.history_compressor_config.frame_token_counts(
            len(images),
            self.num_image_tokens,
            current_frame_qformer=current_frame_qformer,
        )
        image_tokens = "\n".join(
            "<img>" + "<IMG_CONTEXT>" * token_count + "</img>"
            for token_count in image_token_counts
        )
        text = f"{image_tokens}\nUser: {prompt}\nAssistant:"
        tokenizer_out = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        result = {
            "pixel_values": pixel_values,
            "input_ids": tokenizer_out["input_ids"].squeeze(0),
            "attention_mask": tokenizer_out["attention_mask"].squeeze(0),
            "trajectory": trajectory,
            "token": token,
        }
        if mask is not None:
            result["mask"] = mask
        if self.history_compressor_config.requires_history_masks and history_masks is not None and history_mask_valid is not None:
            result["history_masks"] = history_masks
            result["history_mask_valid"] = history_mask_valid
        if probability > 0:
            result["current_frame_qformer"] = current_frame_qformer
        result.update(box_inputs)
        result.update(auxiliary_targets)
        return result

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        return self.get_item(
            idx,
            augment_history_masks=self.history_mask_augmentation,
            sample_current_frame=True,
        )


class _TrackVLASplitDataset(Dataset):
    """Subset-like view that fixes train/validation augmentation per split."""

    def __init__(
        self,
        dataset: TrackVLADataset,
        indices: Sequence[int],
        *,
        augment_history_masks: bool,
        sample_current_frame: bool = False,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.augment_history_masks = bool(augment_history_masks)
        self.sample_current_frame = bool(sample_current_frame)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        return self.dataset.get_item(
            self.indices[index],
            augment_history_masks=self.augment_history_masks,
            sample_current_frame=self.sample_current_frame,
        )


class TrackVLADataModule(pl.LightningDataModule):
    """Lightning DataModule for TrackVLA trajectory and optional mask inputs.

    Metadata is indexed lazily, RGB ZIPs are decoded only on demand, and the
    train/validation split is made at episode granularity rather than per frame.
    """

    def __init__(
        self,
        source_paths: Union[str, Sequence[str]],
        tokenizer,
        input_size: int = 448,
        num_history_frames: int = 4,
        history_frame_interval: int = 0,
        n_future: int = 9,
        trajectory_dim: int = 3,
        batch_size: int = 1,
        num_workers: int = 4,
        val_split: float = 0.1,
        prompt: str = "Predict the follower's future trajectory in the ego coordinate frame.",
        num_image_tokens: int = 256,
        max_length: int = 1200,
        seed: int = 42,
        model_type: str = "internvl",
        processor=None,
        image_token: Optional[str] = None,
        qwen_min_pixels: int = 3136,
        qwen_max_pixels: int = 100_000,
        index_cache_dir: Optional[str] = None,
        max_open_archives: int = 32,
        load_masks: bool = False,
        auxiliary_tasks_config: Optional[
            Union[AuxiliaryTasksConfig, Mapping[str, Any]]
        ] = None,
        history_compressor_config: Optional[
            Union[HistoryCompressorConfig, Mapping[str, Any]]
        ] = None,
        history_mask_augmentation: Union[
            bool,
            HistoryMaskAugmentationConfig,
            Mapping[str, object],
            None,
        ] = True,
    ):
        super().__init__()
        if not 0.0 < val_split < 1.0:
            raise ValueError("val_split must be between 0 and 1")

        self.source_paths = source_paths
        self.tokenizer = tokenizer
        self.input_size = int(input_size)
        self.num_history_frames = int(num_history_frames)
        self.history_frame_interval = int(history_frame_interval)
        self.n_future = int(n_future)
        self.trajectory_dim = int(trajectory_dim)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_split = float(val_split)
        self.prompt = prompt
        self.num_image_tokens = int(num_image_tokens)
        self.max_length = int(max_length)
        self.seed = int(seed)
        self.model_type = model_type
        self.processor = processor
        self.image_token = image_token
        self.qwen_min_pixels = int(qwen_min_pixels)
        self.qwen_max_pixels = int(qwen_max_pixels)
        self.index_cache_dir = index_cache_dir
        self.max_open_archives = int(max_open_archives)
        self.history_compressor_config = HistoryCompressorConfig.from_value(
            history_compressor_config
        )
        ModelExtensionsConfig(
            history_compressor=self.history_compressor_config
        ).validate_for(
            ModelExtensionContext(
                model_type=self.model_type,
                dense_tokens_per_frame=self.num_image_tokens,
            )
        )
        self.load_history_masks = (
            self.history_compressor_config.requires_history_masks or self.history_compressor_config.requires_history_boxes
        )
        self.auxiliary_tasks_config = AuxiliaryTasksConfig.from_value(
            auxiliary_tasks_config
        )
        self.load_masks = bool(load_masks or self.auxiliary_tasks_config.enabled or self.history_compressor_config.current_frame_qformer_use_gt_mask)
        self.history_mask_augmentation_config = (
            HistoryMaskAugmentationConfig.from_value(history_mask_augmentation)
        )

        self.transform = build_transform(self.input_size)
        self.dataset: Optional[TrackVLADataset] = None
        self.train_dataset: Optional[_TrackVLASplitDataset] = None
        self.val_dataset: Optional[_TrackVLASplitDataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.dataset is not None:
            return

        self.dataset = TrackVLADataset(
            source_paths=self.source_paths,
            tokenizer=self.tokenizer,
            transform=self.transform,
            num_history_frames=self.num_history_frames,
            history_frame_interval=self.history_frame_interval,
            n_future=self.n_future,
            trajectory_dim=self.trajectory_dim,
            prompt=self.prompt,
            num_image_tokens=self.num_image_tokens,
            max_length=self.max_length,
            model_type=self.model_type,
            processor=self.processor,
            image_token=self.image_token,
            qwen_min_pixels=self.qwen_min_pixels,
            qwen_max_pixels=self.qwen_max_pixels,
            index_cache_dir=self.index_cache_dir,
            max_open_archives=self.max_open_archives,
            load_masks=self.load_masks,
            auxiliary_tasks_config=self.auxiliary_tasks_config,
            mask_input_size=self.input_size,
            history_compressor_config=self.history_compressor_config,
            history_mask_augmentation=self.history_mask_augmentation_config,
            augmentation_seed=self.seed,
        )

        # Do not split individual frames at random: adjacent frames from one
        # trajectory would otherwise leak almost identical RGB observations
        # across train and validation.
        print(f"[TrackVLADataModule] Grouping {len(self.dataset)} samples by episode...")
        episode_indices = self.dataset.load_or_build_episode_indices()

        episodes = sorted(episode_indices)
        if len(episodes) < 2:
            raise RuntimeError("TrackVLA needs at least two episodes for a train/validation split")
        rng = np.random.default_rng(self.seed)
        rng.shuffle(episodes)
        n_val = min(len(episodes) - 1, max(1, int(len(episodes) * self.val_split)))
        val_episodes = set(episodes[:n_val])
        train_indices = [
            idx for episode, indices in episode_indices.items() if episode not in val_episodes for idx in indices
        ]
        val_indices = [
            idx for episode, indices in episode_indices.items() if episode in val_episodes for idx in indices
        ]
        if not train_indices or not val_indices:
            raise RuntimeError("TrackVLA episode split produced an empty train or validation set")

        self.train_dataset = _TrackVLASplitDataset(
            self.dataset,
            train_indices,
            augment_history_masks=self.history_mask_augmentation_config.enabled,
            sample_current_frame=True,
        )
        self.val_dataset = _TrackVLASplitDataset(
            self.dataset,
            val_indices,
            augment_history_masks=False,
        )
        print(
            f"[TrackVLADataModule] Episodes: train={len(episodes) - n_val}, val={n_val}; "
            f"samples: train={len(train_indices)}, val={len(val_indices)}"
        )

    @staticmethod
    def _pad_token_id(tokenizer) -> int:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None and hasattr(tokenizer, "tokenizer"):
            pad_token_id = getattr(tokenizer.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None and hasattr(tokenizer, "tokenizer"):
            pad_token_id = getattr(tokenizer.tokenizer, "eos_token_id", None)
        return 0 if pad_token_id is None else int(pad_token_id)

    @staticmethod
    def _right_pad_text(
        batch: List[Dict[str, Union[torch.Tensor, str]]],
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        max_len = max(ids.shape[0] for ids in input_ids_list)
        input_ids = torch.full(
            (len(batch), max_len),
            int(pad_token_id),
            dtype=input_ids_list[0].dtype,
        )
        attention_mask = torch.zeros(
            (len(batch), max_len),
            dtype=attention_mask_list[0].dtype,
        )
        for index, (ids, mask) in enumerate(
            zip(input_ids_list, attention_mask_list)
        ):
            length = ids.shape[0]
            input_ids[index, :length] = ids
            attention_mask[index, :length] = mask
        return input_ids, attention_mask

    def collate_fn(self, batch: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict:
        if self.model_type in {"qwen2vl", "qwen3vl"}:
            pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
            image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

            mm_token_type_ids_list = [item["mm_token_type_ids"] for item in batch]
            input_ids, attention_mask = self._right_pad_text(
                batch,
                self._pad_token_id(self.tokenizer),
            )
            max_len = input_ids.shape[1]
            mm_token_type_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
            for idx, token_types in enumerate(mm_token_type_ids_list):
                length = token_types.shape[0]
                mm_token_type_ids[idx, :length] = token_types

            result = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mm_token_type_ids": mm_token_type_ids,
                "trajectories": torch.stack([item["trajectory"] for item in batch]),
                "tokens": [item["token"] for item in batch],
            }
            if self.load_masks:
                result["masks"] = torch.stack([item["mask"] for item in batch])
            if self.history_compressor_config.requires_history_masks:
                result["history_masks"] = torch.stack(
                    [item["history_masks"] for item in batch]
                )
                result["history_mask_valid"] = torch.stack(
                    [item["history_mask_valid"] for item in batch]
                )
            if "current_frame_qformer" in batch[0]:
                result["current_frame_qformer"] = torch.tensor([item["current_frame_qformer"] for item in batch], dtype=torch.bool)
            if self.history_compressor_config.requires_history_boxes:
                result["history_boxes"] = torch.stack([item["history_boxes"] for item in batch])
                result["history_box_valid"] = torch.stack([item["history_box_valid"] for item in batch])
            if self.auxiliary_tasks_config.enabled:
                result["grounding_bbox_target"] = torch.stack(
                    [item["grounding_bbox_target"] for item in batch]
                )
                result["grounding_visibility_target"] = torch.stack(
                    [item["grounding_visibility_target"] for item in batch]
                )
                result["vqa_labels"] = torch.stack(
                    [item["vqa_labels"] for item in batch]
                )
            return result

        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        batch_size, history = pixel_values.shape[:2]
        input_ids, attention_mask = self._right_pad_text(
            batch,
            self._pad_token_id(self.tokenizer),
        )
        # InternVL encodes images as a flat batch while each text prompt remains
        # a B-row sequence. The reshape preserves B,H ordering for placeholder
        # injection: prompt b receives image features b*H through (b+1)*H.
        result = {
            "pixel_values": pixel_values.flatten(0, 1),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "trajectories": torch.stack([item["trajectory"] for item in batch]),
            "tokens": [item["token"] for item in batch],
        }
        if self.load_masks:
            result["masks"] = torch.stack([item["mask"] for item in batch])
        if self.history_compressor_config.requires_history_masks:
            result["history_masks"] = torch.stack(
                [item["history_masks"] for item in batch]
            )
            result["history_mask_valid"] = torch.stack(
                [item["history_mask_valid"] for item in batch]
            )
        if "current_frame_qformer" in batch[0]:
            result["current_frame_qformer"] = torch.tensor([item["current_frame_qformer"] for item in batch], dtype=torch.bool)
        if self.history_compressor_config.requires_history_boxes:
            result["history_boxes"] = torch.stack([item["history_boxes"] for item in batch])
            result["history_box_valid"] = torch.stack([item["history_box_valid"] for item in batch])
        if self.auxiliary_tasks_config.enabled:
            result["grounding_bbox_target"] = torch.stack(
                [item["grounding_bbox_target"] for item in batch]
            )
            result["grounding_visibility_target"] = torch.stack(
                [item["grounding_visibility_target"] for item in batch]
            )
            result["vqa_labels"] = torch.stack(
                [item["vqa_labels"] for item in batch]
            )
        return result

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the training dataloader")
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting the validation dataloader")
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            generator=generator,
        )
