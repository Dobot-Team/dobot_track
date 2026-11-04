#!/usr/bin/env python3
"""Convert Dobot ``training.json`` files to RGB-only TrackVLA format.

The output is a TrackVLA subset that can be consumed by
``data.dataset_type: trackvla``::

    <dst_root>/
    ├── frames/dobot_camera_history/<date>/<segment>/<bag>/<bag>.jpgs
    └── jsonl/dobot_camera_history/<date>/<segment>/<bag>/<bag>.jsonl

Each ``.jpgs`` file is a ZIP archive containing every readable source-camera
image in chronological order as ``frame_000000.jpg``, ``frame_000001.jpg``,
... .  Each JSONL record's ``current`` member is selected by nearest timestamp
to the label. The TrackVLA loader then walks backward from that member to load
the most recent history images. JSONL records reference the archive and its
current member using relative paths. Masks are intentionally not generated or
referenced; this is the RGB-only TrackVLA format.

The TrackVLA loader chooses the number and spacing of history frames from the
training config (``num_history_frames`` and ``history_frame_interval``), so
the manifest stores the complete episode archive rather than an ``images``
list for every sample.

Example for the current RGB-only TrackVLA config::

    python tools/convert_dobot_local.py \
        --json_list /work/shengjun/dobot_trackvla/data/dobot_vla_local/all_training_jsons_local.txt \
        --path_map /mnt/cpfs/dobot_vla_local=/work/shengjun/dobot_trackvla/data/dobot_vla_local \
        --dst_root /work/shengjun/dobot_trackvla/data/dobot-trackvla \
        --n_waypoints 8 \
        --waypoint_step 3 \
        --cam cam1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


PathMap = Tuple[str, str]


def normalize_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def world_to_robot(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    """Rotate a world-frame displacement into the current robot frame."""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        dx * cos_yaw + dy * sin_yaw,
        -dx * sin_yaw + dy * cos_yaw,
    )


def build_trajectory(
    labels: Sequence[Mapping[str, Any]],
    start_idx: int,
    n_waypoints: int,
    step: int,
) -> List[List[float]]:
    """Build future ``[x, y, yaw]`` points in the current ego frame.

    Dobot labels are normally sampled at 10 Hz.  ``step=3`` therefore keeps
    the original converter's 0.3-second waypoint spacing.  TrackVLA does not
    store a separate ``future_dt`` field, so the spacing is represented by the
    selected labels themselves.
    """
    if n_waypoints <= 0:
        raise ValueError("n_waypoints must be positive")
    if step <= 0:
        raise ValueError("waypoint_step must be positive")

    last_idx = start_idx + n_waypoints * step
    if start_idx < 0 or last_idx >= len(labels):
        raise IndexError("not enough future labels for trajectory")

    current_pose = labels[start_idx]["pose"]
    x0 = float(current_pose["x"])
    y0 = float(current_pose["y"])
    yaw0 = float(current_pose["yaw"])

    trajectory: List[List[float]] = []
    for waypoint_idx in range(1, n_waypoints + 1):
        pose = labels[start_idx + waypoint_idx * step]["pose"]
        dx = float(pose["x"]) - x0
        dy = float(pose["y"]) - y0
        robot_x, robot_y = world_to_robot(dx, dy, yaw0)
        robot_yaw = normalize_angle(float(pose["yaw"]) - yaw0)
        trajectory.append([robot_x, robot_y, robot_yaw])
    return trajectory


def frame_member_name(frame_index: int) -> str:
    """Return the normalized ZIP member name used by TrackVLA."""
    return f"frame_{frame_index:06d}.jpg"


def _safe_name(value: str) -> str:
    """Make a path component safe and portable across operating systems."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "episode"


def episode_name(json_path: Path) -> str:
    """Create a stable, collision-resistant episode basename."""
    segment = next(
        (
            part
            for part in reversed(json_path.parts)
            if part.startswith(("F-", "N-"))
        ),
        json_path.parent.name or "episode",
    )
    digest = hashlib.sha1(str(json_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{_safe_name(segment)}_{digest}"


def episode_relative_path(json_path: Path) -> PurePosixPath:
    """Return the logical output directory for one native Dobot bag.

    Native episodes normally use this structure::

        .../post_sync_data/20260709/F-01-.../dobot_bag_0000_0/training.json

    Preserve the date, F/N segment and bag name as nested directories in the
    converted subset. The
    TrackVLA loader discovers JSONL files recursively and resolves each
    record's ``frames_file`` from the subset root, so this hierarchy is
    directly loader-compatible. A hash-based fallback keeps unusual source
    paths collision-resistant.
    """
    parts = json_path.parts
    segment_index = next(
        (
            index
            for index in range(len(parts) - 1, -1, -1)
            if parts[index].startswith(("F-", "N-"))
        ),
        None,
    )
    if segment_index is None or segment_index + 1 >= len(parts):
        return PurePosixPath(episode_name(json_path))

    segment = _safe_name(parts[segment_index])
    parent = parts[segment_index - 1] if segment_index > 0 else ""
    group = _safe_name(parent) if re.fullmatch(r"\d{8}", parent) else None
    bag = _safe_name(parts[segment_index + 1])
    if group is None:
        return PurePosixPath(segment) / bag
    return PurePosixPath(group) / segment / bag


def parse_path_maps(values: Sequence[str]) -> List[PathMap]:
    """Parse repeatable ``OLD=NEW`` path-prefix mappings.

    Native local mirrors are commonly moved from ``/mnt/cpfs`` into a project
    directory.  Both the JSON list and each frame's camera path may retain the
    old absolute prefix.  Keep this conversion-time only: the source list and
    ``training.json`` files remain unchanged.
    """
    mappings: List[PathMap] = []
    for value in values:
        source, separator, destination = value.partition("=")
        source = source.strip().rstrip("/\\")
        destination = destination.strip().rstrip("/\\")
        if not separator or not source or not destination:
            raise ValueError(
                "--path_map entries must use OLD=NEW, for example "
                "/mnt/cpfs/dobot_vla_local=/work/shengjun/dobot_trackvla/data/dobot_vla_local"
            )
        mappings.append((source, destination))

    # Prefer the longest source prefix, allowing a specific remap to override
    # a broader one such as /mnt/cpfs.
    return sorted(mappings, key=lambda mapping: len(mapping[0]), reverse=True)


def remap_path(value: str, path_maps: Sequence[PathMap]) -> str:
    """Return ``value`` with its first matching directory prefix remapped."""
    normalized_value = value.replace("\\", "/")
    for source, destination in path_maps:
        normalized_source = source.replace("\\", "/")
        if normalized_value == normalized_source:
            return str(Path(destination))
        if normalized_value.startswith(normalized_source + "/"):
            relative_parts = normalized_value[len(normalized_source) + 1 :].split("/")
            return str(Path(destination).joinpath(*relative_parts))
    return value


def resolve_image_path(
    value: Any,
    json_path: Path,
    path_maps: Sequence[PathMap] = (),
) -> Optional[Path]:
    """Resolve an image path from a native Dobot frame record."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(remap_path(value.strip(), path_maps))
    if not path.is_absolute():
        path = json_path.parent / path
    return path


def image_is_readable(path: Path) -> bool:
    """Reject missing or corrupt images before putting them in an archive."""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def load_translation_map(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt translation file must contain a JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def readable_camera_paths(
    frames: Sequence[Mapping[str, Any]],
    json_path: Path,
    cam: str,
    path_maps: Sequence[PathMap] = (),
) -> Dict[int, Path]:
    """Return readable source images keyed by their raw camera-frame index."""
    result: Dict[int, Path] = {}
    for frame_index, frame in enumerate(frames):
        source_path = resolve_image_path(frame.get(cam), json_path, path_maps)
        if source_path is not None and image_is_readable(source_path):
            result[frame_index] = source_path
    return result


def write_camera_episode_archive(
    readable_paths: Mapping[int, Path],
    archive_path: Path,
) -> Dict[int, str]:
    """Pack all readable camera images in original frame order."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_name(
        f".{archive_path.name}.{os.getpid()}.tmp"
    )
    valid_members: Dict[int, str] = {}
    try:
        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            ordered_frame_indices = sorted(readable_paths)
            for archive_index, frame_index in enumerate(ordered_frame_indices):
                member = frame_member_name(archive_index)
                source_path = readable_paths[frame_index]
                archive.write(source_path, arcname=member)
                valid_members[frame_index] = member
        os.replace(temp_path, archive_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return valid_members


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write one episode's JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def count_exported_samples(dst_root: Path) -> Tuple[int, int]:
    """Count non-empty JSONL records in the final exported TrackVLA subset."""
    jsonl_root = dst_root / "jsonl"
    if not jsonl_root.is_dir():
        return 0, 0

    manifest_count = 0
    sample_count = 0
    for manifest_path in jsonl_root.rglob("*.jsonl"):
        manifest_count += 1
        with manifest_path.open(encoding="utf-8") as handle:
            sample_count += sum(1 for line in handle if line.strip())
    return manifest_count, sample_count


def nearest_frame_indices(
    frames: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> List[Optional[int]]:
    """Map each label to its nearest raw camera frame with no time cutoff."""
    try:
        frame_timestamps = np.asarray(
            [float(frame["timestamp"]) * 1e-9 for frame in frames],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return [None] * len(labels)
    if len(frame_timestamps) == 0:
        return [None] * len(labels)

    result: List[Optional[int]] = []
    for label in labels:
        try:
            target = float(label["timestamp"])
        except (KeyError, TypeError, ValueError):
            result.append(None)
            continue
        result.append(int(np.argmin(np.abs(frame_timestamps - target))))
    return result


def process_episode(
    json_path: Path,
    dst_root: Path,
    n_waypoints: int,
    waypoint_step: int,
    cam: str,
    translation_map: Optional[Mapping[str, str]],
    include_empty_prompt: bool,
    include_negative_segments: bool,
    path_maps: Sequence[PathMap] = (),
    subset_name: str = "dobot_camera_history",
) -> Optional[Dict[str, Any]]:
    """Convert one native Dobot ``training.json`` into one TrackVLA episode."""
    try:
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"[skip] cannot load {json_path}: {exc}")
        return None

    segment = next(
        (part for part in reversed(json_path.parts) if part.startswith(("F-", "N-"))),
        "",
    )
    if segment.startswith("N-") and not include_negative_segments:
        return None

    bag_meta = data.get("bag_meta", {})
    instruction = str(bag_meta.get("scene_desc", "") or "").strip()
    if not instruction and not include_empty_prompt:
        return None
    if translation_map is not None:
        instruction = translation_map.get(instruction, instruction)

    frames = data.get("frames", [])
    labels = data.get("labels", [])
    if not isinstance(frames, list) or not isinstance(labels, list):
        print(f"[skip] invalid frames/labels in {json_path}")
        return None
    if not frames or len(labels) <= n_waypoints * waypoint_step:
        return None

    readable_paths = readable_camera_paths(frames, json_path, cam, path_maps)
    label_to_frame = nearest_frame_indices(frames, labels)

    episode_rel = episode_relative_path(json_path)
    episode_file_stem = episode_rel.name
    archive_rel = (
        PurePosixPath("frames")
        / subset_name
        / episode_rel
        / f"{episode_file_stem}.jpgs"
    )
    archive_path = dst_root / Path(*archive_rel.parts)
    valid_members = write_camera_episode_archive(readable_paths, archive_path)
    if not valid_members:
        archive_path.unlink(missing_ok=True)
        return None

    max_start = len(labels) - n_waypoints * waypoint_step
    records: List[Dict[str, Any]] = []
    for label_index in range(max_start):
        frame_index = label_to_frame[label_index]
        if frame_index is None or frame_index not in valid_members:
            continue
        try:
            trajectory = build_trajectory(
                labels,
                label_index,
                n_waypoints,
                waypoint_step,
            )
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        records.append(
            {
                "frames_file": archive_rel.as_posix(),
                "current": valid_members[frame_index],
                "instruction": instruction,
                "trajectory": trajectory,
            }
        )

    if not records:
        archive_path.unlink(missing_ok=True)
        return None

    manifest_rel = (
        PurePosixPath("jsonl")
        / subset_name
        / episode_rel
        / f"{episode_file_stem}.jsonl"
    )
    write_jsonl(dst_root / Path(*manifest_rel.parts), records)
    return {
        "episode": str(json_path),
        "archive": archive_rel.as_posix(),
        "manifest": manifest_rel.as_posix(),
        "samples": len(records),
        "archived_camera_frames": len(valid_members),
        "readable_camera_frames": len(readable_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Dobot training.json files to RGB-only TrackVLA JSONL + ZIP archives."
    )
    parser.add_argument(
        "--json_list",
        required=True,
        help="Text file containing one native Dobot training.json path per line.",
    )
    parser.add_argument(
        "--dst_root",
        required=True,
        help="TrackVLA subset root to create; it will contain frames/ and jsonl/.",
    )
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument(
        "--waypoint_step",
        type=int,
        default=3,
        help="Label spacing for future waypoints; 3 is 0.3s at 10 Hz.",
    )
    parser.add_argument("--cam", default="cam1")
    parser.add_argument(
        "--subset_name",
        default="dobot_camera_history",
        help=(
            "Name of the frames/ and jsonl/ child directory. The default "
            "identifies this as a Dobot subset containing camera history."
        ),
    )
    parser.add_argument(
        "--prompt_translation",
        type=Path,
        default=None,
        help="Optional JSON object mapping Chinese scene_desc text to English.",
    )
    parser.add_argument(
        "--path_map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Rewrite a stale directory prefix while reading the JSON list and "
            "camera paths. Repeat this option for multiple mappings; source "
            "files are not modified."
        ),
    )
    parser.add_argument(
        "--include_empty_prompt",
        action="store_true",
        help="Keep episodes whose bag_meta.scene_desc is empty.",
    )
    parser.add_argument(
        "--include_negative_segments",
        action="store_true",
        help="Keep N-prefixed segments; by default they are skipped.",
    )
    args = parser.parse_args()

    if args.n_waypoints <= 0:
        parser.error("--n_waypoints must be positive")
    if args.waypoint_step <= 0:
        parser.error("--waypoint_step must be positive")
    subset_name = _safe_name(args.subset_name)

    try:
        path_maps = parse_path_maps(args.path_map)
    except ValueError as exc:
        parser.error(str(exc))

    json_list = Path(remap_path(args.json_list, path_maps))
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    translation_map = load_translation_map(args.prompt_translation)
    if translation_map is not None:
        print(f"[info] loaded {len(translation_map)} prompt translations")
    for source, destination in path_maps:
        print(f"[info] path map: {source} -> {destination}")

    with json_list.open(encoding="utf-8") as handle:
        json_paths = [
            Path(remap_path(line.strip(), path_maps))
            for line in handle
            if line.strip()
        ]

    results: List[Dict[str, Any]] = []
    for index, json_path in enumerate(json_paths, start=1):
        result = process_episode(
            json_path=json_path,
            dst_root=dst_root,
            n_waypoints=args.n_waypoints,
            waypoint_step=args.waypoint_step,
            cam=args.cam,
            translation_map=translation_map,
            include_empty_prompt=args.include_empty_prompt,
            include_negative_segments=args.include_negative_segments,
            path_maps=path_maps,
            subset_name=subset_name,
        )
        if result is not None:
            results.append(result)
        if index % 10 == 0 or index == len(json_paths):
            print(f"[progress] {index}/{len(json_paths)} source episodes processed")

    run_samples = sum(result["samples"] for result in results)
    manifest_count, exported_samples = count_exported_samples(dst_root)
    print(
        f"[done] converted_episodes={len(results)}, "
        f"converted_samples={run_samples}, "
        f"archived_camera_frames={sum(result['archived_camera_frames'] for result in results)}, "
        f"output={dst_root.resolve()}"
    )
    print(
        f"[done] exported_manifests={manifest_count}, "
        f"exported_samples={exported_samples}"
    )
    print("[info] TrackVLA will split these episodes into train/val at runtime.")


if __name__ == "__main__":
    main()
