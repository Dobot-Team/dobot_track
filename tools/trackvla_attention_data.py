"""Lightweight TrackVLA manifest resolution for offline diagnostics.

The visualisation tools need a small, episode-local view of a TrackVLA
manifest without constructing :class:`vla.data.trackvla.TrackVLADataset`.
Keeping this module independent of the dataset adapter avoids importing torch,
PIL, or a model merely to locate records for one ``.jpgs`` archive.

``dataset.json`` can be large, so array manifests are scanned as byte ranges
and decoded one record at a time.  This follows the same top-level-array
shape accepted by ``TrackVLADataset`` without materialising the whole file.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union


DEFAULT_MAX_SAMPLES = 32


@dataclass(frozen=True)
class ResolvedTrackVLAEpisode:
    """The records in one TrackVLA episode selected from one manifest.

    ``record_indices`` are zero-based indices in the source manifest's valid
    record sequence.  They therefore remain useful in reports when a fallback
    ``dataset.json`` contains samples from many episodes.  ``records`` has the
    matching TrackVLA dictionaries in the same order.
    """

    frames_path: Path
    subset_root: Path
    manifest_path: Path
    source: str
    records: Tuple[Mapping[str, Any], ...]
    record_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.records) != len(self.record_indices):
            raise ValueError("records and record_indices must have the same length")


def parse_indices(value: Optional[Union[str, Sequence[int]]]) -> Optional[list[int]]:
    """Parse comma-separated, zero-based episode-local sample positions.

    ``None`` means that range selection should be used.  Duplicate positions
    are rejected so an accidental repeated CLI value cannot silently duplicate
    visualisation output.
    """

    if value is None:
        return None

    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",")]
        if not pieces or any(not piece for piece in pieces):
            raise ValueError("--indices must be a comma-separated list of integers")
        try:
            parsed = [int(piece, 10) for piece in pieces]
        except ValueError as exc:
            raise ValueError(
                "--indices must be a comma-separated list of integers"
            ) from exc
    else:
        parsed = []
        for item in value:
            # bool is an int subclass but never a useful sample index.
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("indices must contain integers")
            parsed.append(int(item))

    if any(index < 0 for index in parsed):
        raise ValueError("indices must be non-negative")
    if len(set(parsed)) != len(parsed):
        raise ValueError("indices must not contain duplicates")
    return parsed


def select_episode_indices(
    episode_size: int,
    *,
    indices: Optional[Union[str, Sequence[int]]] = None,
    start_index: Optional[int] = None,
    stride: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> list[int]:
    """Select episode-local positions for an offline diagnostic run.

    Explicit ``indices`` and range selection are intentionally mutually
    exclusive.  CLI callers should use ``None`` as the argparse default for
    ``start_index``, ``stride``, and ``max_samples``; absent range arguments
    resolve to ``0``, ``1``, and :data:`DEFAULT_MAX_SAMPLES` respectively.
    """

    if episode_size < 0:
        raise ValueError("episode_size must be non-negative")

    explicit = parse_indices(indices)
    if explicit is not None:
        if start_index is not None or stride is not None or max_samples is not None:
            raise ValueError(
                "--indices cannot be combined with --start-index, --stride, or --max-samples"
            )
        invalid = [index for index in explicit if index >= episode_size]
        if invalid:
            raise IndexError(
                f"episode has {episode_size} samples; index {invalid[0]} is out of range"
            )
        return explicit

    resolved_start = 0 if start_index is None else start_index
    resolved_stride = 1 if stride is None else stride
    resolved_max = DEFAULT_MAX_SAMPLES if max_samples is None else max_samples

    if resolved_start < 0:
        raise ValueError("start_index must be non-negative")
    if resolved_stride <= 0:
        raise ValueError("stride must be positive")
    if resolved_max <= 0:
        raise ValueError("max_samples must be positive")
    if resolved_start >= episode_size:
        return []
    return list(range(resolved_start, episode_size, resolved_stride))[:resolved_max]


def resolve_trackvla_episode(
    frames_path: Union[str, Path],
    samples_path: Optional[Union[str, Path]] = None,
) -> ResolvedTrackVLAEpisode:
    """Resolve TrackVLA records for one ``.jpgs`` archive.

    An explicit manifest always wins.  Without one, the function first checks
    the standard JSONL path mirroring the archive below ``frames/`` (including
    ``samples.jsonl``), then falls back to the subset's ``dataset.json``.
    Every candidate is checked against ``frames_file`` rather than selected by
    filename alone, preventing a nearby but unrelated manifest from being
    used accidentally.
    """

    archive = _normalise_frames_path(frames_path)
    subset_root = _find_subset_root(archive)

    if samples_path is not None:
        manifest = Path(samples_path).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"--samples does not exist: {manifest}")
        _validate_manifest_suffix(manifest)
        records, indices = _matching_records(manifest, archive, subset_root)
        return _build_result(
            archive, subset_root, manifest, "explicit", records, indices
        )

    for manifest in _adjacent_manifest_candidates(archive, subset_root):
        if not manifest.is_file():
            continue
        records, indices = _matching_records(manifest, archive, subset_root)
        if records:
            return _build_result(
                archive, subset_root, manifest, "adjacent_jsonl", records, indices
            )

    dataset_manifest = subset_root / "dataset.json"
    if dataset_manifest.is_file():
        records, indices = _matching_records(dataset_manifest, archive, subset_root)
        if records:
            return _build_result(
                archive, subset_root, dataset_manifest, "dataset_json", records, indices
            )

    candidate_paths = [
        *_adjacent_manifest_candidates(archive, subset_root),
        dataset_manifest,
    ]
    checked = "\n  ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"No TrackVLA manifest records reference archive {archive}. Checked:\n  {checked}"
    )


def _build_result(
    archive: Path,
    subset_root: Path,
    manifest: Path,
    source: str,
    records: list[Mapping[str, Any]],
    indices: list[int],
) -> ResolvedTrackVLAEpisode:
    if not records:
        raise FileNotFoundError(
            f"No TrackVLA records in {manifest} reference archive {archive}"
        )
    return ResolvedTrackVLAEpisode(
        frames_path=archive,
        subset_root=subset_root,
        manifest_path=manifest,
        source=source,
        records=tuple(records),
        record_indices=tuple(indices),
    )


def _normalise_frames_path(value: Union[str, Path]) -> Path:
    archive = Path(value).expanduser().resolve()
    if archive.suffix.lower() != ".jpgs":
        raise ValueError(f"--frames must name a .jpgs archive, got: {archive}")
    if not archive.is_file():
        raise FileNotFoundError(f"--frames does not exist: {archive}")
    return archive


def _find_subset_root(archive: Path) -> Path:
    """Return the parent of the nearest ``frames/`` directory containing it."""

    for ancestor in archive.parents:
        if ancestor.name == "frames":
            return ancestor.parent
    raise ValueError(
        "Cannot infer a TrackVLA subset because --frames is not below a frames/ "
        "directory. Pass an archive from a standard TrackVLA subset."
    )


def _adjacent_manifest_candidates(archive: Path, subset_root: Path) -> tuple[Path, ...]:
    relative_archive = archive.relative_to(subset_root / "frames")
    mirrored_dir = subset_root / "jsonl" / relative_archive.parent
    candidates = (
        mirrored_dir / f"{archive.stem}.jsonl",
        mirrored_dir / "samples.jsonl",
        archive.with_suffix(".jsonl"),
        archive.parent / "samples.jsonl",
    )

    # ``samples.jsonl`` can coincide with the stem-derived candidate for an
    # archive called samples.jpgs. Preserve order while avoiding redundant IO.
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return tuple(unique)


def _validate_manifest_suffix(path: Path) -> None:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(
            f"Unsupported TrackVLA manifest {path}; expected a .json or .jsonl file"
        )


def _matching_records(
    manifest: Path,
    archive: Path,
    subset_root: Path,
) -> tuple[list[Mapping[str, Any]], list[int]]:
    _validate_manifest_suffix(manifest)
    records: list[Mapping[str, Any]] = []
    indices: list[int] = []
    for record_index, record in _iter_manifest_records(manifest):
        if _record_references_archive(record, archive, subset_root):
            records.append(record)
            indices.append(record_index)
    return records, indices


def _record_references_archive(
    record: Mapping[str, Any], archive: Path, subset_root: Path
) -> bool:
    frames_file = record.get("frames_file")
    if not isinstance(frames_file, str) or not frames_file.strip():
        return False
    # Dataset manifests are normally POSIX even on Windows.  ``Path`` accepts
    # forward slashes on Windows, while normalising explicit backslashes keeps
    # hand-written local manifests equivalent.
    reference = Path(frames_file.replace("\\", "/"))
    if not reference.is_absolute():
        reference = subset_root / reference
    return reference.resolve() == archive


def _iter_manifest_records(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl_records(path)
    else:
        yield from _iter_json_array_records(path)


def _iter_jsonl_records(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    record_index = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record is not a JSON object")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                # Match TrackVLADataset's JSONL behavior: a malformed line is
                # skipped, while source indices continue to correspond to its
                # valid-record index rather than physical line number.
                warnings.warn(
                    f"[TrackVLA attention data] Skipping malformed JSONL record "
                    f"{path}:{line_number}: {exc}",
                    stacklevel=2,
                )
                continue
            yield record_index, record
            record_index += 1


def _iter_json_array_records(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open("rb") as handle:
        for record_index, (offset, length) in enumerate(_iter_json_array_objects(path)):
            handle.seek(offset)
            payload = handle.read(length)
            try:
                record = json.loads(payload)
                if not isinstance(record, dict):
                    raise ValueError("record is not a JSON object")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                # Array record positions deliberately remain stable: unlike
                # JSONL, TrackVLADataset indexes every top-level object and
                # only surfaces an invalid object when it is accessed.
                warnings.warn(
                    f"[TrackVLA attention data] Skipping malformed JSON record "
                    f"{record_index} in {path}: {exc}",
                    stacklevel=2,
                )
                continue
            yield record_index, record


def _iter_json_array_objects(path: Path) -> Iterable[tuple[int, int]]:
    """Yield byte ranges for top-level objects in a JSON array.

    This is deliberately equivalent in shape to TrackVLADataset's lazy index
    scanner, but it lives here to keep diagnostics free of data/model imports.
    """

    with path.open("rb") as handle:
        offset = 0
        while True:
            first = handle.read(1)
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

        while chunk := handle.read(1024 * 1024):
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
