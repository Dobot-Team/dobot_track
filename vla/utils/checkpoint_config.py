"""Persist and load training configuration beside Lightning checkpoints."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Mapping, Optional, Union

import torch
import yaml


CHECKPOINT_CONFIG_FILENAME = "config.yaml"


def _load_torch_mapping(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint {path} must contain a mapping")
    return checkpoint


def _deepspeed_model_state_path(checkpoint_dir: Path) -> Path:
    """Resolve one rank-zero DeepSpeed model-state file deterministically."""
    directory = checkpoint_dir.resolve()
    latest_path = directory / "latest"
    search_root = directory
    used_latest_tag = False
    if latest_path.is_file():
        tag = latest_path.read_text(encoding="utf-8").strip()
        if not tag:
            raise ValueError(
                f"DeepSpeed checkpoint latest tag is empty: {latest_path}"
            )
        tagged_root = (directory / tag).resolve()
        try:
            tagged_root.relative_to(directory)
        except ValueError as exc:
            raise ValueError(
                f"DeepSpeed latest tag escapes checkpoint directory: {tag!r}"
            ) from exc
        if not tagged_root.is_dir():
            raise FileNotFoundError(
                f"DeepSpeed latest tag points to no directory: {tagged_root}"
            )
        search_root = tagged_root
        used_latest_tag = True

    candidates = sorted(search_root.rglob("*model_states.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No DeepSpeed *model_states.pt file found under "
            f"{search_root}"
        )

    rank_zero_names = (
        "mp_rank_00_model_states.pt",
        "zero_pp_rank_0_mp_rank_00_model_states.pt",
    )
    rank_zero = [
        path for path in candidates if path.name in rank_zero_names
    ]
    if rank_zero:
        parent_directories = {path.parent for path in rank_zero}
        if not used_latest_tag and len(parent_directories) > 1:
            raise ValueError(
                "DeepSpeed checkpoint directory contains multiple step "
                "directories but no usable latest tag: "
                + ", ".join(
                    str(path) for path in sorted(parent_directories)
                )
            )
        return min(
            rank_zero,
            key=lambda path: (
                rank_zero_names.index(path.name),
                len(path.parts),
                str(path),
            ),
        )
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "DeepSpeed checkpoint has multiple model-state files but no "
        "recognizable rank-zero state: "
        + ", ".join(str(path) for path in candidates)
    )


def load_checkpoint_hparams(
    checkpoint_path: Union[str, Path],
) -> Mapping[str, Any]:
    """Load Lightning hparams from a file or DeepSpeed checkpoint directory."""
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        checkpoint = _load_torch_mapping(path)
    elif path.is_dir():
        checkpoint = _load_torch_mapping(
            _deepspeed_model_state_path(path)
        )
    else:
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    return extract_checkpoint_hparams(checkpoint)


def extract_checkpoint_hparams(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Extract validated Lightning hparams from checkpoint/client state."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    client_state = checkpoint.get("client_state")
    metadata_sources = [checkpoint]
    if isinstance(client_state, Mapping):
        metadata_sources.append(client_state)
    for source in metadata_sources:
        if "hyper_parameters" not in source:
            continue
        hparams = source["hyper_parameters"]
        if hparams is None:
            continue
        if not isinstance(hparams, Mapping):
            raise TypeError(
                "Checkpoint hyper_parameters must be a mapping"
            )
        return hparams
    return {}


def _parse_config_yaml(config_yaml: str, source: str) -> Dict[str, Any]:
    """Parse a VLA configuration and validate the fields entrypoints require."""
    try:
        config = yaml.safe_load(config_yaml)
    except yaml.YAMLError as error:
        raise ValueError(f"Could not parse {source}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError(f"{source} must decode to a mapping")
    if not isinstance(config.get("model"), dict) or not isinstance(config.get("data"), dict):
        raise ValueError(f"{source} must contain model and data mappings")
    return config


def is_rank_zero() -> bool:
    """Whether this process owns shared training artifacts.

    ``RANK`` is set by torchrun. Lightning's subprocess launchers can instead
    expose ``LOCAL_RANK`` and ``NODE_RANK`` before distributed initialization.
    A single-process launch has none of these variables and is rank zero.
    """
    global_rank = os.environ.get("RANK")
    if global_rank not in (None, ""):
        return int(global_rank) == 0

    local_rank = os.environ.get("LOCAL_RANK", "0")
    node_rank = os.environ.get("NODE_RANK", "0")
    return int(local_rank or "0") == 0 and int(node_rank or "0") == 0


def checkpoint_config_path(checkpoint_path: Union[str, Path]) -> Path:
    """Find the nearest visible ``config.yaml`` for a checkpoint artifact.

    A regular Lightning checkpoint lives directly under ``checkpoints/``.
    DeepSpeed checkpoints and converted state-dict files are commonly nested
    one or more levels below that directory.  The training YAML remains the
    source of truth for reconstruction, so search the artifact directory and
    its parents without opening checkpoint contents.

    When no sidecar exists, return the nearest expected location so callers
    can retain a precise error message.
    """
    path = Path(checkpoint_path).expanduser()
    directory = path if path.is_dir() else path.parent
    for candidate_directory in (directory, *directory.parents):
        candidate = candidate_directory / CHECKPOINT_CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return directory / CHECKPOINT_CONFIG_FILENAME


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a VLA YAML configuration from an explicit path."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    return _parse_config_yaml(path.read_text(encoding="utf-8"), f"Config {path}")


def save_checkpoint_config(
    checkpoint_dir: Union[str, Path],
    config_yaml: str,
) -> Path:
    """Atomically save the resolved training YAML beside the checkpoints.

    Only global rank zero writes the sidecar. Replacing a closed temporary file
    keeps readers from observing a partially written configuration.
    """
    directory = Path(checkpoint_dir).expanduser()
    path = directory / CHECKPOINT_CONFIG_FILENAME
    if not is_rank_zero():
        return path

    directory.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{CHECKPOINT_CONFIG_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(str(config_yaml))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_checkpoint_config(
    checkpoint_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load an explicit config override or a checkpoint's ``config.yaml`` sidecar."""
    if config_path is not None:
        return load_config_file(config_path)

    path = checkpoint_config_path(checkpoint_path)
    if not path.is_file():
        raise ValueError(
            f"No checkpoint config found at {path}. "
            "Pass --config to supply a YAML override."
        )
    return load_config_file(path)
