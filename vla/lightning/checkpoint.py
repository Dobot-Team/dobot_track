"""Lightning checkpoint configuration, loading, restoration, and validation."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Collection, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import yaml


CHECKPOINT_CONFIG_FILENAME = "config.yaml"


def _parse_config_yaml(config_yaml: str, source: str) -> Dict[str, Any]:
    """Parse a VLA configuration and validate the fields entrypoints require."""
    try:
        config = yaml.safe_load(config_yaml)
    except yaml.YAMLError as error:
        raise ValueError(f"Could not parse {source}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError(f"{source} must decode to a mapping")
    if not isinstance(config.get("model"), dict) or not isinstance(
        config.get("data"),
        dict,
    ):
        raise ValueError(f"{source} must contain model and data mappings")
    return config


def is_rank_zero() -> bool:
    """Whether this process owns shared training artifacts."""
    global_rank = os.environ.get("RANK")
    if global_rank not in (None, ""):
        return int(global_rank) == 0

    local_rank = os.environ.get("LOCAL_RANK", "0")
    node_rank = os.environ.get("NODE_RANK", "0")
    return int(local_rank or "0") == 0 and int(node_rank or "0") == 0


def checkpoint_config_path(checkpoint_path: Union[str, Path]) -> Path:
    """Return the training sidecar associated with a file or DeepSpeed path."""
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        # A rank-state file belongs to a DeepSpeed tag directory. Reuse the
        # directory rules below so an explicit rank-zero path still finds the
        # training sidecar beside the checkpoint collection.
        if path.name.endswith("model_states.pt"):
            return checkpoint_config_path(path.parent)
        return path.parent / CHECKPOINT_CONFIG_FILENAME

    # Lightning's DeepSpeed strategy normally saves a checkpoint as a
    # ``*.ckpt`` directory containing ``latest``. The training sidecar lives
    # beside all checkpoint entries; a non-.ckpt container owns its sidecar.
    if (path / "latest").is_file():
        if path.suffix.lower() == ".ckpt":
            candidates = (
                path.parent / CHECKPOINT_CONFIG_FILENAME,
                path / CHECKPOINT_CONFIG_FILENAME,
            )
        else:
            candidates = (
                path / CHECKPOINT_CONFIG_FILENAME,
                path.parent / CHECKPOINT_CONFIG_FILENAME,
            )
    elif (path.parent / "latest").is_file():
        candidates = (
            path.parent.parent / CHECKPOINT_CONFIG_FILENAME,
            path.parent / CHECKPOINT_CONFIG_FILENAME,
            path / CHECKPOINT_CONFIG_FILENAME,
        )
    elif path.is_dir() and any(path.glob("*model_states.pt")):
        # A copied DeepSpeed tag directory can omit the original ``latest``.
        candidates = (
            path.parent / CHECKPOINT_CONFIG_FILENAME,
            path.parent.parent / CHECKPOINT_CONFIG_FILENAME,
            path / CHECKPOINT_CONFIG_FILENAME,
        )
    else:
        candidates = (
            path / CHECKPOINT_CONFIG_FILENAME,
            path.parent / CHECKPOINT_CONFIG_FILENAME,
        )
    return next(
        (candidate for candidate in candidates if candidate.is_file()),
        candidates[0],
    )


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
    """Atomically save resolved training YAML beside checkpoint entries."""
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
    """Load a config override or a checkpoint's ``config.yaml`` sidecar."""
    if config_path is not None:
        return load_config_file(config_path)

    path = checkpoint_config_path(checkpoint_path)
    if not path.is_file():
        raise ValueError(
            f"No checkpoint config found at {path}. "
            "Pass --config to supply a YAML override."
        )
    return load_config_file(path)


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


def _resolve_deepspeed_checkpoint_root(
    checkpoint_dir: Path,
) -> tuple[Path, str]:
    """Resolve a DeepSpeed root and inference tag from a root or tag path."""
    requested = checkpoint_dir.expanduser().resolve()
    for root in (requested, requested.parent):
        latest_path = root / "latest"
        if not latest_path.is_file():
            continue
        if (
            requested != root
            and requested.parent == root
            and requested.is_dir()
            and any(requested.rglob("*model_states.pt"))
        ):
            return root, requested.name
        tag = latest_path.read_text(encoding="utf-8").strip()
        if not tag:
            raise ValueError(
                f"DeepSpeed checkpoint latest tag is empty: {latest_path}"
            )
        tag_root = (root / tag).resolve()
        try:
            tag_root.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"DeepSpeed latest tag escapes checkpoint directory: {tag!r}"
            ) from exc
        if not tag_root.is_dir():
            raise FileNotFoundError(
                f"DeepSpeed latest tag points to no directory: {tag_root}"
            )
        if requested == root or requested == tag_root:
            return root, tag

    rank_zero_state = _deepspeed_model_state_path(requested)
    tag_directory = rank_zero_state.parent
    if tag_directory == requested:
        return requested.parent, requested.name
    try:
        tag = tag_directory.relative_to(requested).as_posix()
    except ValueError as exc:
        raise ValueError(
            "Could not infer a DeepSpeed checkpoint root/tag from "
            f"{requested}"
        ) from exc
    return requested, tag


def _load_deepspeed_zero_state_dict(
    checkpoint_root: Path,
    tag: str,
) -> Mapping[str, Any]:
    """Consolidate a DeepSpeed ZeRO checkpoint into a CPU state dictionary."""
    try:
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )
    except Exception as exc:  # noqa: BLE001 - optional dependency import
        raise RuntimeError(
            "DeepSpeed directory inference requires the optional `deepspeed` "
            "package to consolidate ZeRO shards. Install `deepspeed>=0.16.0` "
            "or export a standard VLA .ckpt first."
        ) from exc

    converter = get_fp32_state_dict_from_zero_checkpoint
    try:
        converter_parameters = inspect.signature(converter).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "DeepSpeed directory inference requires `deepspeed>=0.16.0`. "
            "Upgrade DeepSpeed or export a standard VLA .ckpt first."
        ) from exc
    supports_exclude_frozen = (
        "exclude_frozen_parameters" in converter_parameters
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in converter_parameters.values()
        )
    )
    if not supports_exclude_frozen:
        raise RuntimeError(
            "DeepSpeed directory inference requires `deepspeed>=0.16.0`. "
            "Upgrade DeepSpeed or export a standard VLA .ckpt first."
        )
    tag_directory = checkpoint_root / tag
    optimizer_shards = list(tag_directory.rglob("*optim_states.pt"))
    if not optimizer_shards:
        raise FileNotFoundError(
            "DeepSpeed checkpoint is incomplete: no *optim_states.pt ZeRO "
            f"shards found under {tag_directory}"
        )

    try:
        state_dict = converter(
            str(checkpoint_root),
            tag=tag,
            exclude_frozen_parameters=True,
        )
    except Exception as exc:  # noqa: BLE001 - preserve DeepSpeed root cause
        raise RuntimeError(
            "Could not consolidate DeepSpeed checkpoint "
            f"{checkpoint_root} (tag {tag!r}). Ensure every ZeRO shard is "
            "present and enough CPU RAM is available for the FP32 state dict."
        ) from exc
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "DeepSpeed ZeRO consolidation returned a non-mapping state dict"
        )
    return state_dict


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
            raise TypeError("Checkpoint hyper_parameters must be a mapping")
        return hparams
    return {}


def load_checkpoint_hparams(
    checkpoint_path: Union[str, Path],
) -> Mapping[str, Any]:
    """Load Lightning hparams from a file or DeepSpeed checkpoint directory."""
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        checkpoint = _load_torch_mapping(path)
    elif path.is_dir():
        checkpoint = _load_torch_mapping(_deepspeed_model_state_path(path))
    else:
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return extract_checkpoint_hparams(checkpoint)


def load_checkpoint_for_inference(
    checkpoint_path: Union[str, Path],
) -> Dict[str, Any]:
    """Materialize a mutable regular or DeepSpeed checkpoint for inference."""
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        checkpoint = dict(_load_torch_mapping(path))
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            if (
                path.name.endswith("model_states.pt")
                or "client_state" in checkpoint
                or "module" in checkpoint
            ):
                raise ValueError(
                    f"{path} is a DeepSpeed model-state shard, not a portable "
                    "inference checkpoint. Pass its checkpoint directory or "
                    "merge it with scripts/merge_ckpt.py."
                )
            return {
                "state_dict": dict(checkpoint),
                "hyper_parameters": {},
            }
        if not isinstance(state_dict, Mapping):
            raise TypeError("Checkpoint state_dict must be a mapping")
        checkpoint["state_dict"] = dict(state_dict)
        return checkpoint
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checkpoint_root, tag = _resolve_deepspeed_checkpoint_root(path)
    rank_zero_state = _load_torch_mapping(
        _deepspeed_model_state_path(checkpoint_root / tag)
    )
    client_state = rank_zero_state.get("client_state")
    materialized_checkpoint: Dict[str, Any] = {}
    # Lightning's DeepSpeed strategy gives checkpoint hooks the client state.
    # Keep it both flattened and nested so hooks that consume either convention
    # retain their metadata after ZeRO shards are consolidated.
    if client_state is not None:
        materialized_checkpoint["client_state"] = client_state
    if isinstance(client_state, Mapping):
        materialized_checkpoint.update(client_state)
        materialized_checkpoint["client_state"] = dict(client_state)
    materialized_checkpoint.update(
        {
            "state_dict": dict(
                _load_deepspeed_zero_state_dict(checkpoint_root, tag)
            ),
            "hyper_parameters": dict(
                extract_checkpoint_hparams(rank_zero_state)
            ),
        }
    )
    return materialized_checkpoint


def normalize_checkpoint_state_dict(
    state_dict: Mapping[str, Any],
    target_keys: Collection[str],
    *,
    strip_prefixes: Sequence[str] = ("module.",),
    include_only_target_keys: bool = False,
) -> Dict[str, Any]:
    """Normalize distributed state keys and reject collisions deterministically."""
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")
    prefixes = tuple(strip_prefixes)
    if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
        raise ValueError("strip_prefixes must contain only non-empty strings")

    target_key_set = set(target_keys)
    normalized: Dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        if not isinstance(raw_key, str):
            raise TypeError(
                "Checkpoint state_dict keys must be strings, got "
                f"{type(raw_key).__name__}"
            )
        key = raw_key
        while key not in target_key_set:
            prefix = next(
                (candidate for candidate in prefixes if key.startswith(candidate)),
                None,
            )
            if prefix is None:
                break
            key = key[len(prefix) :]
        if include_only_target_keys and key not in target_key_set:
            continue
        if key in normalized:
            raise RuntimeError(
                "Checkpoint state_dict has duplicate keys after removing "
                f"distributed prefixes: {key!r}"
            )
        normalized[key] = value
    return normalized


def _parameter_key(name: str, key_prefix: str) -> str:
    if not key_prefix:
        return name
    return f"{key_prefix.rstrip('.')}.{name}"


def find_missing_trainable_parameter_keys(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    key_prefix: str = "",
) -> Tuple[str, ...]:
    """Return checkpoint keys missing for ``module``'s trainable parameters.

    Frozen parameters and buffers are intentionally ignored. ``key_prefix`` is
    useful when ``state_dict`` belongs to a parent module; both ``"model.head"``
    and ``"model.head."`` are accepted.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"module must be an nn.Module, got {type(module).__name__}"
        )
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"state_dict must be a Mapping, got {type(state_dict).__name__}"
        )
    if not isinstance(key_prefix, str):
        raise TypeError(
            f"key_prefix must be a string, got {type(key_prefix).__name__}"
        )

    return tuple(
        _parameter_key(name, key_prefix)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
        and _parameter_key(name, key_prefix) not in state_dict
    )


def require_complete_trainable_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    key_prefix: str = "",
) -> Tuple[str, ...]:
    """Raise when a checkpoint omits any trainable parameter from ``module``."""
    missing_keys = find_missing_trainable_parameter_keys(
        module,
        state_dict,
        key_prefix=key_prefix,
    )
    if missing_keys:
        formatted_keys = ", ".join(missing_keys)
        raise RuntimeError(
            "Checkpoint state_dict is missing "
            f"{len(missing_keys)} trainable parameter key(s) for "
            f"{type(module).__name__}: {formatted_keys}"
        )
    return missing_keys


def restore_checkpoint_into_module(
    module: Any,
    checkpoint: Mapping[str, Any],
    *,
    strict: bool = True,
    strip_prefixes: Sequence[str] = ("module.",),
) -> Any:
    """Restore a materialized checkpoint through a module's load hook."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Inference checkpoint must be a mapping")
    if not callable(getattr(module, "state_dict", None)) or not callable(
        getattr(module, "load_state_dict", None)
    ):
        raise TypeError(
            "module must provide callable state_dict() and load_state_dict() methods"
        )
    materialized_checkpoint = dict(checkpoint)
    state_dict = materialized_checkpoint.get("state_dict")
    materialized_checkpoint["state_dict"] = normalize_checkpoint_state_dict(
        state_dict,
        module.state_dict().keys(),
        strip_prefixes=strip_prefixes,
    )
    load_hook = getattr(module, "on_load_checkpoint", None)
    if callable(load_hook):
        load_hook(materialized_checkpoint)
    module.load_state_dict(materialized_checkpoint["state_dict"], strict=strict)
    return module


def _module_init_kwargs(
    module_class: Any,
    checkpoint_hparams: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve constructor values from checkpoint hparams and caller overrides."""
    parameters = inspect.signature(module_class.__init__).parameters
    variable_parameter_kinds = (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )
    valid_names = {
        name
        for name, parameter in parameters.items()
        if name != "self" and parameter.kind not in variable_parameter_kinds
    }
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    unknown_overrides = (
        [] if accepts_kwargs else sorted(set(overrides) - valid_names)
    )
    if unknown_overrides:
        raise TypeError(
            "Unknown inference constructor override(s): "
            + ", ".join(unknown_overrides)
        )

    init_kwargs = {
        name: value
        for name, value in checkpoint_hparams.items()
        if name in valid_names or accepts_kwargs
    }
    init_kwargs.update(overrides)
    missing_required = [
        name
        for name, parameter in parameters.items()
        if (
            name != "self"
            and parameter.kind
            not in variable_parameter_kinds
            and parameter.default is inspect.Parameter.empty
            and name not in init_kwargs
        )
    ]
    if missing_required:
        raise ValueError(
            "Checkpoint does not contain enough constructor metadata for "
            "inference. Supply explicit overrides for: "
            + ", ".join(missing_required)
        )
    return init_kwargs


def load_lightning_module_for_inference(
    module_class: Any,
    checkpoint_path: Union[str, Path],
    *,
    init_kwargs: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
    map_location: Any = "cpu",
) -> Any:
    """Load a Lightning module from a file, raw state dict, or DeepSpeed dir.

    Native Lightning files retain Lightning's migration path. Other sources
    are materialized in this utility, then restored through the constructed
    module's existing ``on_load_checkpoint`` hook. ``init_kwargs`` override
    matching checkpoint hparams. They are optional for normal Lightning files
    and DeepSpeed exports that retain all required hparams, but must cover any
    metadata that a raw state-dict file does not carry.
    """
    if init_kwargs is None:
        init_kwargs = {}
    if not isinstance(init_kwargs, Mapping):
        raise TypeError("init_kwargs must be a mapping when provided")
    path = Path(checkpoint_path).expanduser()
    checkpoint = load_checkpoint_for_inference(path)
    if path.is_file() and "pytorch-lightning_version" in checkpoint:
        del checkpoint
        return module_class.load_from_checkpoint(
            str(path),
            strict=strict,
            map_location=map_location,
            **dict(init_kwargs),
        )
    module = module_class(
        **_module_init_kwargs(
            module_class,
            extract_checkpoint_hparams(checkpoint),
            init_kwargs,
        )
    )
    return restore_checkpoint_into_module(module, checkpoint, strict=strict)


def merge_checkpoint(
    checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Save a portable inference checkpoint without dropping VLA metadata."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(load_checkpoint_for_inference(checkpoint_path), temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination
