"""Validate that a state dict contains every trainable module parameter."""

from collections.abc import Mapping
from typing import Any, Tuple

from torch import nn


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
    """Return state-dict keys missing for ``module``'s trainable parameters.

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
    """Raise when a checkpoint omits any trainable parameter from ``module``.

    The empty tuple is returned on success so callers can use the same return
    shape as :func:`find_missing_trainable_parameter_keys`.
    """
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


__all__ = [
    "find_missing_trainable_parameter_keys",
    "require_complete_trainable_state_dict",
]
