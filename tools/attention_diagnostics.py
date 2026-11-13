"""Small, model-agnostic primitives for VLA attention diagnostics.

The visualisation command deliberately keeps its model-specific loading and
dataset handling outside this module.  The utilities here operate on token
spans and ordinary PyTorch modules, which makes them useful both for the
InternVL3 adapter and for small unit-test models.

The two capture contexts are temporary: they install forward hooks only for
the duration of a ``with`` block and remove every hook in ``finally``.  LLM
attention is reduced to per-layer/head source-group statistics immediately;
the full decoder attention matrices are never retained by the collector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import floor, isqrt
from typing import Any, Callable, Iterator, Literal, Optional, Union

import torch
import torch.nn as nn


# Do not use ``typing.TypeAlias`` here: the server's Python 3.9 standard
# library predates it, while ordinary alias assignments work identically at
# runtime and remain understood by type checkers.
SpanLike = Union["TokenSpan", Sequence["TokenSpan"], "TokenGroup"]
AttentionGetter = Callable[[Any], Optional[Sequence[torch.Tensor]]]


@dataclass(frozen=True, order=True)
class TokenSpan:
    """A half-open token interval ``[start, end)`` in a decoder sequence."""

    start: int
    end: int

    def __post_init__(self) -> None:
        for field_name, value in (("start", self.start), ("end", self.end)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer, got {value!r}")
        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                "TokenSpan must contain at least one token, got "
                f"[{self.start}, {self.end})"
            )

    @property
    def length(self) -> int:
        """Number of tokens in the span."""
        return self.end - self.start

    def indices(self, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return this span's token positions as a 1-D long tensor."""
        return torch.arange(self.start, self.end, device=device, dtype=torch.long)

    def to_dict(self) -> dict[str, int]:
        """Return JSON-ready span metadata for a run manifest."""
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class TokenGroup:
    """A named, possibly discontiguous collection of token spans."""

    name: str
    spans: tuple[TokenSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("TokenGroup.name must be a non-empty string")
        if not isinstance(self.spans, tuple) or not self.spans:
            raise ValueError("TokenGroup.spans must be a non-empty tuple of TokenSpan")
        if any(not isinstance(span, TokenSpan) for span in self.spans):
            raise TypeError("TokenGroup.spans must contain only TokenSpan values")
        _validate_disjoint_spans(self.spans, label=f"TokenGroup({self.name!r})")

    @property
    def token_count(self) -> int:
        """Total number of tokens represented by this group."""
        return sum(span.length for span in self.spans)

    def indices(self, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return all positions in ascending order."""
        return torch.cat(
            [span.indices(device=device) for span in sorted(self.spans)]
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready group metadata for a run manifest."""
        return {"name": self.name, "spans": [span.to_dict() for span in self.spans]}


@dataclass(frozen=True)
class AttentionSourceGroup:
    """A key source group and its causal visibility rule.

    ``visibility='causal'`` includes keys at or before a query position;
    ``'strictly_previous'`` is intended for the action-query source group, so
    an action query is compared only with earlier action queries.  ``'all'``
    ignores decoder ordering, which can be useful for non-causal toy models.
    """

    group: TokenGroup
    visibility: Literal["causal", "strictly_previous", "all"] = "causal"

    def __post_init__(self) -> None:
        if self.visibility not in {"causal", "strictly_previous", "all"}:
            raise ValueError(
                "visibility must be 'causal', 'strictly_previous', or 'all', got "
                f"{self.visibility!r}"
            )

    @property
    def name(self) -> str:
        return self.group.name


def _validate_disjoint_spans(spans: Sequence[TokenSpan], *, label: str) -> None:
    """Raise a useful error when spans overlap (adjacent spans are allowed)."""
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end:
            raise ValueError(
                f"{label} contains overlapping token spans "
                f"[{previous.start}, {previous.end}) and [{current.start}, {current.end})"
            )


def _coerce_group(name: str, value: SpanLike) -> TokenGroup:
    if isinstance(value, TokenGroup):
        if value.name == name:
            return value
        return TokenGroup(name=name, spans=value.spans)
    if isinstance(value, TokenSpan):
        return TokenGroup(name=name, spans=(value,))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a TokenSpan, TokenGroup, or sequence of TokenSpan values"
        )
    return TokenGroup(name=name, spans=tuple(value))


def _complement_spans(sequence_length: int, occupied: Sequence[TokenSpan]) -> tuple[TokenSpan, ...]:
    """Return the non-empty gaps in ``[0, sequence_length)``."""
    _validate_disjoint_spans(occupied, label="occupied token spans")
    result: list[TokenSpan] = []
    cursor = 0
    for span in sorted(occupied, key=lambda item: item.start):
        if span.end > sequence_length:
            raise ValueError(
                f"token span [{span.start}, {span.end}) exceeds sequence length "
                f"{sequence_length}"
            )
        if cursor < span.start:
            result.append(TokenSpan(cursor, span.start))
        cursor = span.end
    if cursor < sequence_length:
        result.append(TokenSpan(cursor, sequence_length))
    return tuple(result)


@dataclass(frozen=True)
class AttentionTokenLayout:
    """Token groups needed to explain one assembled VLA decoder sequence.

    ``prompt`` should contain only textual/instruction tokens.  With native
    InternVL3, image features replace ``<IMG_CONTEXT>`` positions *inside* the
    prompt, so ``from_visual_spans`` is preferred: it derives prompt spans as
    the complement of the visual and appended-query spans.
    """

    sequence_length: int
    history_frames: tuple[TokenGroup, ...]
    current_frame: TokenGroup
    prompt: TokenGroup
    action_queries: TokenGroup
    mask_query: Optional[TokenGroup] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sequence_length, int)
            or isinstance(self.sequence_length, bool)
            or self.sequence_length <= 0
        ):
            raise ValueError("sequence_length must be a positive integer")
        if not isinstance(self.history_frames, tuple):
            raise TypeError("history_frames must be a tuple of TokenGroup values")
        if any(not isinstance(group, TokenGroup) for group in self.history_frames):
            raise TypeError("history_frames must contain only TokenGroup values")
        for field_name in ("current_frame", "prompt", "action_queries"):
            if not isinstance(getattr(self, field_name), TokenGroup):
                raise TypeError(f"{field_name} must be a TokenGroup")
        if self.mask_query is not None and not isinstance(self.mask_query, TokenGroup):
            raise TypeError("mask_query must be a TokenGroup or None")

        all_groups = (*self.history_frames, self.current_frame, self.prompt)
        if self.mask_query is not None:
            all_groups = (*all_groups, self.mask_query)
        all_groups = (*all_groups, self.action_queries)
        names = [group.name for group in all_groups]
        if len(set(names)) != len(names):
            raise ValueError(f"token group names must be unique, got {names!r}")
        all_spans = tuple(span for group in all_groups for span in group.spans)
        _validate_disjoint_spans(all_spans, label="AttentionTokenLayout")
        for span in all_spans:
            if span.end > self.sequence_length:
                raise ValueError(
                    f"token span [{span.start}, {span.end}) exceeds sequence_length "
                    f"{self.sequence_length}"
                )

    @classmethod
    def from_visual_spans(
        cls,
        *,
        sequence_length: int,
        history_frame_spans: Sequence[SpanLike],
        current_frame_spans: SpanLike,
        action_query_spans: SpanLike,
        mask_query_spans: Optional[SpanLike] = None,
        prompt_spans: Optional[SpanLike] = None,
    ) -> "AttentionTokenLayout":
        """Build a layout from actual image-placeholder and query positions.

        When ``prompt_spans`` is omitted, it is computed from all unoccupied
        sequence positions.  This is the right default for InternVL3, whose
        visual tokens occur within the text prompt rather than in a separate
        visual prefix.
        """
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
            raise TypeError("sequence_length must be an integer")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        history = tuple(
            _coerce_group(f"history_{index:02d}", spans)
            for index, spans in enumerate(history_frame_spans)
        )
        current = _coerce_group("current", current_frame_spans)
        action = _coerce_group("action_queries", action_query_spans)
        mask = (
            None
            if mask_query_spans is None
            else _coerce_group("mask_query", mask_query_spans)
        )
        occupied_groups = (*history, current, action)
        if mask is not None:
            occupied_groups = (*occupied_groups, mask)
        occupied = tuple(span for group in occupied_groups for span in group.spans)
        if prompt_spans is None:
            prompt = TokenGroup("prompt", _complement_spans(sequence_length, occupied))
        else:
            prompt = _coerce_group("prompt", prompt_spans)
        return cls(
            sequence_length=sequence_length,
            history_frames=history,
            current_frame=current,
            prompt=prompt,
            action_queries=action,
            mask_query=mask,
        )

    @classmethod
    def from_frame_token_counts(
        cls,
        *,
        frame_token_counts: Sequence[int],
        prompt_token_count: int,
        action_query_count: int,
        mask_query_count: int = 0,
        visual_start: int = 0,
    ) -> "AttentionTokenLayout":
        """Build a contiguous-layout convenience representation.

        This helper is useful for tests and architectures with a visual prefix.
        Native InternVL3 normally needs :meth:`from_visual_spans` instead.
        """
        if not isinstance(visual_start, int) or isinstance(visual_start, bool):
            raise TypeError("visual_start must be an integer")
        if visual_start < 0:
            raise ValueError("visual_start must be non-negative")
        counts = tuple(frame_token_counts)
        if not counts:
            raise ValueError("frame_token_counts must contain at least a current frame")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in counts
        ):
            raise ValueError("frame_token_counts must contain positive integers")
        for field_name, value in (
            ("prompt_token_count", prompt_token_count),
            ("action_query_count", action_query_count),
            ("mask_query_count", mask_query_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive")
        if action_query_count <= 0:
            raise ValueError("action_query_count must be positive")

        cursor = visual_start
        frame_groups: list[TokenGroup] = []
        for index, count in enumerate(counts):
            span = TokenSpan(cursor, cursor + count)
            cursor += count
            if index < len(counts) - 1:
                frame_groups.append(TokenGroup(f"history_{index:02d}", (span,)))
            else:
                current = TokenGroup("current", (span,))
        prompt = TokenGroup("prompt", (TokenSpan(cursor, cursor + prompt_token_count),))
        cursor += prompt_token_count
        mask: Optional[TokenGroup] = None
        if mask_query_count:
            mask = TokenGroup("mask_query", (TokenSpan(cursor, cursor + mask_query_count),))
            cursor += mask_query_count
        action = TokenGroup(
            "action_queries", (TokenSpan(cursor, cursor + action_query_count),)
        )
        cursor += action_query_count
        return cls(
            sequence_length=cursor,
            history_frames=tuple(frame_groups),
            current_frame=current,
            prompt=prompt,
            action_queries=action,
            mask_query=mask,
        )

    @property
    def source_groups(self) -> tuple[AttentionSourceGroup, ...]:
        """Source groups in the stable report order.

        The action-query source is explicitly strictly prior, preventing an
        action query's self weight or future query positions from being counted
        as action-query evidence.
        """
        groups: list[AttentionSourceGroup] = [
            AttentionSourceGroup(group, "causal") for group in self.history_frames
        ]
        groups.extend(
            (
                AttentionSourceGroup(self.current_frame, "causal"),
                AttentionSourceGroup(self.prompt, "causal"),
            )
        )
        if self.mask_query is not None:
            groups.append(AttentionSourceGroup(self.mask_query, "causal"))
        groups.append(
            AttentionSourceGroup(self.action_queries, "strictly_previous")
        )
        return tuple(groups)

    @property
    def query_groups(self) -> tuple[TokenGroup, ...]:
        """The instruction/prompt and action-query groups used in reports."""
        return (
            TokenGroup("instruction_prompt", self.prompt.spans),
            TokenGroup("action_queries", self.action_queries.spans),
        )

    def visual_spans(
        self,
        scope: Literal["history", "current", "all"] = "all",
    ) -> tuple[TokenSpan, ...]:
        """Return spans to zero for a requested visual ablation scope."""
        if scope == "history":
            return tuple(span for group in self.history_frames for span in group.spans)
        if scope == "current":
            return self.current_frame.spans
        if scope == "all":
            return (*self.visual_spans("history"), *self.current_frame.spans)
        raise ValueError("scope must be 'history', 'current', or 'all'")

    @property
    def frame_groups(self) -> tuple[TokenGroup, ...]:
        """History frames followed by the current frame, in temporal order."""
        return (*self.history_frames, self.current_frame)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready layout metadata for ``run_manifest.json``."""
        return {
            "sequence_length": self.sequence_length,
            "history_frames": [group.to_dict() for group in self.history_frames],
            "current_frame": self.current_frame.to_dict(),
            "prompt": self.prompt.to_dict(),
            "mask_query": None if self.mask_query is None else self.mask_query.to_dict(),
            "action_queries": self.action_queries.to_dict(),
            "source_groups": [
                {"name": source.name, "visibility": source.visibility}
                for source in self.source_groups
            ],
            "query_groups": [group.name for group in self.query_groups],
        }


@dataclass(frozen=True)
class AttentionAggregation:
    """A compact summary of decoder attention by layer, head and token group.

    ``mass`` and ``density`` have shape
    ``[layer, head, query_group, source_group]``.  ``density`` is the mean
    attention mass per *causally available, non-padding* source key, rather
    than simply mass divided by a static group length.
    """

    mass: torch.Tensor
    density: torch.Tensor
    query_group_names: tuple[str, ...]
    source_group_names: tuple[str, ...]
    available_key_counts: torch.Tensor
    valid_query_counts: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.mass, torch.Tensor) or not isinstance(self.density, torch.Tensor):
            raise TypeError("mass and density must be torch.Tensor values")
        if self.mass.ndim != 4 or tuple(self.density.shape) != tuple(self.mass.shape):
            raise ValueError(
                "mass and density must have matching [layer, head, query_group, "
                f"source_group] shapes, got {tuple(self.mass.shape)} and "
                f"{tuple(self.density.shape)}"
            )
        expected = (self.mass.shape[2], self.mass.shape[3])
        if tuple(self.available_key_counts.shape) != expected:
            raise ValueError(
                "available_key_counts must have [query_group, source_group] shape"
            )
        if tuple(self.valid_query_counts.shape) != (self.mass.shape[2],):
            raise ValueError("valid_query_counts must have [query_group] shape")
        if len(self.query_group_names) != self.mass.shape[2]:
            raise ValueError("query_group_names does not match mass")
        if len(self.source_group_names) != self.mass.shape[3]:
            raise ValueError("source_group_names does not match mass")

    @property
    def num_layers(self) -> int:
        return int(self.mass.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.mass.shape[1])


@dataclass(frozen=True)
class LLMFrameAttentionMaps:
    """Compact direct LLM-to-frame attention maps.

    ``token_maps`` stores one tensor per visual frame with shape
    ``[layer, query_group, frame_tokens]``.  No decoder-head or full-sequence
    matrix is retained: heads, valid query tokens, and batch examples have
    already been averaged.  :meth:`spatial_map` reshapes a selected frame to
    ``[layer, query_group, side, side]`` for plotting.
    """

    token_maps: Mapping[str, torch.Tensor]
    query_group_names: tuple[str, ...]
    grid_sizes: Mapping[str, Optional[int]]

    def __post_init__(self) -> None:
        if not isinstance(self.token_maps, Mapping) or not self.token_maps:
            raise ValueError("token_maps must be a non-empty mapping")
        names = tuple(self.token_maps)
        if set(names) != set(self.grid_sizes):
            raise ValueError("grid_sizes must contain exactly one value per token map")
        expected_prefix: Optional[tuple[int, int]] = None
        for name, values in self.token_maps.items():
            if not isinstance(name, str) or not isinstance(values, torch.Tensor):
                raise TypeError("token_maps must map frame names to torch.Tensor values")
            if values.ndim != 3:
                raise ValueError(
                    f"token map for {name!r} must have [layer, query_group, token] "
                    f"shape, got {tuple(values.shape)}"
                )
            prefix = tuple(values.shape[:2])
            if expected_prefix is None:
                expected_prefix = prefix
            elif prefix != expected_prefix:
                raise ValueError("all token maps must share layer and query-group counts")
            side = self.grid_sizes[name]
            if side is None:
                continue
            if (
                not isinstance(side, int)
                or isinstance(side, bool)
                or side <= 0
                or side * side != values.shape[-1]
            ):
                raise ValueError(
                    f"grid size for {name!r} does not match its token count "
                    f"{values.shape[-1]}"
                )
        if expected_prefix is None or expected_prefix[1] != len(self.query_group_names):
            raise ValueError("query_group_names does not match token maps")

    @property
    def frame_names(self) -> tuple[str, ...]:
        """Frame names in the layout's temporal order."""
        return tuple(self.token_maps)

    @property
    def num_layers(self) -> int:
        return int(next(iter(self.token_maps.values())).shape[0])

    def token_map(self, frame_name: str) -> torch.Tensor:
        """Return a frame's compact ``[layer, query_group, token]`` maps."""
        try:
            return self.token_maps[frame_name]
        except KeyError as exc:
            raise KeyError(f"unknown visual frame {frame_name!r}") from exc

    def spatial_map(self, frame_name: str) -> torch.Tensor:
        """Return ``[layer, query_group, side, side]`` for one visual frame."""
        values = self.token_map(frame_name)
        side = self.grid_sizes[frame_name]
        if side is None:
            raise ValueError(
                f"visual frame {frame_name!r} has a non-square token count; use "
                "token_map() for Q-Former attention composition instead"
            )
        return values.reshape(*values.shape[:-1], side, side)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready metadata; tensor values are intentionally omitted."""
        return {
            "frame_names": list(self.frame_names),
            "query_group_names": list(self.query_group_names),
            "grid_sizes": dict(self.grid_sizes),
            "token_map_shapes": {
                name: list(values.shape) for name, values in self.token_maps.items()
            },
        }


def _concatenate_attention_aggregations(
    layers: Sequence[AttentionAggregation],
) -> AttentionAggregation:
    """Join one-layer reductions without ever reassembling a full matrix."""
    if not layers:
        raise ValueError("at least one reduced attention layer is required")
    first = layers[0]
    for item in layers[1:]:
        if (
            item.query_group_names != first.query_group_names
            or item.source_group_names != first.source_group_names
            or not torch.equal(item.available_key_counts, first.available_key_counts)
            or not torch.equal(item.valid_query_counts, first.valid_query_counts)
        ):
            raise ValueError("reduced attention layers use incompatible token layouts")
    return AttentionAggregation(
        mass=torch.cat([item.mass for item in layers], dim=0),
        density=torch.cat([item.density for item in layers], dim=0),
        query_group_names=first.query_group_names,
        source_group_names=first.source_group_names,
        available_key_counts=first.available_key_counts,
        valid_query_counts=first.valid_query_counts,
    )


def _concatenate_frame_attention_maps(
    layers: Sequence[LLMFrameAttentionMaps],
) -> LLMFrameAttentionMaps:
    """Join compact per-layer frame maps in decoder order."""
    if not layers:
        raise ValueError("at least one reduced frame-attention layer is required")
    first = layers[0]
    for item in layers[1:]:
        if (
            item.frame_names != first.frame_names
            or item.query_group_names != first.query_group_names
            or dict(item.grid_sizes) != dict(first.grid_sizes)
        ):
            raise ValueError("reduced frame-attention layers use incompatible token layouts")
    return LLMFrameAttentionMaps(
        token_maps={
            name: torch.cat([item.token_map(name) for item in layers], dim=0)
            for name in first.frame_names
        },
        query_group_names=first.query_group_names,
        grid_sizes=first.grid_sizes,
    )


def _normalize_attention_layers(
    attentions: Union[torch.Tensor, Sequence[Optional[torch.Tensor]]],
) -> tuple[torch.Tensor, ...]:
    if isinstance(attentions, torch.Tensor):
        if attentions.ndim == 4:
            return (attentions,)
        if attentions.ndim == 5:
            return tuple(attentions[index] for index in range(attentions.shape[0]))
        raise ValueError(
            "attention tensor must be [B,H,Q,K] or [L,B,H,Q,K], got "
            f"{tuple(attentions.shape)}"
        )
    if not isinstance(attentions, Sequence) or isinstance(attentions, (str, bytes)):
        raise TypeError("attentions must be a tensor or sequence of attention tensors")
    layers = tuple(item for item in attentions if item is not None)
    if not layers:
        raise ValueError(
            "no attention weights were captured; load the LLM with eager attention "
            "and ensure it supports output_attentions=True"
        )
    if any(not isinstance(item, torch.Tensor) for item in layers):
        raise TypeError("attention sequence must contain torch.Tensor values or None")
    if any(item.ndim != 4 for item in layers):
        shapes = [tuple(item.shape) for item in layers]
        raise ValueError("each attention layer must have [B,H,Q,K] shape, got " f"{shapes}")
    return layers


def _normalize_token_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor or None")
    if tuple(attention_mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            "attention_mask must have [B, sequence_length] shape, got "
            f"{tuple(attention_mask.shape)}"
        )
    return attention_mask.to(device=device, dtype=torch.bool)


def _causal_key_availability(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    visibility: Literal["causal", "strictly_previous", "all"],
) -> torch.Tensor:
    if visibility == "all":
        return torch.ones(
            query_positions.numel(),
            key_positions.numel(),
            dtype=torch.bool,
            device=query_positions.device,
        )
    if visibility == "causal":
        return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return key_positions.unsqueeze(0) < query_positions.unsqueeze(1)


def aggregate_attention_by_source(
    attentions: Union[torch.Tensor, Sequence[Optional[torch.Tensor]]],
    layout: AttentionTokenLayout,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    to_cpu: bool = True,
) -> AttentionAggregation:
    """Reduce decoder attentions to source mass and key-token density.

    The reduction happens layer by layer and only keeps a small
    ``[heads, query_groups, source_groups]`` result for each layer.  A supplied
    ``attention_mask`` excludes padded query tokens and padded source keys.
    Causal source availability is applied explicitly, so density is not biased
    by action-query positions that a given query cannot yet attend to.
    """
    if not isinstance(layout, AttentionTokenLayout):
        raise TypeError("layout must be an AttentionTokenLayout")
    layers = _normalize_attention_layers(attentions)
    first = layers[0]
    batch_size, num_heads, query_length, key_length = first.shape
    if query_length != layout.sequence_length or key_length != layout.sequence_length:
        raise ValueError(
            "attention sequence dimensions must match layout.sequence_length "
            f"({layout.sequence_length}), got Q={query_length}, K={key_length}"
        )
    if any(
        layer.shape[0] != batch_size
        or layer.shape[1] != num_heads
        or layer.shape[2] != query_length
        or layer.shape[3] != key_length
        for layer in layers
    ):
        raise ValueError("all attention layers must share [B,H,Q,K] dimensions")
    valid_tokens = _normalize_token_mask(
        attention_mask,
        batch_size=batch_size,
        sequence_length=layout.sequence_length,
        device=first.device,
    )
    query_groups = layout.query_groups
    source_groups = layout.source_groups
    query_indices = [group.indices(device=first.device) for group in query_groups]
    source_indices = [source.group.indices(device=first.device) for source in source_groups]

    mass_layers: list[torch.Tensor] = []
    density_layers: list[torch.Tensor] = []
    available_key_counts = torch.zeros(
        len(query_groups), len(source_groups), dtype=torch.float32, device=first.device
    )
    valid_query_counts = torch.zeros(
        len(query_groups), dtype=torch.float32, device=first.device
    )

    # Key/query availability depends only on positions and padding, not layer
    # nor head.  Precompute it once before looping through layers.
    pair_data: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    for query_group_index, positions in enumerate(query_indices):
        query_valid = valid_tokens.index_select(1, positions)
        query_weight = query_valid.to(dtype=torch.float32)
        total_queries = query_weight.sum()
        valid_query_counts[query_group_index] = total_queries
        source_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for source_group, keys in zip(source_groups, source_indices):
            source_valid = valid_tokens.index_select(1, keys)
            causal = _causal_key_availability(
                positions, keys, source_group.visibility
            )
            available = source_valid.unsqueeze(1) & causal.unsqueeze(0)
            counts = available.sum(dim=-1).to(dtype=torch.float32)
            if bool(total_queries > 0):
                available_key_counts[query_group_index, len(source_data)] = (
                    (counts * query_weight).sum() / total_queries
                )
            source_data.append((available, counts, query_weight))
        pair_data.append(source_data)

    for layer in layers:
        # Cast only the compact reductions to float32.  This avoids storing a
        # second copy of the large bf16/fp16 decoder attention matrix.
        head_mass = torch.zeros(
            num_heads,
            len(query_groups),
            len(source_groups),
            dtype=torch.float32,
            device=layer.device,
        )
        head_density = torch.zeros_like(head_mass)
        for query_group_index, positions in enumerate(query_indices):
            selected_queries = layer.index_select(2, positions).float()
            query_weight = pair_data[query_group_index][0][2]
            total_queries = query_weight.sum()
            if not bool(total_queries > 0):
                continue
            for source_group_index, keys in enumerate(source_indices):
                available, counts, _ = pair_data[query_group_index][source_group_index]
                selected = selected_queries.index_select(3, keys)
                selected = selected * available[:, None, :, :].to(selected.dtype)
                per_query_mass = selected.sum(dim=-1)
                weighted_mass = per_query_mass * query_weight[:, None, :]
                head_mass[:, query_group_index, source_group_index] = (
                    weighted_mass.sum(dim=(0, 2)) / total_queries
                )
                per_query_density = torch.where(
                    counts[:, None, :] > 0,
                    per_query_mass / counts[:, None, :].to(per_query_mass.dtype),
                    torch.zeros_like(per_query_mass),
                )
                head_density[:, query_group_index, source_group_index] = (
                    (per_query_density * query_weight[:, None, :]).sum(dim=(0, 2))
                    / total_queries
                )
        mass_layers.append(head_mass)
        density_layers.append(head_density)

    mass = torch.stack(mass_layers, dim=0)
    density = torch.stack(density_layers, dim=0)
    if to_cpu:
        mass = mass.detach().cpu()
        density = density.detach().cpu()
        available_key_counts = available_key_counts.detach().cpu()
        valid_query_counts = valid_query_counts.detach().cpu()
    else:
        mass = mass.detach()
        density = density.detach()
        available_key_counts = available_key_counts.detach()
        valid_query_counts = valid_query_counts.detach()
    return AttentionAggregation(
        mass=mass,
        density=density,
        query_group_names=tuple(group.name for group in query_groups),
        source_group_names=tuple(group.name for group in source_groups),
        available_key_counts=available_key_counts,
        valid_query_counts=valid_query_counts,
    )


def aggregate_llm_attention_frame_maps(
    attentions: Union[torch.Tensor, Sequence[Optional[torch.Tensor]]],
    layout: AttentionTokenLayout,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    frame_grid_sizes: Optional[Mapping[str, int]] = None,
    to_cpu: bool = True,
) -> LLMFrameAttentionMaps:
    """Reduce decoder attention into direct visual-token maps for every frame.

    The output contains each frame separately because historical compressed
    frames (often 4x4 query tokens) and the dense current frame (usually 16x16
    patches) need not have the same spatial resolution.  It is therefore a
    more faithful representation than upsampling everything into one tensor.
    Heads, valid query positions, and batch examples are averaged while each
    layer is processed; the original ``[B,H,Q,K]`` matrices are not retained.
    """
    if not isinstance(layout, AttentionTokenLayout):
        raise TypeError("layout must be an AttentionTokenLayout")
    layers = _normalize_attention_layers(attentions)
    first = layers[0]
    batch_size, num_heads, query_length, key_length = first.shape
    if query_length != layout.sequence_length or key_length != layout.sequence_length:
        raise ValueError(
            "attention sequence dimensions must match layout.sequence_length "
            f"({layout.sequence_length}), got Q={query_length}, K={key_length}"
        )
    if any(tuple(layer.shape) != tuple(first.shape) for layer in layers):
        raise ValueError("all attention layers must share [B,H,Q,K] dimensions")
    valid_tokens = _normalize_token_mask(
        attention_mask,
        batch_size=batch_size,
        sequence_length=layout.sequence_length,
        device=first.device,
    )
    query_groups = layout.query_groups
    frame_groups = layout.frame_groups
    query_indices = [group.indices(device=first.device) for group in query_groups]
    frame_indices = [group.indices(device=first.device) for group in frame_groups]

    resolved_grid_sizes: dict[str, Optional[int]] = {}
    for group, indices in zip(frame_groups, frame_indices):
        explicit = None if frame_grid_sizes is None else frame_grid_sizes.get(group.name)
        inferred = isqrt(int(indices.numel()))
        side: Optional[int] = (
            inferred if inferred * inferred == indices.numel() else None
        )
        if explicit is not None:
            side = explicit
        if (
            side is not None
            and (
                not isinstance(side, int)
                or isinstance(side, bool)
                or side <= 0
                or side * side != indices.numel()
            )
        ):
            raise ValueError(
                f"visual frame {group.name!r} has {indices.numel()} tokens; pass a "
                "matching square frame_grid_sizes value"
            )
        resolved_grid_sizes[group.name] = side

    # Keep only [query_group, frame_token] reductions per layer.  This also
    # preserves the individual compressed-query weights needed for Q-Former
    # attention composition via ``token_map(frame).flatten(-1)``.
    per_frame_layers: dict[str, list[torch.Tensor]] = {
        group.name: [] for group in frame_groups
    }
    for layer in layers:
        for group, key_positions in zip(frame_groups, frame_indices):
            frame_query_maps: list[torch.Tensor] = []
            for query_positions in query_indices:
                query_valid = valid_tokens.index_select(1, query_positions)
                query_weight = query_valid.to(dtype=torch.float32)
                denominator = query_weight.sum() * float(num_heads)
                selected = layer.index_select(2, query_positions).index_select(3, key_positions)
                causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                available = (
                    valid_tokens.index_select(1, key_positions).unsqueeze(1)
                    & causal.unsqueeze(0)
                )
                selected = selected.float() * available[:, None, :, :].to(torch.float32)
                if bool(denominator > 0):
                    reduced = (
                        selected * query_weight[:, None, :, None]
                    ).sum(dim=(0, 1, 2)) / denominator
                else:
                    reduced = torch.zeros(
                        key_positions.numel(), dtype=torch.float32, device=layer.device
                    )
                frame_query_maps.append(reduced)
            per_frame_layers[group.name].append(torch.stack(frame_query_maps, dim=0))

    token_maps: dict[str, torch.Tensor] = {}
    for group in frame_groups:
        values = torch.stack(per_frame_layers[group.name], dim=0).detach()
        token_maps[group.name] = values.cpu() if to_cpu else values
    return LLMFrameAttentionMaps(
        token_maps=token_maps,
        query_group_names=tuple(group.name for group in query_groups),
        grid_sizes=resolved_grid_sizes,
    )


def _default_attention_getter(output: Any) -> Optional[Sequence[torch.Tensor]]:
    """Extract a standard Transformers-style ``attentions`` field."""
    if output is None:
        return None
    attentions = getattr(output, "attentions", None)
    if attentions is not None:
        return attentions
    if isinstance(output, Mapping):
        value = output.get("attentions")
        if value is not None:
            return value
    # ``return_dict=False`` commonly places attention layers as the final tuple
    # component.  Only accept a direct tuple/list of 4-D tensors so a KV cache
    # cannot be mistaken for attention weights.
    if isinstance(output, (tuple, list)):
        for item in reversed(output):
            if (
                isinstance(item, (tuple, list))
                and item
                and all(isinstance(value, torch.Tensor) and value.ndim == 4 for value in item)
            ):
                return item
    return None


class LLMAttentionCollector:
    """Capture compact decoder summaries from a language-model forward output.

    Set ``capture_frame_maps=True`` to also retain direct per-frame visual maps
    with shape ``[layer, query_group, frame_tokens]``.  This remains small and
    is sufficient for early/middle/late spatial plots or Q-Former composition.
    """

    def __init__(
        self,
        layout: AttentionTokenLayout,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_getter: AttentionGetter = _default_attention_getter,
        capture_frame_maps: bool = False,
        frame_grid_sizes: Optional[Mapping[str, int]] = None,
    ) -> None:
        if not isinstance(layout, AttentionTokenLayout):
            raise TypeError("layout must be an AttentionTokenLayout")
        if not callable(attention_getter):
            raise TypeError("attention_getter must be callable")
        if not isinstance(capture_frame_maps, bool):
            raise TypeError("capture_frame_maps must be a bool")
        if frame_grid_sizes is not None and not isinstance(frame_grid_sizes, Mapping):
            raise TypeError("frame_grid_sizes must be a mapping or None")
        self.layout = layout
        self.attention_mask = attention_mask
        self.attention_getter = attention_getter
        self.capture_frame_maps = capture_frame_maps
        self.frame_grid_sizes = frame_grid_sizes
        self._captures: list[AttentionAggregation] = []
        self._frame_map_captures: list[LLMFrameAttentionMaps] = []
        self._pending_layer_captures: list[AttentionAggregation] = []
        self._pending_frame_map_captures: list[LLMFrameAttentionMaps] = []

    @property
    def captures(self) -> tuple[AttentionAggregation, ...]:
        """All compact captures in forward-call order."""
        return tuple(self._captures)

    @property
    def latest(self) -> AttentionAggregation:
        """The latest captured summary, or a clear error if none was emitted."""
        if not self._captures:
            raise RuntimeError(
                "no LLM attentions were captured; eager attention may be required"
            )
        return self._captures[-1]

    @property
    def frame_map_captures(self) -> tuple[LLMFrameAttentionMaps, ...]:
        """Optional direct LLM-to-frame maps in forward-call order."""
        return tuple(self._frame_map_captures)

    @property
    def latest_frame_maps(self) -> LLMFrameAttentionMaps:
        """Latest direct maps, when ``capture_frame_maps=True`` was selected."""
        if not self._frame_map_captures:
            raise RuntimeError(
                "no LLM frame maps were captured; construct the collector with "
                "capture_frame_maps=True"
            )
        return self._frame_map_captures[-1]

    def clear(self) -> None:
        """Discard previous compact captures."""
        self._captures.clear()
        self._frame_map_captures.clear()
        self.discard_pending_layer_capture()

    @property
    def pending_layer_count(self) -> int:
        """Number of decoder-layer reductions awaiting a model-forward finish."""
        return len(self._pending_layer_captures)

    def begin_layer_capture(self) -> None:
        """Start collecting one language-model forward layer by layer.

        The temporary hook context calls this before the model enters decoder
        layers.  It deliberately discards a stale partial capture left by an
        interrupted prior forward, while normal completed captures remain
        available through :attr:`captures`.
        """
        self.discard_pending_layer_capture()

    def discard_pending_layer_capture(self) -> None:
        """Drop an incomplete layerwise capture, e.g. after an exception."""
        self._pending_layer_captures.clear()
        self._pending_frame_map_captures.clear()

    def record_layer(self, attention: torch.Tensor) -> AttentionAggregation:
        """Immediately reduce one decoder layer's ``[B,H,Q,K]`` attention.

        Unlike :meth:`record`, this never stores the square matrix.  Call
        :meth:`finalize_layer_capture` after the language-model forward to
        expose one normal multi-layer capture.
        """
        if not isinstance(attention, torch.Tensor) or attention.ndim != 4:
            raise ValueError(
                "decoder-layer attention must have [B,H,Q,K] shape, got "
                f"{getattr(attention, 'shape', None)}"
            )
        aggregation = aggregate_attention_by_source(
            attention,
            self.layout,
            attention_mask=self.attention_mask,
            to_cpu=True,
        )
        self._pending_layer_captures.append(aggregation)
        if self.capture_frame_maps:
            self._pending_frame_map_captures.append(
                aggregate_llm_attention_frame_maps(
                    attention,
                    self.layout,
                    attention_mask=self.attention_mask,
                    frame_grid_sizes=self.frame_grid_sizes,
                    to_cpu=True,
                )
            )
        return aggregation

    def finalize_layer_capture(self) -> AttentionAggregation:
        """Publish the pending one-layer reductions as one multi-layer result."""
        if not self._pending_layer_captures:
            raise RuntimeError("no decoder-layer attention reductions are pending")
        try:
            aggregation = _concatenate_attention_aggregations(
                self._pending_layer_captures
            )
            self._captures.append(aggregation)
            if self.capture_frame_maps:
                if len(self._pending_frame_map_captures) != len(
                    self._pending_layer_captures
                ):
                    raise RuntimeError("layerwise frame-map capture count is inconsistent")
                self._frame_map_captures.append(
                    _concatenate_frame_attention_maps(self._pending_frame_map_captures)
                )
            return aggregation
        finally:
            self.discard_pending_layer_capture()

    def record(self, attentions: Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]) -> AttentionAggregation:
        """Immediately aggregate one decoder forward's attention tuple."""
        aggregation = aggregate_attention_by_source(
            attentions,
            self.layout,
            attention_mask=self.attention_mask,
            to_cpu=True,
        )
        self._captures.append(aggregation)
        if self.capture_frame_maps:
            self._frame_map_captures.append(
                aggregate_llm_attention_frame_maps(
                    attentions,
                    self.layout,
                    attention_mask=self.attention_mask,
                    frame_grid_sizes=self.frame_grid_sizes,
                    to_cpu=True,
                )
            )
        return aggregation

    def record_output(self, output: Any) -> AttentionAggregation:
        """Extract and aggregate an output object produced by the LLM."""
        attentions = self.attention_getter(output)
        if attentions is None:
            raise RuntimeError(
                "LLM output did not contain attentions. Use eager attention and "
                "verify that output_attentions=True is supported."
            )
        return self.record(attentions)


def _register_pre_hook_with_kwargs(
    module: nn.Module,
    callback: Callable[[nn.Module, tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]],
) -> torch.utils.hooks.RemovableHandle:
    """Register a kwargs-aware pre-hook with an actionable Torch-version error."""
    try:
        return module.register_forward_pre_hook(callback, with_kwargs=True)
    except TypeError as exc:  # pragma: no cover - only relevant to very old torch
        raise RuntimeError(
            "attention diagnostics requires PyTorch forward hooks with "
            "with_kwargs=True"
        ) from exc


def _register_forward_hook_with_kwargs(
    module: nn.Module,
    callback: Callable[[nn.Module, tuple[Any, ...], dict[str, Any], Any], Any],
) -> torch.utils.hooks.RemovableHandle:
    try:
        return module.register_forward_hook(callback, with_kwargs=True)
    except TypeError as exc:  # pragma: no cover - only relevant to very old torch
        raise RuntimeError(
            "attention diagnostics requires PyTorch forward hooks with "
            "with_kwargs=True"
        ) from exc


def _pop_decoder_layer_attention(output: Any) -> tuple[Optional[torch.Tensor], Any]:
    """Return a layer's 4-D attention and an output with that tensor removed.

    Current Qwen decoder layers return tuples such as ``(hidden, attention,
    cache)`` when ``output_attentions=True``.  Replacing the attention element
    with ``None`` before the parent model sees it stops Transformers from
    accumulating an all-layer ``[B,H,S,S]`` tuple.  Unknown output structures
    are intentionally left untouched so the language-model-level fallback can
    still capture them normally.
    """
    if isinstance(output, tuple):
        values = list(output)
        result_type = tuple
    elif isinstance(output, list):
        values = list(output)
        result_type = list
    else:
        return None, output
    for index, value in enumerate(values[1:], start=1):
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            values[index] = None
            return value, result_type(values)
    return None, output


def _resolve_decoder_self_attention_modules(
    decoder_layers: Sequence[nn.Module],
) -> tuple[nn.Module, ...]:
    """Find one self-attention module per decoder layer when exposed.

    Transformers 4.57+'s Qwen decoder layer returns only hidden states; its
    ``self_attn`` submodule is the point where eager attention weights still
    exist.  Returning an empty tuple intentionally asks the caller to use the
    older decoder-layer/top-level capture fallback instead of mixing paths.
    """
    candidates: list[nn.Module] = []
    for layer in decoder_layers:
        attention = None
        for attribute in ("self_attn", "self_attention"):
            value = getattr(layer, attribute, None)
            if isinstance(value, nn.Module):
                attention = value
                break
        if attention is None:
            return ()
        candidates.append(attention)
    return tuple(candidates)


def _uses_top_level_attention_recorder(language_model: nn.Module) -> bool:
    """Whether this Transformers model records attentions outside decoder layers.

    Transformers 4.57 introduced ``check_model_inputs``-based output recording
    for Qwen.  Those models advertise the capability through
    ``_can_record_outputs`` (or its public counterpart).  Older Qwen layers
    also have ``self_attn``, but need their top-level ``output_attentions`` flag
    and therefore must use the decoder-layer fallback instead.
    """
    for attribute in ("_can_record_outputs", "can_record_outputs"):
        advertised = getattr(language_model, attribute, None)
        if callable(advertised):
            try:
                advertised = advertised()
            except TypeError:
                continue
        if isinstance(advertised, Mapping):
            return bool(advertised.get("attentions"))
        if isinstance(advertised, (set, frozenset, tuple, list)):
            return "attentions" in advertised
    return False


@contextmanager
def temporary_llm_attention_capture(
    language_model: nn.Module,
    collector: LLMAttentionCollector,
) -> Iterator[LLMAttentionCollector]:
    """Temporarily force and immediately reduce decoder attentions.

    For current Qwen/InternVL3, hooks attach to every decoder ``self_attn``
    submodule: that is where eager weights exist before Transformers' optional
    top-level attention collector can retain all layers.  Older tuple-returning
    decoder layers and unfamiliar models use compatible fallbacks.  In the
    immediate path, the parent receives a ``None`` placeholder for optional
    attention weights while normal hidden-state predictions are unchanged.
    The model must use an implementation that can materialize weights (for
    InternVL3/Qwen, ``attn_implementation='eager'``).
    """
    if not isinstance(language_model, nn.Module):
        raise TypeError("language_model must be an nn.Module")
    if not isinstance(collector, LLMAttentionCollector):
        raise TypeError("collector must be an LLMAttentionCollector")
    try:
        decoder_layers = resolve_decoder_layers(language_model)
    except ValueError:
        # Small toy models and unfamiliar HF topologies retain the compatible
        # language-model output fallback below.
        decoder_layers = ()
    self_attention_modules = (
        _resolve_decoder_self_attention_modules(decoder_layers)
        if _uses_top_level_attention_recorder(language_model)
        else ()
    )
    if self_attention_modules:
        capture_path = "self_attention"
    elif decoder_layers:
        capture_path = "decoder_layer"
    else:
        capture_path = "language_model"

    def force_output_attentions(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # Transformers forwards expose this by keyword.  Keeping the original
        # args untouched avoids duplicate-argument errors for custom toy modules.
        if capture_path != "language_model":
            collector.begin_layer_capture()
        replacement = dict(kwargs)
        # Qwen 4.57+ globally collects layer weights only when this is true.
        # Eager ``self_attn`` itself still returns its per-layer weights, so
        # leave the collector disabled and intercept them before accumulation.
        replacement["output_attentions"] = capture_path != "self_attention"
        return args, replacement

    def collect_decoder_layer_output(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        attention, stripped_output = _pop_decoder_layer_attention(output)
        if attention is None:
            return None
        collector.record_layer(attention)
        return stripped_output

    def collect_output(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        captured_layers = collector.pending_layer_count
        expected_layers = (
            len(self_attention_modules)
            if capture_path == "self_attention"
            else len(decoder_layers)
        )
        if capture_path == "language_model":
            collector.record_output(output)
        elif captured_layers == expected_layers:
            collector.finalize_layer_capture()
        elif captured_layers == 0 and capture_path == "decoder_layer":
            # Older/unknown layers may still expose a regular top-level tuple.
            collector.record_output(output)
        else:
            collector.discard_pending_layer_capture()
            if capture_path == "self_attention" and captured_layers == 0:
                raise RuntimeError(
                    "decoder self-attention modules did not expose per-head "
                    "weights. Reload the diagnostic model with eager attention."
                )
            raise RuntimeError(
                "only a subset of decoder layers exposed attention tensors "
                f"({captured_layers}/{expected_layers}); cannot aggregate reliably"
            )
        return None

    handles = [
        _register_pre_hook_with_kwargs(language_model, force_output_attentions),
    ]
    capture_modules = (
        self_attention_modules
        if capture_path == "self_attention"
        else decoder_layers
    )
    handles.extend(
        _register_forward_hook_with_kwargs(module, collect_decoder_layer_output)
        for module in capture_modules
    )
    handles.append(_register_forward_hook_with_kwargs(language_model, collect_output))
    try:
        yield collector
    finally:
        collector.discard_pending_layer_capture()
        for handle in reversed(handles):
            handle.remove()


def reshape_qformer_patch_attention(
    attention_weights: torch.Tensor,
    *,
    grid_size: Optional[int] = None,
) -> torch.Tensor:
    """Reshape Q-Former full-feature weights' final patch axis into a grid.

    Input is normally ``[..., heads, queries, patches]`` (or a subset thereof),
    and output is ``[..., heads, queries, side, side]``.  ``patches`` must be a
    square count; with InternVL3 full visual features it is 256 and ``side=16``.
    """
    if not isinstance(attention_weights, torch.Tensor):
        raise TypeError("attention_weights must be a torch.Tensor")
    if attention_weights.ndim < 2:
        raise ValueError(
            "attention_weights must end in [queries, patches], got "
            f"{tuple(attention_weights.shape)}"
        )
    num_patches = int(attention_weights.shape[-1])
    inferred = isqrt(num_patches)
    if grid_size is None:
        grid_size = inferred
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"patch count {num_patches} cannot be reshaped to {grid_size}x{grid_size}"
        )
    return attention_weights.reshape(*attention_weights.shape[:-1], grid_size, grid_size)


def compose_llm_qformer_attention(
    llm_to_compressed_queries: torch.Tensor,
    qformer_patch_attention: torch.Tensor,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Compose LLM-to-query weights with Q-Former query-to-patch weights.

    ``llm_to_compressed_queries`` has shape ``[..., queries]`` and
    ``qformer_patch_attention`` has shape ``[..., queries, height, width]``.
    Leading dimensions follow ordinary PyTorch broadcasting rules.  For the
    common compact per-history-frame case, use
    :func:`compose_llm_qformer_frame_maps`, which selects the final Q-Former
    layer and aligns frame names explicitly.  The result is an attention
    *composition*, not a causal attribution measure.
    """
    if not isinstance(llm_to_compressed_queries, torch.Tensor):
        raise TypeError("llm_to_compressed_queries must be a torch.Tensor")
    if not isinstance(qformer_patch_attention, torch.Tensor):
        raise TypeError("qformer_patch_attention must be a torch.Tensor")
    if llm_to_compressed_queries.ndim < 1 or qformer_patch_attention.ndim < 3:
        raise ValueError(
            "expected llm weights [..., queries] and Q-Former maps "
            "[..., queries, height, width]"
        )
    if llm_to_compressed_queries.shape[-1] != qformer_patch_attention.shape[-3]:
        raise ValueError(
            "LLM compressed-query count must match Q-Former query count, got "
            f"{llm_to_compressed_queries.shape[-1]} and "
            f"{qformer_patch_attention.shape[-3]}"
        )
    weights = llm_to_compressed_queries.to(
        device=qformer_patch_attention.device,
        dtype=qformer_patch_attention.dtype,
    )
    result = (weights.unsqueeze(-1).unsqueeze(-1) * qformer_patch_attention).sum(dim=-3)
    if not normalize:
        return result
    denominator = result.sum(dim=(-2, -1), keepdim=True)
    return torch.where(denominator > 0, result / denominator, torch.zeros_like(result))


def compose_llm_qformer_frame_maps(
    llm_frame_maps: LLMFrameAttentionMaps,
    qformer_patch_maps: torch.Tensor,
    *,
    history_frame_names: Optional[Sequence[str]] = None,
    qformer_layer: int = -1,
    batch_index: int = 0,
    reduce_qformer_heads: bool = True,
    normalize: bool = False,
) -> dict[str, torch.Tensor]:
    """Compose LLM compressed-query weights with final Q-Former patch maps.

    Args:
        llm_frame_maps: Direct compact maps captured by
            :class:`LLMAttentionCollector` for the same model forward.
        qformer_patch_maps: :meth:`QFormerAttentionCollector.patch_maps`
            output with shape ``[B, history_frame, qformer_layer, head, query,
            side, side]``.
        history_frame_names: Layout frame names corresponding to the Q-Former
            frame axis.  Defaults to ``history_`` entries in temporal order.
        qformer_layer: Compressor layer to use; ``-1`` selects the final layer.
        reduce_qformer_heads: Mean Q-Former heads by default.  If false, each
            output includes a head axis after ``[LLM layer, query group]``.

    Returns:
        A map per history frame.  With the default head reduction each value is
        ``[LLM layer, query group, 16, 16]`` (or the checkpoint's patch grid).
        These are attention compositions, not causal attributions.
    """
    if not isinstance(llm_frame_maps, LLMFrameAttentionMaps):
        raise TypeError("llm_frame_maps must be an LLMFrameAttentionMaps")
    if not isinstance(qformer_patch_maps, torch.Tensor) or qformer_patch_maps.ndim != 7:
        raise ValueError(
            "qformer_patch_maps must have [B,F,L,H,Q,S,S] shape, got "
            f"{getattr(qformer_patch_maps, 'shape', None)}"
        )
    batch_size, num_frames, num_layers, _num_heads, num_queries, height, width = (
        qformer_patch_maps.shape
    )
    if not isinstance(batch_index, int) or isinstance(batch_index, bool):
        raise TypeError("batch_index must be an integer")
    if not 0 <= batch_index < batch_size:
        raise ValueError(f"batch_index must be in [0, {batch_size}), got {batch_index}")
    if not isinstance(qformer_layer, int) or isinstance(qformer_layer, bool):
        raise TypeError("qformer_layer must be an integer")
    if not -num_layers <= qformer_layer < num_layers:
        raise ValueError(
            f"qformer_layer must be in [{-num_layers}, {num_layers}), got {qformer_layer}"
        )
    if history_frame_names is None:
        history_frame_names = tuple(
            name for name in llm_frame_maps.frame_names if name.startswith("history_")
        )
    else:
        history_frame_names = tuple(history_frame_names)
    if len(history_frame_names) != num_frames:
        raise ValueError(
            "history_frame_names must match Q-Former frame count, got "
            f"{len(history_frame_names)} and {num_frames}"
        )
    if len(set(history_frame_names)) != len(history_frame_names):
        raise ValueError("history_frame_names must be unique")

    selected = qformer_patch_maps[batch_index, :, qformer_layer]
    # [F,H,Q,S,S] -> [F,Q,S,S] by default.  Keeping head is useful for a
    # specialist report and broadcasts safely against [LLM layer,QG,Q].
    if reduce_qformer_heads:
        selected = selected.mean(dim=1)
    result: dict[str, torch.Tensor] = {}
    for frame_index, name in enumerate(history_frame_names):
        llm_weights = llm_frame_maps.token_map(name)
        if llm_weights.shape[-1] != num_queries:
            raise ValueError(
                f"LLM map for {name!r} has {llm_weights.shape[-1]} tokens but "
                f"Q-Former has {num_queries} queries"
            )
        if not reduce_qformer_heads:
            # [LLM layer,QG,Q] -> [LLM layer,QG,QFormer head,Q] so the
            # Q-Former head is a true broadcast axis rather than colliding
            # with the query-group dimension.
            llm_weights = llm_weights.unsqueeze(-2)
        result[name] = compose_llm_qformer_attention(
            llm_weights,
            selected[frame_index],
            normalize=normalize,
        )
    return result


class QFormerAttentionCollector:
    """Store full-feature Q-Former weights compactly enough for spatial maps.

    The collector stores detached float32 CPU tensors only for ``full_attention``
    modules.  One tensor has shape ``[B * frames, heads, queries, patches]``;
    :meth:`patch_maps` turns a complete capture into
    ``[B, frames, layer, head, query, side, side]``.
    """

    def __init__(self) -> None:
        self._weights: dict[str, list[torch.Tensor]] = {}
        self._module_order: list[str] = []

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(self._module_order)

    @property
    def captures(self) -> Mapping[str, tuple[torch.Tensor, ...]]:
        return {name: tuple(values) for name, values in self._weights.items()}

    def clear(self) -> None:
        self._weights.clear()
        self._module_order.clear()

    def record(self, module_name: str, weights: torch.Tensor) -> None:
        """Record one per-head ``nn.MultiheadAttention`` weight tensor."""
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("module_name must be a non-empty string")
        if not isinstance(weights, torch.Tensor):
            raise TypeError("Q-Former attention weights must be a torch.Tensor")
        if weights.ndim != 4:
            raise ValueError(
                "Q-Former full-attention weights must have [B*F,H,Q,K] shape; "
                f"got {tuple(weights.shape)}. Ensure average_attn_weights=False."
            )
        if weights.shape[-1] <= 0 or weights.shape[-2] <= 0:
            raise ValueError("Q-Former attention weights must have non-empty Q and K")
        if module_name not in self._weights:
            self._weights[module_name] = []
            self._module_order.append(module_name)
        self._weights[module_name].append(weights.detach().float().cpu())

    def latest_weights(self, module_name: str) -> torch.Tensor:
        """Return a module's latest forward-call weights."""
        try:
            values = self._weights[module_name]
        except KeyError as exc:
            raise KeyError(f"no Q-Former attention was captured for {module_name!r}") from exc
        if not values:
            raise RuntimeError(f"no Q-Former attention was captured for {module_name!r}")
        return values[-1]

    def patch_maps(
        self,
        *,
        num_frames: int,
        grid_size: Optional[int] = None,
        call_index: int = -1,
    ) -> torch.Tensor:
        """Return latest/corresponding maps as ``[B,F,L,H,Q,S,S]``.

        ``num_frames`` is the number of history frames passed through the
        compressor.  The first dimension in each captured attention tensor is
        the flattened ``B * F`` dimension used by ``HistoryTokenCompressor``.
        """
        if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        if not self._module_order:
            raise RuntimeError("no Q-Former full-attention modules were captured")
        per_layer: list[torch.Tensor] = []
        expected_shape: Optional[tuple[int, int, int, int]] = None
        for name in self._module_order:
            values = self._weights[name]
            try:
                weights = values[call_index]
            except IndexError as exc:
                raise IndexError(
                    f"module {name!r} has {len(values)} captures; call_index={call_index} is invalid"
                ) from exc
            if weights.shape[0] % num_frames != 0:
                raise ValueError(
                    f"captured batch {weights.shape[0]} for {name!r} is not divisible "
                    f"by num_frames={num_frames}"
                )
            batch_size = weights.shape[0] // num_frames
            trailing = tuple(int(value) for value in weights.shape[1:])
            if expected_shape is None:
                expected_shape = (batch_size, *trailing)
            elif expected_shape != (batch_size, *trailing):
                raise ValueError(
                    "all Q-Former full-attention layers must share B, head, query, "
                    "and patch dimensions"
                )
            maps = reshape_qformer_patch_attention(weights, grid_size=grid_size)
            per_layer.append(maps.reshape(batch_size, num_frames, *maps.shape[1:]))
        return torch.stack(per_layer, dim=2)


def _is_full_attention_module(name: str, module: nn.Module) -> bool:
    """Recognize both current and earlier loadable Q-Former topologies."""
    if not isinstance(module, nn.MultiheadAttention):
        return False
    return (
        name == "full_attention"
        or name.endswith(".full_attention")
        or name == "full_layer.cross_attention"
        or name.endswith(".full_layer.cross_attention")
    )


def discover_qformer_full_attention_modules(
    compressor: nn.Module,
) -> tuple[tuple[str, nn.MultiheadAttention], ...]:
    """Find supported full-feature Q-Former attention modules under a compressor."""
    if not isinstance(compressor, nn.Module):
        raise TypeError("compressor must be an nn.Module")
    found = tuple(
        (name, module)
        for name, module in compressor.named_modules()
        if _is_full_attention_module(name, module)
    )
    return found


@contextmanager
def temporary_qformer_attention_capture(
    compressor: nn.Module,
    collector: QFormerAttentionCollector,
) -> Iterator[QFormerAttentionCollector]:
    """Capture per-head full-feature Q-Former cross-attention temporarily.

    Both current ``layers.*.full_attention`` and an earlier loadable topology's
    ``qformer_layers.*.full_layer.cross_attention`` modules are supported.
    No model weights or module methods are modified.
    """
    if not isinstance(compressor, nn.Module):
        raise TypeError("compressor must be an nn.Module")
    if not isinstance(collector, QFormerAttentionCollector):
        raise TypeError("collector must be a QFormerAttentionCollector")
    modules = discover_qformer_full_attention_modules(compressor)
    if not modules:
        raise ValueError(
            "no supported Q-Former full-attention modules found; this is expected "
            "for adaptive-pool or disabled compression"
        )

    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for name, module in modules:
            def force_per_head_weights(
                _module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                replacement_args = list(args)
                replacement_kwargs = dict(kwargs)
                # MultiheadAttention's fifth and seventh parameters are
                # need_weights and average_attn_weights, respectively.
                if len(replacement_args) > 4:
                    replacement_args[4] = True
                    replacement_kwargs.pop("need_weights", None)
                else:
                    replacement_kwargs["need_weights"] = True
                if len(replacement_args) > 6:
                    replacement_args[6] = False
                    replacement_kwargs.pop("average_attn_weights", None)
                else:
                    replacement_kwargs["average_attn_weights"] = False
                return tuple(replacement_args), replacement_kwargs

            def collect_weights(
                _module: nn.Module,
                _args: tuple[Any, ...],
                _kwargs: dict[str, Any],
                output: Any,
                *,
                module_name: str = name,
            ) -> None:
                if not isinstance(output, tuple) or len(output) < 2:
                    raise RuntimeError(
                        f"Q-Former module {module_name!r} did not return "
                        "(attn_output, attn_weights)"
                    )
                weights = output[1]
                if weights is None:
                    raise RuntimeError(
                        f"Q-Former module {module_name!r} returned no attention "
                        "weights despite need_weights=True"
                    )
                collector.record(module_name, weights)
                return None

            handles.append(_register_pre_hook_with_kwargs(module, force_per_head_weights))
            handles.append(_register_forward_hook_with_kwargs(module, collect_weights))
        yield collector
    finally:
        for handle in reversed(handles):
            handle.remove()


def resolve_decoder_layers(language_model: nn.Module) -> tuple[nn.Module, ...]:
    """Locate a common Hugging Face decoder ``ModuleList`` without hard-coding Qwen."""
    if not isinstance(language_model, nn.Module):
        raise TypeError("language_model must be an nn.Module")
    candidates = (
        ("model", "layers"),
        ("layers",),
        ("decoder", "layers"),
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        value: Any = language_model
        for part in path:
            value = getattr(value, part, None)
            if value is None:
                break
        if isinstance(value, (nn.ModuleList, list, tuple)) and value and all(
            isinstance(layer, nn.Module) for layer in value
        ):
            return tuple(value)
    raise ValueError(
        "could not locate decoder layers. Pass language_model.model.layers (or an "
        "explicit sequence of decoder layers) to the ablation context."
    )


def _coerce_decoder_layers(
    decoder_layers: Union[nn.Module, Sequence[nn.Module]],
) -> tuple[nn.Module, ...]:
    # ModuleList is both an nn.Module and a Sequence.  Treat it as the explicit
    # layer collection users meant to pass, rather than trying to rediscover a
    # nested ``.model.layers`` beneath it.
    if isinstance(decoder_layers, (nn.ModuleList, list, tuple)):
        layers = tuple(decoder_layers)
        if not layers or any(not isinstance(layer, nn.Module) for layer in layers):
            raise ValueError("decoder_layers must be a non-empty sequence of nn.Module")
        return layers
    if isinstance(decoder_layers, nn.Module):
        return resolve_decoder_layers(decoder_layers)
    if not isinstance(decoder_layers, Sequence) or isinstance(decoder_layers, (str, bytes)):
        raise TypeError("decoder_layers must be an nn.Module or sequence of nn.Module")
    layers = tuple(decoder_layers)
    if not layers or any(not isinstance(layer, nn.Module) for layer in layers):
        raise ValueError("decoder_layers must be a non-empty sequence of nn.Module")
    return layers


def layer_cutoff_indices(
    num_layers: int,
    fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[int, ...]:
    """Convert depth fractions to deterministic decoder-layer cutoff indices.

    A cutoff of zero ablates visual hidden states before every decoder layer;
    a cutoff equal to ``num_layers`` installs no hooks and is an exact baseline.
    Fractions are floored, matching the intuitive "from this layer onward"
    semantics and avoiding a cutoff beyond the model depth.
    """
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    result: list[int] = []
    for fraction in fractions:
        if not isinstance(fraction, (float, int)) or isinstance(fraction, bool):
            raise TypeError("cutoff fractions must be numbers")
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError("cutoff fractions must be in [0, 1]")
        index = floor(value * num_layers)
        if index not in result:
            result.append(index)
    return tuple(result)


def zero_visual_hidden_states(
    hidden_states: torch.Tensor,
    visual_spans: Sequence[TokenSpan],
) -> torch.Tensor:
    """Return a clone with selected visual sequence positions zeroed.

    The input is never mutated, which is important because decoder layers can
    retain references to their input during a forward pass.
    """
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor")
    if hidden_states.ndim not in (2, 3):
        raise ValueError(
            "hidden_states must be [S,D] or [B,S,D], got "
            f"{tuple(hidden_states.shape)}"
        )
    spans = tuple(visual_spans)
    if not spans or any(not isinstance(span, TokenSpan) for span in spans):
        raise ValueError("visual_spans must be a non-empty sequence of TokenSpan")
    _validate_disjoint_spans(spans, label="visual_spans")
    sequence_length = int(hidden_states.shape[-2])
    if any(span.end > sequence_length for span in spans):
        raise ValueError(
            f"visual span exceeds hidden-state sequence length {sequence_length}"
        )
    result = hidden_states.clone()
    for span in spans:
        result[..., span.start : span.end, :] = 0
    return result


@contextmanager
def temporary_visual_token_ablation(
    decoder_layers: Union[nn.Module, Sequence[nn.Module]],
    visual_spans: Sequence[TokenSpan],
    *,
    cutoff_layer: int,
) -> Iterator[None]:
    """Zero selected visual token states at each decoder layer from a cutoff.

    ``cutoff_layer`` is zero-based.  All layers with index greater than or
    equal to it receive a cloned, zeroed hidden-state input.  Passing exactly
    the number of layers deliberately installs no hooks, making the 100% depth
    intervention byte-for-byte equivalent to the baseline forward path.
    """
    layers = _coerce_decoder_layers(decoder_layers)
    if (
        not isinstance(cutoff_layer, int)
        or isinstance(cutoff_layer, bool)
        or not 0 <= cutoff_layer <= len(layers)
    ):
        raise ValueError(
            f"cutoff_layer must be an integer in [0, {len(layers)}], got "
            f"{cutoff_layer!r}"
        )
    spans = tuple(visual_spans)
    if not spans or any(not isinstance(span, TokenSpan) for span in spans):
        raise ValueError("visual_spans must be a non-empty sequence of TokenSpan")
    _validate_disjoint_spans(spans, label="visual_spans")

    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for layer in layers[cutoff_layer:]:
            def zero_visual_input(
                _module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                replacement_args = list(args)
                replacement_kwargs = dict(kwargs)
                if replacement_args and isinstance(replacement_args[0], torch.Tensor):
                    replacement_args[0] = zero_visual_hidden_states(
                        replacement_args[0], spans
                    )
                elif isinstance(replacement_kwargs.get("hidden_states"), torch.Tensor):
                    replacement_kwargs["hidden_states"] = zero_visual_hidden_states(
                        replacement_kwargs["hidden_states"], spans
                    )
                else:
                    raise RuntimeError(
                        "decoder layer did not receive hidden_states as its first "
                        "argument or as a hidden_states keyword"
                    )
                return tuple(replacement_args), replacement_kwargs

            handles.append(_register_pre_hook_with_kwargs(layer, zero_visual_input))
        yield None
    finally:
        for handle in reversed(handles):
            handle.remove()


__all__ = [
    "AttentionAggregation",
    "AttentionSourceGroup",
    "AttentionTokenLayout",
    "LLMFrameAttentionMaps",
    "LLMAttentionCollector",
    "QFormerAttentionCollector",
    "TokenGroup",
    "TokenSpan",
    "aggregate_attention_by_source",
    "aggregate_llm_attention_frame_maps",
    "compose_llm_qformer_attention",
    "compose_llm_qformer_frame_maps",
    "discover_qformer_full_attention_modules",
    "layer_cutoff_indices",
    "reshape_qformer_patch_attention",
    "resolve_decoder_layers",
    "temporary_llm_attention_capture",
    "temporary_qformer_attention_capture",
    "temporary_visual_token_ablation",
    "zero_visual_hidden_states",
]
