"""Small, dependency-light plotting helpers for TrackVLA attention diagnostics.

The functions in this module deliberately operate on NumPy arrays, mappings, and
PIL/NumPy images only.  Keeping model and dataset code out of this module makes
the diagnostic runner responsible for token bookkeeping while keeping plotting
usable in tests and offline notebooks.

All writers create their parent directory and return the resolved output path.
Matplotlib is configured with its non-interactive ``Agg`` backend before pyplot
is imported, so the helpers are safe on a headless training machine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Union

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image


PathLike = Union[str, Path]
ArrayLike = Any


def normalize_spatial_map(attention_map: ArrayLike) -> np.ndarray:
    """Return a finite two-dimensional attention map scaled into ``[0, 1]``.

    NaN and infinite values are treated as zero.  Constant and entirely invalid
    maps become all-zero maps, which keeps visualizers stable for masked tokens
    and degenerate attention rows.
    """

    values = np.asarray(attention_map, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            "attention_map must have shape [height, width], "
            f"got {values.shape}."
        )
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.float32)
    clean = np.where(finite, values, 0.0)
    minimum = float(np.min(clean[finite]))
    maximum = float(np.max(clean[finite]))
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.float32)
    normalized = (clean - minimum) / (maximum - minimum)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def compose_query_patch_attention(
    query_weights: ArrayLike,
    query_patch_attention: ArrayLike,
    *,
    normalize_weights: bool = False,
) -> np.ndarray:
    """Compose compressed-query weights with per-query patch attention.

    Args:
        query_weights: Array ending in ``[num_queries]``.  Any preceding
            dimensions (for example layer and head) are broadcast with the
            corresponding leading dimensions of ``query_patch_attention``.
        query_patch_attention: Array ending in
            ``[num_queries, patch_height, patch_width]``.
        normalize_weights: Normalize each query-weight vector before combining.
            The default preserves LLM source mass; enable it when only a
            within-frame spatial distribution is wanted.

    Returns:
        An array ending in ``[patch_height, patch_width]`` with broadcast leading
        dimensions.  It is intentionally not image-normalized; callers can use
        :func:`normalize_spatial_map` for display.
    """

    weights = np.asarray(query_weights, dtype=np.float64)
    patches = np.asarray(query_patch_attention, dtype=np.float64)
    if weights.ndim < 1:
        raise ValueError("query_weights must end in a query dimension.")
    if patches.ndim < 3:
        raise ValueError(
            "query_patch_attention must end in [queries, height, width], "
            f"got {patches.shape}."
        )
    num_queries = patches.shape[-3]
    if weights.shape[-1] != num_queries:
        raise ValueError(
            "Query dimensions do not match: query_weights has "
            f"{weights.shape[-1]}, query_patch_attention has {num_queries}."
        )

    try:
        leading_shape = np.broadcast_shapes(weights.shape[:-1], patches.shape[:-3])
    except ValueError as error:
        raise ValueError(
            "Leading dimensions of query_weights and query_patch_attention "
            f"cannot broadcast: {weights.shape[:-1]} vs {patches.shape[:-3]}."
        ) from error

    broadcast_weights = np.broadcast_to(weights, leading_shape + (num_queries,))
    broadcast_patches = np.broadcast_to(patches, leading_shape + patches.shape[-3:])
    broadcast_weights = np.where(np.isfinite(broadcast_weights), broadcast_weights, 0.0)
    broadcast_patches = np.where(np.isfinite(broadcast_patches), broadcast_patches, 0.0)
    if normalize_weights:
        denominator = np.sum(broadcast_weights, axis=-1, keepdims=True)
        broadcast_weights = np.divide(
            broadcast_weights,
            denominator,
            out=np.zeros_like(broadcast_weights),
            where=np.abs(denominator) > np.finfo(np.float64).eps,
        )
    return np.sum(broadcast_weights[..., :, None, None] * broadcast_patches, axis=-3)


def overlay_attention_on_image(
    image: Image.Image | ArrayLike,
    attention_map: ArrayLike,
    *,
    alpha: float = 0.55,
    cmap: str = "magma",
) -> np.ndarray:
    """Blend an attention map onto an RGB image and return a uint8 RGB array."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
    rgb = _coerce_rgb_image(image)
    normalized = normalize_spatial_map(attention_map)
    resized = _resize_map(normalized, (rgb.shape[1], rgb.shape[0]))
    colors = colormaps.get_cmap(cmap)(resized)[..., :3]
    # Do not tint map regions with no relative attention.
    per_pixel_alpha = alpha * resized[..., None]
    blended = rgb.astype(np.float32) / 255.0
    blended = blended * (1.0 - per_pixel_alpha) + colors * per_pixel_alpha
    return np.clip(np.rint(blended * 255.0), 0, 255).astype(np.uint8)


def render_attention_overview(
    output_path: PathLike,
    attention_mass: Mapping[str, ArrayLike],
    *,
    source_labels: Sequence[str],
    attention_density: Mapping[str, ArrayLike] | None = None,
    title: str | None = None,
    cmap: str = "magma",
) -> Path:
    """Render source-mass trends and layer/head attention heatmaps.

    ``attention_mass`` maps each query group (usually ``"instruction"`` and
    ``"action_query"``) to a ``[layers, heads, sources]`` array.  ``source_labels``
    names its final dimension.  If supplied, ``attention_density`` has the same
    layout and is rendered beside raw mass to account for unequal token counts.
    """

    mass = _validate_group_attention(attention_mass, source_labels, "attention_mass")
    density: dict[str, np.ndarray] | None = None
    if attention_density is not None:
        density = _validate_group_attention(
            attention_density,
            source_labels,
            "attention_density",
            expected_groups=mass,
        )

    group_items = list(mass.items())
    num_columns = 3 if density is not None else 2
    figure, axes = plt.subplots(
        len(group_items),
        num_columns,
        figsize=(6.2 * num_columns, 4.4 * len(group_items)),
        squeeze=False,
        layout="constrained",
    )
    if title:
        figure.suptitle(title, fontsize=14)

    for row, (group_name, values) in enumerate(group_items):
        line_axis = axes[row, 0]
        _draw_source_trends(line_axis, values, source_labels)
        line_axis.set_title(f"{group_name}: source mass")

        _draw_layer_head_heatmap(
            axes[row, 1],
            values,
            source_labels,
            title=f"{group_name}: mass by layer/head",
            cmap=cmap,
        )
        if density is not None:
            _draw_layer_head_heatmap(
                axes[row, 2],
                density[group_name],
                source_labels,
                title=f"{group_name}: mass per key token",
                cmap=cmap,
            )

    return _save_figure(figure, output_path)


def render_spatial_attention_overlays(
    output_path: PathLike,
    image: Image.Image | ArrayLike,
    attention_maps: Mapping[str, ArrayLike] | ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    alpha: float = 0.55,
    cmap: str = "magma",
    columns: int = 4,
) -> Path:
    """Render a source RGB image plus a grid of spatial-attention overlays.

    ``attention_maps`` is either a mapping of labels to ``[height, width]`` maps,
    or an array shaped ``[num_maps, height, width]`` (a single ``[height, width]``
    map is accepted as well).  This is suited to early/middle/late LLM spatial
    maps for one current or historical frame.
    """

    if columns < 1:
        raise ValueError("columns must be at least one.")
    rgb = _coerce_rgb_image(image)
    map_labels, maps = _coerce_named_maps(attention_maps, labels=labels)
    rendered = [
        overlay_attention_on_image(rgb, values, alpha=alpha, cmap=cmap)
        for values in maps
    ]
    panel_count = len(rendered) + 1
    rows = int(np.ceil(panel_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.4 * rows),
        squeeze=False,
        layout="constrained",
    )
    if title:
        figure.suptitle(title, fontsize=14)
    flattened_axes = axes.ravel()
    flattened_axes[0].imshow(rgb)
    flattened_axes[0].set_title("RGB input")
    flattened_axes[0].axis("off")
    for axis, label, overlay in zip(flattened_axes[1:], map_labels, rendered):
        axis.imshow(overlay)
        axis.set_title(label)
        axis.axis("off")
    for axis in flattened_axes[panel_count:]:
        axis.axis("off")
    return _save_figure(figure, output_path)


def render_qformer_attention(
    output_path: PathLike,
    image: Image.Image | ArrayLike,
    query_attention: ArrayLike,
    *,
    query_labels: Sequence[str] | None = None,
    title: str | None = None,
    alpha: float = 0.55,
    cmap: str = "magma",
    columns: int = 5,
) -> Path:
    """Render per-query Q-Former patch maps and their query mean.

    The expected array ends in ``[queries, patch_height, patch_width]``.  Leading
    dimensions such as ``[layers, heads]`` are averaged, which allows a captured
    ``[layers, heads, queries, 16, 16]`` tensor to be rendered directly.
    """

    values = np.asarray(query_attention, dtype=np.float64)
    if values.ndim < 3:
        raise ValueError(
            "query_attention must end in [queries, height, width], "
            f"got {values.shape}."
        )
    if values.shape[-3] < 1:
        raise ValueError("query_attention must contain at least one query.")
    if values.ndim > 3:
        values = _nanmean(values, axis=tuple(range(values.ndim - 3)))
    query_count = values.shape[0]
    if query_labels is None:
        query_labels = [f"query {index}" for index in range(query_count)]
    if len(query_labels) != query_count:
        raise ValueError(
            "query_labels must have one entry per Q-Former query: "
            f"expected {query_count}, got {len(query_labels)}."
        )
    mean_map = _nanmean(values, axis=0)
    maps: dict[str, np.ndarray] = {
        str(label): values[index] for index, label in enumerate(query_labels)
    }
    maps["mean across queries"] = mean_map
    return render_spatial_attention_overlays(
        output_path,
        image,
        maps,
        title=title or "Q-Former query to full-feature attention",
        alpha=alpha,
        cmap=cmap,
        columns=columns,
    )


def render_attention_summary(
    output_path: PathLike,
    series: Mapping[str, ArrayLike],
    *,
    title: str | None = None,
    ylabel: str = "attention mass",
    ci: Mapping[str, ArrayLike | tuple[ArrayLike, ArrayLike]] | None = None,
    layer_positions: Sequence[float] | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> Path:
    """Render batch-level layer trends, with bootstrap 95% CIs by default.

    Each series can be ``[layers]`` or ``[samples, layers]``.  For a supplied
    ``ci`` mapping, each value is either ``[2, layers]`` (low then high) or a
    ``(low, high)`` pair; otherwise a deterministic bootstrap CI is calculated
    for series with more than one sample.
    """

    arrays = _validate_layer_series(series)
    num_layers = next(iter(arrays.values())).shape[-1]
    x_values = _layer_positions(num_layers, layer_positions)
    intervals = _coerce_intervals(ci, arrays) if ci is not None else {}

    figure, axis = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    for index, (label, values) in enumerate(arrays.items()):
        mean = _nanmean(values, axis=0)
        axis.plot(x_values, mean, label=label, linewidth=2.0, marker="o", markersize=3.5)
        if label in intervals:
            low, high = intervals[label]
        elif values.shape[0] > 1:
            low, high = _bootstrap_mean_ci(values, bootstrap_samples, seed + index)
        else:
            continue
        axis.fill_between(x_values, low, high, alpha=0.18)
    axis.set_xlabel("decoder layer")
    axis.set_ylabel(ylabel)
    axis.set_title(title or "Attention summary across samples")
    axis.grid(True, alpha=0.25)
    axis.legend(title="source", loc="best")
    return _save_figure(figure, output_path)


def render_intervention_summary(
    output_path: PathLike,
    metrics: Mapping[str, ArrayLike],
    *,
    cutoffs: Sequence[float],
    scopes: Sequence[str],
    title: str | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> Path:
    """Render trajectory metrics across layer-depth intervention cutoffs.

    Each metrics value has shape ``[samples, scopes, cutoffs]``.  For one sample,
    ``[scopes, cutoffs]`` is accepted; when there is one scope, ``[samples,
    cutoffs]`` and ``[cutoffs]`` are also accepted.  The chart keeps the cutoff
    axis and each visual-token scope explicit, and draws deterministic bootstrap
    95% CIs whenever there are multiple samples.
    """

    if not metrics:
        raise ValueError("metrics must contain at least one intervention metric.")
    if len(cutoffs) == 0:
        raise ValueError("cutoffs must contain at least one value.")
    if len(scopes) == 0:
        raise ValueError("scopes must contain at least one visual-token scope.")
    x_values = np.asarray(cutoffs, dtype=np.float64)
    if x_values.ndim != 1 or not np.all(np.isfinite(x_values)):
        raise ValueError("cutoffs must be a one-dimensional finite sequence.")
    cubes = {
        name: _coerce_intervention_cube(values, len(scopes), len(cutoffs), name)
        for name, values in metrics.items()
    }

    metric_items = list(cubes.items())
    num_columns = min(2, len(metric_items))
    num_rows = int(np.ceil(len(metric_items) / num_columns))
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(7.0 * num_columns, 4.7 * num_rows),
        squeeze=False,
        layout="constrained",
    )
    if title:
        figure.suptitle(title, fontsize=14)
    for metric_index, (metric_name, values) in enumerate(metric_items):
        axis = axes.ravel()[metric_index]
        for scope_index, scope in enumerate(scopes):
            per_sample = values[:, scope_index, :]
            mean = _nanmean(per_sample, axis=0)
            axis.plot(
                x_values,
                mean,
                label=str(scope),
                linewidth=2.0,
                marker="o",
                markersize=4,
            )
            if per_sample.shape[0] > 1 and bootstrap_samples > 0:
                low, high = _bootstrap_mean_ci(
                    per_sample,
                    bootstrap_samples,
                    seed + metric_index * len(scopes) + scope_index,
                )
                axis.fill_between(x_values, low, high, alpha=0.16)
        axis.set_title(str(metric_name))
        axis.set_xlabel("first ablated decoder layer")
        axis.set_ylabel(str(metric_name))
        axis.grid(True, alpha=0.25)
        axis.legend(title="ablated tokens", loc="best")
        if np.all((x_values >= 0.0) & (x_values <= 1.0)):
            axis.set_xticks(x_values, [f"{value:.0%}" for value in x_values])
    for axis in axes.ravel()[len(metric_items):]:
        axis.axis("off")
    return _save_figure(figure, output_path)


def _coerce_rgb_image(image: Image.Image | ArrayLike) -> np.ndarray:
    """Convert a PIL image or a typical NumPy image to an RGB uint8 array."""

    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError(
                "image must be PIL RGB/RGBA or an array shaped [height, width, "
                f"channels], got {array.shape}."
            )
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite = np.where(np.isfinite(array), array, 0.0)
        maximum = float(np.max(finite)) if finite.size else 0.0
        if maximum <= 1.0:
            finite = finite * 255.0
        array = np.rint(finite)
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _resize_map(attention_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a normalized map to a PIL size ``(width, height)``."""

    image = Image.fromarray(attention_map.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, resample=Image.Resampling.BILINEAR), dtype=np.float32)


def _coerce_named_maps(
    attention_maps: Mapping[str, ArrayLike] | ArrayLike,
    *,
    labels: Sequence[str] | None,
) -> tuple[list[str], list[np.ndarray]]:
    if isinstance(attention_maps, Mapping):
        if not attention_maps:
            raise ValueError("attention_maps must contain at least one map.")
        if labels is not None:
            raise ValueError("labels cannot be supplied when attention_maps is a mapping.")
        map_labels = [str(label) for label in attention_maps]
        maps = [np.asarray(values) for values in attention_maps.values()]
    else:
        values = np.asarray(attention_maps)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3:
            raise ValueError(
                "attention_maps array must have shape [maps, height, width] or "
                f"[height, width], got {values.shape}."
            )
        maps = [values[index] for index in range(values.shape[0])]
        if labels is None:
            map_labels = [f"attention map {index}" for index in range(values.shape[0])]
        else:
            map_labels = [str(label) for label in labels]
            if len(map_labels) != len(maps):
                raise ValueError(
                    "labels must have one entry per attention map: "
                    f"expected {len(maps)}, got {len(map_labels)}."
                )
    for values in maps:
        if values.ndim != 2:
            raise ValueError(
                "Each attention map must have shape [height, width], "
                f"got {values.shape}."
            )
    return map_labels, maps


def _validate_group_attention(
    grouped: Mapping[str, ArrayLike],
    source_labels: Sequence[str],
    name: str,
    *,
    expected_groups: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    if not grouped:
        raise ValueError(f"{name} must contain at least one query group.")
    if not source_labels:
        raise ValueError("source_labels must contain at least one source.")
    if expected_groups is not None and set(grouped) != set(expected_groups):
        raise ValueError(
            f"{name} groups must exactly match attention_mass groups: "
            f"{sorted(grouped)} vs {sorted(expected_groups)}."
        )
    result: dict[str, np.ndarray] = {}
    for group_name, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(
                f"{name}[{group_name!r}] must have shape [layers, heads, sources], "
                f"got {array.shape}."
            )
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(f"{name}[{group_name!r}] must have at least one layer and head.")
        if array.shape[-1] != len(source_labels):
            raise ValueError(
                f"{name}[{group_name!r}] has {array.shape[-1]} sources but "
                f"source_labels has {len(source_labels)} entries."
            )
        if expected_groups is not None and array.shape != expected_groups[group_name].shape:
            raise ValueError(
                f"{name}[{group_name!r}] shape {array.shape} must match "
                f"attention_mass shape {expected_groups[group_name].shape}."
            )
        result[str(group_name)] = array
    return result


def _draw_source_trends(axis: plt.Axes, values: np.ndarray, source_labels: Sequence[str]) -> None:
    layer_index = np.arange(values.shape[0])
    source_mass = _nanmean(values, axis=1)
    for source_index, label in enumerate(source_labels):
        axis.plot(
            layer_index,
            source_mass[:, source_index],
            label=str(label),
            linewidth=1.8,
            marker="o",
            markersize=2.8,
        )
    axis.set_xlabel("decoder layer")
    axis.set_ylabel("attention mass")
    axis.grid(True, alpha=0.25)
    axis.legend(title="key source", fontsize=8, title_fontsize=8, loc="best")


def _draw_layer_head_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    source_labels: Sequence[str],
    *,
    title: str,
    cmap: str,
) -> None:
    layers, heads, sources = values.shape
    # Group source blocks vertically.  This preserves both source identity and
    # the requested layer/head resolution without producing one tiny axis per
    # source.
    heatmap = np.transpose(values, (2, 0, 1)).reshape(sources * layers, heads)
    finite = heatmap[np.isfinite(heatmap)]
    upper = float(np.max(finite)) if finite.size else 1.0
    image = axis.imshow(
        np.where(np.isfinite(heatmap), heatmap, 0.0),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=max(upper, np.finfo(np.float64).eps),
    )
    centers = np.arange(sources) * layers + (layers - 1) / 2.0
    axis.set_yticks(centers, [str(label) for label in source_labels])
    axis.set_xticks(np.arange(heads), [str(index) for index in range(heads)])
    axis.set_xlabel("attention head")
    axis.set_ylabel("key source (each block: layers)")
    axis.set_title(title)
    for boundary in np.arange(1, sources) * layers - 0.5:
        axis.axhline(boundary, color="white", alpha=0.42, linewidth=0.6)
    colorbar = axis.figure.colorbar(image, ax=axis, pad=0.01)
    colorbar.ax.set_ylabel("attention", rotation=270, va="bottom")


def _validate_layer_series(series: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
    if not series:
        raise ValueError("series must contain at least one source.")
    result: dict[str, np.ndarray] = {}
    expected_layers: int | None = None
    for label, values in series.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[-1] < 1:
            raise ValueError(
                f"series[{label!r}] must have shape [layers] or [samples, layers], "
                f"got {array.shape}."
            )
        if expected_layers is None:
            expected_layers = array.shape[-1]
        elif array.shape[-1] != expected_layers:
            raise ValueError(
                "All attention-summary series must have the same layer count: "
                f"expected {expected_layers}, got {array.shape[-1]} for {label!r}."
            )
        result[str(label)] = array
    return result


def _layer_positions(num_layers: int, layer_positions: Sequence[float] | None) -> np.ndarray:
    if layer_positions is None:
        return np.arange(num_layers, dtype=np.float64)
    values = np.asarray(layer_positions, dtype=np.float64)
    if values.ndim != 1 or len(values) != num_layers or not np.all(np.isfinite(values)):
        raise ValueError(
            "layer_positions must be a finite sequence with one entry per layer: "
            f"expected {num_layers}, got {values.shape}."
        )
    return values


def _coerce_intervals(
    intervals: Mapping[str, ArrayLike | tuple[ArrayLike, ArrayLike]],
    series: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if set(intervals) != set(series):
        raise ValueError("ci keys must exactly match series keys.")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, interval in intervals.items():
        if isinstance(interval, tuple) and len(interval) == 2:
            low, high = interval
        else:
            array = np.asarray(interval, dtype=np.float64)
            if array.shape != (2, series[label].shape[-1]):
                raise ValueError(
                    f"ci[{label!r}] must have shape [2, layers], got {array.shape}."
                )
            low, high = array[0], array[1]
        low_array = np.asarray(low, dtype=np.float64)
        high_array = np.asarray(high, dtype=np.float64)
        expected_shape = (series[label].shape[-1],)
        if low_array.shape != expected_shape or high_array.shape != expected_shape:
            raise ValueError(
                f"ci[{label!r}] bounds must each have shape {expected_shape}."
            )
        result[label] = (low_array, high_array)
    return result


def _bootstrap_mean_ci(
    values: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least one.")
    sample_count = values.shape[0]
    if sample_count < 2:
        mean = _nanmean(values, axis=0)
        return mean, mean
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, sample_count, size=(bootstrap_samples, sample_count))
    samples = values[indices]
    means = _nanmean(samples, axis=1)
    return (
        np.nanpercentile(means, 2.5, axis=0),
        np.nanpercentile(means, 97.5, axis=0),
    )


def _coerce_intervention_cube(
    values: ArrayLike,
    scopes: int,
    cutoffs: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        cube = array
    elif array.ndim == 2 and array.shape == (scopes, cutoffs):
        cube = array[None, ...]
    elif array.ndim == 2 and scopes == 1 and array.shape[-1] == cutoffs:
        cube = array[:, None, :]
    elif array.ndim == 1 and scopes == 1 and array.shape[0] == cutoffs:
        cube = array[None, None, :]
    else:
        raise ValueError(
            f"metrics[{name!r}] must have [samples, scopes, cutoffs] layout "
            f"(or an accepted single-scope shorthand), got {array.shape}."
        )
    if cube.shape[1:] != (scopes, cutoffs):
        raise ValueError(
            f"metrics[{name!r}] has trailing shape {cube.shape[1:]}; expected "
            f"({scopes}, {cutoffs}) for supplied scopes and cutoffs."
        )
    return cube


def _nanmean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """A warning-free NaN-aware mean that returns NaN for empty reductions."""

    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, array, 0.0), axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=np.float64),
        where=count > 0,
    )


def _save_figure(figure: plt.Figure, output_path: PathLike) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=170, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


__all__ = [
    "compose_query_patch_attention",
    "normalize_spatial_map",
    "overlay_attention_on_image",
    "render_attention_overview",
    "render_attention_summary",
    "render_intervention_summary",
    "render_qformer_attention",
    "render_spatial_attention_overlays",
]
