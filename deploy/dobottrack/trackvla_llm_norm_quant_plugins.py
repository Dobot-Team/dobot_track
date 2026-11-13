"""Build-time emitter patch that makes tensorrt_llm emit the TrackVLA norm/quant plugins.

``tensorrt_llm.quantization.layers`` binds ``smooth_quant_rms_norm`` and
``quantize_per_token`` at import time (``from .functional import ...``), so patching
``tensorrt_llm.quantization.functional`` is not enough: the module globals of
``tensorrt_llm.quantization.layers`` have to be replaced.  Those two names are the only
emitters used by the TrackVLA LLM graph (layers.py lines 80, 341, 1157, 1645, 1984).

The replacements mirror the upstream implementations field for field so the resulting
network is identical apart from the plugin type/namespace, i.e. the TensorRT fusion
boundaries do not move.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple, Union

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_net, default_trtnet
from tensorrt_llm._utils import str_dtype_to_np
from tensorrt_llm.functional import (
    Tensor,
    _add_plugin_info,
    _create_tensor,
    cast,
    constant,
)

PLUGIN_NAMESPACE = "TrackVLALLM"
RMSNORM_PLUGIN_NAME = "TrackVLARmsnormQuant"
QUANTIZE_PLUGIN_NAME = "TrackVLAQuantizePerToken"
PLUGIN_VERSION = "1"


def load_plugin_library(path: str) -> None:
    """dlopen the plugin library so its creators are registered in this process."""
    ctypes.CDLL(path)
    registry = trt.get_plugin_registry()
    for name in (RMSNORM_PLUGIN_NAME, QUANTIZE_PLUGIN_NAME):
        creator = registry.get_plugin_creator(name, PLUGIN_VERSION, PLUGIN_NAMESPACE)
        if creator is None:
            raise RuntimeError(
                f"plugin creator not registered after loading {path}: {name}"
            )
        print(
            f"[trackvla] plugin registered: {name} v{PLUGIN_VERSION} ns={PLUGIN_NAMESPACE}"
        )


def _creator(name: str):
    creator = trt.get_plugin_registry().get_plugin_creator(
        name, PLUGIN_VERSION, PLUGIN_NAMESPACE
    )
    if creator is None:
        raise RuntimeError(f"TrackVLA plugin creator is not registered: {name}")
    return creator


def smooth_quant_rms_norm(
    input: Tensor,
    normalized_shape: Union[int, Tuple[int]],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    scale: Optional[Tensor] = None,
    clamp_val: Optional[Tensor] = None,
    eps: float = 1e-05,
    dynamic_act_scaling: bool = False,
) -> Tensor:
    if clamp_val is not None:
        raise NotImplementedError(
            "the TrackVLA norm+quant plugin does not implement clamp_val"
        )
    if not dynamic_act_scaling:
        raise NotImplementedError(
            "the TrackVLA norm+quant plugin implements the per-token path only"
        )
    if not default_net().plugin_config.rmsnorm_quantization_plugin:
        raise TypeError("Smooth Quant Rms Norm is only supported with plugin")

    plg_creator = _creator(RMSNORM_PLUGIN_NAME)
    pfc = trt.PluginFieldCollection(
        [
            trt.PluginField(
                "eps", np.array(eps, dtype=np.float32), trt.PluginFieldType.FLOAT32
            ),
            trt.PluginField(
                "dyn_act_scaling",
                np.array([int(dynamic_act_scaling)], np.int32),
                trt.PluginFieldType.INT32,
            ),
        ]
    )
    rmsnorm_plug = plg_creator.create_plugin("rmsnorm_quantized", pfc)

    normalized_shape = (
        [normalized_shape] if isinstance(normalized_shape, int) else normalized_shape
    )
    p_dtype = default_net().plugin_config.rmsnorm_quantization_plugin
    if weight is None:
        weight = constant(np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
    if bias is None:
        bias = constant(np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

    # RMS Norm Plugin only supports float32 scale.
    scale = cast(scale, "float32")
    plug_inputs = [
        input.trt_tensor,
        weight.trt_tensor,
        bias.trt_tensor,
        scale.trt_tensor,
    ]
    layer = default_trtnet().add_plugin_v2(plug_inputs, rmsnorm_plug)
    if not default_net().strongly_typed:
        layer.get_output(0).set_dynamic_range(-127, 127)
    _add_plugin_info(layer, plg_creator, "rmsnorm_quantized", pfc)

    return (
        _create_tensor(layer.get_output(0), layer),
        _create_tensor(layer.get_output(1), layer),
    )


def quantize_per_token(x: Tensor, clamp_val: Optional[Tensor] = None) -> Tuple[Tensor]:
    if clamp_val is not None:
        raise NotImplementedError(
            "the TrackVLA per-token quant plugin does not implement clamp_val"
        )
    if not default_net().plugin_config.quantize_per_token_plugin:
        raise TypeError("QuantizePerToken is only supported with plugin")

    plg_creator = _creator(QUANTIZE_PLUGIN_NAME)
    pfc = trt.PluginFieldCollection([])
    quantize_plug = plg_creator.create_plugin("quantize_per_token_plugin", pfc)

    layer = default_trtnet().add_plugin_v2([x.trt_tensor], quantize_plug)
    if not default_net().strongly_typed:
        layer.get_output(0).set_dynamic_range(-127, 127)
    _add_plugin_info(layer, plg_creator, "quantize_per_token_plugin", pfc)

    return (
        _create_tensor(layer.get_output(0), layer),
        _create_tensor(layer.get_output(1), layer),
    )


def install(plugin_library: str, patch: Optional[str] = None) -> None:
    """Register the plugin library and redirect the emitters named by ``patch``.

    ``patch`` (or ``$TRACKVLA_EMITTER_PATCH``) is ``both`` (default), ``rmsnorm`` or
    ``quantize``.  Replacing one emitter at a time isolates which of the two changes a
    rebuilt engine's numerics, which is how the bisect in
    reports/orin_20260921/llm_norm_quant_layers.md was run.
    """
    patch = (patch or os.environ.get("TRACKVLA_EMITTER_PATCH") or "both").lower()
    if patch not in ("both", "rmsnorm", "quantize"):
        raise ValueError(f"unknown emitter patch selector {patch!r}")
    load_plugin_library(plugin_library)
    from tensorrt_llm.quantization import layers as tllm_quant_layers

    if patch in ("both", "rmsnorm"):
        tllm_quant_layers.smooth_quant_rms_norm = smooth_quant_rms_norm
    if patch in ("both", "quantize"):
        tllm_quant_layers.quantize_per_token = quantize_per_token
    print(f"[trackvla] tensorrt_llm.quantization.layers emitters patched: {patch}")
