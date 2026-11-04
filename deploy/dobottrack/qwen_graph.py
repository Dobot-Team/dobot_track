"""Orin build-only graph for the checkpoint-specific SmoothQuant weights (TRT-LLM 0.12).

Consumes external embeddings and returns all final hidden states. No decoding
or KV-cache reuse is performed. Validate numerical agreement before deployment.
"""

from collections import OrderedDict

import tensorrt as trt
from tensorrt_llm._common import default_net
from tensorrt_llm.functional import Tensor
from tensorrt_llm.layers import Attention, KeyValueCacheParams
from tensorrt_llm.models.qwen.model import QWenForCausalLM


class TrackVLAQwen2Jetson(QWenForCausalLM):
    def prepare_inputs(self, max_batch_size, max_input_len, **kwargs):
        if max_batch_size != 1 or max_input_len < 1:
            raise ValueError("Expected batch=1 and a positive max_input_len")
        if default_net().plugin_config.remove_input_padding:
            raise ValueError(
                "Dense embedding inputs require remove_input_padding=disable"
            )
        kwargs["use_cache"] = False
        inputs = super().prepare_inputs(max_batch_size, max_input_len, **kwargs)
        inputs.pop("input_ids", None)
        inputs.pop("position_ids", None)
        inputs["use_cache"] = False
        inputs["inputs_embeds"] = Tensor(
            name="inputs_embeds",
            dtype=trt.float16,
            shape=[1, -1, 896],
            dim_range=OrderedDict(
                [
                    ("batch_size", [[1, 1, 1]]),
                    ("sequence_length", [[1, min(428, max_input_len), max_input_len]]),
                    ("hidden_size", [[896, 896, 896]]),
                ]
            ),
        )
        return inputs

    def forward(self, inputs_embeds, **kwargs):
        hidden = self.transformer.layers.forward(
            inputs_embeds,
            attention_mask=kwargs.get("attention_mask"),
            use_cache=False,
            kv_cache_params=kwargs.get("kv_cache_params") or KeyValueCacheParams(),
            attention_params=Attention.fill_attention_params(
                self, kwargs.get("attention_params")
            ),
        )
        hidden = self.transformer.ln_f(hidden)
        hidden.mark_output("hidden_states", self.config.dtype)
        return hidden
