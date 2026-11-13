"""Fixed-buffer inference optimizations used by the default optimized recipe."""

import numpy as np, torch


def packed_cpu_outputs(values):
    keys = ["trajectory", "grounding_bbox", "grounding_visibility_logits"]
    flat = torch.cat([values[k].reshape(-1).float() for k in keys]).cpu()
    out = {}
    offset = 0
    for k in keys:
        n = values[k].numel()
        out[k] = flat[offset : offset + n].reshape(values[k].shape)
        offset += n
    return out


def attach_cached_prompt(agent):
    cache = {}

    def batch(frames, prompt):
        pixels = agent._internvl_pixel_values(frames)
        key = (len(frames), prompt)
        if key not in cache:
            counts = agent.model_extensions.token_layout(
                len(frames), agent.num_image_tokens
            ).per_frame_counts
            image_tokens = "\n".join(
                "<img>" + "<IMG_CONTEXT>" * n + "</img>" for n in counts
            )
            tokenized = agent.model.tokenizer(
                f"{image_tokens}\nUser: {prompt}\nAssistant:",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=agent.max_length,
            )
            cache[key] = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        return {"pixel_values": pixels, **cache[key]}

    agent._internvl_batch = batch


class BoundRunner:
    """One fixed shape, persistent bindings. Capture only inference, not Python copies."""

    def __init__(self, base, example, graph=False):
        from deploy.dobottrack.runtime_trt import is_host_input
        import tensorrt as trt

        self.base = base
        self.engine = base.engine
        self.context = self.engine.create_execution_context()
        self.inputs = {}
        self.outputs = {}
        self.graph = None
        self.graph_error = None
        for name in base.names:
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            value = example[name]
            dtype = torch.from_numpy(
                np.empty((), trt.nptype(self.engine.get_tensor_dtype(name)))
            ).dtype
            host = is_host_input(name, self.engine.get_tensor_location(name))
            self.inputs[name] = torch.empty(
                value.shape,
                dtype=dtype,
                device="cpu" if host else "cuda",
                pin_memory=True if host else False,
            )
            self.inputs[name].copy_(value)
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise ValueError((name, value.shape))
            self.context.set_tensor_address(name, self.inputs[name].data_ptr())
        for name in base.names:
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            dtype = torch.from_numpy(
                np.empty((), trt.nptype(self.engine.get_tensor_dtype(name)))
            ).dtype
            self.outputs[name] = torch.empty(
                tuple(self.context.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
            self.context.set_tensor_address(name, self.outputs[name].data_ptr())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.execute()
        stream.synchronize()
        if graph:
            try:
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured, stream=stream):
                    self.execute()
                self.graph = captured
            except Exception as exc:
                self.graph_error = str(exc)
        torch.cuda.synchronize()

    def execute(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("Bound TensorRT enqueue failed")

    def __call__(self, values):
        for name in self.inputs:
            value = values[name]
            target = self.inputs[name]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError("Bound shape changed " + name)
            if value.data_ptr() != target.data_ptr():
                target.copy_(value, non_blocking=True)
        if self.graph is not None:
            self.graph.replay()
        else:
            self.execute()
        return self.outputs


class ShapeCache:
    def __init__(self, base, graph):
        self.base = base
        self.graph = graph
        self.runners = {}

    def __call__(self, inputs):
        key = tuple((k, tuple(v.shape)) for k, v in sorted(inputs.items()))
        if key not in self.runners:
            self.runners[key] = BoundRunner(self.base, inputs, self.graph)
        return self.runners[key](inputs)


def configure_agent(agent):
    from deploy.dobottrack.compact_boxes import attach_cpu_box_cache

    attach_cpu_box_cache(agent)
    attach_cached_prompt(agent)


def graph_evidence(policy):
    rows = []
    for name in ["vision", "llm", "heads"]:
        for key, runner in getattr(policy, name).runners.items():
            rows.append(
                {
                    "stage": name,
                    "shape": str(key),
                    "captured": runner.graph is not None,
                    "error": runner.graph_error,
                }
            )
    for k, (runner, *_) in policy.qformer.buckets.items():
        rows.append(
            {
                "stage": "qformer",
                "bucket": k,
                "captured": runner.graph is not None,
                "error": runner.graph_error,
            }
        )
    return rows
