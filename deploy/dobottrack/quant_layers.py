import torch
from torch import nn


class QDQLinear(nn.Module):
    def __init__(self, source, amax):
        super().__init__()
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer(
            "weight_scale", source.weight.detach().abs().amax(1).clamp_min(1e-8) / 127
        )
        self.register_buffer(
            "weight_zero",
            torch.zeros(
                source.weight.shape[0], dtype=torch.int32, device=source.weight.device
            ),
        )
        self.register_buffer(
            "activation_scale",
            torch.tensor(max(amax, 1e-8) / 127, device=source.weight.device),
        )
        self.register_buffer(
            "activation_zero",
            torch.zeros((), dtype=torch.int32, device=source.weight.device),
        )

    def forward(self, x):
        x = torch.fake_quantize_per_tensor_affine(
            x, self.activation_scale, self.activation_zero, -128, 127
        )
        w = torch.fake_quantize_per_channel_affine(
            self.weight, self.weight_scale, self.weight_zero, 0, -128, 127
        )
        return nn.functional.linear(x, w, self.bias)


def install_qdq(model, ranges):
    for name, value in ranges.items():
        parent, key = name.rsplit(".", 1)
        container = model.get_submodule(parent)
        source = getattr(container, key)
        setattr(container, key, QDQLinear(source, value))


def quantize_onnx_linears(model, ranges):
    import numpy as np
    from onnx import helper, numpy_helper, TensorProto

    initial = {x.name: x for x in model.graph.initializer}
    nodes = []
    found = set()
    for node in model.graph.node:
        if node.name not in ranges:
            nodes.append(node)
            continue
        if node.op_type != "MatMul" or node.input[1] not in initial:
            raise ValueError(node.name)
        found.add(node.name)
        prefix = node.name + "_int8"
        w = numpy_helper.to_array(initial[node.input[1]]).astype(np.float32)
        ws = np.maximum(np.max(np.abs(w), axis=0), 1e-8) / 127
        qw = np.clip(np.rint(w / ws), -127, 127).astype(np.int8)
        a = np.float32(max(ranges[node.name], 1e-8) / 127)
        for suffix, value in [
            ("as", np.asarray(a)),
            ("az", np.asarray(0, dtype=np.int8)),
            ("ws", ws.astype(np.float32)),
            ("wz", np.zeros(w.shape[1], dtype=np.int8)),
            ("w", qw),
        ]:
            model.graph.initializer.append(
                numpy_helper.from_array(value, prefix + suffix)
            )
        nodes.extend(
            [
                helper.make_node(
                    "QuantizeLinear",
                    [node.input[0], prefix + "as", prefix + "az"],
                    [prefix + "aq"],
                    name=prefix + "/AQ",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [prefix + "aq", prefix + "as", prefix + "az"],
                    [prefix + "adq"],
                    name=prefix + "/ADQ",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [prefix + "w", prefix + "ws", prefix + "wz"],
                    [prefix + "wdq"],
                    axis=1,
                    name=prefix + "/WDQ",
                ),
            ]
        )
        node.input[0] = prefix + "adq"
        node.input[1] = prefix + "wdq"
        nodes.append(node)
    if found != set(ranges):
        raise ValueError("Missing targets: " + str(set(ranges) - found))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input}
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def quantized_matmul_names(model):
    producers = {name: node for node in model.graph.node for name in node.output}
    return [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and all(
            name in producers and producers[name].op_type == "DequantizeLinear"
            for name in node.input
        )
    ]


def fold_constant_quantizers(model):
    """Fold constant-weight Q nodes with ONNX round-to-nearest-even semantics."""
    import numpy as np
    from onnx import numpy_helper, helper

    initial = {
        tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer
    }
    keep = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or not all(
            name in initial for name in node.input
        ):
            keep.append(node)
            continue
        x, scale = initial[node.input[0]], initial[node.input[1]]
        zero = initial[node.input[2]] if len(node.input) > 2 else np.array(0, np.uint8)
        axis = next(
            (helper.get_attribute_value(a) for a in node.attribute if a.name == "axis"),
            1,
        )
        shape = [1] * x.ndim
        if scale.ndim:
            shape[axis % x.ndim] = scale.size
            scale, zero = scale.reshape(shape), zero.reshape(shape)
        info = np.iinfo(zero.dtype)
        q = np.clip(np.rint(x / scale) + zero, info.min, info.max).astype(zero.dtype)
        model.graph.initializer.append(numpy_helper.from_array(q, node.output[0]))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(keep)
    used = {name for node in keep for name in node.input} | {
        x.name for x in model.graph.output
    }
    tensors = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(tensors)
    return count
