"""Native Orin TensorRT builder. Dump inspector evidence for actual INT8 layers."""

from pathlib import Path
import argparse, json
import tensorrt as trt

p = argparse.ArgumentParser()
p.add_argument("--onnx", required=True)
p.add_argument("--output", required=True)
p.add_argument("--workspace-mb", type=int, default=512)
a = p.parse_args()
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(net, logger)
if not parser.parse_from_file(a.onnx):
    raise RuntimeError(
        "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
    )
for i in range(net.num_inputs):
    t = net.get_input(i)
    if t.dtype == trt.float32 and t.name not in ["pixel_values", "boxes"]:
        t.dtype = trt.float16
for i in range(net.num_outputs):
    t = net.get_output(i)
    if t.dtype == trt.float32:
        t.dtype = trt.float16
c = builder.create_builder_config()
c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_mb * 1024**2)
c.builder_optimization_level = 3
c.set_flag(trt.BuilderFlag.FP16)
c.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
dynamic_inputs = [
    net.get_input(i)
    for i in range(net.num_inputs)
    if -1 in tuple(net.get_input(i).shape)
]
if dynamic_inputs:
    raise ValueError("These component exports must have fixed input shapes")
b = builder.build_serialized_network(net, c)
if b is None:
    raise RuntimeError("TRT build returned None")
p = Path(a.output)
p.parent.mkdir(exist_ok=True, parents=True)
p.write_bytes(b)
rt = trt.Runtime(logger)
engine = rt.deserialize_cuda_engine(b)
inspector = engine.create_engine_inspector()
layers = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
p.with_suffix(".layers.json").write_text(layers)
if "int8" in p.name:
    layer_info = json.loads(layers)
    text = layers.lower()
    assert (
        "int8" in text or "int 8" in text or "int8" in str(layer_info).lower()
    ), "No INT8 layer found in engine inspector"
print("ENGINE_BUILT", p, p.stat().st_size, flush=True)
