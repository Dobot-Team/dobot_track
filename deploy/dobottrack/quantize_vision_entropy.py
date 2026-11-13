import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse, json, numpy as np, onnx
from onnxruntime.quantization import CalibrationDataReader
from modelopt.onnx.quantization.int8 import quantize
from deploy.dobottrack.quant_layers import (
    quantize_onnx_linears,
    quantized_matmul_names,
    fold_constant_quantizers,
)

p = argparse.ArgumentParser()
p.add_argument("--work", required=True)
a = p.parse_args()
r = Path(a.work)
graph = onnx.load(r / "export/vision_sq.onnx")
targets = []
for n in graph.graph.node:
    if n.op_type != "MatMul":
        continue
    if "/mlp/fc1/MatMul" in n.name:
        targets.append(n.name)
    elif (
        "/mlp/fc2/MatMul" in n.name
        and int(n.name.split("/layer.")[1].split("/")[0]) < 12
    ):
        targets.append(n.name)
assert len(targets) == 36, len(targets)


class Reader(CalibrationDataReader):
    def __init__(self):
        self.x = np.load(r / "vision_entropy64.npz")["pixel_values"]
        self.i = 0

    def get_next(self):
        if self.i == len(self.x):
            return None
        value = {"pixel_values": self.x[self.i : self.i + 1]}
        self.i += 1
        return value


# Memory-bounded entropy calibration. Restrict observed tensors to selected MatMul inputs/outputs.
# Same64 calibration images; histogram updates are streamed rather than retaining all activations.
import modelopt.onnx.quantization.int8 as int8_module
from onnxruntime.quantization import calibrate as ortcal
import logging

wanted = {
    x
    for node in graph.graph.node
    if node.name in targets
    for x in [node.input[0], node.output[0]]
}
original_configure = int8_module.configure_ort
original_select = ortcal.CalibraterBase.select_tensors_to_calibrate


def selected_tensors(self, model):
    names, values = original_select(self, model)
    return names.intersection(wanted), values


def collect_stream(self, data_reader):
    if not self.collector:
        self.collector = ortcal.HistogramCollector(
            method=self.method,
            symmetric=self.symmetric,
            num_bins=self.num_bins,
            num_quantized_bins=self.num_quantized_bins,
            percentile=self.percentile,
            scenario=self.scenario,
        )
    names = [x.name for x in self.infer_session.get_outputs()]
    count = 0
    while True:
        data = data_reader.get_next()
        if data is None:
            break
        outputs = self.infer_session.run(None, data)
        self.collector.collect(
            {
                name: [value]
                for name, value in zip(names, outputs)
                if name in self.tensors_to_calibrate
            }
        )
        count += 1
        if count % 8 == 0:
            print(
                "STREAMING_ENTROPY_IMAGES",
                count,
                "observed_tensors",
                len(self.tensors_to_calibrate),
                flush=True,
            )
    self.clear_collected_data()


def bounded_configure(*args, **kwargs):
    result = original_configure(*args, **kwargs)
    ortcal.CalibraterBase.select_tensors_to_calibrate = selected_tensors
    ortcal.HistogramCalibrater.collect_data = collect_stream
    return result


int8_module.configure_ort = bounded_configure
quant = quantize(
    str(r / "export/vision_sq.onnx"),
    calibration_method="entropy",
    calibration_data_reader=Reader(),
    op_types_to_quantize=["MatMul"],
    nodes_to_quantize=targets,
    use_external_data_format=False,
    output_path=str(r / "export/vision_mlp_int8.onnx"),
)
if quant is None:
    quant = onnx.load(r / "export/vision_mlp_int8.onnx")
onnx.save(quant, r / "export/vision_mlp_int8.onnx")
cal = json.loads((r / "vision_calibration.json").read_text())
ranges = {}
for name, value in cal["attention_amax"].items():
    path = "/" + name.replace(".", "/") + "/MatMul"
    parts = path.split("/")
    i = parts.index("layer")
    parts[i : i + 2] = ["layer." + parts[i + 1]]
    ranges["/".join(parts)] = value
assert len(ranges) == 96
quantize_onnx_linears(quant, ranges)
fold_constant_quantizers(quant)
assert (
    len(quantized_matmul_names(quant)) == 132
), "Expected 36 MLP + 96 attention quantized MatMuls"
onnx.checker.check_model(quant)
onnx.save(quant, r / "export/vision_attention_int8.onnx")
print("ENTROPY36_PLUS_ATTENTION96_READY", flush=True)
