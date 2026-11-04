"""PTQ QFormer FFN only. Calibration/validation episodes stay disjoint."""

# Support both direct scripts and python -m deploy.dobottrack.<entry>.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json, copy, sys
from pathlib import Path
import torch, numpy as np
from torch import nn
from deploy.dobottrack.core import BoxJointCore
from deploy.dobottrack.common import model_parser, load

from deploy.dobottrack.quant_layers import QDQLinear, install_qdq

p = model_parser(__doc__)
p.add_argument("--scope", choices=["ffn", "attention_ffn"], default="ffn")
args = p.parse_args()
tag = "qformer_int8" if args.scope == "ffn" else "qformer_qkv_int8"
r, cfg, model, agent = load(args)
core = BoxJointCore(copy.deepcopy(model.history_compressor).float()).cuda().eval()
del model, agent
torch.cuda.empty_cache()
entries = json.loads((r / "samples.json").read_text())
amax = {}
hooks = []
for name, module in core.named_modules():
    if isinstance(module, nn.Linear) and (
        ".ffn." in name
        or (
            args.scope == "attention_ffn"
            and any(x in name for x in [".self_attention.", ".full_attention."])
        )
    ):

        def capture(m, a, name=name):
            amax[name] = max(amax.get(name, 0), a[0].detach().abs().max().item())

        hooks.append(module.register_forward_pre_hook(capture))
with torch.inference_mode():
    for e in entries:
        if e["split"] != "calibration":
            continue
        a = np.load(r / "captured" / e["path"])
        core(
            torch.from_numpy(a["features"]).cuda().float(),
            torch.from_numpy(a["boxes"]).cuda(),
            torch.from_numpy(a["valid"]).cuda(),
        )
for h in hooks:
    h.remove()
quant = copy.deepcopy(core)
install_qdq(quant, amax)
validation = []
with torch.inference_mode():
    for e in entries:
        if e["split"] != "validation":
            continue
        a = np.load(r / "captured" / e["path"])
        inputs = [torch.from_numpy(a[k]).cuda() for k in ["features", "boxes", "valid"]]
        inputs[0] = inputs[0].float()
        ref = core(*inputs)
        actual = quant(*inputs)
        validation.append(
            {
                "index": e["index"],
                "cosine": nn.functional.cosine_similarity(
                    ref.flatten(), actual.flatten(), dim=0
                ).item(),
                "rmse": ((ref - actual) ** 2).mean().sqrt().item(),
                "max_abs": (ref - actual).abs().max().item(),
            }
        )
# Export FP32 Q/DQ graph; TRT keeps nonquantized operators in FP16 as configured.
quant = quant.cpu()
sample = np.load(r / "captured" / entries[0]["path"])
inputs = tuple(
    (
        torch.from_numpy(sample[k]).float()
        if k == "features"
        else torch.from_numpy(sample[k])
    )
    for k in ["features", "boxes", "valid"]
)
with torch.no_grad():
    torch.onnx.export(
        quant,
        inputs,
        str(r / ("export/" + tag + ".onnx")),
        input_names=["features", "boxes", "valid"],
        output_names=["compressed_tokens"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
import onnx

m = onnx.load(r / ("export/" + tag + ".onnx"))
qdq = [x for x in m.graph.node if x.op_type in ["QuantizeLinear", "DequantizeLinear"]]
assert (
    len(amax) == len(core.layers) * (2 if args.scope == "ffn" else 10)
    and len(qdq) >= len(amax) * 2
), (len(amax), len(qdq))
(r / (tag + "_ptq.json")).write_text(
    json.dumps(
        dict(
            scope=args.scope
            + " Linear W8A8; attention matmuls/softmax, norms, input/output projections floating point",
            calibration_amax=amax,
            quantized_linears=len(amax),
            qdq_nodes=len(qdq),
            validation=validation,
            gt_boxes_used=False,
        ),
        indent=2,
    )
)
print("QFORMER_PTQ_COMPLETE", len(amax), len(qdq), flush=True)
