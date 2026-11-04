"""Apply prior Orin alpha=.5 MLP SmoothQuant recipe to NEW trained projector."""

# Support both direct scripts and python -m deploy.dobottrack.<entry>.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy, json, io, tarfile, zipfile
from pathlib import Path
import numpy as np, torch
from torch import nn
from PIL import Image
from deploy.dobottrack.common import load, model_parser
from deploy.dobottrack.components import Vision
from deploy.dobottrack.quant_layers import QDQLinear

p = model_parser(__doc__)
p.add_argument("--data", required=True)
p.add_argument("--fc2-prefix", type=int, choices=range(25), default=12)
args = p.parse_args()
r, cfg, model, agent = load(args)
model.internvl.language_model.cpu()
model.internvl.vision_tower.float()
model.internvl.multi_modal_projector.float()
model.internvl.vision_tower.config._attn_implementation = "eager"
wrapper = Vision(model.internvl).eval()
samples = json.loads((r / "samples.json").read_text())
calib = [e for e in samples if e["split"] == "calibration"]
valid = [e for e in samples if e["split"] == "validation"]
archives = {}


def image(e):
    path = str(Path(args.data) / e["episode"])
    if path not in archives:
        archives[path] = (
            zipfile.ZipFile(path) if zipfile.is_zipfile(path) else tarfile.open(path)
        )
    a = archives[path]
    raw = (
        a.read(e["current"])
        if isinstance(a, zipfile.ZipFile)
        else a.extractfile(e["current"]).read()
    )
    return agent._internvl_pixel_values(
        [np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))]
    ).float()


names = [
    name
    for name, m in wrapper.named_modules()
    if isinstance(m, nn.Linear) and name.endswith("mlp.fc1")
]
assert len(names) == 24, len(names)
ranges = {}
hooks = []
for name in names:

    def capture(m, a, name=name):
        x = a[0].detach().abs().reshape(-1, a[0].shape[-1]).amax(0)
        ranges[name] = x if name not in ranges else torch.maximum(ranges[name], x)

    hooks.append(wrapper.get_submodule(name).register_forward_pre_hook(capture))
with torch.inference_mode():
    for e in calib:
        wrapper(image(e))
for h in hooks:
    h.remove()
with torch.no_grad():
    for name in names:
        fc = wrapper.get_submodule(name)
        norm = wrapper.get_submodule(name.rsplit(".mlp.fc1", 1)[0] + ".layernorm_after")
        w = fc.weight.abs().amax(0).clamp_min(1e-5)
        scale = (ranges[name].clamp_min(1e-5) / w).sqrt().clamp_min(1e-5)
        norm.weight.div_(scale)
        norm.bias.div_(scale)
        fc.weight.mul_(scale[None, :])
# Explicit Q/DQ only for 24 fc1 and the first 12 fc2, matching the old safe scope.
targets = names + [x.rsplit("fc1", 1)[0] + "fc2" for x in names[: args.fc2_prefix]]
amax = {}
hooks = []
for name in targets:

    def capture(m, a, name=name):
        amax[name] = max(amax.get(name, 0), a[0].detach().abs().max().item())

    hooks.append(wrapper.get_submodule(name).register_forward_pre_hook(capture))
with torch.inference_mode():
    for e in calib:
        wrapper(image(e))
for h in hooks:
    h.remove()
validation_inputs = [image(e) for e in valid[:8]]
with torch.inference_mode():
    reference = [wrapper(x).clone() for x in validation_inputs]
for name in targets:
    parent, key = name.rsplit(".", 1)
    setattr(
        wrapper.get_submodule(parent),
        key,
        QDQLinear(wrapper.get_submodule(name), amax[name]),
    )
with torch.inference_mode():
    metrics = [
        {
            "cosine": nn.functional.cosine_similarity(
                ref.flatten(), wrapper(x).flatten(), dim=0
            ).item()
        }
        for x, ref in zip(validation_inputs, reference)
    ]
wrapper = wrapper.cpu()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, 448, 448),),
        str(
            r
            / (
                "export/vision_int8_fc1.onnx"
                if args.fc2_prefix == 0
                else "export/vision_int8.onnx"
            )
        ),
        input_names=["pixel_values"],
        output_names=["image_features"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
(r / ("vision_fc1_ptq.json" if args.fc2_prefix == 0 else "vision_ptq.json")).write_text(
    json.dumps(
        dict(
            quantized_linears=len(targets),
            alpha=0.5,
            fc1=24,
            fc2_prefix=args.fc2_prefix,
            calibration_steps=len(calib),
            validation=metrics,
            trained_projector=json.loads((r / "provenance.json").read_text())[
                "checkpoint_sha256"
            ],
        ),
        indent=2,
    )
)
print("VISION_PTQ_COMPLETE", metrics, flush=True)
