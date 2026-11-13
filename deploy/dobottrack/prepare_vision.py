import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
"""Fresh 448-pixel SmoothQuant192 and entropy64 inputs, calibration episodes only."""
import json, io, zipfile, tarfile
from pathlib import Path
import numpy as np, torch
from torch import nn
from PIL import Image
from deploy.dobottrack.common import load, model_parser as parser
from deploy.dobottrack.components import Vision

p = parser(__doc__)
p.add_argument("--data", required=True)
args = p.parse_args()
r, cfg, m, agent = load(args)
m.internvl.language_model.cpu()
m.internvl.vision_tower.float()
m.internvl.multi_modal_projector.float()
m.internvl.vision_tower.config._attn_implementation = "eager"
v = Vision(m.internvl).eval()
streams = [
    s["records"]
    for s in json.loads((r / "streams.json").read_text())["streams"]
    if s["split"] == "calibration"
]
pixels = []
sources = []
for stream in streams[:6]:
    path = Path(args.data) / stream[0]["frames_file"]
    z = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else tarfile.open(path)
    for e in stream[:32]:
        raw = (
            z.read(e["current"])
            if isinstance(z, zipfile.ZipFile)
            else z.extractfile(e["current"]).read()
        )
        x = agent._internvl_pixel_values(
            [np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))]
        ).float()
        pixels.append(x)
        sources.append([str(path), e["current"]])
    z.close()
assert len(pixels) == 192, len(pixels)
fc1 = [
    n
    for n, x in v.named_modules()
    if isinstance(x, nn.Linear) and n.endswith(".mlp.fc1")
]
attention = [
    n for n, x in v.named_modules() if isinstance(x, nn.Linear) and ".attention." in n
]
assert len(fc1) == 24 and len(attention) == 96
entries = [
    e
    for e in json.loads((r / "samples.json").read_text())
    if e["split"] == "calibration"
][:60]
assert len(entries) == 60, "Recipe requires sixty calibration decisions"
selected = {(str(Path(args.data) / e["episode"]), e["current"]) for e in entries}
ranges = {}
amax = {}
hooks = []
active = [False]
seen = 0
for name in fc1:

    def hook(mod, a, name=name):
        x = a[0].detach().abs().reshape(-1, a[0].shape[-1]).amax(0)
        ranges[name] = x if name not in ranges else torch.maximum(ranges[name], x)

    hooks.append(v.get_submodule(name).register_forward_pre_hook(hook))
for name in attention:

    def hook(mod, a, name=name):
        if active[0]:
            amax[name] = max(amax.get(name, 0), float(a[0].detach().abs().max()))

    hooks.append(v.get_submodule(name).register_forward_pre_hook(hook))
with torch.inference_mode():
    for i, x in enumerate(pixels):
        active[0] = tuple(sources[i]) in selected
        seen += int(active[0])
        v(x)
        if i % 32 == 31:
            print("SQ_CALIB", i + 1, flush=True)
assert seen == 60, (
    seen,
    "Selected attention frames must appear in the first32 frames of six calibration episodes",
)
for h in hooks:
    h.remove()
with torch.no_grad():
    for name in fc1:
        f = v.get_submodule(name)
        norm = v.get_submodule(name.rsplit(".mlp.fc1", 1)[0] + ".layernorm_after")
        scale = (
            (ranges[name].clamp_min(1e-5) / f.weight.abs().amax(0).clamp_min(1e-5))
            .sqrt()
            .clamp_min(1e-5)
        )
        norm.weight.div_(scale)
        norm.bias.div_(scale)
        f.weight.mul_(scale[None])
np.savez_compressed(
    r / "vision_entropy64.npz", pixel_values=torch.cat(pixels[:64]).cpu().numpy()
)
v = v.cpu()
(r / "export").mkdir(exist_ok=True)
with torch.no_grad():
    torch.onnx.export(
        v,
        (torch.randn(1, 3, 448, 448),),
        str(r / "export/vision_sq.onnx"),
        input_names=["pixel_values"],
        output_names=["image_features"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
(r / "vision_calibration.json").write_text(
    json.dumps(
        {
            "sq_samples": 192,
            "entropy_samples": 64,
            "alpha": 0.5,
            "calibration_sources": sources,
            "attention_amax": amax,
            "attention_calibration": "absmax60 native calibration decisions, per-tensor activation/per-channel weight",
            "mlp_quantization": "ModelOpt entropy64 all24fc1 + first12fc2",
            "validation_not_used": True,
        },
        indent=2,
    )
)
print("VISION_SQ_AND_CALIBRATION_READY", flush=True)
