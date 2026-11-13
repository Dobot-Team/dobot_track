"""Check compact-vs-dense QFormer on every captured predicted-box decision."""

import sys, copy, json
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, torch
from deploy.dobottrack.common import model_parser, load, write_json
from deploy.dobottrack.core import BoxJointCore, CompactBoxJointCore
from deploy.dobottrack.compact_boxes import pack_box_indices


def main():
    args = model_parser(__doc__).parse_args()
    r, cfg, model, agent = load(args)
    source = copy.deepcopy(model.history_compressor).cpu().float()
    del model, agent
    torch.cuda.empty_cache()
    dense = BoxJointCore(copy.deepcopy(source)).cuda().eval()
    compact = CompactBoxJointCore(source).cuda().eval()
    rows = []
    with torch.inference_mode():
        for e in json.loads((r / "samples.json").read_text()):
            with np.load(r / "captured" / e["path"]) as d:
                indices, padding = pack_box_indices(d["boxes"], d["valid"])
                features = torch.from_numpy(d["features"]).cuda().float()
                a = dense(
                    features,
                    torch.from_numpy(d["boxes"]).cuda(),
                    torch.from_numpy(d["valid"]).cuda(),
                )
                b = compact(
                    features,
                    torch.from_numpy(indices).cuda(),
                    torch.from_numpy(padding).cuda(),
                )
                torch.testing.assert_close(b, a, atol=1e-4, rtol=1e-4)
                rows.append(
                    {
                        "index": e["index"],
                        "target_count": indices.shape[1],
                        "max_abs": float((a - b).abs().max()),
                    }
                )
    write_json(
        r / "compact_equivalence.json",
        {"samples": len(rows), "all_passed": True, "records": rows},
    )
    print("CAPTURED_COMPACT_EQUIVALENCE_PASS", len(rows))


if __name__ == "__main__":
    main()
