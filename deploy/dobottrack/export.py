"""Export the current Box-before-Action checkpoint and instruction templates."""

# Support both direct scripts and python -m deploy.dobottrack.<entry>.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from deploy.dobottrack.common import model_parser, load, write_json
from deploy.dobottrack.components import Vision, Heads, Mask, query_embeddings
from deploy.dobottrack.core import BoxJointCore


def export(module, inputs, path, names, outputs):
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=names,
            output_names=outputs,
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    print("EXPORTED", path, flush=True)


def main():
    p = model_parser(__doc__)
    p.add_argument(
        "--instructions",
        required=True,
        type=Path,
        help="JSON list of supported instruction strings",
    )
    args = p.parse_args()
    r, cfg, model, agent = load(args)
    out = r / "export"
    out.mkdir(exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)
    instructions = json.loads(args.instructions.read_text())
    if (
        not isinstance(instructions, list)
        or not instructions
        or not all(isinstance(x, str) and x.strip() for x in instructions)
    ):
        raise ValueError(
            "--instructions must be a nonempty JSON list of nonempty strings"
        )
    templates, metadata = {}, []
    with torch.inference_mode():
        queries = query_embeddings(model)
        for i, instruction in enumerate(sorted(set(instructions))):
            counts = model.model_extensions.token_layout(8, 256).per_frame_counts
            text = "\n".join("<img>" + "<IMG_CONTEXT>" * k + "</img>" for k in counts)
            text += "\nUser: " + agent._effective_prompt(instruction) + "\nAssistant:"
            ids = model.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=agent.max_length,
            )["input_ids"].cuda()
            embeddings = model.internvl.get_input_embeddings()(ids)
            template = torch.cat((embeddings, queries.to(embeddings)), 1)
            slots = (ids[0] == model.img_context_token_id).nonzero().reshape(-1)
            if len(slots) != 368:
                raise ValueError("Unexpected visual placeholder layout")
            templates[f"template_{i}"] = (
                template.float().cpu().numpy().astype(np.float16)
            )
            templates[f"ids_{i}"] = ids.cpu().numpy()
            templates[f"slots_{i}"] = slots.cpu().numpy()
            metadata.append(
                {
                    "index": i,
                    "instruction": instruction,
                    "prompt_tokens": template.shape[1],
                }
            )
    np.savez_compressed(r / "templates.npz", **templates)
    write_json(r / "templates.json", metadata)
    # Export only this checkpoint's Qwen weights, including its learned embedding table.
    llm = model.internvl.language_model.cpu()
    llm_config = llm.config.to_dict()
    llm_config.update(
        architectures=["Qwen2ForCausalLM"], tie_word_embeddings=False, rope_scaling=None
    )
    hf = out / "qwen_hf"
    hf.mkdir(exist_ok=True)
    write_json(hf / "config.json", llm_config)
    from safetensors.torch import save_file

    state = {
        "model." + k: v.detach().cpu().contiguous() for k, v in llm.state_dict().items()
    }
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    save_file(state, str(hf / "model.safetensors"), metadata={"format": "pt"})
    del state
    model.cpu().float()
    model.internvl.vision_tower.config._attn_implementation = "eager"
    vision = Vision(model.internvl).eval()
    pixels = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        torch.testing.assert_close(
            vision(pixels), model.internvl.get_image_features(pixel_values=pixels)
        )
    export(vision, (pixels,), out / "vision.onnx", ["pixel_values"], ["image_features"])
    compressor = model.history_compressor.eval()
    core = BoxJointCore(copy.deepcopy(compressor)).eval()
    features = torch.randn(7, 256, 896)
    boxes = torch.tensor([[0.5, 0.5, 0.3, 0.4]] * 7)
    valid = torch.tensor([True, False, True, True, False, True, True])
    with torch.no_grad():
        torch.testing.assert_close(
            core(features, boxes, valid),
            compressor(
                features[None], history_boxes=boxes[None], history_box_valid=valid[None]
            )[0],
            atol=2e-5,
            rtol=2e-4,
        )
    export(
        core,
        (features, boxes, valid),
        out / "qformer_fp.onnx",
        ["features", "boxes", "valid"],
        ["compressed_tokens"],
    )
    query_count = queries.shape[1]
    export(
        Heads(model),
        (torch.randn(1, query_count, 896),),
        out / "heads.onnx",
        ["queries"],
        ["trajectory", "grounding_bbox", "grounding_visibility_logits"],
    )
    if model.mask_auxiliary_head is not None:
        mask = Mask(model)
        mask.head.empty_dense = mask.head.empty_dense.detach()
        mask.head.sam_pe = mask.head.sam_pe.detach()
        export(
            mask,
            (torch.randn(1, 256, 896), torch.randn(1, 896)),
            out / "mask.onnx",
            ["features", "query"],
            ["mask_logits"],
        )
    write_json(
        r / "deployment.json",
        {
            "schema": 1,
            "query_count": query_count,
            "mask_query": model.mask_query_token is not None,
            "box_index": int(model.mask_query_token is not None),
            "action_start": 1 + int(model.mask_query_token is not None),
            "action_count": model.num_queries,
            "hidden_size": 896,
            "llm_layers": llm.config.num_hidden_layers,
            "kv_heads": llm.config.num_key_value_heads,
            "head_size": llm.config.hidden_size // llm.config.num_attention_heads,
            "max_sequence_length": int(cfg["data"].get("max_length", 1200))
            + query_count,
            "provenance": json.loads((r / "provenance.json").read_text()),
        },
    )


if __name__ == "__main__":
    main()
