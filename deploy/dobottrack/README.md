# DobotTrack on Jetson Orin

The deployment follows the desktop Box/Joint16 quantization method, adapted to the current model's **Mask → Box → eight Action queries**. Mask is optional; when enabled, its query remains in the LLM even if mask decoding is skipped. Historical targets use cached **predicted** boxes, with the same clipping, tiny-box fallback and no-target token as training code.

Supported recipe: InternVL3-1B, Q-identity16, bbox selection without box embedding, eight observations at 448×448, eight future waypoints. Other shapes/backbones fail validation instead of silently reusing these graphs.

## Environment

Export/calibration needs the project's model environment, SAM2, `onnx`, and `safetensors`. Native builds and replay additionally need **TensorRT and TensorRT-LLM 0.12.x** in the compatible Orin environment used by the original method. `qwen_graph.py` uses that release's builder API. Build engines on the target device; desktop/older engines are not portable artifacts.

```bash
python -m pip install onnx safetensors
```

Use the NVIDIA PyTorch/TensorRT environment appropriate to the target JetPack. Do not install a generic desktop TensorRT wheel over the device's runtime.

## 1. Export the trained checkpoint

Run from the repository root. `config.yaml` must describe a checkpoint trained with **Box before Action**; moving a learned query at deployment time does not convert an old Box-after-Action policy.

```bash
export WORK="$PWD/outputs/orin_dobottrack"
export CKPT=/path/to/checkpoints/model.ckpt
export CONFIG=/path/to/checkpoints/config.yaml
export BASE=/path/to/InternVL3-1B-hf
export SAM=/path/to/sam2.1_hiera_base_plus.pt
export DATA=/path/to/frame_archives

python deploy/dobottrack/export.py \
  --work "$WORK" --checkpoint "$CKPT" --config "$CONFIG" \
  --base "$BASE" --sam "$SAM" --instructions instructions.json
```

`instructions.json` is a nonempty JSON list of the exact instructions used for replay/deployment. Export includes their text embeddings and the checkpoint's learned query embeddings. An unexported instruction fails explicitly. On the original NVIDIA torch 2.5 Jetson environment, add `--jetson` for the SDPA compatibility adjustment.

Outputs include FP ONNX components, this checkpoint's Qwen weights, templates, query-layout metadata, and checkpoint/config SHA256 provenance. Use a fresh work directory for a different checkpoint, configuration or query order. Old engines, calibration ranges and prompt templates must all be regenerated.

## 2. Collect calibration and validation streams

Prepare `streams.json` from chronological episode records, with explicit episode-disjoint splits:

```json
{
  "streams": [
    {
      "split": "calibration",
      "records": [
        {
          "frames_file": "episode_a.jpgs",
          "current": "000001.jpg",
          "instruction": "Follow the person in the red shirt.",
          "trajectory": [[0.1, 0.0, 0.0]]
        }
      ]
    },
    {
      "split": "validation",
      "records": [
        {
          "frames_file": "episode_b.jpgs",
          "current": "000001.jpg",
          "instruction": "Follow the person in the red shirt.",
          "trajectory": [[0.1, 0.0, 0.0]]
        }
      ]
    }
  ]
}
```

This shows the record schema only: supply at least **24 successive records per episode** for the default 14-observation warmup plus ten measured decisions, and **eight `(x,y,yaw)` future points per record**. `frames_file` is relative to `$DATA`; ZIP and TAR frame archives are accepted. Keep record cadence consistent with training. All instructions must be included in `instructions.json`. Use multiple representative episodes in each split; the historical desktop experiment used six calibration and seven validation episodes.

```bash
python deploy/dobottrack/collect.py \
  --work "$WORK" --checkpoint "$CKPT" --config "$CONFIG" \
  --base "$BASE" --sam "$SAM" --data "$DATA" --manifest streams.json
```

Only calibration episodes enter `llm_calibration.npz` and Q-identity/vision activation ranges. Warmup predictions update the historical Box cache causally; ground-truth boxes are never supplied. Each new source frame is encoded once.

## 3. Quantize

LLM: SmoothQuant **alpha 0.8**, per-channel weights/per-token activations, using the newly captured visual/text/**Box-before-Action** embeddings. Set `--samples` to `calibration_samples` in `collection.json`.

```bash
python deploy/dobottrack/quantize_llm.py \
  --model-dir "$WORK/export/qwen_hf" --calibration "$WORK/llm_calibration.npz" \
  --samples 60 --smoothing-strength 0.8 --output "$WORK/export/llm_checkpoint"
```

Q-identity: the default FP graph is the reference. The optional FFN candidate quantizes six Linear layers across three blocks; the wider candidate quantizes thirty attention-projection/FFN Linear layers. Attention matrix products, softmax, normalization, residuals and input/output projections remain floating point.

```bash
python deploy/dobottrack/quantize_core.py \
  --work "$WORK" --checkpoint "$CKPT" --config "$CONFIG" \
  --base "$BASE" --sam "$SAM" --scope ffn
# Optional wider scope: --scope attention_ffn
```

Optional vision candidate: SmoothQuant alpha 0.5, all 24 fc1 layers and the first 12 fc2 layers, static absmax W8A8 ranges from this checkpoint's calibration streams. This is the self-contained max-calibration variant, not the desktop's later entropy-calibrated graph reuse.

```bash
python deploy/dobottrack/quantize_vision.py \
  --work "$WORK" --checkpoint "$CKPT" --config "$CONFIG" \
  --base "$BASE" --sam "$SAM" --data "$DATA" --fc2-prefix 12
```

## 4. Build on Orin

Transfer the **same work directory** and base tokenizer to the target. Keep all component weights and calibration results from the same export. This path uses native TRT-LLM SmoothQuant plugins, without requiring the desktop machine's private norm-fusion `.so`.

```bash
python deploy/dobottrack/build.py --work "$WORK"
# Candidate build (after the corresponding quantization steps):
python deploy/dobottrack/build.py --work "$WORK" \
  --vision vision_int8 --qformer qformer_int8
```

The default uses FP vision/Q-identity/heads plus W8A8 LLM. Candidate builds replace the selected engine files in that work directory; retain separate work directories if comparing different TRT-LLM builds. `build_logs/`, per-engine `.layers.json` and `engine_manifest.json` record build evidence and hashes. `--dry-run` prints commands without requiring TensorRT. The LLM profile covers the configured prompt limit, rather than hard-coding the old 512-token limit.

## 5. Causal replay and application use

```bash
python deploy/dobottrack/replay.py \
  --work "$WORK" --base "$BASE" --data "$DATA"
# Compare a built candidate:
python deploy/dobottrack/replay.py \
  --work "$WORK" --base "$BASE" --data "$DATA" \
  --vision vision_int8 --qformer qformer_int8
```

Replay reports held-out ADE/FDE, P50/P95/mean latency and per-decision trajectory/Box outputs. Timing includes preprocessing, one new vision encoding, cached history assembly, Q-identity, LLM, heads and CPU postprocessing; archive decoding, loading and simulator/robot time are excluded. Quantized candidates require trajectory/Box agreement and held-out validation before use. The desktop's old latency numbers belong to a different checkpoint and optimized runtime, and are not measured results for this export.

For application integration, use `runtime_trt.Policy` with `TrackVLAHabitatAgent` and `common.StreamingVision` as in `replay.py`: call `agent.observe(rgb)` then `agent.predict(instruction)`, and reset both agent and feature cache at episode boundaries. No robot commands are sent by these tools. `--skip-mask-decoder` skips mask output while preserving the trained Mask query and Box/Action order.
