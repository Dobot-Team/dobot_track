# DobotTrack optimized deployment on Jetson Orin

The **default `optimized` recipe** contains the complete low-latency path: native
TensorRT INT8 Vision, FP16 physically Box-selected Joint QFormer, SmoothQuant W8A8
LLM with the included RMSNorm/quant CUDA plugin, persistent buffers, CPU prompt/Box
caches and CUDA Graphs. It preserves the trained **Mask → Box → eight Action
queries**. Mask decoding is skipped; its learned query remains in the LLM.

Supported model: InternVL3-1B, Q-identity16, bbox target selection without box
embedding, eight 448×448 observations, eight future waypoints. Unsupported
architectures fail validation. Engines, prompts and calibration are checkpoint
specific; build engines on the target device.

## Environments

Use the project's model environment and pinned SAM2 source from the root README
for desktop export/collection. The reference desktop environment uses Torch 2.8,
Transformers 4.57.6 and ONNX 1.17. For desktop entropy calibration, install the
isolated dependencies:

```bash
python -m pip install -r deploy/dobottrack/requirements-calibration.txt
```

Orin build/replay requires the compatible NVIDIA Torch/TensorRT environment,
**TensorRT 10.4 and TensorRT-LLM 0.12.x**, CUDA compiler and C++ toolchain. The
reference runtime uses NVIDIA Torch 2.5 and Transformers 4.46.2. Do not install
the desktop calibration or training requirements over that pinned runtime.
The included LLM plugin targets `sm_87`; `NVCC`, `TENSORRT_ROOT` and `OUTPUT_DIR` can override
its build paths. No prebuilt private plugin, engine or experiment directory is
required. No custom ViT plugin is used.

## 1. Export on the desktop

Use a fresh work directory. `config.yaml` must describe a checkpoint **trained
with Box before Action**; moving queries at deployment does not convert an old
Box-after-Action checkpoint.

```bash
export WORK="$PWD/outputs/orin_dobottrack"
export CKPT=/path/to/model.ckpt
export CONFIG=/path/to/config.yaml
export BASE=/path/to/InternVL3-1B-hf
export SAM=/path/to/sam2.1_hiera_base_plus.pt
export DATA=/path/to/frame_archives

python deploy/dobottrack/export.py \
  --work "$WORK" --checkpoint "$CKPT" --config "$CONFIG" \
  --base "$BASE" --sam "$SAM" --instructions instructions.json
```

`instructions.json` is a nonempty JSON list of the exact deployment instructions.
Export saves their embeddings and the learned queries; an unexported instruction
fails explicitly. The optimized LLM profile rounds the largest exported prompt
up to a multiple of 128 tokens, with a minimum of 512. It does not truncate
exported templates. Add `--jetson` if exporting with NVIDIA Torch 2.5's SDPA compatibility requirement.

## 2. Collect calibration and validation episodes

Create `streams.json` with explicit, episode-disjoint splits:

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

The JSON above shows the schema, not a sufficient calibration set. Supply **six
calibration episodes with at least 32 consecutive records each**, and separate
validation episodes with at least 24 records each. Every record needs all eight
`(x,y,yaw)` future points. Keep the training observation cadence and include all
instructions in the exported list. `frames_file` is relative to `$DATA`; ZIP and
TAR frame archives are supported. Choose representative calibration episodes.

```bash
python deploy/dobottrack/collect.py \
  --work "$WORK" --checkpoint "$CKPT" --config "$CONFIG" \
  --base "$BASE" --sam "$SAM" --data "$DATA" --manifest streams.json
```

The default is 14 warmup observations plus ten measured decisions per episode.
Only calibration episodes enter the ranges. Historical Box predictions update
causally; **ground-truth boxes are not used**. Each new RGB frame is encoded once
and historical features are cached.

## 3. Calibrate Vision on the desktop

```bash
python deploy/dobottrack/prepare_vision.py \
  --work "$WORK" --checkpoint "$CKPT" --config "$CONFIG" \
  --base "$BASE" --sam "$SAM" --data "$DATA"
python deploy/dobottrack/quantize_vision_entropy.py --work "$WORK"
```

The recipe uses 192 frames for SmoothQuant alpha 0.5; entropy calibration on the
first 64 frames for all 24 fc1 and the first 12 fc2 projections; absmax on 60 native
calibration decisions for 96 Q/K/V/output projections. These **132 MatMuls** use
W8A8, with per-output-channel weights. Attention matrix multiplication, softmax,
normalization and residuals remain floating point. Compile-time constant weight
quantizers are folded to INT8 initializers, preserving ONNX rounding/saturation.
Final graph: `export/vision_attention_int8.onnx`.

## 4. Quantize the LLM and build on Orin

Transfer the entire work directory and use the same source revision. Set `WORK`,
`BASE` and `DATA` to device-local paths. Run in the pinned TensorRT-LLM environment:

```bash
python deploy/dobottrack/quantize_llm.py \
  --model-dir "$WORK/export/qwen_hf" --calibration "$WORK/llm_calibration.npz" \
  --samples 60 --smoothing-strength 0.8 --output "$WORK/export/llm_checkpoint"
python deploy/dobottrack/build.py --work "$WORK"
```

The default build compiles the included LLM plugin locally, builds the 132-MatMul
Vision graph, compact FP16 QFormer and FP16 task heads, then builds the fused
W8A8 LLM. It forces one context-only optimization profile. The physical target
selection gathers Box-covered patch tokens into buckets 32/64/128/256, retaining
no-target fallback/padding semantics. Query-only self-attention and full 256-token
visual cross-attention are unchanged. The dense and compact FP32 exports are
checked for equivalence; optionally verify all captured native inputs:

```bash
python deploy/dobottrack/verify_compact.py \
  --work "$WORK" --checkpoint "$CKPT" --config "$CONFIG" \
  --base "$BASE" --sam "$SAM"
```

`build_logs/`, `*.layers.json` and `engine_manifest.json` record build evidence
and engine/plugin SHA256 hashes. `build.py --dry-run` prints commands without
requiring TensorRT. No Mask decoder engine is built in the optimized recipe.

## 5. Causal replay and application integration

```bash
python deploy/dobottrack/replay.py --work "$WORK" --base "$BASE" --data "$DATA"
# Separate synchronized module timing (adds overhead; not policy latency):
python deploy/dobottrack/replay.py \
  --work "$WORK" --base "$BASE" --data "$DATA" --instrument
```

Results are `replay_vision_attention_int8_qformer_compact_fp_no_mask.json` and
`*_stages.json`. Policy percentiles cover all measured decisions; ADE/FDE cover
only validation episodes. Calibration-disjoint does **not** establish that data
were unseen in training. These trajectory regression metrics are not Habitat SR.
Replay timing includes RGB preprocessing, cached Vision/history assembly,
QFormer, LLM, task heads and CPU postprocessing. It excludes model loading,
archive decoding and simulator or robot execution.
The report contains CUDA Graph capture evidence. Optimized initialization fails
if graph capture fails rather than silently timing an unfused fallback.

For an application, follow `replay.py`: create `runtime_trt.Policy`, the
`TrackVLAHabitatAgent`, call `optimized.configure_agent(agent)`, and install
`common.StreamingVision`. Call `agent.observe(rgb)` then
`agent.predict(instruction)`; reset both the agent and feature cache at episode
boundaries. Keep the CPU historical Box cache hook: it enables physical target
gather without a GPU-to-CPU synchronization each frame. These tools send no
robot commands. A known prompt's first-use initialization and model loading are
outside steady-state latency.

## Resumable single-environment workflow

If one environment contains all stage dependencies:

```bash
python deploy/dobottrack/workflow.py \
  --work /local_disk/dobottrack_run \
  --checkpoint /models/policy.ckpt --config /models/config.yaml \
  --base /models/InternVL3-1B-hf --sam /models/sam2.1_hiera_base_plus.pt \
  --data /datasets/frames --manifest /datasets/streams.json \
  --instructions /datasets/instructions.json --samples 60
```

Add `--dry-run` to inspect the complete plan. The wrapper runs export → causal
collection → LLM quantization → Vision calibration → optimized build → replay.
It records input/source/output hashes and resumes verified stages. Changed
checkpoint/config/instructions/archives/sources require a new work directory.
The wrapper does not submit stages to another machine; use the numbered split
workflow above when the desktop and Orin have separate dependencies.

## Explicit baseline

The baseline path is available with `--recipe baseline`.
Use that flag consistently for **export, build and replay**, in a separate work
directory. It uses native FP16 Vision, dense FP16 QFormer and upstream W8A8 LLM;
Mask is decoded unless `replay.py --skip-mask-decoder` is passed. Baseline-only
INT8 candidates are separate experiments, not the optimized release default.
Never reuse templates, scales or engines across checkpoint/query-order changes.
