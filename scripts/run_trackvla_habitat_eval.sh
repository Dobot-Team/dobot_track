#!/usr/bin/env bash
# Launch TrackVLA Habitat evaluation in GPU-sized dataset shards.
# Run this script from TrackOpen's compatible environment, normally:
# bash scripts/run_trackvla_habitat_eval.sh --checkpoint outputs/trackvla_internvl3_mlp6_mutitoken/checkpoints/last.ckpt --chunks 140 \
# --save-video --seed 42 --config outputs/trackvla_internvl3_mlp6_mutitoken/trackvla_internvl3_vla.yaml \
# --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml --save-path outputs/trackvla_internvl3_mlp6_mutitoken/stt

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACKVLA_ROOT="${TRACKVLA_ROOT:-$ROOT_DIR/trackvla_bench}"
TRACKVLA_CODE_ROOT="${TRACKVLA_CODE_ROOT:-}"
TRACKVLA_ASSETS_ROOT="${TRACKVLA_ASSETS_ROOT:-}"
CONFIG="${CONFIG:-}"
CHECKPOINT="${CHECKPOINT:-}"
EXP_CONFIG="${EXP_CONFIG:-habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml}"
GPUS="${GPUS:-0}"
CHUNKS="${CHUNKS:-128}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"
SEED="${SEED:-0}"
SAVE_PATH="${SAVE_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-}"
MODEL_PATH="${MODEL_PATH:-}"
CONTROL_DT="${CONTROL_DT:-0.1}"
WAYPOINT_INDEX="${WAYPOINT_INDEX:-0}"
INFERENCE_INTERVAL="${INFERENCE_INTERVAL:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
SHOW_MASKS="${SHOW_MASKS:-false}"
VIDEO_FPS="${VIDEO_FPS:-10}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
EXTRA_OPTS=()

usage() {
    cat <<'EOF'
Usage:
  conda run -n evt_bench bash scripts/run_trackvla_habitat_eval.sh \
    --checkpoint outputs/.../checkpoints/last.ckpt \
    --save-path outputs/trackvla_habitat_eval [options] [-- habitat.option=value]

Options:
  --checkpoint PATH       Lightning checkpoint or converted DeepSpeed state; finds nearest config.yaml by default
  --config PATH           Optional YAML override when no suitable config.yaml is present
  --trackvla-root PATH    Full TrackOpen checkout or local asset pack (default: trackvla_bench)
  --trackvla-code-root PATH
                            TrackOpen checkout with habitat-lab and evt_bench
  --trackvla-assets-root PATH
                            Asset pack with data/; overrides assets from --trackvla-root
  --exp-config PATH       Habitat experiment config relative to TrackOpen code
  --gpus LIST             Comma-separated CUDA IDs, default: 0
  --chunks N              Dataset split count, default: 128
  --start-idx N           First split to launch, default: 0
  --end-idx N             Last split to launch; -1 launches all
  --seed N                Habitat simulator seed
  --save-path PATH        Output root for episode JSON and shard summaries
  --model-path PATH       Override model.model_path from the checkpoint config or --config
  --device DEVICE         Torch device passed to each worker
  --control-dt SEC        TrackVLA label integration interval, default: 0.1
  --waypoint-index N      Predicted waypoint used for the action, default: 0
  --inference-interval N  Infer every N simulator ticks, default: 1
  --save-video            Save and annotate every rendered RGB frame in MP4 files
  --show-masks            Overlay available SAM2 masks in saved video frames
  --video-fps N           MP4 playback FPS, default: 10
  --continue-on-error     Continue after failed splits (default); exits 1 at the end
  --fail-fast             Stop after the first batch containing a failed split
  -h, --help              Show this help
EOF
}

while (($#)); do
    case "$1" in
        --checkpoint|--config|--trackvla-root|--trackvla-code-root|--trackvla-assets-root|--exp-config|--gpus|--chunks|--start-idx|--end-idx|--seed|--save-path|--model-path|--device|--control-dt|--waypoint-index|--inference-interval|--video-fps)
            if (($# < 2)); then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --checkpoint) CHECKPOINT="$2" ;;
                --config) CONFIG="$2" ;;
                --trackvla-root) TRACKVLA_ROOT="$2" ;;
                --trackvla-code-root) TRACKVLA_CODE_ROOT="$2" ;;
                --trackvla-assets-root) TRACKVLA_ASSETS_ROOT="$2" ;;
                --exp-config) EXP_CONFIG="$2" ;;
                --gpus) GPUS="$2" ;;
                --chunks) CHUNKS="$2" ;;
                --start-idx) START_IDX="$2" ;;
                --end-idx) END_IDX="$2" ;;
                --seed) SEED="$2" ;;
                --save-path) SAVE_PATH="$2" ;;
                --model-path) MODEL_PATH="$2" ;;
                --device) DEVICE="$2" ;;
                --control-dt) CONTROL_DT="$2" ;;
                --waypoint-index) WAYPOINT_INDEX="$2" ;;
                --inference-interval) INFERENCE_INTERVAL="$2" ;;
                --video-fps) VIDEO_FPS="$2" ;;
            esac
            shift 2
            ;;
        --save-video)
            SAVE_VIDEO=true
            shift
            ;;
        --show-masks)
            SHOW_MASKS=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --fail-fast)
            CONTINUE_ON_ERROR=false
            shift
            ;;
        --)
            shift
            EXTRA_OPTS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "Set --checkpoint PATH or CHECKPOINT=/path/to/last.ckpt." >&2
    exit 2
fi
if [[ -z "$SAVE_PATH" ]]; then
    SAVE_PATH="$ROOT_DIR/outputs/trackvla_habitat_eval"
fi
if ! [[ "$START_IDX" =~ ^[0-9]+$ ]]; then
    echo "--start-idx must be a non-negative integer." >&2
    exit 2
fi
if ((START_IDX >= CHUNKS)); then
    echo "--start-idx must be smaller than --chunks." >&2
    exit 2
fi
if ((END_IDX != -1 && END_IDX < START_IDX)); then
    echo "--end-idx must be -1 or no smaller than --start-idx." >&2
    exit 2
fi
if [[ "$CONTINUE_ON_ERROR" != "true" && "$CONTINUE_ON_ERROR" != "false" ]]; then
    echo "CONTINUE_ON_ERROR must be true or false." >&2
    exit 2
fi

if [[ -n "$CONFIG" ]]; then
    CONFIG="$(realpath -e "$CONFIG")"
fi
CHECKPOINT="$(realpath -e "$CHECKPOINT")"
TRACKVLA_ROOT="$(realpath -e "$TRACKVLA_ROOT")"
if [[ -n "$TRACKVLA_CODE_ROOT" ]]; then
    TRACKVLA_CODE_ROOT="$(realpath -e "$TRACKVLA_CODE_ROOT")"
fi
if [[ -n "$TRACKVLA_ASSETS_ROOT" ]]; then
    TRACKVLA_ASSETS_ROOT="$(realpath -e "$TRACKVLA_ASSETS_ROOT")"
fi
if [[ "$EXP_CONFIG" == /* ]]; then
    EXP_CONFIG="$(realpath -e "$EXP_CONFIG")"
fi
SAVE_PATH="$(realpath -m "$SAVE_PATH")"
mkdir -p "$SAVE_PATH"
FAILED_SPLITS_FILE="$SAVE_PATH/failed_splits.tsv"
if [[ "$CONTINUE_ON_ERROR" == "true" ]]; then
    printf "split_id\texit_code\n" > "$FAILED_SPLITS_FILE"
fi

trap 'echo "Stopping child jobs"; kill $(jobs -p) 2>/dev/null || true; exit 130' INT TERM

GPUS="${GPUS//,/ }"
read -r -a GPU_LIST <<< "$GPUS"
if ((${#GPU_LIST[@]} == 0)); then
    echo "--gpus must contain at least one CUDA device ID." >&2
    exit 2
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

IDX="$START_IDX"
ANY_FAILED=0
while ((IDX < CHUNKS)); do
    PIDS=()
    SPLIT_IDS=()
    for GPU in "${GPU_LIST[@]}"; do
        if ((IDX >= CHUNKS)); then
            break
        fi
        if ((END_IDX != -1 && IDX > END_IDX)); then
            echo "Reached END_IDX=$END_IDX, stop."
            break 2
        fi

        echo "[sim-eval] Launching split $IDX/$CHUNKS on CUDA device $GPU"
        COMMAND=(
            "$PYTHON_BIN" "$ROOT_DIR/scripts/eval_trackvla_habitat.py"
            --checkpoint "$CHECKPOINT"
            --trackvla-root "$TRACKVLA_ROOT"
            --exp-config "$EXP_CONFIG"
            --split-num "$CHUNKS"
            --split-id "$IDX"
            --save-path "$SAVE_PATH"
            --control-dt "$CONTROL_DT"
            --waypoint-index "$WAYPOINT_INDEX"
            --inference-interval "$INFERENCE_INTERVAL"
        )
        if [[ -n "$CONFIG" ]]; then
            COMMAND+=(--config "$CONFIG")
        fi
        if [[ -n "$MODEL_PATH" ]]; then
            COMMAND+=(--model-path "$MODEL_PATH")
        fi
        if [[ -n "$TRACKVLA_CODE_ROOT" ]]; then
            COMMAND+=(--trackvla-code-root "$TRACKVLA_CODE_ROOT")
        fi
        if [[ -n "$TRACKVLA_ASSETS_ROOT" ]]; then
            COMMAND+=(--trackvla-assets-root "$TRACKVLA_ASSETS_ROOT")
        fi
        if [[ -n "$DEVICE" ]]; then
            COMMAND+=(--device "$DEVICE")
        fi
        if [[ "$SAVE_VIDEO" == "true" ]]; then
            COMMAND+=(
                --save-video
                --video-fps "$VIDEO_FPS"
            )
        fi
        if [[ "$SHOW_MASKS" == "true" ]]; then
            COMMAND+=(--show-masks)
        fi
        COMMAND+=("${EXTRA_OPTS[@]}" "habitat.simulator.seed=$SEED")
        CUDA_VISIBLE_DEVICES="$GPU" "${COMMAND[@]}" &
        PIDS+=("$!")
        SPLIT_IDS+=("$IDX")
        ((IDX += 1))
    done

    FAILED=0
    for PID_INDEX in "${!PIDS[@]}"; do
        PID="${PIDS[$PID_INDEX]}"
        SPLIT_ID="${SPLIT_IDS[$PID_INDEX]}"
        if wait "$PID"; then
            :
        else
            EXIT_CODE=$?
            FAILED=1
            ANY_FAILED=1
            echo "[sim-eval] Split $SPLIT_ID failed with exit code $EXIT_CODE." >&2
            if [[ "$CONTINUE_ON_ERROR" == "true" ]]; then
                printf "%s\t%s\n" "$SPLIT_ID" "$EXIT_CODE" >> "$FAILED_SPLITS_FILE"
            fi
        fi
    done
    if ((FAILED)); then
        if [[ "$CONTINUE_ON_ERROR" == "true" ]]; then
            echo "At least one Habitat shard failed; continuing with later splits." >&2
        else
            echo "At least one Habitat shard failed; stopping." >&2
            exit 1
        fi
    fi
done

if ((ANY_FAILED)); then
    echo "Evaluation finished with failed splits; see $FAILED_SPLITS_FILE" >&2
    exit 1
fi
