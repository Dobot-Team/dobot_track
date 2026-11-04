#!/usr/bin/env bash
set -euo pipefail

# Strict-control-variable real-data conversion:
#   * simulation manifests and targets are read-only;
#   * real RGB/current frames are sampled at 10 Hz;
#   * eight real history frames are sampled at 5 Hz;
#   * real metric velocities are matched to simulation normalized actions;
#   * eight 10 Hz real future points are re-integrated in simulation label space.

ROOT_DIR="${ROOT_DIR:-/home/jovyan/work/embodyla_dobot_trackvla}"
REAL_DATASET="${REAL_DATASET:-/home/jovyan/work/real_trackvla/dataset.json}"
RAW_REAL_ROOT="${RAW_REAL_ROOT:-/home/jovyan/work/post_sync_res/pipeline_v2}"
SIM_ROOT="${SIM_ROOT:-/home/jovyan/work/dataset/sample}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jovyan/work/real_trackvla_sim_aligned_p95}"
REPORT_DIR="${REPORT_DIR:-${ROOT_DIR}/outputs/motion_audit_strict}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

STATS_JSON="${REPORT_DIR}/real_vs_sim_strict.json"
STATS_CSV="${REPORT_DIR}/real_vs_sim_strict.csv"

cd "${ROOT_DIR}"

if [[ "${SKIP_AUDIT:-0}" == "1" ]]; then
  echo "[1/3] Reuse completed motion audit: ${STATS_JSON}"
  test -f "${STATS_JSON}"
else
  echo "[1/3] Full real/simulation motion audit"
  python -u tools/audit_real_sim_motion.py \
    --real-dataset "${REAL_DATASET}" \
    --sim-root "${SIM_ROOT}" \
    --record-dt 0.1 \
    --integration-dt 0.025 \
    --output-json "${STATS_JSON}" \
    --output-csv "${STATS_CSV}" \
    2>&1 | tee "${LOG_DIR}/real_vs_sim_strict.log"
fi

eval "$(python - "${STATS_JSON}" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    report = json.load(handle)

scales = report["real_to_sim_normalization"]
values = {
    "FORWARD_SCALE": scales["forward_speed_scale_mps"]["p95"]["real_speed_scale"],
    "LATERAL_SCALE": scales["lateral_speed_scale_mps"]["p95"]["real_speed_scale"],
    "ANGULAR_SCALE": scales["angular_speed_scale_radps"]["p95"]["real_speed_scale"],
}
for name, value in values.items():
    if value is None or float(value) <= 0:
        raise RuntimeError(f"Invalid P95 scale {name}={value!r}")
    print(f"{name}={shlex.quote(format(float(value), '.12g'))}")
PY
)"

printf '[scales] forward=%s m/s lateral=%s m/s angular=%s rad/s\n' \
  "${FORWARD_SCALE}" "${LATERAL_SCALE}" "${ANGULAR_SCALE}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_ROOT}" >&2
  echo "Choose a new OUTPUT_ROOT or move the existing directory first." >&2
  exit 2
fi

echo "[2/3] Convert real data into unchanged simulation target space"
python -u tools/resample_real_trackvla.py \
  --source-dataset "${REAL_DATASET}" \
  --raw-source-root "${RAW_REAL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --current-hz 10 \
  --history-hz 5 \
  --history-frames 8 \
  --trajectory-hz 10 \
  --forward-speed-scale "${FORWARD_SCALE}" \
  --lateral-speed-scale "${LATERAL_SCALE}" \
  --angular-speed-scale "${ANGULAR_SCALE}" \
  --clip-normalized-speed \
  --max-alignment-error-ms 75 \
  --max-pose-alignment-error-ms 75 \
  --archive-mode hardlink \
  2>&1 | tee "${LOG_DIR}/real_trackvla_sim_aligned_conversion.log"

echo "[3/3] Validate converted manifest"
python -u tools/validate_real_trackvla_alignment.py \
  --dataset "${OUTPUT_ROOT}/dataset.json" \
  --output "${OUTPUT_ROOT}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/real_trackvla_sim_aligned_validation.log"

echo "Completed: ${OUTPUT_ROOT}"
