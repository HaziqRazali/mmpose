#!/usr/bin/env bash
set -euo pipefail

# MediaPipe Pose batch runner (with TEST_MODE + SINGLE_VIDEO safety switches)
# Mirrors run_mmpose_small_mocap_dataset.sh — same env vars, same video glob,
# same SKIP/RUN/FORCE logic, same parallel job pool.
#
# Input videos:
#   <DATA_ROOT>/<split>/<subject>/videos/<camera>/<action>.{mp4,avi}
#
# Output:
#   <DATA_ROOT>/<split>/<subject>/mediapipe/<camera>/<action>.json
#
# Usage (same patterns as the MMPose runner):
#   TEST_MODE=1 FORCE=0 DATA_ROOT_BASE=/data/haziq/mocap/data ./run_mediapipe_mocap_dataset.sh fit3d
#   PARALLEL_JOBS=10 TEST_MODE=1 FORCE=0 DATA_ROOT_BASE=/data/haziq/mocap/data ./mediapipe/run_mediapipe_mocap_dataset.sh fit3d | tee mediapipe.txt
#   SINGLE_VIDEO=1 DATA_ROOT_BASE=/data/haziq/mocap/data ./mediapipe/run_mediapipe_mocap_dataset.sh fit3d
#   FORCE=1 DATA_ROOT_BASE=/data/haziq/mocap/data ./mediapipe/run_mediapipe_mocap_dataset.sh fit3d
#
# Parallelism notes:
#   - PARALLEL_JOBS controls how many inference processes run simultaneously.
#   - MediaPipe runs on CPU; set PARALLEL_JOBS to however many cores you want
#     to saturate (default: 1 = sequential).
#   - SINGLE_VIDEO=1 forces sequential mode (exit semantics need the main shell).
#
# If you don't pass an arg, it defaults to "kit".

# Absolute path to the directory containing this script (works regardless of cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------- ARG: dataset name ----------------
DATASET_NAME="${1:-kit}"
DATA_ROOT_BASE="${DATA_ROOT_BASE:-/home/haziq/datasets/mocap/data}"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_BASE}/${DATASET_NAME}}"

# Optional sanity check (helps catch typos like "kti")
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT does not exist: ${DATA_ROOT}"
  echo "        You ran: $0 ${DATASET_NAME}"
  exit 1
fi

# ---------------- settings ----------------
TEST_MODE="${TEST_MODE:-0}"
SINGLE_VIDEO="${SINGLE_VIDEO:-0}"
FORCE="${FORCE:-0}"          # FORCE=1 -> do not skip even if JSON exists
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"   # number of simultaneous inference workers

# SINGLE_VIDEO exit semantics only work in the main shell, not subshells
if [[ "${SINGLE_VIDEO}" == "1" ]]; then
  PARALLEL_JOBS=1
fi

shopt -s nullglob

# Export all variables that the worker function needs
export SCRIPT_DIR DATA_ROOT DATASET_NAME
export TEST_MODE FORCE SINGLE_VIDEO

# ---------------------------------------------------------------------------
# Worker function: process a single video file.
# Runs in a subshell when PARALLEL_JOBS > 1, so uses return instead of exit.
# ---------------------------------------------------------------------------
_process_video() {
  local f="$1"

  local action_file action_name camera subject split
  action_file="$(basename "$f")"                                               # e.g., w_raise.mp4
  action_name="${action_file%.*}"                                              # e.g., w_raise
  camera="$(basename "$(dirname "$f")")"
  subject="$(basename "$(dirname "$(dirname "$(dirname "$f")")")")"
  split="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$f")")")")")"

  local out_dir out_json
  out_dir="${DATA_ROOT}/${split}/${subject}/mediapipe/${camera}"
  out_json="${out_dir}/${action_name}.json"

  echo "Processing dataset=${DATASET_NAME} split=${split} subject=${subject} camera=${camera} action=${action_name}"
  echo "  input:    ${f}"
  echo "  out_json: ${out_json}"

  # Always print decision status (even in TEST_MODE)
  if [[ -f "${out_json}" ]]; then
    if [[ "${FORCE}" == "1" ]]; then
      echo "  [FORCE] JSON exists -> will re-run (FORCE=1)."
    else
      echo "  [SKIP]  JSON exists -> will skip (FORCE!=1)."
    fi
  else
    echo "  [RUN]   JSON missing -> will run."
  fi

  # Skip logic
  if [[ "${FORCE}" != "1" && -f "${out_json}" ]]; then
    echo
    return 0
  fi

  if [[ "${TEST_MODE}" == "1" ]]; then
    echo "  TEST_MODE=1 -> not creating dirs, not running inference."
    echo
    return 0
  fi

  mkdir -p "${out_dir}"

  python "${SCRIPT_DIR}/run.py" \
    --input "${f}" \
    --output-root "${out_dir}" \
    --save-predictions \
    --save-video

  echo
}
export -f _process_video

# ---------------------------------------------------------------------------
# Job pool: dispatch up to PARALLEL_JOBS workers; wait -n frees a slot when
# any one worker finishes.  SINGLE_VIDEO=1 is forced to sequential above.
# ---------------------------------------------------------------------------
_job_count=0

for f in "${DATA_ROOT}"/{train,val}/*/videos/*/*.{avi,mp4}; do
  if (( PARALLEL_JOBS > 1 )); then
    _process_video "$f" &
    (( ++_job_count ))
    if (( _job_count >= PARALLEL_JOBS )); then
      wait -n 2>/dev/null || true   # free one slot; bash 4.3+ required
      (( --_job_count ))
    fi
  else
    # Sequential path — SINGLE_VIDEO exit semantics work normally here
    _process_video "$f"
    if [[ "${SINGLE_VIDEO}" == "1" ]]; then
      echo "SINGLE_VIDEO=1 -> stopping after first video."
      break
    fi
  fi
done

# Wait for all remaining background workers
wait
