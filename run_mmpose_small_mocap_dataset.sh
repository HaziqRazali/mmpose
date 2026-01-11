#!/usr/bin/env bash
set -euo pipefail

# MMPose batch runner (with TEST_MODE + SINGLE_VIDEO safety switches)
# https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
#
# Input videos:
#   /media/haziq/Haziq/mocap/data/<dataset>/<train|val>/<subject>/videos/<camera>/<action>.mp4
#
# Desired outputs:
#   /media/haziq/Haziq/mocap/data/<dataset>/<train|val>/<subject>/mmpose/<model>/<camera>/<action>.json
#   /media/haziq/Haziq/mocap/data/<dataset>/<train|val>/<subject>/mmpose/<model>/<camera>/<action>.mp4
#
# NOTE:
# - Keeps --save-predictions.
# - By default, SKIPS videos if the output JSON already exists (resume-friendly).
# - Set FORCE=1 to re-run inference even if outputs already exist.
# - In TEST_MODE=1, it will still PRINT whether it would SKIP/RUN (and why).
# - Whether an output mp4 is produced depends on your demo script + flags.
#
# Usage:
#   TEST_MODE=1 FORCE=1 ./run_mmpose_small_mocap_dataset.sh self
#   TEST_MODE=1 FORCE=0 ./run_mmpose_small_mocap_dataset.sh self
#   SINGLE_VIDEO=1 ./run_mmpose_small_mocap_dataset.sh self
#   FORCE=1 ./run_mmpose_small_mocap_dataset.sh self
#   ./run_mmpose_small_mocap_dataset.sh self | tee mocap_self.txt
#
# If you don’t pass an arg, it defaults to "kit".

# ---------------- ARG: dataset name ----------------
DATASET_NAME="${1:-kit}"
#DATA_ROOT="/home/haziqmr/datasets/mocap/data/${DATASET_NAME}"
DATA_ROOT="/media/haziq/Haziq/mocap/data/${DATASET_NAME}"

# Optional sanity check (helps catch typos like "kti")
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT does not exist: ${DATA_ROOT}"
  echo "        You ran: $0 ${DATASET_NAME}"
  exit 1
fi

# ---------------- settings ----------------
MODEL_NAME="rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122"

TEST_MODE="${TEST_MODE:-0}"
SINGLE_VIDEO="${SINGLE_VIDEO:-0}"
FORCE="${FORCE:-0}"   # FORCE=1 -> do not skip even if JSON exists

DET_CFG="demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py"
DET_CKPT="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
POSE_CFG="configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py"
POSE_CKPT="https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth"

shopt -s nullglob

for f in "${DATA_ROOT}"/{train,val}/*/videos/*/*.{avi,mp4}; do
  # Expected:
  # .../<dataset>/<split>/<subject>/videos/<camera>/<action>.avi

  action_file="$(basename "$f")"      # e.g., 001.avi
  action_name="${action_file%.*}"     # e.g., 001

  camera="$(basename "$(dirname "$f")")"  # e.g., cam1 or 50591643

  # dirname 1 -> <camera>
  # dirname 2 -> videos
  # dirname 3 -> <subject>
  # dirname 4 -> <split>
  subject="$(basename "$(dirname "$(dirname "$(dirname "$f")")")")"
  split="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$f")")")")")"

  out_dir="${DATA_ROOT}/${split}/${subject}/mmpose/${MODEL_NAME}/${camera}"
  out_json="${out_dir}/results_${action_name}.json"
  out_mp4="${out_dir}/${action_name}.mp4"

  echo "Processing dataset=${DATASET_NAME} split=${split} subject=${subject} camera=${camera} action=${action_name}"
  echo "  input:     ${f}"
  echo "  out_json:  ${out_json}"
  echo "  out_video: ${out_mp4}  (only if your demo script actually writes a video)"

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

  # Skip logic:
  # - Default: if JSON exists, assume this video is done and skip it.
  # - FORCE=1: do NOT skip; always re-run.
  if [[ "${FORCE}" != "1" && -f "${out_json}" ]]; then
    echo
    if [[ "${SINGLE_VIDEO}" == "1" ]]; then
      echo "SINGLE_VIDEO=1 -> stopping after first encountered video (skipped)."
      exit 0
    fi
    continue
  fi

  if [[ "${TEST_MODE}" == "1" ]]; then
    echo "  TEST_MODE=1 -> not creating dirs, not running inference."
    echo
    if [[ "${SINGLE_VIDEO}" == "1" ]]; then
      echo "SINGLE_VIDEO=1 -> stopping after first encountered video (test mode)."
      exit 0
    fi
    continue
  fi

  mkdir -p "${out_dir}"

  python demo/topdown_demo_with_mmdet.py \
    "${DET_CFG}" \
    "${DET_CKPT}" \
    "${POSE_CFG}" \
    "${POSE_CKPT}" \
    --input "${f}" \
    --output-root "${out_dir}" \
    --save-predictions \
    --save-video \
    --det-interval 1 \
    --pick-center-person

  echo

  if [[ "${SINGLE_VIDEO}" == "1" ]]; then
    echo "SINGLE_VIDEO=1 -> stopping after first processed video."
    exit 0
  fi
done
