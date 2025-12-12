#!/usr/bin/env bash
set -euo pipefail

# HumanEva MMPose batch runner (with TEST_MODE + SINGLE_VIDEO safety switches)
#
# Input videos:
#   /media/haziq/Haziq/mocap/data/humaneva/<train|val>/<subject>/videos/<camera>/<action>.avi
#
# Output predictions (desired format):
#   /media/haziq/Haziq/mocap/data/humaneva/<train|val>/<subject>/mmpose/<model>/<camera>/<action>.json
#
# IMPORTANT:
# - This script keeps --save-predictions.
# - It does NOT skip existing outputs.
# - Whether an output *video* is saved depends on your demo script + flags.
#   This script DOES NOT add/assume any video-saving flags.

# TEST_MODE=1 ./run_humaneva.sh
# SINGLE_VIDEO=1 ./run_humaneva.sh

DATA_ROOT="/media/haziq/Haziq/mocap/data/humaneva"
MODEL_NAME="rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122"

# Safety switches:
#   TEST_MODE=1      -> print planned outputs only; do NOT mkdir; do NOT run python
#   SINGLE_VIDEO=1   -> run only the first matched video then exit (ignored if TEST_MODE=1)
TEST_MODE="${TEST_MODE:-0}"
SINGLE_VIDEO="${SINGLE_VIDEO:-0}"

# MMPose configs / checkpoints (same ones you used before)
DET_CFG="demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_CKPT="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CFG="configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py"
POSE_CKPT="https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth"

shopt -s nullglob

for f in "${DATA_ROOT}"/{train,val}/*/videos/*/*.avi; do
  # Expected f format:
  # /media/haziq/Haziq/mocap/data/humaneva/<split>/<subject>/videos/<camera>/<action>.avi

  action_file="$(basename "$f")"          # e.g., Box.avi
  action_name="${action_file%.*}"         # e.g., Box

  camera="$(basename "$(dirname "$f")")"  # e.g., C3

  # Correct hierarchy:
  # .../<split>/<subject>/videos/<camera>/<action>.avi
  # dirname 1 -> <camera>
  # dirname 2 -> videos
  # dirname 3 -> <subject>
  # dirname 4 -> <split>
  subject="$(basename "$(dirname "$(dirname "$(dirname "$f")")")")"
  split="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$f")")")")")"

  out_dir="${DATA_ROOT}/${split}/${subject}/mmpose/${MODEL_NAME}/${camera}"
  out_json="${out_dir}/${action_name}.json"

  # This is just a "would be" path for a rendered video (ONLY if your demo script is configured to output one).
  out_mp4="${out_dir}/${action_name}.mp4"

  echo "Processing split=${split} subject=${subject} camera=${camera} action=${action_name}"
  echo "  input:     ${f}"
  echo "  out_json:  ${out_json}"
  echo "  out_video: ${out_mp4}  (only if your demo script actually writes a video)"

  if [[ "${TEST_MODE}" == "1" ]]; then
    echo "  TEST_MODE=1 -> not creating dirs, not running inference."
    echo
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
    --save-predictions

  echo

  if [[ "${SINGLE_VIDEO}" == "1" ]]; then
    echo "SINGLE_VIDEO=1 -> stopping after first processed video."
    exit 0
  fi
done
