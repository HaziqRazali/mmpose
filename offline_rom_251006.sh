#!/usr/bin/env bash
set -euo pipefail

#FOLDERNAME="$HOME/datasets/telept/data/ipad/20251001-hh"
FOLDERNAME="/media/haziq/Haziq/telept/251006-hhcc"

DET_CFG="demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_CKP="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CFG="configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py"
POSE_CKP="https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth"

# ----- Flexion jobs -----
# Format: rgb_base,depth_base,rom,t1,t2,extra_flags
# Put any per-job flags in the 6th field (space-separated, no commas).
JOBS1=(
  "rgb_1759733264296,,right_elbow_flexion,00:00:10.833"
  "rgb_1759733264296,,right_elbow_flexion,00:00:16.793"
  "rgb_1759733264296,,right_elbow_flexion,00:00:26.229"
  "rgb_1759733264296,,right_elbow_flexion,00:00:40.943"
  "rgb_1759733264296,,right_elbow_flexion,00:01:04.374"
  "rgb_1759733264296,,right_elbow_flexion,00:01:11.574"
)

JOBS2=(
  "rgb_1759734715452,,right_elbow_flexion,00:00:00.346"
  "rgb_1759734715452,,right_elbow_flexion,00:00:28.146"
  "rgb_1759734715452,,right_elbow_flexion,00:00:42.523"
  "rgb_1759734715452,,right_elbow_flexion,00:00:51.703"
  "rgb_1759734715452,,right_elbow_flexion,00:01:05.040"
)

# ----- Combine which ones you want -----
# Comment out lines below to skip whole blocks
JOBS=(
  "${JOBS1[@]}"
  "${JOBS2[@]}"
  # "${JOBS4[@]}"
  # "${JOBS5[@]}"
)

for job in "${JOBS[@]}"; do
  # Parse up to 6 fields. If the 6th (extra_flags) is missing, it becomes empty.
  IFS=, read -r rgb_base depth_base rom t1 t2 extra_flags <<< "$job"

  INPUT_ARG=( --input "${FOLDERNAME}/${rgb_base}.mp4" )
  if [ -n "${depth_base}" ]; then
    DEPTH_ARG=( --depth "${FOLDERNAME}/${depth_base}.zip" )
  else
    DEPTH_ARG=()
  fi

  # shellcheck disable=SC2086  # we intentionally want word splitting of $extra_flags
  python demo/offline_rom.py \
    "$DET_CFG" "$DET_CKP" "$POSE_CFG" "$POSE_CKP" \
    "${INPUT_ARG[@]}" "${DEPTH_ARG[@]}" \
    --rom-test "$rom" \
    --t1 "$t1" --t2 "$t2" \
    --out-dir "${FOLDERNAME}/rom" \
    --save-frames \
    #--debug-boxes \
    $extra_flags
done

#    --show \