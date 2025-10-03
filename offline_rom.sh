#!/usr/bin/env bash
set -euo pipefail

FOLDERNAME="/home/haziq/datasets/telept/data/ipad/20251001-hh"

DET_CFG="demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_CKP="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CFG="configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py"
POSE_CKP="https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth"

# ----- Flexion jobs -----
# Format: rgb_base,depth_base,rom,t1,t2,extra_flags
# Put any per-job flags in the 6th field (space-separated, no commas).
JOBS1=(
  "rgb_1759295456074,depth_1759295456074,left_shoulder_flexion,00:00:02.000,00:00:33.000,--show-3d --show-3d-both --pcd-voxel 0.005 --t2-offset 0.25"
  "rgb_1759295588273,depth_1759295456074,left_shoulder_flexion,00:00:02.000,00:00:21.000,--show-3d --show-3d-both --pcd-voxel 0.005 --t2-offset 0.25"
  "rgb_1759295647580,depth_1759295456074,left_shoulder_flexion,00:00:05.000,00:00:30.000,--show-3d --show-3d-both --pcd-voxel 0.005 --t2-offset 0.25"
)

JOBS2=(
  "rgb_1759295705887,,left_shoulder_extension,00:00:03.500,00:00:20.000,"
  "rgb_1759295750311,,left_shoulder_extension,00:00:03.000,00:00:14.000,"
  "rgb_1759295793727,,left_shoulder_extension,00:00:02.500,00:00:12.000,"
)

JOBS3=(
  "rgb_1759296232878,,left_shoulder_abduction,00:00:02.000,00:00:07.500,"
  "rgb_1759296269895,,left_shoulder_abduction,00:00:04.000,00:00:08.000,"
  "rgb_1759296304535,,left_shoulder_abduction,00:00:00.000,00:00:05.000,"
)

JOBS4=(
  "rgb_1759295872201,depth_1759295872201,left_shoulder_external_rotation,00:00:00.100,00:00:25.500,"
  "rgb_1759295948541,depth_1759295948541,left_shoulder_external_rotation,00:00:01.000,00:00:13.000,"
  "rgb_1759295986117,depth_1759295986117,left_shoulder_external_rotation,00:00:04.000,00:00:08.200,"
)

JOBS5=(
  "rgb_1759296036023,depth_1759296036023,left_shoulder_internal_rotation,00:00:00.500,00:00:09.000,--show-3d --show-3d-both --pcd-voxel 0.005 --t2-offset 0.25"
  "rgb_1759296084639,depth_1759296084639,left_shoulder_internal_rotation,00:00:02.000,00:00:05.000,--show-3d --show-3d-both --pcd-voxel 0.005 --t2-offset 0.25"
  "rgb_1759296119263,depth_1759296119263,left_shoulder_internal_rotation,00:00:01.000,00:00:05.000,--show-3d --show-3d-both --pcd-voxel 0.005 --t2-offset 0.25"
)

# ----- Combine which ones you want -----
# Comment out lines below to skip whole blocks
JOBS=(
  "${JOBS1[@]}"
  # "${JOBS2[@]}"
  # "${JOBS3[@]}"
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
    --save-frames --debug-boxes \
    $extra_flags
done
