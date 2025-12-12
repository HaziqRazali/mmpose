
for f in /media/haziq/Haziq/fit3d/data/train/*/videos/*/*.mp4; do
  subject=$(echo "$f" | cut -d'/' -f8)                  # extract subject, e.g., s05
  recording_num=$(echo "$f" | cut -d'/' -f10)           # extract recording number, e.g., 50591643
  action=$(basename "$f")                               # e.g., band_pull_apart.mp4
  action_name="${action%.*}"                            # remove .mp4 extension

  out_dir="/media/haziq/Haziq/fit3d/data/train/${subject}/mmpose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122/${recording_num}"

  echo "Processing $subject / $recording_num / $action_name"
  
  # Create output directory if it doesn't exist
  mkdir -p "$out_dir"

  python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
    --input "$f" \
    --output-root "$out_dir" \
    --save-predictions
done