
#################### test

# rtmw-l_8xb1024-270e_cocktail14-256x192
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input tests/data/coco/000000196141.jpg \
--output-root vis_results/ --save-predictions

# webcam
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input "webcam" --show

# realsense
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input "realsense" --show --rom_test "right_elbow_rom"

# home desktop + creative
python demo/topdown_demo_with_mmdet.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
  --input webcam \
  --show \
  --voice --voice-mic "plughw:CARD=L1080p,DEV=0" \
  --output-root out \
  --auto-rom \
  --rom-v-go 30 \
  --rom-v-stop 10 \
  --rom-hold-sec 0.6 \
  --rom-std-max 1.5 \
  --rom-min-amplitude 15 \
  --rom-start-amp 8 \
  --rom-baseline-tol 12 \
  --rom-baseline-hold-sec 0.5 \
  --rom-timeout-sec 25 \
  --plot-seconds 0
  
# astar aftershock laptop + creative
python demo/topdown_demo_with_mmdet.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
  --input "/dev/video2" \
  --show \
  --voice --voice-mic "plughw:CARD=L1080p,DEV=0" \
  --output-root out \
  --auto-rom \
  --rom-v-go 30 \
  --rom-v-stop 10 \
  --rom-hold-sec 0.6 \
  --rom-std-max 1.5 \
  --rom-min-amplitude 15 \
  --rom-start-amp 8 \
  --rom-baseline-tol 12 \
  --rom-baseline-hold-sec 0.5 \
  --rom-timeout-sec 25 \
  --plot-seconds 0

# astar aftershock laptop + realsense
python demo/topdown_demo_with_mmdet.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
  --input "realsense" \
  --show \
  --show-result \
  --panel-window gallery \
  --voice --voice-mic "default" \
  --output-root out \
  --auto-rom \
  --rom-v-go 20 \
  --rom-v-stop 10 \
  --rom-hold-sec 0.6 \
  --rom-std-max 4.0 \
  --rom-min-amplitude 15 \
  --rom-start-amp 8 \
  --rom-baseline-tol 12 \
  --rom-baseline-hold-sec 0.5 \
  --rom-timeout-sec 25 \
  --plot-seconds 0 \
  --zero --abs-angle

  --debug

# video
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input "/home/haziq/datasets/telept/recordings_from_brett/video-20250711-175311-72cee329.mov" \
--output-root vis_results/ --save-predictions

# rtmw-l_8xb320-270e_cocktail14-384x288
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth \
--input tests/data/coco/000000196141.jpg \
--output-root vis_results/ --save-predictions

# look for mp4 files in /home/haziq/datasets/physio/ and uses them as input
for f in /home/haziq/datasets/physio/*.mp4; do
  python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
    --input "$f" \
    --output-root vis_results/ \
    --save-predictions
done

python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input "/home/haziq/datasets/fit3d/data/fit3d/train/s05/videos/50591643/band_pull_apart.mp4" \
--output-root "/home/haziq/datasets/fit3d/data/fit3d/train/s05/mmpose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122/" \
--save-predictions

# /home/haziq/datasets/fit3d/data/train/*/videos/*/*.mp4
# /media/haziq/Haziq/fit3d/data/train/*/videos/*/*.mp4

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












#################### wholebody
CUDA_VISIBLE_DEVICES=0 python tools/train.py projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py --resume
CUDA_VISIBLE_DEVICES=1 python tools/train.py projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-420e_coco-wholebody-256x192.py --resume

#################### wholebody, rtmw-l_8xb1024-270e_cocktail14-256x192 and rtmw-l_8xb1024-270e_cocktail14-256x192, fine tune on coco
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py --resume configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192_fine_tune.py --resume configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth --trainable_layers coco --resume_optimizer 0 --resume_param_scheduler 0
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192_fine_tune_coco.py --resume configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth --trainable_layers coco --resume_optimizer 0 --resume_param_scheduler 0

python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192_fine_tune.py \
work_dirs/rtmw-l_8xb1024-270e_cocktail14-256x192_fine_tune/best_coco_AP_epoch_160.pth \
--input tests/data/coco/000000196141.jpg \
--output-root vis_results/ --save-predictions

#################### wholebody, rtmw-l_8xb1024-270e_cocktail14-256x192 and rtmw-l_8xb1024-270e_cocktail14-256x192, fine tune on mpii
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192_fine_tune_mpii.py --resume configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth --trainable_layers coco --resume_optimizer 0 --resume_param_scheduler 0

#################### body
CUDA_VISIBLE_DEVICES=0 python tools/train.py projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py





















#################### test

# wholebody
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py

# wholebody, rtmw-l_8xb1024-270e_cocktail14-256x192
# https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
/home/haziq/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth


