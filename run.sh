# rtmw-l_8xb1024-270e_cocktail14-256x192
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input tests/data/coco/000000000785.jpg \
--output-root vis_results/ --save-predictions

python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input /home/haziq/4D-Humans/example_data/videos/walking_new.mp4 \
--output-root vis_results/ --save-predictions

# webcam
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input "webcam" --device "cpu" --show
  
# rtmw-l_8xb320-270e_cocktail14-384x288
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth \
--input tests/data/coco/000000196141.jpg \
--output-root vis_results/ --save-predictions

python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input "/home/haziq/datasets/fit3d/data/fit3d/train/s05/videos/50591643/band_pull_apart.mp4" \
--output-root "/home/haziq/datasets/fit3d/data/fit3d/train/s05/mmpose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122/" \
--save-predictions

# # # # # # # # # # #
# for fit3d dataset #
# # # # # # # # # # #

for f in /media/haziq/Haziq/fit3d/data/temp/*/videos/*/*.mp4; do
  subject=$(echo "$f" | cut -d'/' -f8)                  # extract subject, e.g., s05
  recording_num=$(echo "$f" | cut -d'/' -f10)           # extract recording number, e.g., 50591643
  action=$(basename "$f")                               # e.g., band_pull_apart.mp4
  action_name="${action%.*}"                            # remove .mp4 extension

  out_dir="/media/haziq/Haziq/fit3d/data/temp/${subject}/mmpose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122/${recording_num}"

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

# # # # # # # # # # # #
# for custom dataset  #
# # # # # # # # # # # #

python demo/topdown_demo_with_mmdet.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
  --input /home/haziq/datasets/fit3d/data/others/wave.mp4 \
  --output-root /home/haziq/datasets/fit3d/data/others/wave \
  --save-predictions







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


