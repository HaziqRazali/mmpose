<<<<<<< HEAD

# https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose

# nano
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth \
--input "webcam" --device "cpu" --show

# large
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input "webcam" --device "cpu" --show
=======

# https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose

CUDA_VISIBLE_DEVICES=0 python demo/synthium_demo.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
  --input "/home/haziq/datasets/mocap/data/fit3d/val/s10/videos/50591643/barbell_shrug.mp4" \
  --device cuda:0 \
  --config-path "/home/haziq/Collab_AI/experiments/synthium/kpts2smpl" \
  --config-name "v1" \
  --override "run_name=fit3d_mmposelarge_mlp_lr1e-3_thr0_kptsmask0" \
  --override "optimization.batch_size=1" \
  --weights "/home/haziq/Collab_AI/weights/synthium/kpts2smpl/fit3d_mmposelarge_mlp_lr1e-3_thr0_kptsmask0/all_epoch_0384_best_0384_state_dict.pt"

CUDA_VISIBLE_DEVICES=0 python demo/synthium_demo.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
  --input webcam \
  --device cuda:0 \
  --config-path "/home/haziq/Collab_AI/experiments/synthium/kpts2smpl" \
  --config-name "v1" \
  --override "run_name=fit3d_mmposelarge_mlp_lr1e-3_thr0_kptsmask0" \
  --override "optimization.batch_size=1" \
  --weights "/home/haziq/Collab_AI/weights/synthium/kpts2smpl/fit3d_mmposelarge_mlp_lr1e-3_thr0_kptsmask0/all_epoch_0384_best_0384_state_dict.pt"

# nano
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth \
--input "webcam" --device "cpu" --show

# large
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input "webcam" --device "cpu" --show
>>>>>>> 8e05453455735dda4af8839c71f8c83ed23a1aee
