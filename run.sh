
#################### test

# rtmw-l_8xb1024-270e_cocktail14-256x192
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--input tests/data/coco/000000196141.jpg \
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

for f in /home/haziq/datasets/fit3d/data/fit3d/train/*/videos/*/*.mp4; do
  python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
    --input "$f" \
    --output-root /home/haziq/datasets/fit3d/data/fit3d/train/*/videos/*/ \
    --save-predictions
done


#################### train

# wholebody
CUDA_VISIBLE_DEVICES=0 python tools/train.py projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py --resume
CUDA_VISIBLE_DEVICES=1 python tools/train.py projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-420e_coco-wholebody-256x192.py --resume

# wholebody, rtmw-l_8xb1024-270e_cocktail14-256x192
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py --resume configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192_fine_tune.py --resume configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth --trainable_layers coco --resume_optimizer 0 --resume_param_scheduler 0

# body
CUDA_VISIBLE_DEVICES=0 python tools/train.py projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py

#################### test

# wholebody
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py

# wholebody, rtmw-l_8xb1024-270e_cocktail14-256x192
# https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
/home/haziq/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth

#################### TTD

# run test command to make sure loading works
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
/home/haziq/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth