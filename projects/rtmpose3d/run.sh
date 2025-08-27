cd /home/haziq/mmpose/projects/rtmpose3d/
export PYTHONPATH=$(pwd):$PYTHONPATH

python ./demo/body3d_img2pose_demo.py \
    configs/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    ./configs/rtmw3d-l_8xb64_cocktail14-384x288.py \
    rtmw3d-l_cock14-0d4ad840_20240422.pth \
    --input /home/haziq/mmpose/tests/data/coco/000000000785.jpg \
    --output-root /home/haziq/mmpose \
    --device cpu
	
python tools/train.py "/home/haziq/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py"