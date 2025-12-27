python demo/online_rom.py \
--det-cfg $HOME/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
--det-ckpt https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
--pose-cfg $HOME/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
--pose-ckpt https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
--device cuda:0