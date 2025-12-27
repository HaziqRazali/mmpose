#!/usr/bin/env python3
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

from hydra import initialize_config_dir, compose

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except Exception:
    has_mmdet = False

# make project imports work
sys.path.append(os.path.join(os.path.expanduser("~"), "Collab_AI"))
sys.path.append(os.path.join(os.path.expanduser("~"), "Collab_AI", "dataloaders", "variables"))
from fit3d_variables import coco_wholebody

sys.path.append(os.path.expanduser('~/datasets/mocap/my_scripts/'))
from utils_draw import render_simple_pyrender

sys.path.append('/home/haziq/datasets/mocap/my_scripts/imar_vision_datasets_tools/')
from util.smplx_util import SMPLXHelper
from util.dataset_util import plot_over_image

sys.path.append('/home/haziq/datasets/mocap/my_scripts/imar_vision_datasets_tools/util')
from dataset_util import rot6d_to_matrix

def run_detector(args, img_bgr, detector):
    det_result = inference_detector(detector, img_bgr)
    pred_instance = det_result.pred_instances.cpu().numpy()

    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)

    keep = np.logical_and(
        pred_instance.labels == args.det_cat_id,
        pred_instance.scores > args.bbox_thr
    )
    bboxes = bboxes[keep]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
    return bboxes


def run_pose(img_bgr, pose_estimator, bboxes_xyxy):
    pose_results = inference_topdown(pose_estimator, img_bgr, bboxes_xyxy)
    data_samples = merge_data_samples(pose_results)
    return data_samples.get("pred_instances", None)


def normalize_kpts_bbox(kpts_xy, bbox_xyxy):
    # placeholder – replace later with your real preprocessing
    x1, y1, x2, y2 = bbox_xyxy
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    out = kpts_xy.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] - x1) / w
    out[:, 1] = (out[:, 1] - y1) / h
    return out


def load_hydra_cfg(config_path, config_name, overrides):
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def build_kpts2smpl_model(cfg, weights_path, device):
    import importlib

    module = importlib.import_module(cfg.architecture.module_path)
    net = module.model(cfg).to(device)
    net.eval()

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # raw state_dict

    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"[WEIGHTS] loaded {weights_path}")
    print(f"[WEIGHTS] missing={len(missing)} unexpected={len(unexpected)}")

    return net


def open_input_source(input_arg: str):
    """
    input_arg:
      - "webcam" / "cam" / "0" -> webcam index 0
      - any integer string -> webcam index int(input_arg)
      - else -> treat as video file path
    """
    s = (input_arg or "").strip()

    # webcam aliases
    if s.lower() in {"webcam", "cam"}:
        return cv2.VideoCapture(0), "webcam:0"

    # numeric webcam index, e.g. --input 1
    if s.isdigit():
        idx = int(s)
        return cv2.VideoCapture(idx), f"webcam:{idx}"

    # otherwise treat as file
    cap = cv2.VideoCapture(s)
    return cap, s

def draw_kpts_and_limbs(
    frame_bgr,
    kpts_xy,
    connections=None,
    with_ids=False,
):
    """
    Draw keypoints and limbs directly on an image.

    Args:
        frame_bgr (np.ndarray): H×W×3 BGR image (modified copy is returned)
        kpts_xy (np.ndarray): [N, 2] keypoints in pixel coords
        connections (list): [(i, j), ...] limb indices
        with_ids (bool): draw joint index

    Returns:
        np.ndarray: drawn image (BGR)
    """
    img = frame_bgr.copy()
    num_points = kpts_xy.shape[0]

    # draw keypoints
    for i in range(num_points):
        x, y = int(kpts_xy[i, 0]), int(kpts_xy[i, 1])
        cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

        if with_ids:
            cv2.putText(
                img,
                str(i),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # draw limbs
    if connections is not None:
        for i, j in connections:
            if i < num_points and j < num_points:
                x1, y1 = map(int, kpts_xy[i])
                x2, y2 = map(int, kpts_xy[j])
                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color=(255, 0, 0),
                    thickness=2,
                )

    return img

def main():
    parser = ArgumentParser()

    # MMDet + MMPose
    parser.add_argument("det_config")
    parser.add_argument("det_checkpoint")
    parser.add_argument("pose_config")
    parser.add_argument("pose_checkpoint")

    parser.add_argument("--input", default="webcam")
    parser.add_argument("--show", default=True)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--det-cat-id", type=int, default=0)
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)

    # Hydra
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--override", action="append", default=[])

    # model weights
    parser.add_argument("--weights", required=True)

    args = parser.parse_args()

    if not has_mmdet:
        raise RuntimeError("mmdet not installed")

    # detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    )
    relevant_joint_idxs = sorted(set(i for pair in coco_wholebody["truncated_hand_skeleton_links"] for i in pair))

    # kpts2smpl
    cfg = load_hydra_cfg(args.config_path, args.config_name, args.override)
    input_key = cfg.params.kpts2smpl.input  # "kpts_normalized_filtered"
    net = build_kpts2smpl_model(cfg, args.weights, args.device)
    smplx_helper = SMPLXHelper('/home/haziq/datasets/mocap/data/models_smplx_v1_1/models/')
    smplx_helper.smplx_model = smplx_helper.smplx_model.to(args.device)
    faces       = smplx_helper.smplx_model.faces
    faces       = np.asarray(faces)

    # input source (webcam or video file)
    cap, src_name = open_input_source(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input source: {src_name}")

    # Set webcam size (does nothing for most video files)
    if src_name.startswith("webcam:"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_id = 0

    with torch.no_grad():
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_id += 1
            print(frame_id)

            bboxes = run_detector(args, frame_bgr, detector)
            if bboxes is None or len(bboxes) == 0:
                continue

            pred_instances = run_pose(frame_bgr, pose_estimator, bboxes)
            if pred_instances is None or len(pred_instances) == 0:
                continue

            # take first person
            kpts_xy = pred_instances.keypoints[0]
            bbox_xyxy = bboxes[0]

            kpts_in = normalize_kpts_bbox(kpts_xy, bbox_xyxy)
            kpts_t = torch.from_numpy(kpts_in).unsqueeze(0).to(args.device)
            kpts_t = kpts_t[:,relevant_joint_idxs]
            mask_t = torch.ones((1, 22, 1), device=args.device, dtype=kpts_t.dtype)

            # draw on image
            drawn_image = draw_kpts_and_limbs(
                frame_bgr,
                kpts_xy[relevant_joint_idxs],
                coco_wholebody["truncated_hand_skeleton_links"],
                with_ids=False
            )
            
            out = net({input_key: kpts_t, "mask": mask_t}, mode="val")
            pred_smpl = out["pred_smpl"][0]  # [22,6]

            global_orient   = pred_smpl[0:1]                    # [1,  6]
            global_orient   = rot6d_to_matrix(global_orient)    # [1,  3, 3]
            global_orient   = global_orient[None]               # [1, 1,  3, 3]
            body_pose       = pred_smpl[1:]                     # [21, 6]
            body_pose       = rot6d_to_matrix(body_pose)        # [21, 3, 3]
            body_pose       = body_pose[None]                   # [1, 21, 3, 3]

            #################### TTD: ADD TEMPORAL FILTER
            
            #################### form smplx model
            world_smplx_params = {
                "transl": torch.zeros((1, 3), device=args.device, dtype=torch.float32),
                "global_orient": global_orient.to(device=args.device, dtype=torch.float32),
                "body_pose": body_pose.to(device=args.device, dtype=torch.float32),
            }

            if args.show == 1:
                world_posed_data    = smplx_helper.smplx_model(**world_smplx_params)
                vertices            = world_posed_data.vertices[0]   # [10475, 3]
                vertices            = vertices.detach().cpu().numpy()

                # show o3d
                _, _, image = render_simple_pyrender(
                            vertices,
                            faces,
                            "temp.png",
                            img_width=900,
                            img_height=900,
                            ref_center=None,
                            ref_radius=None,
                            center_offset=(0.0, 0.0, 0.0),
                            label="TRUE",
                            save=False,
                        )
                concat_img = np.concatenate([drawn_image, image],axis=1)

                cv2.imshow('smplx', concat_img)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
