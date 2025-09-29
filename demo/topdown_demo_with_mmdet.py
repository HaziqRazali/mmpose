import os
import time
import logging
import mimetypes
import cv2
import mmcv
import mmengine
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

from mmengine.logging import print_log
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector

from utils_variables import get_vectors_for_preset, get_show_kpt_subset, CATEGORY_SIDES
from utils import handle_hotkeys_for_presets

import pyrealsense2 as rs
from utils_3d import extract_intrinsics_from_depth, compute_joint_xyz_for_person, angle as angle_nd

# ---------- Person extraction ----------

def _extract_top_person_arrays(pred_instances):
    if pred_instances is None:
        return None, None
    try:
        scores = pred_instances.get('bbox_scores', None) or pred_instances.get('scores', None)
        if scores is None: return None, None
        scores = np.asarray(scores)
        if scores.ndim == 0 or scores.size == 0: return None, None
        idx = int(np.argmax(scores))
        kpts = np.asarray(pred_instances['keypoints'])
        kpt_scores = np.asarray(pred_instances.get('keypoint_scores', np.ones((kpts.shape[1],), dtype=np.float32)))
        if kpts.ndim != 3 or kpts.shape[1] < 3: return None, None
        return kpts[idx], kpt_scores[idx]
    except Exception:
        return None, None

def _subset_from_preset(name):
    try:
        subset = get_show_kpt_subset(name)
        return subset if subset else None
    except Exception:
        return None

# ---------- Vector-pair angle helpers ----------

def _avg_point_2d(point_spec, kpts_xy, kpt_scores, kpt_thr):
    if kpts_xy is None or kpt_scores is None:
        return np.array([np.nan, np.nan], dtype=float)
    def _one(idx):
        j = int(idx)
        if 0 <= j < kpts_xy.shape[0] and kpt_scores[j] >= kpt_thr:
            return kpts_xy[j, :2].astype(float)
        return np.array([np.nan, np.nan], dtype=float)
    if isinstance(point_spec, (list, tuple)):
        pts = np.array([_one(i) for i in point_spec], dtype=float)
        return np.nanmean(pts, axis=0)
    return _one(point_spec)

def _avg_point_3d(point_spec, joint_xyz, kpt_scores, kpt_thr):
    if joint_xyz is None or kpt_scores is None:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    def _one(idx):
        j = int(idx)
        if 0 <= j < joint_xyz.shape[0] and kpt_scores[j] >= kpt_thr:
            p = joint_xyz[j].astype(float)
            return p if np.all(np.isfinite(p)) else np.array([np.nan, np.nan, np.nan], dtype=float)
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    if isinstance(point_spec, (list, tuple)):
        pts = np.array([_one(i) for i in point_spec], dtype=float)
        return np.nanmean(pts, axis=0)
    return _one(point_spec)

def _vector_angle_2d(P0, P1, Q0, Q1):
    u, v = P1 - P0, Q1 - Q0
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if not (np.isfinite(nu) and np.isfinite(nv)) or nu < 1e-6 or nv < 1e-6:
        return None
    cu, cv = u / nu, v / nv
    cosang = np.clip(float(np.dot(cu, cv)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _build_top_xyz(depth_img, kpts_xy, kpt_scores, kpt_thr):
    if depth_img is None or kpts_xy is None or kpt_scores is None:
        return None
    intr = extract_intrinsics_from_depth(depth_img)
    if not intr:
        return None
    try:
        xyz = compute_joint_xyz_for_person(kpts_xy[:, :2], kpt_scores, depth_img, intr, kpt_thr)
        return xyz if np.isfinite(xyz).any() else None
    except Exception:
        return None

def _compute_angle_from_vectors(vec_def, kpts_xy, kpt_scores, top_xyz, kpt_thr):
    (P0s, P1s), (Q0s, Q1s) = vec_def

    # 3D first
    if top_xyz is not None:
        P0_3d = _avg_point_3d(P0s, top_xyz, kpt_scores, kpt_thr)
        P1_3d = _avg_point_3d(P1s, top_xyz, kpt_scores, kpt_thr)
        Q0_3d = _avg_point_3d(Q0s, top_xyz, kpt_scores, kpt_thr)
        Q1_3d = _avg_point_3d(Q1s, top_xyz, kpt_scores, kpt_thr)
        if np.all(np.isfinite([P0_3d, P1_3d, Q0_3d, Q1_3d])):
            a = angle_nd(P1_3d - P0_3d, np.zeros(3), Q1_3d - Q0_3d)
            if np.isfinite(a):
                P0_2d = _avg_point_2d(P0s, kpts_xy, kpt_scores, kpt_thr)
                P1_2d = _avg_point_2d(P1s, kpts_xy, kpt_scores, kpt_thr)
                Q0_2d = _avg_point_2d(Q0s, kpts_xy, kpt_scores, kpt_thr)
                Q1_2d = _avg_point_2d(Q1s, kpts_xy, kpt_scores, kpt_thr)
                draw = dict(P0=P0_2d, P1=P1_2d, Q0=Q0_2d, Q1=Q1_2d)
                return float(a), "3D", draw

    # 2D fallback
    P0_2d = _avg_point_2d(P0s, kpts_xy, kpt_scores, kpt_thr)
    P1_2d = _avg_point_2d(P1s, kpts_xy, kpt_scores, kpt_thr)
    Q0_2d = _avg_point_2d(Q0s, kpts_xy, kpt_scores, kpt_thr)
    Q1_2d = _avg_point_2d(Q1s, kpts_xy, kpt_scores, kpt_thr)
    if np.all(np.isfinite([P0_2d, P1_2d, Q0_2d, Q1_2d])):
        a2 = _vector_angle_2d(P0_2d, P1_2d, Q0_2d, Q1_2d)
        if a2 is not None:
            draw = dict(P0=P0_2d, P1=P1_2d, Q0=Q0_2d, Q1=Q1_2d)
            return float(a2), "2D", draw

    return None, None, {}

def _draw_vectors_overlay(frame_bgr, draw_pts, mode_tag=None):
    if not draw_pts: return
    def _pt(name):
        p = draw_pts.get(name, None)
        if p is None or not np.all(np.isfinite(p)): return None
        return (int(round(p[0])), int(round(p[1])))
    P0, P1, Q0, Q1 = _pt("P0"), _pt("P1"), _pt("Q0"), _pt("Q1")
    colA, colB = (0,255,0), (255,128,0)
    if P0: cv2.circle(frame_bgr, P0, 4, colA, -1, cv2.LINE_AA)
    if P1: cv2.circle(frame_bgr, P1, 4, colA, -1, cv2.LINE_AA)
    if Q0: cv2.circle(frame_bgr, Q0, 4, colB, -1, cv2.LINE_AA)
    if Q1: cv2.circle(frame_bgr, Q1, 4, colB, -1, cv2.LINE_AA)
    if P0 and P1: cv2.line(frame_bgr, P0, P1, colA, 2, cv2.LINE_AA)
    if Q0 and Q1: cv2.line(frame_bgr, Q0, Q1, colB, 2, cv2.LINE_AA)
    if mode_tag in ("3D","2D"):
        cv2.putText(frame_bgr, mode_tag, (10,58), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2,cv2.LINE_AA)

# ---------- Compare canvas ----------

def _compose_compare_canvas(frameA, txtA, frameB, txtB, rom_lines):
    pad, bg = 16, (18, 18, 18)
    font = cv2.FONT_HERSHEY_SIMPLEX
    def _box(im): return np.full((360, 640, 3), bg, np.uint8) if im is None else im
    A, B = _box(frameA), _box(frameB)
    hA, hB = A.shape[0], B.shape[0]; H = max(hA, hB)
    if hA < H: A = np.vstack([A, np.full((H - hA, A.shape[1], 3), bg, np.uint8)])
    if hB < H: B = np.vstack([B, np.full((H - hB, B.shape[1], 3), bg, np.uint8)])
    row = np.hstack([A, np.full((H, pad, 3), bg, np.uint8), B])
    title_h, footer_h = 36, (80 if rom_lines else 0)
    W = row.shape[1]
    panel = np.full((title_h + H + footer_h, W, 3), bg, np.uint8)
    cv2.putText(panel, "Frame A (4)", (8, 25), font, 0.8, (255,255,255), 2)
    cv2.putText(panel, "Frame B (5)", (W//2 + pad + 8, 25), font, 0.8, (255,255,255), 2)
    panel[title_h:title_h + H] = row
    if txtA: cv2.putText(panel, txtA, (8, title_h + 28), font, 0.7, (0,255,255), 2)
    if txtB: cv2.putText(panel, txtB, (W//2 + pad + 8, title_h + 28), font, 0.7, (0,255,255), 2)
    for i, line in enumerate(rom_lines):
        cv2.putText(panel, line, (8, title_h + H + 36 + i * 26), font, 0.8, (255,255,255), 2)
    return panel

# ---------- Per-frame pipeline ----------

def process_one_image(args, color_img, depth_img, detector, pose_estimator, visualizer=None, show_interval=0):
    init_default_scope('mmdet')
    det_result = inference_detector(detector, color_img)
    init_default_scope('mmpose')

    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id, pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    pose_results = inference_topdown(pose_estimator, color_img, bboxes)
    data_samples = merge_data_samples(pose_results)

    if isinstance(color_img, np.ndarray):
        color_img_rgb = mmcv.bgr2rgb(color_img)
    else:
        color_img_rgb = mmcv.imread(color_img, channel_order="rgb")

    vec_list = get_vectors_for_preset(args.rom_test)
    is_rom_mode = len(vec_list) > 0

    # Draw skeleton unless only-rom-lines is set (and not full_body/133)
    if visualizer is not None and (not args.only_rom_lines or not is_rom_mode):
        subset = _subset_from_preset(args.rom_test)
        visualizer.add_datasample(
            "result",
            color_img_rgb,
            depth_img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            show_kpt_subset=subset,
            skeleton_style=args.skeleton_style,
            show=False,
            wait_time=0,
            kpt_thr=args.kpt_thr,
        )
        frame_rgb = visualizer.get_image()
    else:
        frame_rgb = color_img_rgb

    frame_bgr = mmcv.rgb2bgr(frame_rgb)

    # Extract top person
    pred_instances = data_samples.get("pred_instances", None)
    kpts, kscores = _extract_top_person_arrays(pred_instances)
    top_xyz = _build_top_xyz(depth_img, kpts, kscores, args.kpt_thr)

    # Live angle overlay
    if is_rom_mode:
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(vec_list) == 1:
            ang, mode, draw = _compute_angle_from_vectors(vec_list[0], kpts, kscores, top_xyz, args.kpt_thr)
            txt = "Angle: --" if ang is None else f"Angle: {ang:.1f} deg"
            cv2.putText(frame_bgr, txt, (10,30), font,0.9,(0,255,255),2,cv2.LINE_AA)
            _draw_vectors_overlay(frame_bgr, draw, mode)
        else:
            label_L, label_R = "Left","Right"
            for _cat, sides in CATEGORY_SIDES.items():
                if sides.get("both") == args.rom_test:
                    label_L = sides.get("left","Left").split("_")[0].capitalize()
                    label_R = sides.get("right","Right").split("_")[0].capitalize()
                    break
            angL, modeL, drawL = _compute_angle_from_vectors(vec_list[0], kpts, kscores, top_xyz, args.kpt_thr)
            angR, modeR, drawR = _compute_angle_from_vectors(vec_list[1], kpts, kscores, top_xyz, args.kpt_thr)
            sL = f"{label_L}: --" if angL is None else f"{label_L}: {angL:.1f} deg"
            sR = f"{label_R}: --" if angR is None else f"{label_R}: {angR:.1f} deg"
            cv2.putText(frame_bgr, f"{sL}   {sR}", (10,30), font,0.9,(0,255,255),2,cv2.LINE_AA)
            _draw_vectors_overlay(frame_bgr, drawL, modeL)
            _draw_vectors_overlay(frame_bgr, drawR, modeR)

    key = -1
    if args.show:
        cv2.imshow("Pose", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        handle_hotkeys_for_presets(args, key)
        if args.show_interval > 0:
            time.sleep(args.show_interval)

    return data_samples.get("pred_instances", None), top_xyz, key, frame_bgr

# ---------- Main ----------

def main():
    parser = ArgumentParser()
    parser.add_argument("det_config")
    parser.add_argument("det_checkpoint")
    parser.add_argument("pose_config")
    parser.add_argument("pose_checkpoint")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--save-predictions", action="store_true", default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--det-cat-id", type=int, default=0)
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--draw-heatmap", action="store_true", default=False)
    parser.add_argument("--show-kpt-idx", action="store_true", default=False)
    parser.add_argument("--skeleton-style", default="mmpose", choices=["mmpose","openpose"])
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--show-interval", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--rom_test", default="full_body")
    parser.add_argument("--only-rom-lines", action="store_true", default=False,
                        help="If True and preset is ROM, skip skeleton and draw only the ROM vectors.")
    parser.add_argument("--show-depth", action="store_true", default=False,
                        help="Show a colorized RealSense depth window.")
    args = parser.parse_args()

    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(args.pose_config, args.pose_checkpoint, device=args.device)
    init_default_scope('mmpose')

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    output_file, video_writer = None, None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        if args.input and args.input.lower() not in ("webcam","realsense"):
            base = os.path.basename(args.input); name, _ = os.path.splitext(base)
            output_file = os.path.join(args.output_root, f"{name}.mp4")
        else:
            output_file = os.path.join(args.output_root, "stream.mp4")

    # RealSense colorizer for depth
    depth_colorizer = rs.colorizer() if args.show_depth else None

    # ----- Input handling -----
    input_src = (args.input or "").strip()
    cap = None; depth_pipe = None; align = None; is_image = False; frames = None

    if input_src.lower() == "realsense":
        try:
            depth_pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 640,480,rs.format.z16,30)
            cfg.enable_stream(rs.stream.color, 640,480,rs.format.bgr8,30)
            depth_pipe.start(cfg)
            align = rs.align(rs.stream.color)
            print("[INFO] RealSense started.")
        except Exception as e:
            print(f"[WARN] RealSense not started: {e}")
            depth_pipe = None
    elif input_src and os.path.exists(input_src):
        mime, _ = mimetypes.guess_type(input_src)
        if mime and mime.startswith("image/"):
            frames = [mmcv.imread(input_src)]; is_image = True
        else:
            cap = cv2.VideoCapture(input_src)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open input file: {input_src}"); return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No webcam found (and no other input specified)."); return

    # ----- Main loops -----
    try:
        if depth_pipe is not None:
            while True:
                frameset = depth_pipe.wait_for_frames()
                frameset = align.process(frameset) if align else frameset
                depth_frame = frameset.get_depth_frame()
                color_frame = frameset.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Depth window
                if depth_colorizer is not None:
                    depth_color = np.asanyarray(depth_colorizer.colorize(depth_frame).get_data())
                    cv2.imshow("Depth", depth_color)

                color = np.asanyarray(color_frame.get_data())
                depth = depth_frame

                pred_instances, top_xyz, key, frame_bgr = process_one_image(
                    args, color, depth, detector, pose_estimator, visualizer, args.show_interval
                )

                if output_file:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    if video_writer is None:
                        H, W = frame_bgr.shape[:2]
                        video_writer = cv2.VideoWriter(output_file, fourcc, 30.0, (W, H))
                    video_writer.write(frame_bgr)

                if key in (ord('4'), ord('5'), ord('6')):
                    kpts, kscores = _extract_top_person_arrays(pred_instances)
                    if key == ord('4'):
                        snapA = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": top_xyz}
                    elif key == ord('5'):
                        snapB = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": top_xyz}
                    elif key == ord('6'):
                        # compute and print ROM mode inside function
                        last_panel = _compose_compare_canvas(snapA["img"], "", snapB["img"], "", [])
                        # just to ensure window exists
                        _ = last_panel
                        # delegate to local compute below
                        pass

                # Handle Esc in the main loop as well
                if args.show and (cv2.waitKey(1) & 0xFF) == 27:
                    break

        elif cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                pred_instances, top_xyz, key, frame_bgr = process_one_image(
                    args, frame, None, detector, pose_estimator, visualizer, args.show_interval
                )

                if output_file:
                    if video_writer is None:
                        _H, _W = frame_bgr.shape[:2]
                        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (_W, _H))
                    video_writer.write(frame_bgr)

                if args.show and (cv2.waitKey(1) & 0xFF) == 27:
                    break

        elif is_image:
            frame = frames[0]
            _ = process_one_image(args, frame, None, detector, pose_estimator, visualizer, args.show_interval)
            if args.show:
                cv2.waitKey(0)

    finally:
        if cap is not None:
            cap.release()
        if 'depth_pipe' in locals() and depth_pipe is not None:
            try:
                depth_pipe.stop()
            except Exception:
                pass
        if video_writer is not None:
            video_writer.release()
            print_log(f"the output video has been saved at {output_file}", logger="current", level=logging.INFO)
        if args.show_depth:
            cv2.destroyWindow("Depth")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
