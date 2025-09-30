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

from utils_variables import rom_test, CATEGORY_SIDES, get_vectors_for_preset
from utils import handle_hotkeys_for_presets

from utils_3d import (
    extract_intrinsics_from_depth,
    compute_joint_xyz_for_person,
    resolve_point,
    angle_from_vecpair,
)
import pyrealsense2 as rs


# =========================
# Helpers: persons & angles
# =========================

def _extract_top_person_arrays(pred_instances):
    """Top-scoring person's (K,2) keypoints and (K,) scores arrays."""
    if pred_instances is None:
        return None, None
    try:
        scores = pred_instances.get('bbox_scores', None) or pred_instances.get('scores', None)
        if scores is None:
            return None, None
        scores = np.asarray(scores)
        if scores.ndim == 0 or scores.size == 0:
            return None, None
        idx = int(np.argmax(scores))

        kpts = np.asarray(pred_instances['keypoints'])
        if 'keypoint_scores' in pred_instances:
            kpt_scores = np.asarray(pred_instances['keypoint_scores'])
        else:
            kpt_scores = np.ones((kpts.shape[1],), dtype=np.float32)

        if kpts.ndim != 3 or kpts.shape[1] < 3:
            return None, None
        return kpts[idx], kpt_scores[idx]
    except Exception:
        return None, None


def _compose_compare_canvas(frameA, txtA, frameB, txtB, rom_lines):
    """Side-by-side A|B with optional header texts and a ROM footer."""
    pad, bg = 16, (18, 18, 18)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _box(im):
        return np.full((360, 640, 3), bg, np.uint8) if im is None else im

    A, B = _box(frameA), _box(frameB)
    hA, hB = A.shape[0], B.shape[0]
    H = max(hA, hB)
    if hA < H:
        A = np.vstack([A, np.full((H - hA, A.shape[1], 3), bg, np.uint8)])
    if hB < H:
        B = np.vstack([B, np.full((H - hB, B.shape[1], 3), bg, np.uint8)])

    row = np.hstack([A, np.full((H, pad, 3), bg, np.uint8), B])

    title_h = 36
    footer_h = (80 if rom_lines else 0)

    W = row.shape[1]
    panel = np.full((title_h + H + footer_h, W, 3), bg, np.uint8)

    cv2.putText(panel, "Frame A (4)", (8, 25), font, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, "Frame B (5)", (W // 2 + pad + 8, 25), font, 0.8, (255, 255, 255), 2)

    panel[title_h:title_h + H] = row
    if txtA:
        cv2.putText(panel, txtA, (8, title_h + 28), font, 0.7, (0, 255, 255), 2)
    if txtB:
        cv2.putText(panel, txtB, (W // 2 + pad + 8, title_h + 28), font, 0.7, (0, 255, 255), 2)

    for i, line in enumerate(rom_lines):
        cv2.putText(panel, line, (8, title_h + H + 36 + i * 26), font, 0.8, (255, 255, 255), 2)

    return panel


# ---------- Vector-pair angle helpers (2D/3D with fallback) ----------

def _resolve_point_2d_with_scores(kpts_xy, kpt_scores, spec, kpt_thr):
    """
    Like resolve_point, but for 2D with score-thresholding.
    Average only over ids with score >= kpt_thr.
    """
    arr = np.asarray(kpts_xy, dtype=float)
    K = arr.shape[0]
    def _one(i):
        try:
            j = int(i)
            if j < 0 or j >= K:
                return None
            if kpt_scores[j] < kpt_thr:
                return None
            p = arr[j, :2]
            if not np.all(np.isfinite(p)):
                return None
            return p
        except Exception:
            return None

    if isinstance(spec, (list, tuple)):
        pts = [p for p in (_one(i) for i in spec) if p is not None]
        if not pts:
            return np.array([np.nan, np.nan], dtype=float)
        return np.nanmean(np.vstack(pts), axis=0)
    p = _one(spec)
    if p is None:
        return np.array([np.nan, np.nan], dtype=float)
    return p


def _angle_from_vecpair_auto(kpts_xy, kpt_scores, joint_xyz, pair, kpt_thr):
    """
    Compute angle using 3D if all endpoints valid; else fall back to 2D.
    Returns (angle_deg or None, source_str '3D'/'2D'/None)
    """
    # Try 3D first
    if joint_xyz is not None and isinstance(joint_xyz, np.ndarray) and joint_xyz.ndim == 2 and joint_xyz.shape[1] == 3:
        ang3d = angle_from_vecpair(joint_xyz, pair)
        if np.isfinite(ang3d):
            return float(ang3d), "3D"

    # Fallback to 2D with score filtering
    if kpts_xy is not None and kpt_scores is not None:
        try:
            (P0, P1), (Q0, Q1) = pair
        except Exception:
            return None, None
        A = _resolve_point_2d_with_scores(kpts_xy, kpt_scores, P1, kpt_thr)
        B = _resolve_point_2d_with_scores(kpts_xy, kpt_scores, P0, kpt_thr)
        C = _resolve_point_2d_with_scores(kpts_xy, kpt_scores, Q1, kpt_thr)
        if not (np.any(np.isnan(A)) or np.any(np.isnan(B)) or np.any(np.isnan(C))):
            u = A - B
            v = C - B
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 1e-6 and nv > 1e-6:
                u /= nu; v /= nv
                cosang = float(np.clip(np.dot(u, v), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cosang)))
                return ang, "2D"
    return None, None


# =========================
# Per-frame pipeline
# =========================

def process_one_image(args, color_img, depth_img, detector, pose_estimator, visualizer=None, show_interval=0):
    """
    Detector -> pose -> draw skeleton (+ live angle text with 3D/2D tag).
    Returns (pred_instances, key_code, frame_bgr, joint_xyz_top_or_None).
    """
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

    if visualizer is not None:
        visualizer.add_datasample(
            "result",
            color_img_rgb,
            depth_img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            show_kpt_subset=rom_test[args.rom_test],
            skeleton_style=args.skeleton_style,
            show=False,
            wait_time=0,
            kpt_thr=args.kpt_thr,
        )
        frame_rgb = visualizer.get_image()
    else:
        frame_rgb = color_img_rgb

    frame_bgr = mmcv.rgb2bgr(frame_rgb)

    # Prepare possible 3D for the top person
    kpts_xy_top, kpt_scores_top = _extract_top_person_arrays(data_samples.get("pred_instances", None))
    joint_xyz_top = None
    if (depth_img is not None) and (kpts_xy_top is not None):
        intrin = extract_intrinsics_from_depth(depth_img)
        if intrin is not None:
            try:
                joint_xyz_top = compute_joint_xyz_for_person(
                    person_kpts=kpts_xy_top[:, :2],
                    person_vis=kpt_scores_top if kpt_scores_top is not None else np.ones((kpts_xy_top.shape[0],), dtype=np.float32),
                    depth_img=depth_img,
                    intrin_dict=intrin,
                    kpt_thr=args.kpt_thr,
                    window_size=3,
                    reducer="median",
                    ignore_zeros=True,
                    depth_min=0.1,
                    depth_max=4.0,
                )
            except Exception as e:
                print(f"[WARN] 3D back-projection failed: {e}")
                joint_xyz_top = None

    # Live angle text on Pose window for vector-pairs
    vecpairs = get_vectors_for_preset(args.rom_test)
    if vecpairs:
        font = cv2.FONT_HERSHEY_SIMPLEX
        src_tag = None
        if len(vecpairs) == 1:
            ang, src = _angle_from_vecpair_auto(
                kpts_xy_top[:, :2] if kpts_xy_top is not None else None,
                kpt_scores_top,
                joint_xyz_top,
                vecpairs[0],
                args.kpt_thr
            )
            src_tag = src
            txt = "Angle: --" if ang is None else f"Angle: {ang:.1f} deg ({src})"
            cv2.putText(frame_bgr, txt, (10, 30), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            # "both" presets: try to label Left/Right from CATEGORY_SIDES
            label_L, label_R = "Left", "Right"
            for _cat, sides in CATEGORY_SIDES.items():
                if sides.get("both") == args.rom_test:
                    label_L = sides.get("left", "Left")
                    label_R = sides.get("right", "Right")
                    if "_" in label_L: label_L = label_L.split("_")[0].capitalize()
                    if "_" in label_R: label_R = label_R.split("_")[0].capitalize()
                    break
            angL, srcL = _angle_from_vecpair_auto(
                kpts_xy_top[:, :2] if kpts_xy_top is not None else None,
                kpt_scores_top,
                joint_xyz_top,
                vecpairs[0],
                args.kpt_thr
            )
            angR, srcR = _angle_from_vecpair_auto(
                kpts_xy_top[:, :2] if kpts_xy_top is not None else None,
                kpt_scores_top,
                joint_xyz_top,
                vecpairs[1],
                args.kpt_thr
            )
            sL = f"{label_L}: --" if angL is None else f"{label_L}: {angL:.1f} deg ({srcL})"
            sR = f"{label_R}: --" if angR is None else f"{label_R}: {angR:.1f} deg ({srcR})"
            cv2.putText(frame_bgr, f"{sL}   {sR}", (10, 30), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    key = -1
    if args.show:
        cv2.imshow("Pose", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        handle_hotkeys_for_presets(args, key)
        if args.show_interval > 0:
            time.sleep(args.show_interval)

    return data_samples.get("pred_instances", None), key, frame_bgr, joint_xyz_top


# =========================
# Main
# =========================

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
    parser.add_argument("--skeleton-style", default="mmpose", choices=["mmpose", "openpose"])
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--show-interval", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--rom_test", default="full_body")
    # ROM behavior is handled in utils_rom/utils_results; unchanged here, we supply angles.

    args = parser.parse_args()

    # Init models
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(args.pose_config, args.pose_checkpoint, device=args.device)
    init_default_scope('mmpose')

    # Visualizer
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    # Output writer (optional)
    output_file = None
    video_writer = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        if args.input and args.input.lower() not in ("webcam", "realsense"):
            base = os.path.basename(args.input)
            name, _ext = os.path.splitext(base)
            output_file = os.path.join(args.output_root, f"{name}.mp4")
        else:
            output_file = os.path.join(args.output_root, "stream.mp4")

    def _ensure_writer(frame_bgr, cap=None, depth_pipe=None):
        nonlocal video_writer
        if output_file and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            H, W = frame_bgr.shape[:2]
            fps = 25.0
            if cap is not None:
                fps_probe = cap.get(cv2.CAP_PROP_FPS)
                if fps_probe and fps_probe > 1:
                    fps = float(fps_probe)
            elif depth_pipe is not None:
                fps = 30.0
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (W, H))

    # Compare state
    compare_win = "Compare"
    snapA = {"img": None, "kpts": None, "scores": None, "xyz": None}
    snapB = {"img": None, "kpts": None, "scores": None, "xyz": None}
    last_rom_lines = []
    session_root = None  # set on first save (key 6)

    def _update_compare_overlay():
        if not args.show:
            return None
        txtA = "" if get_vectors_for_preset(args.rom_test) else "Select a specific test"
        txtB = txtA
        panel = _compose_compare_canvas(snapA["img"], txtA, snapB["img"], txtB, last_rom_lines)
        cv2.imshow(compare_win, panel)
        return panel

    def _save_compare(panel):
        nonlocal session_root
        if panel is None:
            return
        if session_root is None:
            session_root = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(session_root, exist_ok=True)
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir = os.path.join(session_root, f"{args.rom_test}_{run_ts}")
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, "result.png")
        cv2.imwrite(out_path, panel)
        print(f"[INFO] Saved: {out_path}")

    def _format_rom_lines(label_to_value_src):
        """
        label_to_value_src: list[(label, value_or_None, src_or_None)]
        Output: ["label: value deg (SRC)" or "label: N/A"]
        """
        lines = []
        for label, val, src in label_to_value_src:
            if val is None:
                lines.append(f"{label}: N/A")
            else:
                tag = f" ({src})" if src else ""
                lines.append(f"{label}: {val:.1f} deg{tag}")
        return lines

    def _compute_show_and_save_rom():
        """Compute ROM with 3D-first fallback 2D, refresh Compare, then save panel."""
        nonlocal last_rom_lines
        last_rom_lines = []

        vecpairs = get_vectors_for_preset(args.rom_test)
        if not vecpairs:
            last_rom_lines = ["Select a specific test"]
            panel = _update_compare_overlay()
            _save_compare(panel)
            return
        if snapA["img"] is None or snapB["img"] is None:
            last_rom_lines = ["Snap both frames (4 then 5) before computing ROM (6)."]
            panel = _update_compare_overlay()
            _save_compare(panel)
            return

        label_vals = []
        if len(vecpairs) == 1:
            angA, srcA = _angle_from_vecpair_auto(
                snapA["kpts"][:, :2] if snapA["kpts"] is not None else None, snapA["scores"], snapA["xyz"], vecpairs[0], args.kpt_thr
            )
            angB, srcB = _angle_from_vecpair_auto(
                snapB["kpts"][:, :2] if snapB["kpts"] is not None else None, snapB["scores"], snapB["xyz"], vecpairs[0], args.kpt_thr
            )
            if angA is None or angB is None:
                val = None; src = None
            else:
                val = abs(angB - angA)
                # if one is 3D and the other is 2D, mark as "mixed"
                src = srcA if (srcA == srcB) else "mixed"
            label_vals.append((args.rom_test, val, src))
        else:
            # left/right
            left_name, right_name = None, None
            for _cat, sides in CATEGORY_SIDES.items():
                if sides.get("both") == args.rom_test:
                    left_name = sides.get("left", "Left")
                    right_name = sides.get("right", "Right")
                    break
            for lab, pair in zip((left_name, right_name), vecpairs):
                angA, srcA = _angle_from_vecpair_auto(
                    snapA["kpts"][:, :2] if snapA["kpts"] is not None else None, snapA["scores"], snapA["xyz"], pair, args.kpt_thr
                )
                angB, srcB = _angle_from_vecpair_auto(
                    snapB["kpts"][:, :2] if snapB["kpts"] is not None else None, snapB["scores"], snapB["xyz"], pair, args.kpt_thr
                )
                if angA is None or angB is None:
                    val = None; src = None
                else:
                    val = abs(angB - angA)
                    src = srcA if (srcA == srcB) else "mixed"
                label_vals.append((lab or "", val, src))

        last_rom_lines = _format_rom_lines(label_vals)
        panel = _update_compare_overlay()
        _save_compare(panel)

    # ---------- Input handling ----------
    input_src = (args.input or "").strip()
    cap = None
    depth_pipe = None
    align = None
    is_image = False
    frames = None

    if input_src.lower() == "realsense":
        try:
            depth_pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            depth_pipe.start(cfg)
            align = rs.align(rs.stream.color)
            print("[INFO] RealSense started.")
        except Exception as e:
            print(f"[WARN] RealSense not started: {e}")
            depth_pipe = None

    elif input_src and os.path.exists(input_src):
        mime, _ = mimetypes.guess_type(input_src)
        if mime and mime.startswith("image/"):
            frames = [mmcv.imread(input_src)]
            is_image = True
        else:
            cap = cv2.VideoCapture(input_src)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open input file: {input_src}")
                return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No webcam found (and no other input specified).")
            return

    _update_compare_overlay()

    # ---------- Main loops ----------
    try:
        if depth_pipe is not None:
            while True:
                frameset = depth_pipe.wait_for_frames()
                frameset = align.process(frameset) if align else frameset
                depth_frame = frameset.get_depth_frame()
                color_frame = frameset.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                depth = depth_frame

                pred_instances, key, frame_bgr, joint_xyz_top = process_one_image(
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
                        snapA = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": joint_xyz_top}
                        _update_compare_overlay()
                    elif key == ord('5'):
                        snapB = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": joint_xyz_top}
                        _update_compare_overlay()
                    elif key == ord('6'):
                        _compute_show_and_save_rom()

                if key in (ord('1'), ord('2'), ord('3')):
                    snapA = {"img": None, "kpts": None, "scores": None, "xyz": None}
                    snapB = {"img": None, "kpts": None, "scores": None, "xyz": None}
                    last_rom_lines.clear()
                    _update_compare_overlay()

                if args.show and (cv2.waitKey(1) & 0xFF) == 27:
                    break

        elif cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                pred_instances, key, frame_bgr, joint_xyz_top = process_one_image(
                    args, frame, None, detector, pose_estimator, visualizer, args.show_interval
                )

                if output_file:
                    _ensure_writer(frame_bgr, cap=cap, depth_pipe=None)
                    if video_writer is not None:
                        video_writer.write(frame_bgr)

                if key in (ord('4'), ord('5'), ord('6')):
                    kpts, kscores = _extract_top_person_arrays(pred_instances)
                    if key == ord('4'):
                        snapA = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": None}
                        _update_compare_overlay()
                    elif key == ord('5'):
                        snapB = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": None}
                        _update_compare_overlay()
                    elif key == ord('6'):
                        _compute_show_and_save_rom()

                if key in (ord('1'), ord('2'), ord('3')):
                    snapA = {"img": None, "kpts": None, "scores": None, "xyz": None}
                    snapB = {"img": None, "kpts": None, "scores": None, "xyz": None}
                    last_rom_lines.clear()
                    _update_compare_overlay()

                if args.show and (cv2.waitKey(1) & 0xFF) == 27:
                    break

        elif is_image:
            frame = frames[0]
            pred_instances, key, frame_bgr, joint_xyz_top = process_one_image(
                args, frame, None, detector, pose_estimator, visualizer, args.show_interval
            )
            _update_compare_overlay()
            if key in (ord('4'), ord('5'), ord('6')):
                kpts, kscores = _extract_top_person_arrays(pred_instances)
                if key == ord('4'):
                    snapA = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": None}
                elif key == ord('5'):
                    snapB = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": None}
                elif key == ord('6'):
                    _compute_show_and_save_rom()
            if args.show:
                cv2.waitKey(0)

    finally:
        if cap is not None:
            cap.release()
        if depth_pipe is not None:
            try:
                depth_pipe.stop()
            except Exception:
                pass
        if video_writer is not None:
            video_writer.release()
            print_log(f"the output video has been saved at {output_file}", logger="current", level=logging.INFO)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
