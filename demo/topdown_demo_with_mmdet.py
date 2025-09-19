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

from utils_variables import rom_test, CATEGORY_SIDES
from utils import handle_hotkeys_for_presets

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


def _normalize_triplet(trip):
    """Return exactly 3 joint IDs; handle ankle triplets like [13,15,[17,18]]."""
    if not isinstance(trip, (list, tuple)) or len(trip) < 3:
        return None
    a = int(trip[0]); b = int(trip[1]); c_raw = trip[2]
    c = int(c_raw[0]) if isinstance(c_raw, (list, tuple)) else int(c_raw)
    return [a, b, c]


def _angle_deg_from_triplet_2d(kpts_xy, kpt_scores, triplet, kpt_thr=0.3):
    """Angle at middle joint B between BA and BC, degrees in [0,180]."""
    if kpts_xy is None or kpt_scores is None:
        return None
    tri = _normalize_triplet(triplet)
    if tri is None:
        return None
    a, b, c = tri
    K = kpts_xy.shape[0]
    if not (0 <= a < K and 0 <= b < K and 0 <= c < K):
        return None
    if not (kpt_scores[a] >= kpt_thr and kpt_scores[b] >= kpt_thr and kpt_scores[c] >= kpt_thr):
        return None

    A, B, C = kpts_xy[a].astype(np.float32), kpts_xy[b].astype(np.float32), kpts_xy[c].astype(np.float32)
    v1, v2 = A - B, C - B
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _current_triplets_for_preset(preset_name):
    """
    For a given rom_test name:
      - If single-side: return [triplet]
      - If group (both): return [left_triplet, right_triplet]
      - If full_body / 133: return []
    """
    if preset_name in ("full_body", "133"):
        return []
    for cat, sides in CATEGORY_SIDES.items():
        if sides.get("both") == preset_name:
            return [rom_test[sides["left"]], rom_test[sides["right"]]]
        if preset_name in sides.values() and preset_name in rom_test:
            return [rom_test[preset_name]]
    entry = rom_test.get(preset_name)
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        return [entry]
    return []


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

    # >>> Increased footer height <<<
    title_h = 36
    footer_h = (80 if rom_lines else 0)  # was ~52

    W = row.shape[1]
    panel = np.full((title_h + H + footer_h, W, 3), bg, np.uint8)

    cv2.putText(panel, "Frame A (4)", (8, 25), font, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, "Frame B (5)", (W // 2 + pad + 8, 25), font, 0.8, (255, 255, 255), 2)

    panel[title_h:title_h + H] = row
    if txtA:
        cv2.putText(panel, txtA, (8, title_h + 28), font, 0.7, (0, 255, 255), 2)
    if txtB:
        cv2.putText(panel, txtB, (W // 2 + pad + 8, title_h + 28), font, 0.7, (0, 255, 255), 2)

    # Draw footer lines (more spacing between lines for readability)
    for i, line in enumerate(rom_lines):
        cv2.putText(panel, line, (8, title_h + H + 36 + i * 26), font, 0.8, (255, 255, 255), 2)

    return panel


# =========================
# Per-frame pipeline
# =========================

def process_one_image(args, color_img, depth_img, detector, pose_estimator, visualizer=None, show_interval=0):
    """Detector -> pose -> draw skeleton only (+ live angle text). Returns (pred_instances, key_code, frame_bgr)."""
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

    # Live angle text on Pose window every frame (except full_body/133)
    trips = _current_triplets_for_preset(args.rom_test)
    if args.rom_test not in ("full_body", "133") and len(trips) > 0:
        pred_instances = data_samples.get("pred_instances", None)
        kpts, kscores = _extract_top_person_arrays(pred_instances)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(trips) == 1:
            ang = _angle_deg_from_triplet_2d(kpts, kscores, trips[0], args.kpt_thr) if kpts is not None else None
            txt = "Angle: --" if ang is None else f"Angle: {ang:.1f} deg"
            cv2.putText(frame_bgr, txt, (10, 30), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            angL = _angle_deg_from_triplet_2d(kpts, kscores, trips[0], args.kpt_thr) if kpts is not None else None
            angR = _angle_deg_from_triplet_2d(kpts, kscores, trips[1], args.kpt_thr) if kpts is not None else None
            label_L, label_R = "Left", "Right"
            for _cat, sides in CATEGORY_SIDES.items():
                if sides.get("both") == args.rom_test:
                    label_L = sides.get("left", "Left")
                    label_R = sides.get("right", "Right")
                    if "_" in label_L: label_L = label_L.split("_")[0].capitalize()
                    if "_" in label_R: label_R = label_R.split("_")[0].capitalize()
                    break
            sL = f"{label_L}: --" if angL is None else f"{label_L}: {angL:.1f} deg"
            sR = f"{label_R}: --" if angR is None else f"{label_R}: {angR:.1f} deg"
            cv2.putText(frame_bgr, f"{sL}   {sR}", (10, 30), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    key = -1
    if args.show:
        cv2.imshow("Pose", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        handle_hotkeys_for_presets(args, key)
        if args.show_interval > 0:
            time.sleep(args.show_interval)

    return data_samples.get("pred_instances", None), key, frame_bgr


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
    snapA = {"img": None, "kpts": None, "scores": None}
    snapB = {"img": None, "kpts": None, "scores": None}
    last_rom_lines = []
    session_root = None  # set on first save (key 6)

    # Footer text builder (no spacing padding; single space after colon)
    def _format_rom_lines(label_to_value):
        """
        label_to_value: list[(label, value_or_None)]
        Output: ["label: value deg" or "label: N/A"]
        """
        lines = []
        for label, val in label_to_value:
            if val is None:
                lines.append(f"{label}: N/A")
            else:
                lines.append(f"{label}: {val:.1f} deg")
        return lines

    def _update_compare_overlay():
        if not args.show:
            return None
        # No re-printing of angles in panes (already baked into snaps)
        if args.rom_test in ("full_body", "133"):
            txtA = "Select a specific test"
            txtB = "Select a specific test"
        else:
            txtA = ""
            txtB = ""
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

    def _compute_show_and_save_rom():
        """Compute ROM (if possible), refresh Compare, then save panel to disk."""
        nonlocal last_rom_lines
        last_rom_lines = []

        if args.rom_test in ("full_body", "133"):
            last_rom_lines = ["Select a specific test"]
            panel = _update_compare_overlay()
            _save_compare(panel)
            return

        trips = _current_triplets_for_preset(args.rom_test)
        if snapA["img"] is None or snapB["img"] is None:
            last_rom_lines = ["Snap both frames (4 then 5) before computing ROM (6)."]
            panel = _update_compare_overlay()
            _save_compare(panel)
            return
        if len(trips) == 0:
            last_rom_lines = ["No valid triplet for this test."]
            panel = _update_compare_overlay()
            _save_compare(panel)
            return

        # Build label->value pairs
        label_vals = []
        if len(trips) == 1:
            angA = _angle_deg_from_triplet_2d(snapA["kpts"], snapA["scores"], trips[0], args.kpt_thr)
            angB = _angle_deg_from_triplet_2d(snapB["kpts"], snapB["scores"], trips[0], args.kpt_thr)
            val = None if (angA is None or angB is None) else abs(angB - angA)
            label_vals.append((args.rom_test, val))
        else:
            for _cat, sides in CATEGORY_SIDES.items():
                if sides.get("both") == args.rom_test:
                    left_name = sides.get("left", "Left")
                    right_name = sides.get("right", "Right")
                    break
            for preset_name, trip in zip((left_name, right_name), trips):
                angA = _angle_deg_from_triplet_2d(snapA["kpts"], snapA["scores"], trip, args.kpt_thr)
                angB = _angle_deg_from_triplet_2d(snapB["kpts"], snapB["scores"], trip, args.kpt_thr)
                val = None if (angA is None or angB is None) else abs(angB - angA)
                label_vals.append((preset_name, val))

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
        # webcam or unspecified -> webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No webcam found (and no other input specified).")
            return

    # Open Compare window immediately with blank panes
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

                pred_instances, key, frame_bgr = process_one_image(
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
                        snapA = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores}
                        _update_compare_overlay()
                    elif key == ord('5'):
                        snapB = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores}
                        _update_compare_overlay()
                    elif key == ord('6'):
                        _compute_show_and_save_rom()

                if key in (ord('1'), ord('2'), ord('3')):
                    snapA = {"img": None, "kpts": None, "scores": None}
                    snapB = {"img": None, "kpts": None, "scores": None}
                    last_rom_lines.clear()
                    _update_compare_overlay()

                if args.show and (cv2.waitKey(1) & 0xFF) == 27:
                    break

        elif cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                pred_instances, key, frame_bgr = process_one_image(
                    args, frame, None, detector, pose_estimator, visualizer, args.show_interval
                )

                if output_file:
                    _ensure_writer(frame_bgr, cap=cap, depth_pipe=None)
                    if video_writer is not None:
                        video_writer.write(frame_bgr)

                if key in (ord('4'), ord('5'), ord('6')):
                    kpts, kscores = _extract_top_person_arrays(pred_instances)
                    if key == ord('4'):
                        snapA = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores}
                        _update_compare_overlay()
                    elif key == ord('5'):
                        snapB = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores}
                        _update_compare_overlay()
                    elif key == ord('6'):
                        _compute_show_and_save_rom()

                if key in (ord('1'), ord('2'), ord('3')):
                    snapA = {"img": None, "kpts": None, "scores": None}
                    snapB = {"img": None, "kpts": None, "scores": None}
                    last_rom_lines.clear()
                    _update_compare_overlay()

                if args.show and (cv2.waitKey(1) & 0xFF) == 27:
                    break

        elif is_image:
            frame = frames[0]
            pred_instances, key, frame_bgr = process_one_image(
                args, frame, None, detector, pose_estimator, visualizer, args.show_interval
            )
            _update_compare_overlay()
            if key in (ord('4'), ord('5'), ord('6')):
                kpts, kscores = _extract_top_person_arrays(pred_instances)
                if key == ord('4'):
                    snapA = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores}
                elif key == ord('5'):
                    snapB = {"img": frame_bgr.copy(), "kpts": kpts, "scores": kscores}
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
