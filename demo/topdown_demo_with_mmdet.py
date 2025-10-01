import os
import time
import json
import logging
import mimetypes
import cv2
import mmcv
import mmengine
import numpy as np

from pathlib import Path
from typing import List, Dict
from datetime import datetime
from argparse import ArgumentParser
from collections import deque
from collections import deque
from utils_filters import make_filter_1d

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
from utils_filters import make_filter_1d
from utils_visualization import draw_rom_lines
import pyrealsense2 as rs

# =========================
# Helpers: offline batch I/O
# =========================

def _parse_hhmmss(ts: str) -> float:
    """Return seconds as float for 'HH:MM:SS' or 'HH:MM:SS.mmm'."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad timestamp '{ts}'")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return float(h * 3600 + m * 60) + s

def _frame_idx_from_ts(ts: str, fps: float) -> int:
    ts = ts.strip()
    if ts.startswith("#"):
        try:
            return max(0, int(ts[1:]) - 1)  # 1-based -> 0-based
        except Exception:
            raise ValueError(f"Bad index spec '{ts}'")
    return int(max(0, int(_parse_hhmmss(ts) * float(fps))))

def _try_read_rgb(rgb_dir: str, idx: int):
    base_variants = [
        f"{idx:06d}", f"{idx:05d}", f"{idx:04d}", f"{idx:03d}", str(idx),
        f"frame_{idx:06d}", f"frame_{idx:05d}", f"{'frame_'}{idx:04d}", f"frame_{idx:03d}", f"frame_{idx}"
    ]
    for base in base_variants:
        for ext in (".png", ".jpg", ".jpeg"):
            p = Path(rgb_dir) / f"{base}{ext}"
            if p.is_file():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                return img, str(p)

    # Fallback: sorted listing
    cache = getattr(_try_read_rgb, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_try_read_rgb, "_cache", cache)
    files = cache.get(rgb_dir)
    if files is None:
        files = []
        for ext in (".png", ".jpg", ".jpeg"):
            files.extend(sorted(str(p) for p in Path(rgb_dir).glob(f"*{ext}")))
        cache[rgb_dir] = files

    if 0 <= idx < len(files):
        p = files[idx]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        return img, p
    if 1 <= idx <= len(files):  # 1-based fallback
        p = files[idx - 1]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        return img, p
    return None, None

def _compose_compare_panel_simple(imgA, imgB, header_lines: List[str], footer_lines: List[str] = None):
    """Simple side-by-side compositor with header and optional footer."""
    if imgA is None or imgB is None:
        return None
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    target_h = max(hA, hB)
    def _resize_h(img, h):
        if img.shape[0] == h:
            return img
        scale = h / float(img.shape[0])
        w = int(round(img.shape[1] * scale))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    A = _resize_h(imgA, target_h)
    B = _resize_h(imgB, target_h)
    panel = cv2.hconcat([A, B])

    # header
    header_h = 48
    header = np.zeros((header_h, panel.shape[1], 3), dtype=np.uint8)
    y = 32
    for i, line in enumerate(header_lines or []):
        cv2.putText(header, line, (12, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    canv = cv2.vconcat([header, panel])

    if footer_lines:
        footer_h = 36
        footer = np.zeros((footer_h, canv.shape[1], 3), dtype=np.uint8)
        yy = 24
        for i, line in enumerate(footer_lines):
            cv2.putText(footer, line, (12, yy + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        canv = cv2.vconcat([canv, footer])
    return canv

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
            #print("P0", B, "P1", A, "Q1", C, "len(u)", nu, "len(v)", nv)
            if nu > 1e-6 and nv > 1e-6:
                u /= nu; v /= nv
                cosang = float(np.clip(np.dot(u, v), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cosang)))
                return ang, "2D"
    return None, None


# =========================
# Smoothing state (CLI-driven)
# =========================

ANGLE_MODE = "median"  # "none"|"median"|"moving_average"|"exp"
# Single-angle buffers/filters
ANGLE_BUF = None
ANGLE_FILT_1 = None
LAST_ANG = None
LAST_SRC = None
# L/R buffers/filters
ANGLE_BUF_L = None
ANGLE_BUF_R = None
ANGLE_FILT_L = None
ANGLE_FILT_R = None
LAST_L = LAST_R = None
LAST_LSRC = LAST_RSRC = None


def _append_and_disp_from_buf(buf, ang, src):
    """Append sample and return (ang_disp, src_disp) with 3D-first temporal median."""
    if ang is not None and src:
        buf.append((time.time(), float(ang), src))
    v3 = [a for _, a, s in buf if s == "3D"]
    v2 = [a for _, a, s in buf if s == "2D"]
    if v3:
        return float(np.median(v3)), "3D"
    if v2:
        return float(np.median(v2)), "2D"
    return None, None


# =========================
# Per-frame pipeline
# =========================

def process_one_image(args, color_img, depth_img, detector, pose_estimator, visualizer=None, show_interval=0):
    """
    Detector -> pose -> draw skeleton or ROM lines (+ live angle text with 3D/2D tag).
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

    # Visualization branch
    use_rom_only = bool(getattr(args, "only_rom_lines", False)) and (args.rom_test not in ("133", "full_body"))
    if (visualizer is not None) and (not use_rom_only):
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
                    window_size=int(args.depth_win),
                    reducer=str(args.depth_reducer),
                    ignore_zeros=bool(args.depth_ignore_zeros),
                    depth_min=float(args.depth_min),
                    depth_max=float(args.depth_max),
                )
            except Exception as e:
                print(f"[WARN] 3D back-projection failed: {e}")
                joint_xyz_top = None

    # If ROM-only mode, draw only the vectors used for the current preset
    if use_rom_only:
        if (kpts_xy_top is not None) and (kpt_scores_top is not None):
            vecpairs = get_vectors_for_preset(args.rom_test)
            frame_bgr = draw_rom_lines(
                frame_bgr,
                kpts_xy_top[:, :2],
                kpt_scores_top,
                vecpairs,
                kpt_thr=args.kpt_thr,
                thickness=args.thickness,
            )

    # Live angle text on Pose window for vector-pairs
    vecpairs = get_vectors_for_preset(args.rom_test)
    font = cv2.FONT_HERSHEY_SIMPLEX

    global LAST_ANG, LAST_SRC, LAST_L, LAST_LSRC, LAST_R, LAST_RSRC

    if vecpairs:
        if len(vecpairs) == 1:
            # Single angle
            ang, src = _angle_from_vecpair_auto(
                kpts_xy_top[:, :2] if kpts_xy_top is not None else None,
                kpt_scores_top,
                joint_xyz_top,
                vecpairs[0],
                args.kpt_thr
            )

            if ANGLE_MODE == "median":
                ang_disp, src_disp = _append_and_disp_from_buf(ANGLE_BUF, ang, src)
            elif ANGLE_MODE == "moving_average":
                ang_disp = None if ang is None else ANGLE_FILT_1.update(ang); src_disp = src
            elif ANGLE_MODE == "exp":
                ang_disp = None if ang is None else ANGLE_FILT_1.update(ang); src_disp = src
            else:
                ang_disp, src_disp = ang, src

            LAST_ANG, LAST_SRC = ang_disp, src_disp
            txt = "Angle: --" if ang_disp is None else f"Angle: {ang_disp:.1f} deg ({src_disp})"
            cv2.putText(frame_bgr, txt, (10, 30), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        else:
            # Left/Right angles
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

            if ANGLE_MODE == "median":
                angL_disp, srcL_disp = _append_and_disp_from_buf(ANGLE_BUF_L, angL, srcL)
                angR_disp, srcR_disp = _append_and_disp_from_buf(ANGLE_BUF_R, angR, srcR)
            elif ANGLE_MODE == "moving_average":
                angL_disp = None if angL is None else ANGLE_FILT_L.update(angL); srcL_disp = srcL
                angR_disp = None if angR is None else ANGLE_FILT_R.update(angR); srcR_disp = srcR
            elif ANGLE_MODE == "exp":
                angL_disp = None if angL is None else ANGLE_FILT_L.update(angL); srcL_disp = srcL
                angR_disp = None if angR is None else ANGLE_FILT_R.update(angR); srcR_disp = srcR
            else:
                angL_disp, srcL_disp = angL, srcL
                angR_disp, srcR_disp = angR, srcR

            LAST_L, LAST_LSRC = angL_disp, srcL_disp
            LAST_R, LAST_RSRC = angR_disp, srcR_disp

            sL = f"{label_L}: --" if angL_disp is None else f"{label_L}: {angL_disp:.1f} deg ({srcL_disp})"
            sR = f"{label_R}: --" if angR_disp is None else f"{label_R}: {angR_disp:.1f} deg ({srcR_disp})"
            cv2.putText(frame_bgr, f"{sL}   {sR}", (10, 30), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    key = -1
    if args.show:
        cv2.imshow("Pose", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        handle_hotkeys_for_presets(args, key)
        if args.show_interval > 0:
            time.sleep(args.show_interval)

    return data_samples.get("pred_instances", None), key, frame_bgr, joint_xyz_top

def run_offline_batch(args, detector, pose_estimator, visualizer):
    """
    Batch mode:
      - Read tasks JSON
      - For each job: set args.rom_test, map t1/t2 -> frame indices, run pose on both frames,
        read displayed angles from LAST_* globals, compose a compare panel, and save.
    """
    tasks_path = Path(args.tasks)
    rgb_dir = Path(args.rgb_dir)
    if not tasks_path.is_file():
        raise FileNotFoundError(f"--tasks not found: {tasks_path}")
    if not rgb_dir.is_dir():
        raise NotADirectoryError(f"--rgb-dir not a directory: {rgb_dir}")

    with open(tasks_path, "r") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list) or len(jobs) == 0:
        raise ValueError("tasks JSON must be a non-empty list")

    # Session output root
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) if args.output_root else Path(".")
    out_root = out_root / f"batch_{session_stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Disable live windows unless user wants it
    show_windows = False if args.no_show else bool(args.show)

    # Init temporal smoothing globals (same as interactive)
    global ANGLE_MODE, ANGLE_BUF, ANGLE_BUF_L, ANGLE_BUF_R
    global ANGLE_FILT_1, ANGLE_FILT_L, ANGLE_FILT_R

    ANGLE_MODE = getattr(args, "angle_filter", "none")

    if ANGLE_MODE == "median":
        win = int(getattr(args, "angle_window", 9))
        ANGLE_BUF   = deque(maxlen=max(1, win))
        ANGLE_BUF_L = deque(maxlen=max(1, win))
        ANGLE_BUF_R = deque(maxlen=max(1, win))
        ANGLE_FILT_1 = ANGLE_FILT_L = ANGLE_FILT_R = None

    elif ANGLE_MODE in ("moving_average", "exp"):
        kind = "moving_average" if ANGLE_MODE == "moving_average" else "exponential"
        ANGLE_BUF = ANGLE_BUF_L = ANGLE_BUF_R = None
        ANGLE_FILT_1 = make_filter_1d(
            kind=kind,
            ma_window=getattr(args, "ma_window", 9),
            ma_robust=getattr(args, "ma_robust", False),
            exp_alpha=getattr(args, "exp_alpha", 0.2),
        )
        ANGLE_FILT_L = make_filter_1d(
            kind=kind,
            ma_window=getattr(args, "ma_window", 9),
            ma_robust=getattr(args, "ma_robust", False),
            exp_alpha=getattr(args, "exp_alpha", 0.2),
        )
        ANGLE_FILT_R = make_filter_1d(
            kind=kind,
            ma_window=getattr(args, "ma_window", 9),
            ma_robust=getattr(args, "ma_robust", False),
            exp_alpha=getattr(args, "exp_alpha", 0.2),
        )

    else:  # none
        ANGLE_MODE = "none"
        ANGLE_BUF = ANGLE_BUF_L = ANGLE_BUF_R = None
        ANGLE_FILT_1 = ANGLE_FILT_L = ANGLE_FILT_R = None

    processed = 0
    failed = 0

    for job in jobs:
        # Two shapes supported:
        #   {"t1":"HH:MM:SS(.ms)","t2":"HH:MM:SS(.ms)","test":"left_elbow_flexion","label":"optional"}
        #   {"ts":[...], "test":"...", "label":"optional"} -> runs adjacent pairs
        label = str(job.get("label", f"job{processed+1}"))
        test = str(job.get("test", "full_body"))

        def _run_pair(t1: str, t2: str, pair_idx: int):
            nonlocal processed, failed
            try:
                idx1 = _frame_idx_from_ts(t1, args.fps)
                idx2 = _frame_idx_from_ts(t2, args.fps)

                # load RGB frames
                rgb1, p1 = _try_read_rgb(str(rgb_dir), idx1)
                rgb2, p2 = _try_read_rgb(str(rgb_dir), idx2)
                if rgb1 is None or rgb2 is None:
                    raise FileNotFoundError(f"Frames not found for indices {idx1},{idx2}")

                # Set current ROM preset
                setattr(args, "rom_test", test)

                # Run inference on frame 1
                pred1, key1, vis1, joint_xyz1 = process_one_image(
                    args, rgb1, None, detector, pose_estimator, visualizer, show_interval=0
                )
                # Capture angle(s)
                vecpairs = get_vectors_for_preset(args.rom_test)
                if vecpairs and len(vecpairs) == 1:
                    a1, a1_src = LAST_ANG, LAST_SRC
                elif vecpairs and len(vecpairs) >= 2:
                    # For L/R tests we report both and main ROM is max delta across sides
                    a1, a1_src = (LAST_L, LAST_LSRC)
                    b1, b1_src = (LAST_R, LAST_RSRC)
                else:
                    a1 = None

                # Run inference on frame 2
                pred2, key2, vis2, joint_xyz2 = process_one_image(
                    args, rgb2, None, detector, pose_estimator, visualizer, show_interval=0
                )
                if vecpairs and len(vecpairs) == 1:
                    a2, a2_src = LAST_ANG, LAST_SRC
                    rom_deg = None if (a1 is None or a2 is None) else float(abs(a2 - a1))
                    footer = [] if rom_deg is None else [f"ROM = {rom_deg:.1f} deg"]
                    hdr = [f"Test: {test}",
                           f"T1 {t1} -> idx {idx1} | T2 {t2} -> idx {idx2}",
                           f"Angles: {('N/A' if a1 is None else f'{a1:.1f} deg')} -> {('N/A' if a2 is None else f'{a2:.1f} deg')}"]
                else:
                    a2, a2_src = (LAST_L, LAST_LSRC)
                    b2, b2_src = (LAST_R, LAST_RSRC)
                    # Compute per-side deltas if both sides exist
                    dL = None if (a1 is None or a2 is None) else float(abs(a2 - a1))
                    dR = None if ('b1' not in locals() or 'b2' not in locals() or b1 is None or b2 is None) else float(abs(b2 - b1))
                    # Report max available as ROM
                    rom_candidates = [d for d in (dL, dR) if d is not None]
                    rom_deg = None if not rom_candidates else float(max(rom_candidates))
                    footer = [] if rom_deg is None else [f"ROM = {rom_deg:.1f} deg"]
                    hdr = [f"Test: {test}",
                           f"T1 {t1} -> idx {idx1} | T2 {t2} -> idx {idx2}",
                           f"L: {('N/A' if a1 is None else f'{a1:.1f}')} -> {('N/A' if a2 is None else f'{a2:.1f}')},  "
                           f"R: {('N/A' if 'b1' not in locals() or b1 is None else f'{b1:.1f}')} -> {('N/A' if 'b2' not in locals() or b2 is None else f'{b2:.1f}')} deg"]

                # Compose compare panel
                panel = _compose_compare_panel_simple(vis1, vis2, header_lines=hdr, footer_lines=footer)

                # Save
                out_dir = out_root / f"{label}_{test}"
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{t1.replace(':','-')}_{t2.replace(':','-')}"
                panel_path = out_dir / f"panel_{stem}.png"
                cv2.imwrite(str(panel_path), panel)

                # Write a small JSON per pair
                meta = {
                    "label": label,
                    "test": test,
                    "t1": t1, "t2": t2,
                    "idx1": idx1, "idx2": idx2,
                    "angle_mode": "single" if (vecpairs and len(vecpairs) == 1) else "lr",
                    "angle_t1": None if vecpairs and len(vecpairs) != 1 else a1,
                    "angle_t2": None if vecpairs and len(vecpairs) != 1 else a2,
                    "rom_deg": None if panel is None else (footer and rom_deg),
                    "panel_path": str(panel_path)
                }
                with open(out_dir / f"result_{stem}.json", "w") as jf:
                    json.dump(meta, jf, indent=2)
                processed += 1

                # Optional view
                if show_windows:
                    cv2.imshow("Compare", panel)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        return  # ESC closes batch early

            except Exception as e:
                failed += 1
                print(f"[ERROR] Batch pair failed for {label}/{test} ({t1},{t2}): {e}")

        # one pair
        if "t1" in job and "t2" in job:
            _run_pair(str(job["t1"]), str(job["t2"]), 0)
        # sequence
        elif "ts" in job and isinstance(job["ts"], list) and len(job["ts"]) >= 2:
            ts = [str(x) for x in job["ts"]]
            for i in range(len(ts) - 1):
                _run_pair(ts[i], ts[i+1], i)
        else:
            print(f"[WARN] Skipping malformed job: {job}")

    if show_windows:
        cv2.destroyAllWindows()

    print(f"[BATCH DONE] processed={processed}, failed={failed}, output={out_root}")

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

    # New flag: ROM-only visualization
    parser.add_argument("--only-rom-lines", dest="only_rom_lines", action="store_true", help="If set, hide skeleton for specific ROM tests and draw only the two ROM vectors.")
    parser.set_defaults(only_rom_lines=False)

    # --- depth back-projection args ---
    parser.add_argument("--depth-win", type=int, default=3, help="Depth window size (odd).")
    parser.add_argument("--depth-reducer", default="median", choices=["median", "mean"], help="Depth reducer within window.")
    parser.add_argument("--depth-min", type=float, default=0.1, help="Min valid depth in meters.")
    parser.add_argument("--depth-max", type=float, default=4.0, help="Max valid depth in meters.")
    parser.add_argument("--no-depth-ignore-zeros", dest="depth_ignore_zeros", action="store_false",
                        help="If set, do not ignore zeros in depth window.")
    parser.set_defaults(depth_ignore_zeros=True)

    # --- NEW: depth viewing and probing ---
    parser.add_argument("--show-depth", action="store_true",
                        help="Open a separate window to visualize the depth stream when using RealSense.")
    parser.add_argument("--probe-depth", action="store_true",
                        help="Left-click in 'Pose' or 'Depth' window to print depth at that pixel (meters).")

    # --- temporal angle smoothing args ---
    parser.add_argument("--angle-filter", default="median",
                        choices=["none", "median", "moving_average", "exp"],
                        help="Temporal smoothing mode for angles.")
    parser.add_argument("--angle-window", type=int, default=15, help="Window length for median mode.")
    parser.add_argument("--ma-window", type=int, default=5, help="Window for moving average.")
    parser.add_argument("--ma-robust", action="store_true", help="Use median in moving average (robust MA).")
    parser.add_argument("--exp-alpha", type=float, default=0.25, help="EMA alpha for exp mode.")

    # --- NEW: offline RGB batch mode ---
    parser.add_argument("--rgb-dir", type=str, default="", help="Directory of RGB frames named 000001.png/jpg etc.")
    parser.add_argument("--tasks", type=str, default="", help="JSON file describing timestamped jobs.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used to map HH:MM:SS timestamps to frame indices.")
    parser.add_argument("--no-show", action="store_true", help="Disable any UI windows in batch mode.")

    args = parser.parse_args()

    # Init models
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(args.pose_config, args.pose_checkpoint, device=args.device)
    init_default_scope('mmpose')

    # Visualizer
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    if args.tasks and args.rgb_dir != "":
        if args.no_show:
            args.show = False
        run_offline_batch(args, detector, pose_estimator, visualizer)
        return

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
    snapA = {"img": None, "kpts": None, "scores": None, "xyz": None, "ang": None, "src": None,
             "angL": None, "srcL": None, "angR": None, "srcR": None}
    snapB = {"img": None, "kpts": None, "scores": None, "xyz": None, "ang": None, "src": None,
             "angL": None, "srcL": None, "angR": None, "srcR": None}
    last_rom_lines = []
    session_root = None  # set on first save (key 6)

    # Init smoothing globals based on args
    global ANGLE_MODE, ANGLE_BUF, ANGLE_BUF_L, ANGLE_BUF_R
    global ANGLE_FILT_1, ANGLE_FILT_L, ANGLE_FILT_R
    ANGLE_MODE = args.angle_filter
    if ANGLE_MODE == "median":
        ANGLE_BUF = deque(maxlen=max(1, int(args.angle_window)))
        ANGLE_BUF_L = deque(maxlen=max(1, int(args.angle_window)))
        ANGLE_BUF_R = deque(maxlen=max(1, int(args.angle_window)))
    elif ANGLE_MODE in ("moving_average", "exp"):
        kind = "moving_average" if ANGLE_MODE == "moving_average" else "exponential"
        def mk():
            return make_filter_1d(kind=kind,
                                  ma_window=args.ma_window, ma_robust=args.ma_robust,
                                  exp_alpha=args.exp_alpha)
        ANGLE_FILT_1 = mk(); ANGLE_FILT_L = mk(); ANGLE_FILT_R = mk()
    # else: "none" -> nothing to init

    # --- Depth probe state and callbacks ---
    depth_probe = {"frame": None}  # latest aligned rs.depth_frame

    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        df = depth_probe.get("frame", None)
        if df is None:
            print("[DEPTH] No depth frame available.")
            return
        try:
            z = float(df.get_distance(int(x), int(y)))
            if np.isfinite(z) and z > 0.0:
                print(f"[DEPTH] ({x},{y}) = {z:.3f} m")
            else:
                print(f"[DEPTH] ({x},{y}) = invalid/0")
        except Exception as e:
            print(f"[DEPTH] probe failed: {e}")

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
        lines = []
        for label, val, src in label_to_value_src:
            if val is None:
                lines.append(f"{label}: N/A")
            else:
                tag = f" ({src})" if src else ""
                lines.append(f"{label}: {val:.1f} deg{tag}")
        return lines

    def _compute_show_and_save_rom():
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
            if (snapA.get("ang") is not None) and (snapB.get("ang") is not None):
                val = abs(float(snapB["ang"]) - float(snapA["ang"]))
                src = snapA["src"] if (snapA["src"] == snapB["src"]) else "mixed"
            else:
                from_this = vecpairs[0]
                angA, srcA = _angle_from_vecpair_auto(
                    snapA["kpts"][:, :2] if snapA["kpts"] is not None else None, snapA["scores"], snapA["xyz"], from_this, args.kpt_thr
                )
                angB, srcB = _angle_from_vecpair_auto(
                    snapB["kpts"][:, :2] if snapB["kpts"] is not None else None, snapB["scores"], snapB["xyz"], from_this, args.kpt_thr
                )
                if angA is None or angB is None:
                    val = None; src = None
                else:
                    val = abs(angB - angA)
                    src = srcA if (srcA == srcB) else "mixed"
            label_vals.append((args.rom_test, val, src))
        else:
            left_name, right_name = None, None
            for _cat, sides in CATEGORY_SIDES.items():
                if sides.get("both") == args.rom_test:
                    left_name = sides.get("left", "Left")
                    right_name = sides.get("right", "Right")
                    break
            for side, lab, pair in (("L", left_name, vecpairs[0]), ("R", right_name, vecpairs[1])):
                key_ang = "angL" if side == "L" else "angR"
                key_src = "srcL" if side == "L" else "srcR"
                if (snapA.get(key_ang) is not None) and (snapB.get(key_ang) is not None):
                    angA = float(snapA[key_ang]); srcA = snapA[key_src]
                    angB = float(snapB[key_ang]); srcB = snapB[key_src]
                else:
                    angA, srcA = _angle_from_vecpair_auto(
                        snapA["kpts"][:, :2] if snapA["kpts"] is not None else None, snapA["scores"], snapA["xyz"], pair, args.kpt_thr
                    )
                    angB, srcB = _angle_from_vecpair_auto(
                        snapB["kpts"][:, :2] if snapB["kpts"] is not None else None, snapB["scores"], snapB["xyz"], pair, args.kpt_thr
                    )
                if (angA is None) or (angB is None):
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

    # Depth visualization helper
    colorizer = rs.colorizer() if args.show_depth else None

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

    # Precreate windows for mouse callbacks if needed
    if args.show:
        cv2.namedWindow("Pose", cv2.WINDOW_NORMAL)
    if args.show_depth:
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    if args.probe_depth:
        # Attach callbacks only to Pose and Depth (not Compare)
        cv2.setMouseCallback("Pose", _on_mouse)
        if args.show_depth:
            cv2.setMouseCallback("Depth", _on_mouse)

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

                # update probe frame
                depth_probe["frame"] = depth_frame

                # Optional depth window
                if args.show_depth:
                    dvis = np.asanyarray(colorizer.colorize(depth_frame).get_data()) if colorizer else None
                    if dvis is not None:
                        cv2.imshow("Depth", dvis)

                pred_instances, key, frame_bgr, joint_xyz_top = process_one_image(
                    args, color, depth_frame, detector, pose_estimator, visualizer, args.show_interval
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
                        snapA = {
                            "img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": joint_xyz_top,
                            "ang": LAST_ANG, "src": LAST_SRC,
                            "angL": LAST_L, "srcL": LAST_LSRC, "angR": LAST_R, "srcR": LAST_RSRC
                        }
                        _update_compare_overlay()
                    elif key == ord('5'):
                        snapB = {
                            "img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": joint_xyz_top,
                            "ang": LAST_ANG, "src": LAST_SRC,
                            "angL": LAST_L, "srcL": LAST_LSRC, "angR": LAST_R, "srcR": LAST_RSRC
                        }
                        _update_compare_overlay()
                    elif key == ord('6'):
                        _compute_show_and_save_rom()

                if key in (ord('1'), ord('2'), ord('3')):
                    snapA = {"img": None, "kpts": None, "scores": None, "xyz": None, "ang": None, "src": None,
                             "angL": None, "srcL": None, "angR": None, "srcR": None}
                    snapB = {"img": None, "kpts": None, "scores": None, "xyz": None, "ang": None, "src": None,
                             "angL": None, "srcL": None, "angR": None, "srcR": None}
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
                        snapA = {
                            "img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": None,
                            "ang": LAST_ANG, "src": LAST_SRC,
                            "angL": LAST_L, "srcL": LAST_LSRC, "angR": LAST_R, "srcR": LAST_RSRC
                        }
                        _update_compare_overlay()
                    elif key == ord('5'):
                        snapB = {
                            "img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": None,
                            "ang": LAST_ANG, "src": LAST_SRC,
                            "angL": LAST_L, "srcL": LAST_LSRC, "angR": LAST_R, "srcR": LAST_RSRC
                        }
                        _update_compare_overlay()
                    elif key == ord('6'):
                        _compute_show_and_save_rom()

                if key in (ord('1'), ord('2'), ord('3')):
                    snapA = {"img": None, "kpts": None, "scores": None, "xyz": None, "ang": None, "src": None,
                             "angL": None, "srcL": None, "angR": None, "srcR": None}
                    snapB = {"img": None, "kpts": None, "scores": None, "xyz": None, "ang": None, "src": None,
                             "angL": None, "srcL": None, "angR": None, "srcR": None}
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
                    snapA = {
                        "img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": None,
                        "ang": LAST_ANG, "src": LAST_SRC,
                        "angL": LAST_L, "srcL": LAST_LSRC, "angR": LAST_R, "srcR": LAST_RSRC
                    }
                elif key == ord('5'):
                    snapB = {
                        "img": frame_bgr.copy(), "kpts": kpts, "scores": kscores, "xyz": None,
                        "ang": LAST_ANG, "src": LAST_SRC,
                        "angL": LAST_L, "srcL": LAST_LSRC, "angR": LAST_R, "srcR": LAST_RSRC
                    }
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
