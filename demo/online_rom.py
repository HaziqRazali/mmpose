#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from mmengine.registry import init_default_scope, DefaultScope
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

# ---------------- State ----------------
STATE = {"live_pose": False}
RESULT = {"rom": None, "angle": None, "mode": "n/a"}  # used to annotate t1/t2 after compute

# -------------- Project imports --------------
DEMO_DIR = "/home/haziq/mmpose/demo"
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from utils import presets, calculators, viz
try:
    import utils.drawers_internal_rotation  # noqa: F401
except Exception:
    pass
try:
    import utils.drawers_segments  # noqa: F401
except Exception:
    pass

# -------------- Data holders --------------
@dataclass
class Snapshot:
    image_bgr: np.ndarray
    keypoints: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    timestamp: float

# -------------- Helpers --------------
def switch_scope(name: str):
    init_default_scope(name)
    try:
        DefaultScope.get_current_instance()
    except Exception:
        pass

def largest_person_bbox(det_result, min_score=0.35):
    # mmdet 3.x → DetDataSample
    if isinstance(det_result, DetDataSample):
        inst = det_result.pred_instances
        if inst is None or not hasattr(inst, "bboxes"):
            return None
        bboxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        labels = inst.labels.detach().cpu().numpy()
        mask = (labels == 0)
        if not np.any(mask):
            return None
        b = bboxes[mask]
        s = scores[mask]
        cand = np.hstack([b, s[:, None]])
    else:
        # legacy list/tuple
        if isinstance(det_result, tuple):
            bboxes, labels = det_result
            mask = (labels == 0)
            cand = bboxes[mask] if np.any(mask) else np.empty((0, 5))
        else:
            cand = det_result[0] if len(det_result) > 0 else np.empty((0, 5))

    if cand.shape[0] == 0:
        return None
    areas = (cand[:, 2] - cand[:, 0]) * (cand[:, 3] - cand[:, 1])
    scores = cand[:, 4] if cand.shape[1] >= 5 else np.ones((cand.shape[0],), float)
    areas[scores < min_score] = -1
    idx = int(np.argmax(areas))
    if areas[idx] <= 0:
        return None
    x1, y1, x2, y2, sc = cand[idx, :5]
    return int(x1), int(y1), int(x2), int(y2), float(sc)

def run_pose_on_bbox(pose_model, frame_bgr, bbox_xyxy) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    switch_scope("mmpose")
    det_nd = np.array([[x1, y1, x2, y2]], dtype=np.float32)  # (1,4)
    pose_results = inference_topdown(pose_model, frame_bgr, det_nd, bbox_format="xyxy")
    if not pose_results:
        return None
    ds = merge_data_samples(pose_results)
    kpts_xy = ds.pred_instances.keypoints[0]
    kpts_sc = ds.pred_instances.keypoint_scores[0]
    return np.concatenate([kpts_xy, kpts_sc[:, None]], axis=1)

def capture_snapshot(frame_bgr: np.ndarray, det_model, pose_model) -> Optional[Snapshot]:
    switch_scope("mmdet")
    det = inference_detector(det_model, frame_bgr)
    bb = largest_person_bbox(det, min_score=0.35)
    if bb is None:
        return None
    x1, y1, x2, y2, _ = bb
    kpts = run_pose_on_bbox(pose_model, frame_bgr, (x1, y1, x2, y2))
    if kpts is None:
        return None
    return Snapshot(frame_bgr.copy(), kpts.copy(), (x1, y1, x2, y2), time.time())

def estimate_pose_once(frame_bgr, det_model, pose_model):
    switch_scope("mmdet")
    det = inference_detector(det_model, frame_bgr)
    bb = largest_person_bbox(det, min_score=0.35)
    if bb is None: return None, None
    x1, y1, x2, y2, _ = bb
    kpts = run_pose_on_bbox(pose_model, frame_bgr, (x1, y1, x2, y2))
    return kpts, (x1, y1, x2, y2)

# -------------- Preview panels --------------
def build_panel_from_snapshot(s: Optional[Snapshot], rom_name: str, when_label: str,
                              angle_deg: Optional[float] = None, mode_used: str = "n/a") -> np.ndarray:
    H, W = 720, 1280  # large, clean canvas
    pane = np.zeros((H, W, 3), np.uint8)

    if s is None:
        cv2.putText(pane, "Empty", (24, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA)
        return viz.annotate_panel(pane, rom_name, mode_used, when_label, angle_deg, None)

    img = s.image_bgr
    h, w = img.shape[:2]
    scale = min(W / w, H / h)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    pane[:resized.shape[0], :resized.shape[1]] = resized

    vec_pairs = presets.get_vectors_for_preset(rom_name)
    if vec_pairs:
        k = s.keypoints.copy()
        k[:, :2] *= scale
        pane = viz.draw_vectors2d(pane, k, vec_pairs[0], thickness=2)

    ctx = viz.VizContext(
        frame=pane,
        kpts_xy=s.keypoints * np.array([scale, scale, 1.0])[None, :],
        when_label=when_label,
        angle_deg=angle_deg,
        rgb_pos_sec=None,
        vec_pair=vec_pairs[0] if vec_pairs else None,
        depth_frame=None,
        rgb_size=(resized.shape[1], resized.shape[0]),
        median_k=5,
    )
    viz.draw_for_rom(rom_name, ctx)

    return viz.annotate_panel(pane, rom_name, mode_used, when_label, angle_deg, None)

def show_preview(rom_name: str, t1: Optional[Snapshot], t2: Optional[Snapshot]):
    angle = RESULT["angle"] if RESULT["rom"] == rom_name else None
    mode  = RESULT["mode"]  if RESULT["rom"] == rom_name else "n/a"
    if presets.needs_t2(rom_name):
        L = build_panel_from_snapshot(t1, rom_name, "t1", angle, mode)
        R = build_panel_from_snapshot(t2, rom_name, "t2", angle, mode)
        panel = viz.stack_side_by_side(L, R)
    else:
        panel = build_panel_from_snapshot(t1, rom_name, "t1", angle, mode)
    cv2.imshow("ROM Preview (t1 / t2)", panel)

# -------------- Compute --------------
def compute_and_overlay(rom_name, t1, t2, live_frame):
    need_t2 = presets.needs_t2(rom_name)
    if t1 is None:
        msg = "Compute blocked: capture t1 first."
        print(msg); viz.overlay_text(live_frame, [msg], org=(20, 90)); cv2.imshow("ONLINE ROM (Live)", live_frame); return
    if need_t2 and t2 is None:
        msg = "Compute blocked: this ROM needs t2."
        print(msg); viz.overlay_text(live_frame, [msg], org=(20, 90)); cv2.imshow("ONLINE ROM (Live)", live_frame); return

    H, W = live_frame.shape[:2]
    try:
        if need_t2:
            res = calculators.compute(rom_name, t1.keypoints, t2.keypoints, None, None, rgb_size=(W, H), median_k=5)
            angle = getattr(res, "segment_rotation_t1_to_t2_deg", None) or getattr(res, "delta_deg", None)
        else:
            res = calculators.compute(rom_name, t1.keypoints, None, None, None, rgb_size=(W, H), median_k=5)
            angle = getattr(res, "ang1_deg", None)

        if angle is None or not np.isfinite(angle):
            msg = f"{rom_name}: compute failed (degenerate/missing keypoints)."
            print(msg); viz.overlay_text(live_frame, [msg], org=(20, 90))
        else:
            mode = getattr(res, "mode_used", "n/a")
            RESULT.update({"rom": rom_name, "angle": float(angle), "mode": mode})
            msg = f"RESULT — {rom_name}: {angle:.2f} deg  [{mode}]"
            print(msg); viz.overlay_text(live_frame, [msg], org=(20, 90))
            show_preview(rom_name, t1, t2)  # refresh preview with result
    except Exception as e:
        msg = f"Compute blocked: {e}"
        print(msg); viz.overlay_text(live_frame, [msg], org=(20, 90))

    cv2.imshow("ONLINE ROM (Live)", live_frame)

# -------------- Camera helpers --------------
def try_open_cam(index: int, width: int, height: int, backends: List[int]) -> Optional[cv2.VideoCapture]:
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            cap.release(); continue
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _ = cap.read()
        if ok:
            print(f"[camera] Opened index {index} with backend {be}"); return cap
        cap.release()
    return None

def open_any_camera(preferred: int, width: int, height: int) -> cv2.VideoCapture:
    print(f"[camera] Trying preferred index {preferred} ...")
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    cap = try_open_cam(preferred, width, height, backends)
    if cap: return cap
    print(f"[camera] Preferred index {preferred} failed; scanning indices 0..6 ...")
    for idx in range(0, 7):
        cap = try_open_cam(idx, width, height, backends)
        if cap: print(f"[camera] Selected index {idx}"); return cap
    raise SystemExit("Could not open any /dev/video* device.")

# -------------- Main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--det-cfg", required=True)
    ap.add_argument("--det-ckpt", required=True)
    ap.add_argument("--pose-cfg", required=True)
    ap.add_argument("--pose-ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("[init] Loading models ...")
    init_default_scope("mmdet")
    det_model = init_detector(args.det_cfg, args.det_ckpt, device=args.device)
    init_default_scope("mmpose")
    pose_model = init_pose_model(args.pose_cfg, args.pose_ckpt, device=args.device)
    pose_model.cfg = adapt_mmdet_pipeline(pose_model.cfg)  # local single-arg signature
    print("[init] Models loaded.")

    cap = open_any_camera(args.cam, args.width, args.height)

    rom_names = presets.list_roms()
    pin = ["left_elbow_flexion", "right_elbow_flexion", "left_knee_flexion", "right_knee_flexion", "left_shoulder_flexion"]
    rom_names = [r for r in pin if r in rom_names] + [r for r in rom_names if r not in pin]
    idx = 0

    t1: Optional[Snapshot] = None
    t2: Optional[Snapshot] = None

    LIVE_WIN = "ONLINE ROM (Live)"
    PREV_WIN = "ROM Preview (t1 / t2)"
    cv2.namedWindow(LIVE_WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PREV_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_WIN, 1280, 720)
    cv2.resizeWindow(PREV_WIN, 1280, 720)

    fps_hist, t_prev = [], time.time()
    print("[run] [ / ] switch ROM | 1=t1 | 2=t2 | 3=compute | p=live | q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[camera] Read failed; trying to continue ...")
            time.sleep(0.01)
            continue

        # Clean vs Display buffers
        frame_raw = frame.copy()   # capture/live-pose source (no HUD)
        frame_disp = frame.copy()  # live window (HUD and any transient messages)

        # FPS
        t_now = time.time()
        dt = t_now - t_prev; t_prev = t_now
        fps = 1.0 / dt if dt > 1e-3 else 0.0
        fps_hist = (fps_hist + [fps])[-30:]
        fps_avg = sum(fps_hist) / max(1, len(fps_hist))

        rom = rom_names[idx]
        need_t2 = presets.needs_t2(rom)

        # --- Live pose (optional) ---
        live_ang = None
        if STATE["live_pose"]:
            k_live, bb = estimate_pose_once(frame_raw, det_model, pose_model)
            if k_live is not None:
                vec_pairs = presets.get_vectors_for_preset(rom)
                if vec_pairs:
                    frame_disp = viz.draw_vectors2d(frame_disp, k_live, vec_pairs[0], thickness=2)
                if not need_t2:
                    try:
                        H, W = frame_disp.shape[:2]
                        res = calculators.compute(rom, k_live, None, None, None, rgb_size=(W, H), median_k=3)
                        live_ang = getattr(res, "ang1_deg", None)
                    except Exception:
                        pass

        # --- Single compact HUD (one overlay call only) ---
        lines = [
            f"ROM: {rom}   FPS: {fps_avg:4.1f}",
            "[ / ] prev/next  |  1 t1  2 t2  3 compute  |  p live  q quit",
        ]
        if not need_t2:
            lines.append("Note: 2 is ignored (single-pose ROM).")
        if STATE["live_pose"]:
            s = "LIVE POSE: ON" + (f"  |  {rom}: {live_ang:.1f} deg" if (live_ang is not None) else "")
            lines.append(s)
        viz.overlay_text(frame_disp, lines, org=(20, 40))

        # Preview (no HUD)
        show_preview(rom, t1, t2)

        cv2.imshow(LIVE_WIN, frame_disp)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[run] Quit."); break
        elif key == ord('p'):
            STATE["live_pose"] = not STATE["live_pose"]
            print(f"[live-pose] {'ON' if STATE['live_pose'] else 'OFF'}")
        elif key in (ord('['), ord('{')):
            idx = (idx - 1) % len(rom_names); RESULT["rom"] = None; print(f"[rom] {rom_names[idx]}")
        elif key in (ord(']'), ord('}')):
            idx = (idx + 1) % len(rom_names); RESULT["rom"] = None; print(f"[rom] {rom_names[idx]}")
        elif key == ord('1'):
            snap = capture_snapshot(frame_raw, det_model, pose_model)
            if snap is None: print("[t1] capture failed: no person or pose.")
            else: t1 = snap; print("[t1] saved.")
        elif key == ord('2'):
            if need_t2:
                snap = capture_snapshot(frame_raw, det_model, pose_model)
                if snap is None: print("[t2] capture failed: no person or pose.")
                else: t2 = snap; print("[t2] saved.")
            else:
                print("[t2] ignored: single-pose ROM.")
        elif key == ord('3'):
            compute_and_overlay(rom, t1, t2, live_frame=frame_disp)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
