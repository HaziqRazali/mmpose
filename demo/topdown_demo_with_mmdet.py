# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

import ast
from utils_variables import presets, rom_test
from utils import set_kpt_preset, handle_hotkeys_for_presets
from utils_3d import extract_intrinsics_from_depth, compute_3d_skeletons, visualize_3d_skeletons, angle

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    import pyrealsense2 as rs
except (ImportError, ModuleNotFoundError):
    rs = None
    pass


# =========================
# Stability Tracking Helpers
# =========================

@dataclass
class MotionStability:
    """
    Tracks whether a signal is 'stable' over a sliding window and measures how long
    it has remained stable. Supports two modes:

    - 'angle': value is a float in degrees. Stability uses std dev (deg) and median speed (deg/s).
    - 'position': value is an array [K,3] in meters. Stability uses mean joint std (mm)
                  and median joint speed (mm/s) across the tracked joints.

    A small dropout_tolerance allows brief instability without resetting the timer.
    """
    mode: str  # 'angle' or 'position'
    window_sec: float
    std_thr: float
    speed_thr: float
    min_stable_sec: float
    dropout_tolerance: float = 0.3
    history: Deque = field(default_factory=deque)
    stable_since: Optional[float] = None
    unstable_since: Optional[float] = None
    missing_since: Optional[float] = None

    def update(self, value, t_now: float):

        """
        value: float (deg) for 'angle' or np.ndarray shape [K,3] in meters for 'position'
        t_now: seconds (time.time())
        """

        # Detect missing input for this frame
        if self.mode == 'angle':
            is_missing = not (isinstance(value, (int, float, np.floating)) and np.isfinite(value))
        else:  # position
            v = np.asarray(value)
            is_missing = (v.ndim != 2) or (v.shape[1] != 3) or (not np.all(np.isfinite(v)))

        # Always prune old samples in history
        while self.history and (t_now - self.history[0][0]) > self.window_sec:
            self.history.popleft()

        if is_missing:
            # Do NOT append missing to history; it would poison the window
            if self.missing_since is None:
                self.missing_since = t_now

            # Only reset the stable timer if missing lasts longer than tolerance
            if self.stable_since is not None and (t_now - self.missing_since) > self.dropout_tolerance:
                self.stable_since = None

            # Report unknown metrics during missing frames
            stable_dur = 0.0 if self.stable_since is None else (t_now - self.stable_since)
            return False, stable_dur, {"std": float('nan'), "speed": float('nan')}

        # If we got here, we have a valid sample; clear missing timer and append
        self.missing_since = None        
        self.history.append((t_now, value))

        # prune old
        #while self.history and (t_now - self.history[0][0]) > self.window_sec:
        #    self.history.popleft()

        stable, metrics = self._evaluate()

        if stable:
            self.unstable_since = None
            if self.stable_since is None:
                # anchor stability timer at oldest element in current window
                self.stable_since = self.history[0][0]
        else:
            if self.unstable_since is None:
                self.unstable_since = t_now
            # reset only if instability lasts longer than tolerance
            if self.stable_since is not None and (t_now - self.unstable_since) > self.dropout_tolerance:
                self.stable_since = None

        stable_dur = 0.0 if self.stable_since is None else (t_now - self.stable_since)
        ready = stable and (self.stable_since is not None) and (stable_dur >= self.min_stable_sec)
        return ready, stable_dur, metrics

    def _evaluate(self):
        if len(self.history) < 2:
            return False, {"std": float('inf'), "speed": float('inf')}

        times = np.array([t for t, _ in self.history], dtype=float)

        if self.mode == 'angle':
            vals = np.array([v for _, v in self.history], dtype=float)  # deg
            times = np.array([t for t, _ in self.history], dtype=float)

            valid = np.isfinite(vals)
            if np.sum(valid) < 2:
                return False, {"std": float('nan'), "speed": float('nan')}

            vals_v = vals[valid]
            times_v = times[valid]

            std = float(np.std(vals_v))  # all finite
            dvals = np.diff(vals_v)
            dt = np.diff(times_v)
            speed = float(np.median(np.abs(dvals) / np.maximum(dt, 1e-6)))  # deg/s

            ok = (std <= self.std_thr) and (speed <= self.speed_thr)
            return ok, {"std": std, "speed": speed}

        # position mode
        arr = np.stack([v for _, v in self.history], axis=0)   # [T,K,3] m
        times = np.array([t for t, _ in self.history], dtype=float)

        # Keep only timesteps where all coords are finite
        mask_t = np.all(np.isfinite(arr), axis=(1, 2))
        if np.sum(mask_t) < 2:
            return False, {"std": float('nan'), "speed": float('nan')}

        arr_v = arr[mask_t]          # [Tv,K,3]
        times_v = times[mask_t]      # [Tv]

        # Positional std across the window (average per-joint magnitude), in mm
        stds = np.std(arr_v, axis=0)  # [K,3] m
        std_mm = float(np.mean(np.linalg.norm(stds, axis=-1)) * 1000.0)

        # Positional speed: median of per-step joint speeds, in mm/s
        diffs = np.diff(arr_v, axis=0)                         # [Tv-1,K,3] m
        dt = np.diff(times_v)[:, None, None]                   # [Tv-1,1,1] s
        vels = np.linalg.norm(diffs / np.maximum(dt, 1e-6), axis=-1)  # [Tv-1,K] m/s
        vel_mm_s = float(np.median(vels) * 1000.0)

        ok = (std_mm <= self.std_thr) and (vel_mm_s <= self.speed_thr)
        return ok, {"std": std_mm, "speed": vel_mm_s}

class SessionState:
    def __init__(self, args):
        self.angle_tracker = MotionStability(
            mode='angle',
            window_sec=args.stable_window,
            std_thr=args.angle_std_thr,
            speed_thr=args.angle_speed_thr,
            min_stable_sec=args.min_stable_sec,
            dropout_tolerance=args.dropout_tolerance
        )
        self.pos_tracker = MotionStability(
            mode='position',
            window_sec=args.stable_window,
            std_thr=args.pos_std_mm,
            speed_thr=args.pos_speed_mm,
            min_stable_sec=args.min_stable_sec,
            dropout_tolerance=args.dropout_tolerance
        )
        self.rom_started = False
        self.rom_baseline = None
        self.last_angle = None
        self.last_key = -1
        self.last_metrics_a = {}
        self.last_metrics_p = {}
        self.delta_ema = None
        self.angle_series = deque()

# =========================
# Core Processing Function
# =========================

def process_one_image(args,
                      color_img,
                      depth_img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      state=None):
    """Visualize predicted keypoints (and heatmaps) of one image + stability gate."""

    #################### prediction

    # predict bbox
    det_result = inference_detector(detector, color_img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, color_img, bboxes)
    data_samples = merge_data_samples(pose_results)

    #################### 2D visualization

    if isinstance(color_img, str):
        color_img = mmcv.imread(color_img, channel_order='rgb')
    elif isinstance(color_img, np.ndarray):
        color_img = mmcv.bgr2rgb(color_img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            color_img,
            depth_img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            show_kpt_subset=args.show_kpt_subset,
            skeleton_style=args.skeleton_style,
            show=False,  # show in our own cv2 window below
            wait_time=0, # wait handled below
            kpt_thr=args.kpt_thr)

    # Prepare frame for cv2 overlay
    frame_rgb = visualizer.get_image()
    frame_bgr = mmcv.rgb2bgr(frame_rgb)

    #################### save predictions for debugging

    if args.save_predictions and data_samples.get('pred_instances', None) is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # add ms to avoid collision
        os.makedirs("snapshots", exist_ok=True)

        # Save RGB image (already BGR here)
        cv2.imwrite(f"snapshots/image_{timestamp}.png", frame_bgr)

        # Save Depth image (raw uint16)
        if hasattr(depth_img, "get_data"):
            depth_array = np.asanyarray(depth_img.get_data())  # uint16
            cv2.imwrite(f"snapshots/depth_{timestamp}.png", depth_array)

            # Save intrinsics
            try:
                depth_intrin = depth_img.profile.as_video_stream_profile().get_intrinsics()
                intrin_dict = {
                    "width": depth_intrin.width,
                    "height": depth_intrin.height,
                    "fx": depth_intrin.fx,
                    "fy": depth_intrin.fy,
                    "ppx": depth_intrin.ppx,
                    "ppy": depth_intrin.ppy,
                    "model": str(depth_intrin.model),
                    "coeffs": depth_intrin.coeffs
                }
                with open(f"snapshots/intrinsics_{timestamp}.json", "w") as f:
                    json.dump(intrin_dict, f, indent=2)
            except Exception as e:
                print(f"[!] Failed to save intrinsics: {e}")
        else:
            # non-RealSense input
            pass

        # Save prediction JSON
        pred_dict = {}
        pred = data_samples.pred_instances
        for k, v in pred.items():
            if hasattr(v, "cpu"):
                pred_dict[k] = v.cpu().numpy().tolist()
            elif isinstance(v, np.ndarray):
                pred_dict[k] = v.tolist()
            else:
                pred_dict[k] = v  # already JSON-safe

        # Save transformed keypoints if available
        if hasattr(pred, "transformed_keypoints"):
            pred_dict["transformed_keypoints"] = pred.transformed_keypoints.cpu().numpy().tolist()

        # Save visualizer skeleton
        if hasattr(visualizer, "skeleton"):
            pred_dict["skeleton"] = visualizer.skeleton
        else:
            pred_dict["skeleton"] = []

        # Save final JSON
        with open(f"snapshots/pred_{timestamp}.json", "w") as f:
            json.dump(pred_dict, f, indent=2)

        print(f"[✔] Frame saved: snapshots/image_{timestamp}.png")

    #################### 3D computation + STABILITY HUD

    hud_lines = []
    t_now = time.time()
    pred = data_samples.get('pred_instances', None)

    # Persistent HUD after ROM has started (always draw something)
    if state is not None and state.rom_started:
        if state.rom_baseline is not None:
            hud_lines.append(f"ROM STARTED  (baseline {state.rom_baseline:.1f} deg)")
        else:
            hud_lines.append("ROM STARTED")
        
        # Show the last known angle even if this frame misses detections
        #if getattr(state, "last_angle", None) is not None:
        #    delta = state.last_angle - (state.rom_baseline or 0.0)
        #    hud_lines.append(f"Angle: {state.last_angle:.1f} deg   Delta: {delta:+.1f} deg")

        if getattr(state, "last_angle", None) is not None and state.rom_baseline is not None:
            raw_delta = state.last_angle - state.rom_baseline

            # EMA smoothing for display (optional)
            if args.delta_ema_alpha > 0.0:
                if state.delta_ema is None or not np.isfinite(state.delta_ema):
                    state.delta_ema = float(raw_delta)
                else:
                    a = float(args.delta_ema_alpha)
                    state.delta_ema = a * float(raw_delta) + (1.0 - a) * state.delta_ema
                disp_delta = state.delta_ema
            else:
                disp_delta = raw_delta

            hud_lines.append(f"Angle: {state.last_angle:.1f} deg   d={disp_delta:+.1f} deg")

    if (depth_img is not None) and (pred is not None):
        intrin_dict = extract_intrinsics_from_depth(depth_img)
        if intrin_dict is not None:
            # Prefer transformed_keypoints if present
            keypoints = getattr(pred, 'transformed_keypoints', None)
            if keypoints is None:
                keypoints = pred.keypoints
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()  # [N,J,2]

            visibility = getattr(pred, 'keypoints_visible', None)
            if visibility is None:
                visibility = np.ones(keypoints.shape[:2])
            elif hasattr(visibility, 'cpu'):
                visibility = visibility.cpu().numpy()

            if keypoints.shape[0] > 0:
                # 3D joints for person 0
                joints_xyz = compute_3d_skeletons(keypoints, visibility, depth_img, intrin_dict, args.kpt_thr)
                joints_xyz = np.array(joints_xyz)  # [N,J,3] meters

                # -------------- Angle for the ROM test --------------
                angle_t = np.nan
                try:
                    kpt_ids = rom_test[args.rom_test]  # e.g., [shoulder, elbow, wrist]
                    angle_t = angle(joints_xyz[0, kpt_ids[0]],
                                    joints_xyz[0, kpt_ids[1]],
                                    joints_xyz[0, kpt_ids[2]])
                except Exception as e:
                    hud_lines.append(f"Angle err: {e}")

                if state is not None and np.isfinite(angle_t):
                    state.last_angle = float(angle_t)

                # Add to rolling series for sparkline and prune by time
                if args.plot_seconds > 0 and np.isfinite(angle_t):
                    state.angle_series.append((t_now, float(angle_t)))
                    cutoff = t_now - float(args.plot_seconds)
                    while state.angle_series and state.angle_series[0][0] < cutoff:
                        state.angle_series.popleft()

                # -------------- STABILITY --------------
                if state is not None and not state.rom_started:
                    ready_a = duration_a = None
                    metrics_a = {}
                    ready_p = duration_p = None
                    metrics_p = {}

                    if args.stability_mode in ('angle', 'both') and np.isfinite(angle_t):
                        ready_a, duration_a, metrics_a = state.angle_tracker.update(float(angle_t), t_now)

                    if args.stability_mode in ('position', 'both'):
                        # track the same ROM joints
                        tracked = joints_xyz[0, kpt_ids, :]  # [3,3] or [K,3]
                        ready_p, duration_p, metrics_p = state.pos_tracker.update(tracked, t_now)

                    if args.stability_mode == 'angle':
                        ready = bool(ready_a)
                        duration = duration_a or 0.0
                    elif args.stability_mode == 'position':
                        ready = bool(ready_p)
                        duration = duration_p or 0.0
                    else:  # both
                        ready = bool(ready_a) and bool(ready_p)
                        duration = min(duration_a or 0.0, duration_p or 0.0)

                    # Cache last good metrics (steady HUD across brief dropouts)
                    if metrics_a and (np.isfinite(metrics_a.get('std', np.nan)) or np.isfinite(metrics_a.get('speed', np.nan))):
                        state.last_metrics_a = metrics_a
                    if metrics_p and (np.isfinite(metrics_p.get('std', np.nan)) or np.isfinite(metrics_p.get('speed', np.nan))):
                        state.last_metrics_p = metrics_p

                    hud_lines.append(f"Stable? {'YES' if ready else 'NO'} [{duration:.2f}s / {args.min_stable_sec:.2f}s]")
                    if args.stability_mode in ('angle', 'both'):
                        src = metrics_a if metrics_a else state.last_metrics_a
                        a_std = float((src or {}).get('std', float('nan')))
                        a_spd = float((src or {}).get('speed', float('nan')))
                        hud_lines.append(f"Angle std={a_std:.1f} deg, speed={a_spd:.1f} deg/s")

                    if args.stability_mode in ('position', 'both'):
                        src = metrics_p if metrics_p else state.last_metrics_p
                        p_std = float((src or {}).get('std', float('nan')))
                        p_spd = float((src or {}).get('speed', float('nan')))
                        hud_lines.append(f"Pos std={p_std:.1f} mm, speed={p_spd:.1f} mm/s")

                    if ready and not state.rom_started:
                        state.rom_started = True
                        # Robust baseline = median angle over the current stable window (angle tracker history)
                        vals = np.array([v for (_, v) in state.angle_tracker.history], dtype=float) if state.angle_tracker else np.array([])
                        vals = vals[np.isfinite(vals)]
                        if vals.size >= 1:
                            state.rom_baseline = float(np.median(vals))
                        elif np.isfinite(angle_t):
                            # Fallback if we somehow don't have angle history but do have an angle
                            state.rom_baseline = float(angle_t)
                        else:
                            state.rom_baseline = None

                    #elif state is not None and state.rom_started:
                    #    # Keep showing a compact HUD after gating
                    #    hud_lines.append(f"ROM STARTED  (baseline {state.rom_baseline:.1f}°)")
                    #    if np.isfinite(angle_t):
                    #        hud_lines.append(f"Angle: {angle_t:.1f}°   Δ: {angle_t - state.rom_baseline:+.1f}°")

                # Optional: 3D visualization
                if getattr(args, "show3d", False):
                    try:
                        skeleton = visualizer.skeleton if hasattr(visualizer, "skeleton") else None
                        visualize_3d_skeletons(
                            joints_xyz[0], skeleton, args.show_kpt_subset, args.kpt_thr
                        )
                    except Exception as e:
                        hud_lines.append(f"3D viz err: {e}")

    # Draw HUD (top-left)
    if args.show:
        y = 22
        for line in hud_lines:
            cv2.putText(frame_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 0), 2, cv2.LINE_AA)
            y += 22

        # ----- Sparkline: angle over last X seconds -----
        if args.plot_seconds > 0 and len(state.angle_series) >= 2:
            # Filter out any non-finite samples first
            series = [(t, a) for (t, a) in state.angle_series if np.isfinite(a)]
            if len(series) >= 2:
                H, W = frame_bgr.shape[:2]
                pw, ph, pm = int(args.plot_width), int(args.plot_height), int(args.plot_margin)
                x0 = max(0, W - pw - pm)
                y0 = pm
                x1 = min(W - 1, x0 + pw)
                y1 = min(H - 1, y0 + ph)

                # time window
                t0 = t_now - float(args.plot_seconds)  # reuse the frame timestamp you already have
                ts = [max(t0, t) for (t, _) in series]
                ys = [a for (_, a) in series]

                # y scale (pad if flat)
                y_min = min(ys); y_max = max(ys)
                if not np.isfinite(y_min) or not np.isfinite(y_max) or abs(y_max - y_min) < 1e-6:
                    y_min, y_max = (0.0, 1.0) if not np.isfinite(y_min) or not np.isfinite(y_max) else (y_min - 1.0, y_max + 1.0)

                def map_pt(t, y):
                    # time maps left->right
                    tx = (t - t0) / max(args.plot_seconds, 1e-6)
                    tx = 0.0 if not np.isfinite(tx) else min(max(tx, 0.0), 1.0)
                    px = int(round(x0 + tx * (x1 - x0)))
                    # angle maps bottom->top
                    ty = (y - y_min) / max((y_max - y_min), 1e-6)
                    ty = 0.0 if not np.isfinite(ty) else min(max(ty, 0.0), 1.0)
                    py = int(round(y1 - ty * (y1 - y0)))
                    return px, py

                # background + border
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (32, 32, 32), thickness=-1)
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (180, 180, 180), thickness=1)

                # polyline
                pts = [map_pt(t, y) for (t, y) in zip(ts, ys)]
                cv2.polylines(frame_bgr, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                # labels
                if getattr(state, "last_angle", None) is not None and np.isfinite(state.last_angle):
                    cv2.putText(frame_bgr, f"{state.last_angle:.1f} deg", (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame_bgr, f"[{y_min:.1f}, {y_max:.1f}]", (x0 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
        # ----- end sparkline -----

        cv2.imshow("Pose", frame_bgr)
        key = cv2.pollKey() if hasattr(cv2, "pollKey") else cv2.waitKey(1)
        handle_hotkeys_for_presets(args, key)
        if args.show_interval > 0:
            time.sleep(args.show_interval)
    else:
        key = -1  # headless mode

    return data_samples.get('pred_instances', None), key


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth = param.get_distance(x, y)
        print(f"[mouse_callback] Mouse = ({x}, {y}) → Distance: {depth:.3f} meters")


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=3,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    #################### my extra arguments
    parser.add_argument('--show_kpt_subset', default="right_elbow_rom", type=str)
    parser.add_argument('--rom_test', default="right_elbow_flexion", type=str)
    parser.add_argument('--show3d', action='store_true')

    # Stability gating knobs
    parser.add_argument('--stability-mode', default='angle', choices=['angle', 'position', 'both'],
                        help='What to monitor before starting ROM')
    
    parser.add_argument('--stable-window', type=float, default=2.0,
                        help='Seconds of history used to test stability')
    
    parser.add_argument('--min-stable-sec', type=float, default=3.0,
                        help='How long the signal must stay stable before ROM starts')
    
    parser.add_argument('--angle-std-thr', type=float, default=2.0,
                        help='Max angular std dev (deg) inside window')
    
    parser.add_argument('--angle-speed-thr', type=float, default=5.0,
                        help='Max median angular speed (deg/s) inside window')

    parser.add_argument('--pos-std-mm', type=float, default=5.0,
                        help='Max positional std dev (mm) inside window')
    
    parser.add_argument('--pos-speed-mm', type=float, default=10.0,
                        help='Max median positional speed (mm/s) inside window')
    
    parser.add_argument('--dropout-tolerance', type=float, default=0.3,
                        help='Seconds of allowed instability before timer resets')

    parser.add_argument('--delta-ema-alpha', type=float, default=0.3,
                    help='EMA smoothing for delta after ROM starts (0 disables)')

    parser.add_argument('--plot-seconds', type=float, default=5.0,
                    help='Show a rolling sparkline of the angle over the last X seconds (0 to disable)')
    parser.add_argument('--plot-width', type=int, default=240,
                        help='Sparkline width in pixels')
    parser.add_argument('--plot-height', type=int, default=80,
                        help='Sparkline height in pixels')
    parser.add_argument('--plot-margin', type=int, default=10,
                        help='Margin from edges for the sparkline rectangle')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    #################### parse args
    args = parser.parse_args()
    # args.show_kpt_subset = presets[args.show_kpt_subset]
    args.kpt_preset_name = args.show_kpt_subset
    set_kpt_preset(args, args.kpt_preset_name)

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    # session state (stability gate)
    state = SessionState(args)

    # determine input type
    if args.input == 'webcam':
        input_type = 'webcam'
    elif args.input == 'realsense':
        input_type = 'realsense'
    else:
        mt = mimetypes.guess_type(args.input)[0]
        input_type = mt.split('/')[0] if mt else 'video'

    if input_type == 'image':

        # inference
        pred_instances, key = process_one_image(
            args, args.input, None, detector, pose_estimator, visualizer, 0, state)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type == 'realsense':

        if rs is None:
            raise ImportError("pyrealsense2 is required for 'realsense' input mode.")

        # Setup pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline_profile = pipeline.start(config)

        # Get depth scale (unused but available)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # in meters per unit

        # Create align object (align depth to color)
        align = rs.align(rs.stream.color)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = depth_frame

                # run pose estimator
                pred_instances, key = process_one_image(
                    args, color_image, depth_image, detector, pose_estimator, visualizer, 0.001, state)

                if key == 27:  # ESC
                    break

        finally:
            pipeline.stop()

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # topdown pose estimation
            pred_instances, key = process_one_image(
                args, frame, None, detector, pose_estimator, visualizer, 0.001, state)

            if key == 27:
                break

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

        if video_writer:
            video_writer.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type_print = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type_print} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()