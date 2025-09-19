# topdown_demo_with_mmdet.py
# End-to-end demo with:
# - Detector (mmdet) -> per-person bboxes
# - Top-down pose estimation (mmpose)
# - ROM auto-baseline (auto-zero) + auto-ROM min/max pairing
# - HUD as separate strips (text + optional sparkline) under the video
# - Voice control for presets / start / zero / next / prev
#
# Layout:
#   [ video frame ]
#   [ text strip: Angle/ROM + Test | Zeroed | Armed (+ auto-zero status) ]
#   [ sparkline (if --plot-seconds > 0) ]
#
# Notes:
# - DEBUG + full_body: show four instantaneous angles in the HUD strip every frame:
#     left/right elbow, left/right shoulder (no baseline subtraction).
# - VideoWriter is enabled when --output-root is provided. It writes the fully
#   composited HUD stream (combined_bgr) to an MP4.

import os
import time
import logging
import mimetypes
import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

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

from utils_variables import rom_test
from utils import set_kpt_preset, handle_hotkeys_for_presets

# Voice utilities
from utils_voice import (
    VoiceController,
    apply_voice_command,
    add_voice_args,
    init_voice_from_args,
)

# ROM mechanics
from utils_rom import (
    SessionState,
    reset_auto_rom_state,
    maybe_start_trial,
    confirm_hold_and_lock_extremum,
    attach_locked_frame_if_pending,  # expected to be present in your utils_rom
)

# 3D utilities
from utils_3d import extract_intrinsics_from_depth, compute_3d_skeletons, angle

# Visualization HUD
from utils_visualization import render_sparkline_strip, render_text_strip

# Finalization (persistence + panel)
from utils_results import finalize_rom_trial

try:
    import pyrealsense2 as rs
except (ImportError, ModuleNotFoundError):
    rs = None


# =========================
# Helpers
# =========================

def _rel_angle_deg(state, raw_angle_deg):
    """Return angle relative to baseline if available; else raw angle."""
    if raw_angle_deg is None or not np.isfinite(raw_angle_deg):
        return raw_angle_deg
    if state.baseline_deg is None:
        return float(raw_angle_deg)
    return float(raw_angle_deg - state.baseline_deg)


def _compute_instant_angle(ids, pred, depth_img, intrin_dict, kpt_thr):
    """
    Compute instantaneous angle (degrees) for joint triplet `ids`.
    Prefers 3D (if RealSense intrinsics available) else 2D.
    Returns np.nan if not computable.
    """
    try:
        if pred is None or ids is None or len(ids) != 3:
            return np.nan

        keypoints = getattr(pred, "transformed_keypoints", None) or pred.keypoints
        if hasattr(keypoints, "cpu"):
            keypoints = keypoints.cpu().numpy()
        visibility = getattr(pred, "keypoints_visible", None)
        visibility = (
            np.ones(keypoints.shape[:2])
            if visibility is None
            else (visibility.cpu().numpy() if hasattr(visibility, "cpu") else visibility)
        )
        if keypoints.shape[0] <= 0:
            return np.nan

        a, b, c = ids

        # Try 3D first
        if (depth_img is not None) and (intrin_dict is not None):
            joints_xyz_list = compute_3d_skeletons(keypoints, visibility, depth_img, intrin_dict, kpt_thr)
            joints_xyz = np.array(joints_xyz_list)  # [N, J, 3]
            if joints_xyz.shape[0] > 0:
                return float(angle(joints_xyz[0, a], joints_xyz[0, b], joints_xyz[0, c]))

        # Fallback: 2D
        k = keypoints[0]
        v = visibility[0]
        if np.all(v[[a, b, c]] >= kpt_thr):
            ax, ay = k[a][:2]
            bx, by = k[b][:2]
            cx, cy = k[c][:2]
            return float(angle([ax, ay], [bx, by], [cx, cy]))
    except Exception:
        pass
    return np.nan


# =========================
# Main per-frame processing
# =========================

def process_one_image(
    args,
    color_img,
    depth_img,
    detector,
    pose_estimator,
    visualizer=None,
    show_interval=0,
    state: SessionState = None,
    voice: VoiceController = None,
):
    """Run one frame: det -> pose -> angle/ROM -> HUD -> finalize -> voice/hotkeys."""

    # 1) Detector (ensure mmdet scope)
    init_default_scope('mmdet')
    det_result = inference_detector(detector, color_img)
    init_default_scope('mmpose')

    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[
        np.logical_and(pred_instance.labels == args.det_cat_id, pred_instance.scores > args.bbox_thr)
    ]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # 2) Pose
    pose_results = inference_topdown(pose_estimator, color_img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # Visualizer expects RGB image
    if isinstance(color_img, str):
        color_img_rgb = mmcv.imread(color_img, channel_order="rgb")
    elif isinstance(color_img, np.ndarray):
        color_img_rgb = mmcv.bgr2rgb(color_img)
    else:
        raise TypeError("Unsupported color_img type.")

    # Initialize / react to preset changes
    if state is not None:
        if state.active_rom is None:
            state.active_rom = args.rom_test
            state.baseline_deg = None
            state.baseline_set_ts = None
        elif args.rom_test != state.active_rom:
            # finalize ongoing trial first (if any) — but NEVER in --debug
            if (not args.debug) and args.auto_rom and state.trial_active:
                finalize_rom_trial(args, state, time.time(), mmcv.rgb2bgr(color_img_rgb.copy()), combined_bgr=None)

            state.active_rom = args.rom_test
            state.last_angle = None
            state.angle_series.clear()

            # Auto-zero arming only when zeroing is enabled
            if args.zero:
                state.auto_zero_pending = True
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                state.baseline_deg = None
                state.baseline_set_ts = None
                state.first_auto_arm_consumed = False
                state.last_zero_source = 'auto'
            else:
                state.auto_zero_pending = False
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                state.baseline_deg = None
                state.baseline_set_ts = None

    # Draw pose (no HUD text here — keep video clean)
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
    frame_bgr = mmcv.rgb2bgr(frame_rgb)
    t_now = time.time()
    pred = data_samples.get("pred_instances", None)

    # 3) Angle calc for active ROM (prefer 3D; else 2D)
    angle_t = np.nan
    ids = rom_test.get(args.rom_test, [])
    is_angle_mode = isinstance(ids, (list, tuple)) and len(ids) == 3

    intrin_dict = None
    if depth_img is not None:
        try:
            intrin_dict = extract_intrinsics_from_depth(depth_img)
        except Exception:
            intrin_dict = None

    if pred is not None and is_angle_mode:
        # Try 3D
        if intrin_dict is not None:
            try:
                keypoints = getattr(pred, "transformed_keypoints", None) or pred.keypoints
                if hasattr(keypoints, "cpu"):
                    keypoints = keypoints.cpu().numpy()
                visibility = getattr(pred, "keypoints_visible", None)
                visibility = (
                    np.ones(keypoints.shape[:2])
                    if visibility is None
                    else (visibility.cpu().numpy() if hasattr(visibility, "cpu") else visibility)
                )
                if keypoints.shape[0] > 0:
                    a, b, c = ids
                    joints_xyz = compute_3d_skeletons(keypoints, visibility, depth_img, intrin_dict, args.kpt_thr)
                    joints_xyz = np.array(joints_xyz)
                    angle_t = angle(joints_xyz[0, a], joints_xyz[0, b], joints_xyz[0, c])
            except Exception:
                pass

        # Fallback 2D
        if (not np.isfinite(angle_t)) and pred is not None:
            try:
                keypoints = getattr(pred, "transformed_keypoints", None) or pred.keypoints
                if hasattr(keypoints, "cpu"):
                    keypoints = keypoints.cpu().numpy()
                visibility = getattr(pred, "keypoints_visible", None)
                visibility = (
                    np.ones(keypoints.shape[:2])
                    if visibility is None
                    else (visibility.cpu().numpy() if hasattr(visibility, "cpu") else visibility)
                )
                if keypoints.shape[0] > 0:
                    a, b, c = ids
                    k = keypoints[0]
                    v = visibility[0]
                    if np.all(v[[a, b, c]] >= args.kpt_thr):
                        ax, ay = k[a][:2]
                        bx, by = k[b][:2]
                        cx, cy = k[c][:2]
                        angle_t = angle([ax, ay], [bx, by], [cx, cy])
            except Exception:
                pass

    # Store raw angle for zeroing
    state.current_raw_angle = angle_t if np.isfinite(angle_t) else np.nan

    # 4) Auto-zero collector / status
    autozero_line = None
    if is_angle_mode and args.zero and state.auto_zero_pending:
        if state.auto_zero_start_time is None:
            if np.isfinite(state.current_raw_angle):
                state.auto_zero_start_time = t_now
                state.auto_zero_buffer = [float(state.current_raw_angle)]
                autozero_line = ("Auto-zero: capturing...", (0, 200, 255))
            else:
                autozero_line = ("Auto-zero: waiting for stable angle...", (0, 200, 255))
        else:
            if np.isfinite(state.current_raw_angle):
                state.auto_zero_buffer.append(float(state.current_raw_angle))
            autozero_line = ("Auto-zero: capturing...", (0, 200, 255))

            if (t_now - state.auto_zero_start_time) >= state.auto_zero_window_sec:
                buf = np.array(state.auto_zero_buffer, dtype=float)
                buf = buf[np.isfinite(buf)]
                if buf.size >= state.auto_zero_min_samples:
                    std_ok = (np.nanstd(buf) <= state.auto_zero_max_std_deg)
                    if std_ok:
                        baseline = float(np.nanmedian(buf))
                        state.baseline_deg = baseline
                        state.baseline_set_ts = t_now
                        state.auto_zero_pending = False
                        state.auto_zero_start_time = None
                        state.auto_zero_buffer.clear()
                        state.angle_series.clear()
                        state.last_angle = None
                        reset_auto_rom_state(state, keep_trial_ready=False)

                        # Arming rules after zero
                        if state.arm_after_baseline:
                            state.trial_armed = True
                            state.arm_after_baseline = False
                            state.first_auto_arm_consumed = True
                        elif (state.last_zero_source == "auto") and (not state.first_auto_arm_consumed):
                            state.trial_armed = True
                            state.first_auto_arm_consumed = True
                        else:
                            state.trial_armed = False

                        print(f"[AUTO-ZERO] Baseline locked at {baseline:.2f}° for {args.rom_test}")
                        autozero_line = None  # done
                    else:
                        state.auto_zero_start_time = t_now
                        state.auto_zero_buffer = []
                        print("[AUTO-ZERO] Too jittery; extending capture window.")
                else:
                    state.auto_zero_start_time = t_now
                    state.auto_zero_buffer = []
                    print("[AUTO-ZERO] Not enough valid samples; extending capture window.")

    # Angle after baseline subtraction (if any)
    angle_disp = _rel_angle_deg(state, angle_t)
    angle_disp = abs(angle_disp) if (args.abs_angle and np.isfinite(angle_disp)) else angle_disp

    # 5) Auto-ROM core (disabled in debug)
    if (not args.debug) and args.auto_rom and is_angle_mode and np.isfinite(angle_disp):
        if state.filt_angle is None:
            state.filt_angle = float(angle_disp)
            state.prev_filt_angle = float(angle_disp)
            state.prev_ts = t_now
        else:
            alpha = float(args.rom_ema_alpha)
            state.filt_angle = alpha * float(angle_disp) + (1.0 - alpha) * state.filt_angle

            dt = max(t_now - (state.prev_ts or t_now), 1e-6)
            state.vel_dps = (state.filt_angle - (state.prev_filt_angle if state.prev_filt_angle is not None else state.filt_angle)) / dt

            # Hysteresis counters
            if abs(state.vel_dps) > args.rom_v_go:
                state.moving_counter += 1
            else:
                state.moving_counter = 0

            if abs(state.vel_dps) < args.rom_v_stop:
                state.stopped_counter += 1
                if state.stopped_since_ts is None:
                    state.stopped_since_ts = t_now
            else:
                state.stopped_counter = 0
                state.stopped_since_ts = None

            # Movement sign
            if state.vel_dps > args.rom_v_stop:
                state.last_move_sign = +1
            elif state.vel_dps < -args.rom_v_stop:
                state.last_move_sign = -1
            elif abs(state.vel_dps) > 1.0 and state.last_move_sign == 0:
                state.last_move_sign = 1 if state.vel_dps > 0 else -1

            # Hold window when near-stopped
            if abs(state.vel_dps) < args.rom_v_stop:
                state.hold_window.append((t_now, state.filt_angle))
                while state.hold_window and (t_now - state.hold_window[0][0] > args.rom_hold_sec + 0.2):
                    state.hold_window.popleft()
            else:
                state.hold_window.clear()

            # Clear latch when fresh motion arrives
            if state.after_lock_motion_needed and abs(state.vel_dps) > args.rom_v_go:
                state.after_lock_motion_needed = False

            # Gate trial start and lock extrema
            maybe_start_trial(args, state, t_now)
            if state.trial_active:
                confirm_hold_and_lock_extremum(args, state, t_now, frame_bgr)

            # Update history
            state.prev_filt_angle = state.filt_angle
            state.prev_ts = t_now

    # 6) Update series for sparkline
    if state is not None and args.plot_seconds > 0 and is_angle_mode and np.isfinite(angle_disp):
        state.last_angle = float(angle_disp)
        state.angle_series.append((t_now, float(angle_disp)))
        cutoff = t_now - float(args.plot_seconds)
        while state.angle_series and state.angle_series[0][0] < cutoff:
            state.angle_series.popleft()

    # 7) Build text strip (HUD)
    H, W = frame_bgr.shape[:2]

    minimal_debug = (args.debug and args.rom_test == "full_body")

    lines = []

    if minimal_debug:
        # show ONLY the 4 instantaneous angles (no standard lines, no auto-zero line)
        try:
            le_ids = rom_test.get("left_elbow_flexion")
            re_ids = rom_test.get("right_elbow_flexion")
            ls_ids = rom_test.get("left_shoulder_abduction")
            rs_ids = rom_test.get("right_shoulder_abduction")

            le = _compute_instant_angle(le_ids, pred, depth_img, intrin_dict, args.kpt_thr)
            re = _compute_instant_angle(re_ids, pred, depth_img, intrin_dict, args.kpt_thr)
            ls = _compute_instant_angle(ls_ids, pred, depth_img, intrin_dict, args.kpt_thr)
            rs = _compute_instant_angle(rs_ids, pred, depth_img, intrin_dict, args.kpt_thr)

            fmt = lambda x: f"{x:.1f}" if np.isfinite(x) else "--.-"
            lines.append((f"left elbow: {fmt(le)}", (255, 255, 255)))
            lines.append((f"right elbow: {fmt(re)}", (255, 255, 255)))
            lines.append((f"left shoulder: {fmt(ls)}", (255, 255, 255)))
            lines.append((f"right shoulder: {fmt(rs)}", (255, 255, 255)))
        except Exception:
            lines.extend([
                ("left elbow: --.-", (255, 255, 255)),
                ("right elbow: --.-", (255, 255, 255)),
                ("left shoulder: --.-", (255, 255, 255)),
                ("right shoulder: --.-", (255, 255, 255)),
            ])
    else:
        # original two standard lines (+ auto-zero status if present)
        if autozero_line is not None:
            lines.append(autozero_line)

        if is_angle_mode:
            if np.isfinite(angle_disp):
                lines.append((f"Angle: {angle_disp:.1f} deg", (0, 240, 0)))
            else:
                lines.append(("Angle: --.- deg", (0, 240, 240)))
        else:
            lines.append((f"ROM: {args.rom_test} (no angle)", (0, 240, 240)))

        arm_text = "Active" if state.trial_active else ("Armed" if state.trial_armed else "Disarmed")
        zero_text = "Zeroed" if state.baseline_deg is not None else "Unzeroed"
        lines.append((f"Test: {args.rom_test}   |   {zero_text}   |   {arm_text}", (200, 200, 200)))

    text_strip = render_text_strip(
        width=W,
        height=int(args.text_strip_height),
        lines=lines,
        bg_mode=str(args.text_strip_bg),
    )

    # 8) Optional sparkline strip
    sparkline = (
        render_sparkline_strip(
            width=W,
            height=int(args.plot_height),
            series=list(state.angle_series) if (state is not None and is_angle_mode) else [],
            t_now=t_now,
            window_sec=float(args.plot_seconds),
            label_text="Angle history (s)",
            gap_sec=float(args.plot_gap_sec),
        )
        if args.plot_seconds > 0
        else None
    )

    # 9) Compose final frame: video + text + (optional) sparkline
    rows = [frame_bgr, text_strip]
    if sparkline is not None:
        rows.append(sparkline)
    combined_bgr = np.vstack(rows)

    # Attach HUD frame to the just-locked extremum, if any
    if not args.debug:
        attach_locked_frame_if_pending(state, combined_bgr)

    # 10) Voice + hotkeys
    if voice:
        while True:
            cmd = voice.poll()
            if cmd is None:
                break
            apply_voice_command(args, state, cmd, reset_auto_rom_state)

    if args.show:
        cv2.imshow("Pose", combined_bgr)
        key = cv2.pollKey() if hasattr(cv2, "pollKey") else cv2.waitKey(1)
        handle_hotkeys_for_presets(args, key)

        # Manual baseline zero via 'b'
        if key == ord("b"):
            if np.isfinite(state.current_raw_angle):
                state.baseline_deg = float(state.current_raw_angle)
                state.baseline_set_ts = time.time()
                state.angle_series.clear()
                state.last_angle = None
                state.auto_zero_pending = False
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                reset_auto_rom_state(state, keep_trial_ready=False)
                state.trial_armed = False
                state.first_auto_arm_consumed = True
                state.last_zero_source = 'manual'
                print(f"[KPT] Baseline set to {state.baseline_deg:.2f}°")
            else:
                print("[KPT] Cannot zero: no valid angle this frame.")

        # Arm start via 's'
        if key == ord("s"):
            if state.baseline_deg is None:
                state.arm_after_baseline = True
                print("[ROM] Start queued. Will arm as soon as baseline locks.")
            else:
                state.trial_armed = True
                state.arm_after_baseline = False
                state.first_auto_arm_consumed = True
                print("[ROM] Armed for a single repetition. Move when ready.")

        if args.show_interval > 0:
            time.sleep(args.show_interval)
    else:
        key = -1

    # 11) Finalize (after HUD is drawn so saved frames include it)
    if (not args.debug) and args.auto_rom and is_angle_mode:
        finalize_rom_trial(args, state, t_now, frame_bgr, combined_bgr)

    return data_samples.get("pred_instances", None), key, combined_bgr


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth = param.get_distance(x, y)
        print(f"[mouse_callback] Mouse = ({x}, {y}) → Distance: {depth:.3f} m")


def main():
    parser = ArgumentParser()

    parser.add_argument("--abs-angle", action="store_true", help="Use |angle - baseline| for ROM (peaks become MAX).")

    parser.add_argument("det_config")
    parser.add_argument("det_checkpoint")
    parser.add_argument("pose_config")
    parser.add_argument("pose_checkpoint")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument(
        "--show-result",
        action="store_true",
        default=False,
        help="Pop a ROM panel window & save ROM_PANEL.png on finalize (requires --show).",
    )
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--save-predictions", action="store_true", default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--det-cat-id", type=int, default=0)
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--draw-heatmap", action="store_true", default=False)
    parser.add_argument("--show-kpt-idx", action="store_true", default=False)
    parser.add_argument("--skeleton-style", default="mmpose", type=str, choices=["mmpose", "openpose"])
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--show-interval", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--draw-bbox", action="store_true")

    parser.add_argument("--rom_test", default="full_body", type=str)
    parser.add_argument("--show3d", action="store_true")

    # Sparkline controls
    parser.add_argument("--plot-seconds", type=float, default=5.0)
    parser.add_argument("--plot-height", type=int, default=80)
    parser.add_argument("--plot-gap-sec", type=float, default=0.4)

    # Text strip controls
    parser.add_argument(
        "--text-strip-height",
        type=int,
        default=100,  # a bit higher to fit 4 extra debug lines comfortably
        help="Height (px) of the HUD text strip under the video.",
    )
    parser.add_argument(
        "--text-strip-bg",
        type=str,
        default="dark",
        choices=["dark", "light"],
        help="Background theme of the HUD text strip.",
    )

    # Auto-ROM parameters
    parser.add_argument("--rom-ema-alpha", type=float, default=0.35)
    parser.add_argument("--rom-v-go", type=float, default=6.0)
    parser.add_argument("--rom-v-stop", type=float, default=3.0)
    parser.add_argument("--rom-hold-sec", type=float, default=0.35)
    parser.add_argument("--rom-std-max", type=float, default=4.0)
    parser.add_argument("--rom-lock-refractory-sec", type=float, default=0.50)
    parser.add_argument("--rom-start-amp", type=float, default=5.0)
    parser.add_argument("--rom-min-amplitude", type=float, default=10.0)
    parser.add_argument("--rom-baseline-tol", type=float, default=3.5)
    parser.add_argument("--rom-baseline-hold-sec", type=float, default=0.35)
    parser.add_argument("--rom-timeout-sec", type=float, default=8.0)

    # Modes / flags
    parser.add_argument("--zero", action="store_true", default=True, help="Enable baseline-zeroing logic.")
    parser.add_argument("--auto-rom", action="store_true", default=True, help="Enable auto-ROM logic.")
    parser.add_argument("--debug", action="store_true", default=False, help="DEBUG mode: no persistence/finalization.")

    # Voice args
    add_voice_args(parser)

    args = parser.parse_args()

    # Init detector & pose estimator
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    # Ensure mmdet pipeline is adapted for array inputs
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_pose_estimator(args.pose_config, args.pose_checkpoint, device=args.device)

    # Set initial scope to mmpose; flip to mmdet only during detection
    init_default_scope('mmpose')

    # Init visualizer
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    # State
    state = SessionState()
    state.result_window_name = "ROM_RESULT"

    # Voice
    voice = init_voice_from_args(args)

    # ----- Output video path & writer (restored) -----
    output_file = None
    video_writer = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        if args.input and args.input.lower() != "webcam":
            base = os.path.basename(args.input)
            name, _ext = os.path.splitext(base)
            output_file = os.path.join(args.output_root, f"{name}.mp4")
        else:
            output_file = os.path.join(args.output_root, "webcam.mp4")

    def _ensure_writer(frame_bgr):
        nonlocal video_writer
        if output_file and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            H, W = frame_bgr.shape[:2]
            # Try to infer FPS; fall back to 25
            fps = 25.0
            if cap is not None:
                fps_probe = cap.get(cv2.CAP_PROP_FPS)
                if fps_probe and fps_probe > 1:
                    fps = float(fps_probe)
            elif depth_pipe is not None:
                fps = 30.0
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (W, H))

    # Input handling: file / directory / webcam / RealSense
    input_src = args.input or ""
    cap = None
    depth_pipe = None
    depth_profile = None
    align = None

    # If input is empty and RealSense is available, use RealSense
    if input_src == "" and rs is not None:
        try:
            depth_pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            depth_profile = depth_pipe.start(cfg)
            align = rs.align(rs.stream.color)
            print("[INFO] RealSense started.")
        except Exception as e:
            print(f"[WARN] RealSense not started: {e}")
            depth_pipe = None

    if input_src and os.path.exists(input_src):
        # Video file or image
        mime, _ = mimetypes.guess_type(input_src)
        if mime and mime.startswith("image/"):
            frames = [mmcv.imread(input_src)]
            is_image = True
        else:
            cap = cv2.VideoCapture(input_src)
            is_image = False
    else:
        is_image = False

    # Main loop
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
                depth = depth_frame  # keep rs.frame for intrinsics + get_distance()

                pred_instances, key, combined_bgr = process_one_image(
                    args, color, depth, detector, pose_estimator, visualizer, args.show_interval, state, voice
                )

                # Write video
                if output_file:
                    _ensure_writer(combined_bgr)
                    if video_writer is not None:
                        video_writer.write(combined_bgr)

                if args.show:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        elif cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                pred_instances, key, combined_bgr = process_one_image(
                    args, frame, None, detector, pose_estimator, visualizer, args.show_interval, state, voice
                )

                # Write video
                if output_file:
                    _ensure_writer(combined_bgr)
                    if video_writer is not None:
                        video_writer.write(combined_bgr)

                if args.show:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        elif input_src and is_image:
            # Single image path (no video writing)
            frame = frames[0]
            pred_instances, key, combined_bgr = process_one_image(
                args, frame, None, detector, pose_estimator, visualizer, args.show_interval, state, voice
            )
            if args.show:
                cv2.waitKey(0)

        else:
            # Webcam fallback
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] No input source available.")
                return
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                pred_instances, key, combined_bgr = process_one_image(
                    args, frame, None, detector, pose_estimator, visualizer, args.show_interval, state, voice
                )

                # Write video
                if output_file:
                    _ensure_writer(combined_bgr)
                    if video_writer is not None:
                        video_writer.write(combined_bgr)

                if args.show:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

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
