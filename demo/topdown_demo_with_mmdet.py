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
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
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
    attach_locked_frame_if_pending,
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
# Main per-frame processing
# =========================

def _rel_angle_deg(state, raw_angle_deg):
    """
    Returns angle relative to baseline if available; otherwise raw angle.
    Lets ROM run in --no-zero (baseline=None) without crashing.
    """
    if raw_angle_deg is None or not np.isfinite(raw_angle_deg):
        return raw_angle_deg
    if state.baseline_deg is None:
        # no baseline (either not yet locked or --no-zero)
        return float(raw_angle_deg)
    return float(raw_angle_deg - state.baseline_deg)


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
    """Run one frame: detect -> pose -> angle/ROM -> HUD -> finalize/save (if due)."""

    # 1) Detector
    det_result = inference_detector(detector, color_img)
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
                # No-zero mode: keep baseline None and ensure nothing waits for it
                state.auto_zero_pending = False
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                state.baseline_deg = None
                state.baseline_set_ts = None
                # do not arm here; arming decisions will use raw angle

    # Draw pose (no HUD text here — we keep video clean)
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

    # 3) Angle calculation (prefer 3D with depth; fallback to 2D)
    angle_t = np.nan
    ids = rom_test.get(args.rom_test, [])
    is_angle_mode = isinstance(ids, (list, tuple)) and len(ids) == 3

    if (depth_img is not None) and (pred is not None):
        intrin_dict = None
        try:
            intrin_dict = extract_intrinsics_from_depth(depth_img)
        except Exception:
            intrin_dict = None

        if intrin_dict is not None:
            keypoints = getattr(pred, "transformed_keypoints", None) or pred.keypoints
            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()
            visibility = getattr(pred, "keypoints_visible", None)
            visibility = (
                np.ones(keypoints.shape[:2])
                if visibility is None
                else (visibility.cpu().numpy() if hasattr(visibility, "cpu") else visibility)
            )
            if keypoints.shape[0] > 0 and is_angle_mode:
                joints_xyz = compute_3d_skeletons(keypoints, visibility, depth_img, intrin_dict, args.kpt_thr)
                joints_xyz = np.array(joints_xyz)
                try:
                    a, b, c = ids
                    angle_t = angle(joints_xyz[0, a], joints_xyz[0, b], joints_xyz[0, c])
                except Exception:
                    pass

    if (depth_img is None) and (pred is not None) and is_angle_mode:
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

    # ---------------------------------
    # 4) Auto-zero collector / status
    # ---------------------------------
    autozero_line = None  # status line for the text strip

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
                        # too jittery -> extend window
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

    # ---------------------------------
    # 5) Auto-ROM core
    # ---------------------------------
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

    # 7) Build text strip (no text drawn on the video itself)
    H, W = frame_bgr.shape[:2]
    lines = []
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
                state.last_zero_source = "manual"
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
        default=72,
        help="Height (px) of the HUD text strip under the video.",
    )
    parser.add_argument(
        "--text-strip-bg",
        type=str,
        default="dark",
        choices=["dark", "light"],
        help="Background mode for the HUD text strip.",
    )

    # Panel view mode: single (default) or gallery
    parser.add_argument("--panel-window", choices=["single", "gallery"], default="single",
                        help="single: one latest result window; gallery: one window showing a grid of recent results.")
    parser.add_argument("--gallery-size", type=int, default=4,
                        help="How many recent panels to keep in gallery (max items).")
    parser.add_argument("--gallery-cols", type=int, default=1,
                        help="Number of columns in gallery grid.")
    parser.add_argument("--gallery-cell-width", type=int, default=800,
                        help="Target width per panel cell in gallery (px).")

    # Voice args
    add_voice_args(parser)

    # ---- ZEROING CONTROL ----------------------------------------------------
    # Default: auto-zero enabled unless explicitly disabled with --no-zero.
    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument(
        "--zero",
        dest="zero",
        action="store_true",
        help="Enable baseline zeroing (auto/manual). Default if neither flag given."
    )
    zero_group.add_argument(
        "--no-zero",
        dest="zero",
        action="store_false",
        help="Disable baseline zeroing (auto/manual). Angles use raw values."
    )
    parser.set_defaults(zero=True)

    # ---- DEBUG MODE ---------------------------------------------------------
    # Display-only: still runs detection/pose/angle + HUD,
    # but disables baseline/arming/locking/finalize/saving completely.
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Display-only: compute angles & draw HUD; disable ROM tracking and all saving."
    )

    # AUTO-ROM thresholds
    parser.add_argument("--auto-rom", action="store_true", help="Auto-detect min/max/ROM, save images + JSON")
    parser.add_argument("--rom-ema-alpha", type=float, default=0.25)
    parser.add_argument("--rom-v-go", type=float, default=12.0)
    parser.add_argument("--rom-v-stop", type=float, default=6.0)
    parser.add_argument("--rom-hold-sec", type=float, default=0.4)
    parser.add_argument("--rom-std-max", type=float, default=2.0)
    parser.add_argument("--rom-min-amplitude", type=float, default=10.0)
    parser.add_argument("--rom-timeout-sec", type=float, default=12.0)
    parser.add_argument("--rom-start-amp", type=float, default=5.0)
    parser.add_argument("--rom-baseline-tol", type=float, default=5.0)
    parser.add_argument("--rom-baseline-hold-sec", type=float, default=0.6)
    parser.add_argument("--rom-refractory-sec", type=float, default=1.0)
    parser.add_argument("--rom-lock-refractory-sec", type=float, default=0.8)

    args = parser.parse_args()

    if args.auto_rom:
        assert args.output_root != "", "--auto-rom requires --output-root to save results."

    args.kpt_preset_name = args.rom_test
    set_kpt_preset(args, args.kpt_preset_name)

    assert args.show or (args.output_root != "")
    assert args.input != ""
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, os.path.basename(args.input))
        if args.input == "webcam":
            output_file += ".mp4"

    if args.save_predictions:
        assert args.output_root != ""
        args.pred_save_path = f"{args.output_root}/results_{os.path.splitext(os.path.basename(args.input))[0]}.json"

    # Init detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Init pose estimator (disable flip-test/TTA for speed)
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(
                test_cfg=dict(
                    output_heatmaps=args.draw_heatmap,
                    flip_test=False,  # speed-up
                )
            )
        ),
    )

    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    state = SessionState()

    # Voice (optional)
    voice = init_voice_from_args(args)

    # Determine input type
    if args.input == "webcam":
        input_type = "webcam"
    elif args.input == "realsense":
        input_type = "realsense"
    else:
        mt = mimetypes.guess_type(args.input)[0]
        input_type = mt.split("/")[0] if mt else "video"

    if input_type == "image":
        pred_instances, key, combined_bgr = process_one_image(
            args, args.input, None, detector, pose_estimator, visualizer, 0, state, voice
        )
        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)
        if output_file:
            mmcv.imwrite(combined_bgr, output_file)

    elif input_type == "realsense":
        if rs is None:
            raise ImportError("pyrealsense2 is required for 'realsense' input mode.")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline_profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = depth_frame

                pred_instances, key, combined_bgr = process_one_image(
                    args, color_image, depth_image, detector, pose_estimator, visualizer, 0.001, state, voice
                )

                if key == 27:  # ESC
                    break
        finally:
            pipeline.stop()
            if voice:
                voice.stop()

    elif input_type in ["webcam", "video"]:
        cap = cv2.VideoCapture(0) if args.input == "webcam" else cv2.VideoCapture(args.input)
        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                frame_idx += 1
                if not success:
                    break

                pred_instances, key, combined_bgr = process_one_image(
                    args, frame, None, detector, pose_estimator, visualizer, 0.001, state, voice
                )

                if key == 27:
                    break

                if args.save_predictions:
                    pred_instances_list.append(dict(frame_id=frame_idx, instances=split_instances(pred_instances)))

                if output_file:
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        H, W = combined_bgr.shape[:2]
                        video_writer = cv2.VideoWriter(output_file, fourcc, 25, (W, H))
                    video_writer.write(combined_bgr)
        finally:
            if video_writer:
                video_writer.release()
            cap.release()
            if voice:
                voice.stop()

    else:
        args.save_predictions = False
        if voice:
            voice.stop()
        raise ValueError(f"file {os.path.basename(args.input)} has invalid format.")

    if args.save_predictions:
        with open(args.pred_save_path, "w") as f:
            json.dump(
                dict(meta_info=pose_estimator.dataset_meta, instance_info=pred_instances_list), f, indent="\t"
            )
        print(f"predictions have been saved at {args.pred_save_path}")

    if output_file:
        input_type_print = input_type.replace("webcam", "video")
        print_log(f'the output {input_type_print} has been saved at {output_file}', logger="current", level=logging.INFO)


if __name__ == "__main__":
    main()
