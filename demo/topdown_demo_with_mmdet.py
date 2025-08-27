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

from collections import deque

from utils_variables import rom_test
from utils import set_kpt_preset, handle_hotkeys_for_presets
from utils_3d import extract_intrinsics_from_depth, compute_3d_skeletons, visualize_3d_skeletons, angle

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
# Session State
# =========================

class SessionState:
    def __init__(self):
        self.last_angle = None
        self.angle_series = deque()
        self.active_rom = None
        self.baseline_deg = None          # baseline offset (None = unzeroed)
        self.current_raw_angle = np.nan   # latest raw angle each frame


# =========================
# Sparkline
# =========================

def render_sparkline_strip(width, height, series, t_now, window_sec, label_text=None, gap_sec=0.4):
    strip = np.zeros((height, width, 3), dtype=np.uint8)

    if window_sec <= 0 or len(series) < 2:
        if label_text:
            cv2.putText(strip, label_text, (10, int(0.65 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return strip

    t0 = t_now - float(window_sec)
    filtered = [(max(t0, t), a) for (t, a) in series if (t >= t0 and np.isfinite(a))]
    if len(filtered) < 2:
        if label_text:
            cv2.putText(strip, label_text, (10, int(0.65 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return strip

    ts = [t for (t, _) in filtered]
    ys = [a for (_, a) in filtered]

    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or abs(y_max - y_min) < 1e-9:
        y_min, y_max = (0.0, 1.0) if not np.isfinite(y_min) or not np.isfinite(y_max) else (y_min - 1.0, y_max + 1.0)

    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (32, 32, 32), thickness=-1)
    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (180, 180, 180), thickness=1)

    def map_x(t):
        tx = (t - t0) / max(window_sec, 1e-6)
        tx = 0.0 if not np.isfinite(tx) else min(max(tx, 0.0), 1.0)
        return int(round(tx * (width - 1)))

    def map_y(y):
        ty = (y - y_min) / max((y_max - y_min), 1e-6)
        ty = 0.0 if not np.isfinite(ty) else min(max(ty, 0.0), 1.0)
        return int(round((height - 1) - ty * (height - 1)))

    segments = []
    seg = [(map_x(ts[0]), map_y(ys[0]))]
    for i in range(1, len(ts)):
        if (ts[i] - ts[i - 1]) > gap_sec:
            if len(seg) >= 2:
                segments.append(np.array(seg, dtype=np.int32))
            seg = []
        seg.append((map_x(ts[i]), map_y(ys[i])))
    if len(seg) >= 2:
        segments.append(np.array(seg, dtype=np.int32))

    for poly in segments:
        cv2.polylines(strip, [poly], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.putText(strip, f"[{y_min:.1f}, {y_max:.1f}]", (10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    last_y = ys[-1]
    if np.isfinite(last_y):
        cv2.putText(strip, f"{last_y:.1f} deg", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if label_text:
        txt_w = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(strip, label_text, (width - txt_w - 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return strip


# =========================
# Core Processing
# =========================

def process_one_image(args,
                      color_img,
                      depth_img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      state=None):

    det_result = inference_detector(detector, color_img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    pose_results = inference_topdown(pose_estimator, color_img, bboxes)
    data_samples = merge_data_samples(pose_results)

    if isinstance(color_img, str):
        color_img = mmcv.imread(color_img, channel_order='rgb')
    elif isinstance(color_img, np.ndarray):
        color_img = mmcv.bgr2rgb(color_img)

    # Reset state if preset changed
    if state is not None:
        if state.active_rom is None:
            state.active_rom = args.rom_test
            state.baseline_deg = None
        elif args.rom_test != state.active_rom:
            state.active_rom = args.rom_test
            state.last_angle = None
            state.angle_series.clear()
            state.baseline_deg = None

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
            show_kpt_subset=rom_test[args.rom_test],
            skeleton_style=args.skeleton_style,
            show=False,
            wait_time=0,
            kpt_thr=args.kpt_thr)

    frame_rgb = visualizer.get_image()
    frame_bgr = mmcv.rgb2bgr(frame_rgb)

    t_now = time.time()
    pred = data_samples.get('pred_instances', None)

    angle_t = np.nan
    current_ids = rom_test.get(args.rom_test, [])
    is_angle_mode = isinstance(current_ids, (list, tuple)) and len(current_ids) == 3

    if (depth_img is not None) and (pred is not None):
        intrin_dict = extract_intrinsics_from_depth(depth_img)
        if intrin_dict is not None:
            keypoints = getattr(pred, 'transformed_keypoints', None)
            if keypoints is None:
                keypoints = pred.keypoints
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()

            visibility = getattr(pred, 'keypoints_visible', None)
            if visibility is None:
                visibility = np.ones(keypoints.shape[:2])
            elif hasattr(visibility, 'cpu'):
                visibility = visibility.cpu().numpy()

            if keypoints.shape[0] > 0:
                joints_xyz = compute_3d_skeletons(keypoints, visibility, depth_img, intrin_dict, args.kpt_thr)
                joints_xyz = np.array(joints_xyz)

                try:
                    if is_angle_mode:
                        a, b, c = current_ids
                        angle_t = angle(joints_xyz[0, a], joints_xyz[0, b], joints_xyz[0, c])
                except Exception as e:
                    cv2.putText(frame_bgr, f"Angle err: {e}", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # store raw
    state.current_raw_angle = angle_t if np.isfinite(angle_t) else np.nan

    # apply baseline
    if np.isfinite(angle_t) and (state.baseline_deg is not None):
        angle_disp = angle_t - state.baseline_deg
    else:
        angle_disp = angle_t

    if state is not None and args.plot_seconds > 0 and is_angle_mode and np.isfinite(angle_disp):
        state.last_angle = float(angle_disp)
        state.angle_series.append((t_now, float(angle_disp)))
        cutoff = t_now - float(args.plot_seconds)
        while state.angle_series and state.angle_series[0][0] < cutoff:
            state.angle_series.popleft()

    # HUD
    if is_angle_mode:
        if np.isfinite(angle_disp):
            cv2.putText(frame_bgr, f"Angle: {angle_disp:.1f} deg", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Angle: --.- deg", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 240), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame_bgr, f"ROM: {args.rom_test} (no angle)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 240), 2, cv2.LINE_AA)

    # test name + zero status
    zero_text = "Zeroed" if state.baseline_deg is not None else "Unzeroed"
    cv2.putText(frame_bgr, f"Test: {args.rom_test} | {zero_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    # sparkline
    H, W = frame_bgr.shape[:2]
    sparkline = render_sparkline_strip(
        width=W,
        height=int(args.plot_height),
        series=list(state.angle_series) if (state is not None and is_angle_mode) else [],
        t_now=t_now,
        window_sec=float(args.plot_seconds),
        label_text="Angle history (s)",
        gap_sec=float(args.plot_gap_sec)
    ) if args.plot_seconds > 0 else None

    if sparkline is not None:
        combined_bgr = np.vstack([frame_bgr, sparkline])
    else:
        combined_bgr = frame_bgr

    # window + hotkeys
    if args.show:
        cv2.imshow("Pose", combined_bgr)
        key = cv2.pollKey() if hasattr(cv2, "pollKey") else cv2.waitKey(1)
        handle_hotkeys_for_presets(args, key)
        # hotkey: b to zero
        if key == ord('b'):
            if np.isfinite(state.current_raw_angle):
                state.baseline_deg = float(state.current_raw_angle)
                state.angle_series.clear()
                state.last_angle = None
                print(f"[KPT] Baseline set to {state.baseline_deg:.2f}°")
            else:
                print("[KPT] Cannot zero: no valid angle this frame.")
        if args.show_interval > 0:
            time.sleep(args.show_interval)
    else:
        key = -1

    return data_samples.get('pred_instances', None), key, combined_bgr


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth = param.get_distance(x, y)
        print(f"[mouse_callback] Mouse = ({x}, {y}) → Distance: {depth:.3f} meters")


def main():
    parser = ArgumentParser()
    parser.add_argument('det_config')
    parser.add_argument('det_checkpoint')
    parser.add_argument('pose_config')
    parser.add_argument('pose_checkpoint')
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--output-root', type=str, default='')
    parser.add_argument('--save-predictions', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--det-cat-id', type=int, default=0)
    parser.add_argument('--bbox-thr', type=float, default=0.3)
    parser.add_argument('--nms-thr', type=float, default=0.3)
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--draw-heatmap', action='store_true', default=False)
    parser.add_argument('--show-kpt-idx', action='store_true', default=False)
    parser.add_argument('--skeleton-style', default='mmpose', type=str,
                        choices=['mmpose', 'openpose'])
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--thickness', type=int, default=3)
    parser.add_argument('--show-interval', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--draw-bbox', action='store_true')

    parser.add_argument('--rom_test', default="right_elbow_flexion", type=str)
    parser.add_argument('--show3d', action='store_true')

    parser.add_argument('--plot-seconds', type=float, default=5.0)
    parser.add_argument('--plot-height', type=int, default=80)
    parser.add_argument('--plot-gap-sec', type=float, default=0.4)

    assert has_mmdet, 'Please install mmdet to run the demo.'
    args = parser.parse_args()

    args.kpt_preset_name = args.rom_test
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

    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    state = SessionState()

    if args.input == 'webcam':
        input_type = 'webcam'
    elif args.input == 'realsense':
        input_type = 'realsense'
    else:
        mt = mimetypes.guess_type(args.input)[0]
        input_type = mt.split('/')[0] if mt else 'video'

    if input_type == 'image':

        # inference
        pred_instances, key, combined_bgr = process_one_image(
            args, args.input, None, detector, pose_estimator, visualizer, 0, state)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            mmcv.imwrite(combined_bgr, output_file)

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

        # (Optional) depth scale
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # meters per unit

        # Align depth to color
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
                    args, color_image, depth_image, detector, pose_estimator, visualizer, 0.001, state)

                if key == 27:  # ESC
                    break

        finally:
            pipeline.stop()

    elif input_type in ['webcam', 'video']:

        cap = cv2.VideoCapture(0) if args.input == 'webcam' else cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            pred_instances, key, combined_bgr = process_one_image(
                args, frame, None, detector, pose_estimator, visualizer, 0.001, state)

            if key == 27:
                break

            if args.save_predictions:
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos (save the concatenated frame)
            if output_file:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    H, W = combined_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,
                        (W, H))
                video_writer.write(combined_bgr)

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