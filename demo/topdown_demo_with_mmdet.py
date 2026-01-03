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

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def run_detector_once(args, img, detector):
    """Run detector and return person bboxes (x1, y1, x2, y2)."""
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()

    # [x1, y1, x2, y2, score]
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)

    keep = np.logical_and(
        pred_instance.labels == args.det_cat_id,
        pred_instance.scores > args.bbox_thr)
    bboxes = bboxes[keep]

    # NMS and drop score column
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
    return bboxes


def pick_center_person_bbox(img, bboxes):
    """Keep only the bbox whose center is closest to the image center."""
    if bboxes is None or len(bboxes) == 0:
        return bboxes

    if isinstance(img, str):
        img_arr = mmcv.imread(img)
    else:
        img_arr = img

    h, w = img_arr.shape[:2]
    cx_img, cy_img = w * 0.5, h * 0.5

    centers = (bboxes[:, :2] + bboxes[:, 2:4]) * 0.5
    d2 = (centers[:, 0] - cx_img) ** 2 + (centers[:, 1] - cy_img) ** 2
    idx = int(np.argmin(d2))
    return bboxes[idx:idx + 1]


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      last_bboxes=None,
                      frame_idx=0):
    """Run detector+pose (with optional skip-frame bbox caching) and visualize.

    Returns:
        pred_instances (InstanceData | None), last_bboxes (np.ndarray)
    """

    # Decide whether to run detector on this frame
    need_detector = (last_bboxes is None) or (frame_idx % args.det_interval == 0)

    if need_detector:
        bboxes = run_detector_once(args, img, detector)
        last_bboxes = bboxes
    else:
        bboxes = last_bboxes

    # Optional single-person-only mode
    if args.pick_center_person and bboxes is not None and len(bboxes) > 1:
        bboxes = pick_center_person_bbox(img, bboxes)
        last_bboxes = bboxes

    # Hard error if no detections (your requirement)
    assert bboxes is not None and len(bboxes) > 0, (
        f"[ERROR] No person detected on frame {frame_idx}. "
        f"Increase --bbox-thr? Check --det-cat-id? Bad detector checkpoint?")

    # Predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # Visualization (optional / headless-safe)
    if visualizer is not None:
        if isinstance(img, str):
            img_vis_in = mmcv.imread(img, channel_order='rgb')
        elif isinstance(img, np.ndarray):
            img_vis_in = mmcv.bgr2rgb(img)
        else:
            img_vis_in = img

        visualizer.add_datasample(
            'result',
            img_vis_in,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval if args.show else 0,
            kpt_thr=args.kpt_thr)

    return data_samples.get('pred_instances', None), last_bboxes


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')

    parser.add_argument('--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show', action='store_true', default=False, help='whether to show img')

    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output file. '
        'For images: saves visualized image. '
        'For video/webcam: used with --save-video to save mp4. '
        'Also required for --save-predictions.')

    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results (json)')

    parser.add_argument(
        '--save-video',
        action='store_true',
        default=False,
        help='Draw overlays and save an output mp4 (headless-safe). '
        'Requires --output-root.')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

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

    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bboxes of instances')

    # A) Detector skip-frame logic
    parser.add_argument(
        '--det-interval',
        type=int,
        default=1,
        help='Run detector every N frames (1 = run on every frame). '
        'When N>1, bboxes are cached between detector frames.')

    # B) Optional single person mode only
    parser.add_argument(
        '--pick-center-person',
        action='store_true',
        default=False,
        help='If set, keep only the detected person whose bbox center is closest '
        'to the image center. Requires --det-interval 1.')

    assert has_mmdet, 'Please install mmdet to run the demo.'
    args = parser.parse_args()

    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    assert args.det_interval >= 1, '--det-interval must be >= 1'

    if args.pick_center_person:
        assert args.det_interval == 1, (
            '--pick-center-person requires --det-interval 1 (no bbox caching).')

    if args.save_video:
        assert args.output_root != '', '--save-video requires --output-root'

    if args.save_predictions:
        assert args.output_root != '', '--save-predictions requires --output-root'

    # Decide input type
    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        mt = mimetypes.guess_type(args.input)[0]
        assert mt is not None, f'Cannot infer file type from: {args.input}'
        input_type = mt.split('/')[0]

    # Output paths
    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'  # webcam always video output

    if args.save_predictions:
        args.pred_save_path = (
            f'{args.output_root}/results_'
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'
        )

    # Build detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # C) Headless-safe visualizer: only build when needed
    visualizer = None
    need_visualizer = args.show or args.save_video or (input_type == 'image' and args.output_root != '')
    if need_visualizer:
        pose_estimator.cfg.visualizer.radius = args.radius
        pose_estimator.cfg.visualizer.alpha = args.alpha
        pose_estimator.cfg.visualizer.line_width = args.thickness
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if input_type == 'image':
        # For images, saving visualization requires visualizer.
        assert args.show or (args.output_root != ''), (
            'For image input, set --show or --output-root to produce output.')

        pred_instances, _ = process_one_image(
            args, args.input, detector, pose_estimator,
            visualizer=visualizer, show_interval=args.show_interval,
            last_bboxes=None, frame_idx=0)

        pred_instances_list = None
        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if args.output_root:
            assert visualizer is not None, 'Internal error: visualizer should exist for image output.'
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

        if args.save_predictions:
            with open(args.pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=[dict(frame_id=0, instances=pred_instances_list)]
                    ),
                    f,
                    indent='\t')
            print(f'predictions have been saved at {args.pred_save_path}')

    elif input_type in ['webcam', 'video']:
        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
            # keep same defaults as your optimized runner
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0
        last_bboxes = None

        t_start = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1

            pred_instances, last_bboxes = process_one_image(
                args, frame, detector, pose_estimator,
                visualizer=visualizer,
                show_interval=0.001,
                last_bboxes=last_bboxes,
                frame_idx=frame_idx)

            if args.save_predictions:
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # Save video (YES even if --show is false)
            if args.save_video:
                assert output_file is not None and args.output_root != ''
                assert visualizer is not None, 'Internal error: visualizer required for --save-video.'
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            # Show (optional)
            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                time.sleep(args.show_interval)

            # Optional end-to-end FPS print (kept, since it's useful for your runs)
            elapsed = time.time() - t_start
            fps = frame_idx / max(elapsed, 1e-9)
            print(f"Total FPS (capture + inference): {fps:.2f}")

        if video_writer:
            video_writer.release()
        cap.release()

        if args.save_predictions:
            with open(args.pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_list),
                    f,
                    indent='\t')
            print(f'predictions have been saved at {args.pred_save_path}')

        if args.save_video and output_file:
            print_log(
                f'the output video has been saved at {output_file}',
                logger='current',
                level=logging.INFO)

    else:
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')


if __name__ == '__main__':
    main()
