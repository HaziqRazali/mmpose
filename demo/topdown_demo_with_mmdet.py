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
from utils_variables import presets
from utils_3d import extract_intrinsics_from_depth, compute_3d_skeletons, visualize_3d_skeletons

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    import pyrealsense2 as rs
except (ImportError, ModuleNotFoundError):
    sys.exit()

def process_one_image(args,
                      color_img,
                      depth_img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

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
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    #################### save predictions for debugging

    if args.save_predictions and data_samples.get('pred_instances', None) is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # add ms to avoid collision
        os.makedirs("snapshots", exist_ok=True)

        # Save RGB image
        cv2.imwrite(f"snapshots/image_{timestamp}.png", cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

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
            print("Warning: depth_img is not a RealSense frame")

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

    #################### 3D computation and visualization                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

    if depth_img is not None:
        intrin_dict = extract_intrinsics_from_depth(depth_img)
        if intrin_dict is not None:
            pred = data_samples.pred_instances

            # Convert to numpy (safe for torch.Tensor)
            keypoints = pred.get('transformed_keypoints', pred.keypoints)
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()  # [N, J, 2]

            visibility = pred.get('keypoints_visible', None)
            if visibility is None:
                visibility = np.ones(keypoints.shape[:2])
            elif hasattr(visibility, 'cpu'):
                visibility = visibility.cpu().numpy()

            if keypoints.shape[0] > 0:
                # ---- 3D computation (no plotting here) ----
                joint_xyz_list = compute_3d_skeletons(
                    keypoints, visibility, depth_img, intrin_dict, args.kpt_thr
                )

                # ---- 3D visualization (optional) ----
                if getattr(args, "show3d", False):
                    visualize_3d_skeletons(
                        joint_xyz_list, skeleton, args.show_kpt_subset, args.kpt_thr
                    )

    # try:
    #     depth_intrin = depth_img.profile.as_video_stream_profile().get_intrinsics()
    #     intrin_dict = {
    #         "fx": depth_intrin.fx,
    #         "fy": depth_intrin.fy,
    #         "ppx": depth_intrin.ppx,
    #         "ppy": depth_intrin.ppy,
    #         "depth_scale": 0.001  # RealSense default
    #     }
    # except Exception as e:
    #     print(f"[!] Cannot extract intrinsics for 3D viz: {e}")
    #     return data_samples.get('pred_instances', None)

    # pred = data_samples.pred_instances

    # # Convert to numpy (safe for torch.Tensor)
    # keypoints = pred.get('transformed_keypoints', pred.keypoints)
    # if hasattr(keypoints, 'cpu'):
    #     keypoints = keypoints.cpu().numpy()  # [N, J, 2]
    # visibility = pred.get('keypoints_visible', None)
    # if visibility is None:
    #     visibility = np.ones(keypoints.shape[:2])
    # elif hasattr(visibility, 'cpu'):
    #     visibility = visibility.cpu().numpy()

    # if keypoints.shape[0] == 0:
    #     return pred  # no person detected

    # # Plot setup
    # fig = plt.gcf()
    # fig.clf()  # Clear only the current figure (preserve window)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_title("3D Skeleton")

    # for person_kpts, person_vis in zip(keypoints, visibility):
    #     joint_xyz = []

    #     for (x, y), v in zip(person_kpts, person_vis):
    #         if v > args.kpt_thr and 0 <= int(x) < depth_img.width and 0 <= int(y) < depth_img.height:
    #             depth_val = depth_img.get_distance(int(x), int(y))
    #             if depth_val and depth_val > 0:
    #                 joint_xyz.append(deproject(depth_val, x, y, intrin_dict))
    #             else:
    #                 joint_xyz.append([np.nan, np.nan, np.nan])
    #         else:
    #             joint_xyz.append([np.nan, np.nan, np.nan])

    #     joint_xyz = np.array(joint_xyz)

    #     for j, (x, y, z) in enumerate(joint_xyz):
    #         if not np.isnan(z) and j in args.show_kpt_subset:
    #             ax.scatter(x, y, z, c='green', s=10)

    #     for idx1, idx2 in skeleton:
    #         if idx1 < len(joint_xyz) and idx2 < len(joint_xyz):
    #             pt1, pt2 = joint_xyz[idx1], joint_xyz[idx2]
    #             if not np.any(np.isnan(pt1)) and not np.any(np.isnan(pt2)):
    #                 ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='blue', linewidth=2)
                    
    # ax.set_xlim(-1.0, 1.0)
    # ax.set_ylim(-1.0, 1.0)
    # ax.set_zlim(0.0, 3.0)
    # set_axes_equal(ax)

    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.view_init(elev=-80, azim=-90)
    # plt.pause(0.001)  # Refresh the plot without blocking

    return data_samples.get('pred_instances', None)

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
    parser.add_argument(
        '--save_predictions', action='store_true', help='Save predictions to a folder')
    parser.add_argument(
        '--show_kpt_subset', default="full_body", type=str)
    parser.add_argument(
        '--show3d', action='store_true')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    #################### parse args
    args = parser.parse_args()
    args.show_kpt_subset = presets[args.show_kpt_subset]

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

    if args.input == 'webcam':
        input_type = 'webcam'
    elif args.input == "realsense":
        input_type = "realsense"
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input, detector, pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type == "realsense":

        # Setup pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline_profile = pipeline.start(config)

        # Get depth scale
        depth_sensor    = pipeline_profile.get_device().first_depth_sensor()
        depth_scale     = depth_sensor.get_depth_scale()  # in meters per unit

        # Create align object (align depth to color)
        align = rs.align(rs.stream.color)

        # Mouse callback
        #mouse_x, mouse_y = -1, -1

        #cv2.namedWindow("RGB | Depth | Overlay")
        #cv2.setMouseCallback("RGB | Depth | Overlay", mouse_callback)

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
                pred_instances  = process_one_image(args, color_image, depth_image, detector, pose_estimator, visualizer, 0.001)

                if args.show:
                    # press ESC to exit
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    time.sleep(args.show_interval)

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
            pred_instances = process_one_image(args, frame, detector, pose_estimator, visualizer, 0.001)

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

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

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
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
