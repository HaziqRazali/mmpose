#!/usr/bin/env python3
"""
MediaPipe Pose inference on a single video.
Saves predictions in the same JSON schema as MMPose --save-predictions.

Uses the MediaPipe Tasks API (mediapipe >= 0.10.x).
Model is auto-downloaded to ~/.cache/mediapipe/ on first run.

Usage:
    python mediapipe/run.py --input <video> --output-root <dir> --save-predictions
"""

import argparse
import json
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model auto-download
# ---------------------------------------------------------------------------
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
_MODEL_CACHE = os.path.expanduser("~/.cache/mediapipe/pose_landmarker_heavy.task")


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_CACHE):
        os.makedirs(os.path.dirname(_MODEL_CACHE), exist_ok=True)
        print(f"[mediapipe] Downloading model -> {_MODEL_CACHE} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
        print("[mediapipe] Model downloaded.")
    return _MODEL_CACHE

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark names (matches PoseLandmark enum order)
# ---------------------------------------------------------------------------
_KEYPOINT_ID2NAME = {
    0:  "nose",
    1:  "left_eye_inner",
    2:  "left_eye",
    3:  "left_eye_outer",
    4:  "right_eye_inner",
    5:  "right_eye",
    6:  "right_eye_outer",
    7:  "left_ear",
    8:  "right_ear",
    9:  "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

_META_INFO = {
    "dataset_name": "mediapipe_pose",
    "num_keypoints": 33,
    "keypoint_id2name":  {str(k): v for k, v in _KEYPOINT_ID2NAME.items()},
    "keypoint_name2id":  {v: k for k, v in _KEYPOINT_ID2NAME.items()},
}

_BBOX_PAD_PX = 10  # padding around the landmark bounding box

# Skeleton connections (pairs of landmark indices) — same as MediaPipe's official skeleton
_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 7),          # nose -> left eye chain -> left ear
    (0, 4), (4, 5), (5, 6), (6, 8),          # nose -> right eye chain -> right ear
    (9, 10),                                  # mouth
    (11, 12),                                 # shoulders
    (11, 13), (13, 15),                       # left arm
    (12, 14), (14, 16),                       # right arm
    (15, 17), (15, 19), (15, 21),             # left hand
    (16, 18), (16, 20), (16, 22),             # right hand
    (11, 23), (12, 24), (23, 24),             # torso
    (23, 25), (25, 27), (27, 29), (27, 31),   # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),   # right leg
]

_KPT_COLOR  = (0, 255, 0)    # BGR green
_BONE_COLOR = (255, 128, 0)  # BGR orange
_KPT_RADIUS = 4
_BONE_THICK = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _landmarks_to_instance(landmarks: list, frame_w: int, frame_h: int) -> dict:
    """Convert a list of MediaPipe NormalizedLandmark (Tasks API) to an MMPose instance dict."""
    keypoints = []
    keypoint_scores = []
    xs, ys = [], []

    for lm in landmarks:
        x_px = float(lm.x * frame_w)
        y_px = float(lm.y * frame_h)
        keypoints.append([x_px, y_px])
        keypoint_scores.append(float(lm.visibility if lm.visibility is not None else 0.0))
        xs.append(x_px)
        ys.append(y_px)

    x1 = max(0.0, min(xs) - _BBOX_PAD_PX)
    y1 = max(0.0, min(ys) - _BBOX_PAD_PX)
    x2 = min(float(frame_w), max(xs) + _BBOX_PAD_PX)
    y2 = min(float(frame_h), max(ys) + _BBOX_PAD_PX)

    return {
        "keypoints":        keypoints,
        "keypoint_scores":  keypoint_scores,
        "bbox":             [[x1, y1, x2, y2]],
        "bbox_score":       1.0,
    }


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------
def _draw_pose(frame, keypoints: list, keypoint_scores: list, score_thr: float = 0.3):
    """Draw skeleton and keypoints on a BGR frame in-place."""
    for (i, j) in _SKELETON:
        if keypoint_scores[i] >= score_thr and keypoint_scores[j] >= score_thr:
            p1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            p2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, p1, p2, _BONE_COLOR, _BONE_THICK)
    for idx, (kp, sc) in enumerate(zip(keypoints, keypoint_scores)):
        if sc >= score_thr:
            cx, cy = int(kp[0]), int(kp[1])
            cv2.circle(frame, (cx, cy), _KPT_RADIUS, _KPT_COLOR, -1)


# ---------------------------------------------------------------------------
# Core inference  (uses MediaPipe Tasks API — mediapipe >= 0.10.x)
# ---------------------------------------------------------------------------
def run_inference(input_path: str, out_video_path: str | None = None) -> dict:
    model_path = _ensure_model()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    instance_info = []
    frame_id = 0

    # Optional video writer
    writer = None
    if out_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Tasks VIDEO mode requires monotonically increasing timestamps (ms)
            timestamp_ms = int(frame_id * 1000.0 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                instances = [_landmarks_to_instance(result.pose_landmarks[0], w, h)]
            else:
                instances = []

            if writer is not None:
                vis = frame.copy()
                for inst in instances:
                    _draw_pose(vis, inst["keypoints"], inst["keypoint_scores"])
                writer.write(vis)

            instance_info.append({"frame_id": frame_id, "instances": instances})
            frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    return {
        "meta_info":     _META_INFO,
        "instance_info": instance_info,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe Pose batch inference (MMPose-compatible output)"
    )
    parser.add_argument("--input",            required=True, help="Path to input video")
    parser.add_argument("--output-root",      required=True, help="Directory to save output JSON")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save predictions to JSON (always true; flag kept for interface compatibility)")
    parser.add_argument("--save-video", action="store_true",
                        help="Write visualisation video alongside the JSON")
    args = parser.parse_args()

    action_name = os.path.splitext(os.path.basename(args.input))[0]
    out_json    = os.path.join(args.output_root, f"{action_name}.json")
    out_mp4     = os.path.join(args.output_root, f"{action_name}.mp4") if args.save_video else None

    print(f"[mediapipe] Running on:     {args.input}")
    print(f"[mediapipe] Output JSON:    {out_json}")
    if out_mp4:
        print(f"[mediapipe] Output video:   {out_mp4}")

    predictions = run_inference(args.input, out_video_path=out_mp4)

    n_frames   = len(predictions["instance_info"])
    n_detected = sum(1 for fi in predictions["instance_info"] if fi["instances"])
    print(f"[mediapipe] Processed {n_frames} frames, pose detected in {n_detected} frames.")

    os.makedirs(args.output_root, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(predictions, fh)

    print(f"[mediapipe] Saved JSON:     {out_json}")
    if out_mp4:
        print(f"[mediapipe] Saved video:    {out_mp4}")


if __name__ == "__main__":
    main()
