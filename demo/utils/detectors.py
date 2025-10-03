# detectors.py
# Run detection+pose for a single RGB frame and (optionally) draw all detected boxes.

from typing import Optional, Tuple
import numpy as np
import cv2

from mmdet.apis import inference_detector
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples


def _to_np(x):
    """Torch tensor -> numpy (or passthrough if already numpy-like)."""
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def largest_person_bbox(det_result, det_cat_id: int = 0, score_thr: float = 0.3):
    """
    Pick all person boxes above threshold and return (bboxes, scores, idx_of_largest_area).
    If none kept, returns (None, None, None).
    """
    pred = det_result.pred_instances
    b = _to_np(pred.bboxes)
    labels = _to_np(pred.labels)
    scores = _to_np(pred.scores)

    keep = np.logical_and(labels == det_cat_id, scores > score_thr)
    b = b[keep]
    s = scores[keep]

    if b.size == 0:
        return None, None, None

    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    idx = int(np.argmax(areas))
    return b, s, idx


def run_once(frame_bgr: np.ndarray,
             det_model,
             pose_model,
             det_cat_id: int,
             score_thr: float,
             prefer_nearer: bool = False,
             depth_frame: Optional[dict] = None,
             rgb_size: Optional[Tuple[int, int]] = None):
    """
    Full pass for one frame:
      1) detect persons
      2) pick the largest-area person box
      3) run top-down pose on that box
    Returns:
      (kpts_xy[NK,3], chosen_bbox[4], err_str_or_None, all_bboxes, all_scores, chosen_idx)
    On failure, kpts and chosen_bbox are None, and err_str is "no-person" or "no-pose".
    """
    det_result = inference_detector(det_model, frame_bgr)
    kept_bboxes, kept_scores, idx_area = largest_person_bbox(det_result, det_cat_id, score_thr)

    if kept_bboxes is None:
        return None, None, "no-person", None, None, None

    # (Optional) policy: could switch to a nearer box if using depth_frame; currently fixed to largest-area
    chosen_idx = idx_area
    chosen_bbox = kept_bboxes[chosen_idx]

    pose_results = inference_topdown(pose_model, frame_bgr, bboxes=chosen_bbox[None, :4])
    if not pose_results:
        return None, None, "no-pose", kept_bboxes, kept_scores, chosen_idx

    data = merge_data_samples(pose_results)
    if data is None or not hasattr(data, "pred_instances") or len(data.pred_instances) == 0:
        return None, None, "no-pose", kept_bboxes, kept_scores, chosen_idx

    k = _to_np(data.pred_instances.keypoints)[0]  # [K, 3] (x,y,score)
    return k, chosen_bbox, None, kept_bboxes, kept_scores, chosen_idx


def draw_bboxes(frame: np.ndarray,
                bboxes: Optional[np.ndarray],
                scores: Optional[np.ndarray] = None,
                chosen_idx: Optional[int] = None,
                show_indices: bool = True) -> np.ndarray:
    """
    Draw all detected person boxes; highlight the chosen one in yellow.
    """
    if bboxes is None:
        return frame

    for i, b in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, b[:4])
        color = (0, 165, 255)  # orange for others
        if chosen_idx is not None and i == chosen_idx:
            color = (0, 255, 255)  # yellow for chosen
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        if show_indices:
            label = f"{i}"
            if scores is not None and len(scores) > i:
                label += f":{scores[i]:.2f}"
            # drop-shadow text
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
