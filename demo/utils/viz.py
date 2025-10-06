# viz.py
# Generic visualization toolbox + optional per-ROM drawer hooks.

from typing import Optional, Tuple, Dict, Callable, Any
from dataclasses import dataclass
import numpy as np
import cv2


# ------------------------
# Text overlays & layout
# ------------------------

def overlay_text(frame, lines, org=(16, 32), scale=0.7, box_alpha=0.5, pad=6):
    # compute box size
    w_max, h_tot = 0, 0
    for ln in lines:
        (w, h), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        w_max = max(w_max, w)
        h_tot += h + int(10 * scale)
    x0, y0 = org
    x1, y1 = x0 + w_max + 2*pad, y0 + h_tot + 2*pad

    # semi-transparent box
    box = frame.copy()
    cv2.rectangle(box, (x0 - pad, y0 - int(20*scale) - pad), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(box, box_alpha, frame, 1 - box_alpha, 0, frame)

    # text
    y = y0
    for ln in lines:
        cv2.putText(frame, ln, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(28 * scale)
    return frame

def annotate_panel(frame_rgb,
                   rom_name,
                   mode_used,
                   when_label,
                   angle_deg,
                   rgb_pos_sec,
                   scale: float = 0.7):
    # compact header
    line1 = f"{when_label} | {rom_name}"
    line2 = f"Mode: {mode_used}" + (f" | Angle: {angle_deg:.1f} deg" if angle_deg is not None else "")
    line3 = (f"RGB: {rgb_pos_sec:.3f}s" if rgb_pos_sec is not None else None)
    lines = [line1, line2] + ([line3] if line3 else [])
    return overlay_text(frame_rgb, lines, scale=scale)

def stack_side_by_side(img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
    """
    Horizontal stack with height match. Preserves aspect ratio.
    """
    h = max(img_left.shape[0], img_right.shape[0])

    def resize_to_h(img, H):
        return cv2.resize(img, (int(round(img.shape[1] * (H / img.shape[0]))), H))

    L = resize_to_h(img_left, h)
    R = resize_to_h(img_right, h)
    return np.hstack([L, R])


# ------------------------
# 2D vector drawing
# ------------------------

def _avg_point(kpts_xy: np.ndarray, spec) -> Optional[np.ndarray]:
    """
    Accepts an int (joint id) or list/tuple (average of provided joints).
    Returns [x, y] or None.
    """
    if isinstance(spec, (list, tuple)):
        pts = []
        for s in spec:
            p = _avg_point(kpts_xy, s)
            if p is not None:
                pts.append(p)
        if not pts:
            return None
        arr = np.array(pts, dtype=float)
        return arr.mean(axis=0)
    else:
        idx = int(spec)
        if idx < 0 or idx >= len(kpts_xy):
            return None
        return kpts_xy[idx][:2]


def draw_vectors2d(frame_bgr: np.ndarray,
                   kpts_xy: np.ndarray,
                   vec_pair,
                   thickness: int = 3) -> np.ndarray:
    """
    Draw two vectors defined by keypoints on top of a frame.
    vec_pair = [[P0, P1], [Q0, Q1]] where each element is int or list[int] (avg).
    """
    (P0, P1), (Q0, Q1) = vec_pair
    p0 = _avg_point(kpts_xy, P0); p1 = _avg_point(kpts_xy, P1)
    q0 = _avg_point(kpts_xy, Q0); q1 = _avg_point(kpts_xy, Q1)

    def ok(pt): return pt is not None and np.isfinite(pt).all()

    if ok(p0) and ok(p1):
        cv2.line(frame_bgr, tuple(np.int32(p0)), tuple(np.int32(p1)), (255, 0, 0), thickness, cv2.LINE_AA)
        cv2.circle(frame_bgr, tuple(np.int32(p0)), 5, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, tuple(np.int32(p1)), 5, (255, 0, 0), -1, cv2.LINE_AA)

    if ok(q0) and ok(q1):
        cv2.line(frame_bgr, tuple(np.int32(q0)), tuple(np.int32(q1)), (0, 200, 0), thickness, cv2.LINE_AA)
        cv2.circle(frame_bgr, tuple(np.int32(q0)), 5, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, tuple(np.int32(q1)), 5, (0, 200, 0), -1, cv2.LINE_AA)

    return frame_bgr


# ------------------------
# Per-ROM custom drawers (hook registry)
# ------------------------

@dataclass
class VizContext:
    """
    Everything a custom drawer might need for a single panel.
    - frame: BGR image to draw on (in-place)
    - kpts_xy: [K,3] keypoints (x,y,score)
    - when_label: "t1" | "t2" | custom
    - angle_deg: per-panel angle if applicable (else None)
    - rgb_pos_sec: actual RGB timestamp
    - vec_pair: the main vec_pair used for this ROM, if any (viz purpose only)
    - depth_frame: optional dict with {depth, fx, fy, ox, oy, h, w, ...}
    - rgb_size: (W, H)
    - median_k: patch size used by depth samplers if needed
    """
    frame: np.ndarray
    kpts_xy: np.ndarray
    when_label: str
    angle_deg: Optional[float]
    rgb_pos_sec: Optional[float]
    vec_pair: Optional[list]
    depth_frame: Optional[dict]
    rgb_size: Tuple[int, int]
    median_k: int = 5


DrawerFn = Callable[[VizContext], None]
_CUSTOM_DRAWERS: Dict[str, DrawerFn] = {}


def register_drawer(rom_name: str, fn: DrawerFn) -> None:
    """
    Register a function that draws additional, ROM-specific visuals on a panel.
    Call at import time from your drawers_* modules.
    """
    _CUSTOM_DRAWERS[rom_name] = fn


def draw_for_rom(rom_name: str, ctx: VizContext) -> None:
    """
    Invoke the registered drawer for `rom_name` if one exists.
    """
    fn = _CUSTOM_DRAWERS.get(rom_name, None)
    if fn is not None:
        fn(ctx)
