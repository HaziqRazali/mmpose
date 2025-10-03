# viz.py
# Generic visualization toolbox + optional per-ROM drawer hooks.

from typing import Optional, Tuple, Dict, Callable, Any
from dataclasses import dataclass
import numpy as np
import cv2


# ------------------------
# Text overlays & layout
# ------------------------

def overlay_text(frame: np.ndarray,
                 lines: list,
                 org: Tuple[int, int] = (20, 40),
                 scale: float = 0.8) -> np.ndarray:
    """
    Draws a simple left-aligned multi-line text block with light outline.
    """
    y = org[1]
    for line in lines:
        cv2.putText(frame, line, (org[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (org[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(28 * scale)
    return frame


def annotate_panel(frame_rgb: np.ndarray,
                   rom_name: str,
                   mode_used: str,
                   when_label: str,
                   angle_deg: Optional[float],
                   rgb_pos_sec: Optional[float]) -> np.ndarray:
    """
    Standard header used by the coordinator. Keep this minimal so ROM-specific drawers can add extras.
    """
    lines = [when_label,
             f"Mode: {mode_used}",
             f"ROM: {rom_name}"]
    if angle_deg is not None:
        lines.append(f"Angle: {angle_deg:.2f} deg")
    lines.append(f"RGB actual: {rgb_pos_sec:.3f}s" if rgb_pos_sec is not None else "RGB actual: n/a")
    return overlay_text(frame_rgb, lines)


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
