# drawers_segments.py
# Minimal overlays that highlight specific limb segments used by some ROMs.
# These are purely visual (don’t affect measurements).

import numpy as np
import cv2
from .viz import VizContext, register_drawer

_SKELETON_EDGES = [
    (5,7), (7,9), (6,8), (8,10),          # arms
    (11,13), (13,15), (12,14), (14,16),   # legs
    (5,6), (11,12), (5,11), (6,12),       # shoulders/hips
    # add head/feet connections if you want, e.g. (0,5), (0,6), (15,17), (16,18)
]

def _draw_segment(ctx: VizContext, j0: int, j1: int, color=(0, 255, 255), radius=5, thickness=3):
    if ctx.kpts_xy.shape[0] <= max(j0, j1):
        return
    p = ctx.kpts_xy[j0][:2]
    q = ctx.kpts_xy[j1][:2]
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        return
    cv2.line(ctx.frame, tuple(np.int32(p)), tuple(np.int32(q)), color, thickness, cv2.LINE_AA)
    cv2.circle(ctx.frame, tuple(np.int32(p)), radius, color, -1, cv2.LINE_AA)
    cv2.circle(ctx.frame, tuple(np.int32(q)), radius, color, -1, cv2.LINE_AA)

# ---- drawers ----

def _drawer_left_upper_arm(ctx: VizContext):
    # Shoulder(5) -> Elbow(7)
    _draw_segment(ctx, 5, 7, color=(0, 255, 255), thickness=3)  # yellow

def _drawer_right_upper_arm(ctx: VizContext):
    # Shoulder(6) -> Elbow(8)
    _draw_segment(ctx, 6, 8, color=(0, 255, 255), thickness=3)  # yellow

def _drawer_left_forearm(ctx: VizContext):
    # Elbow(7) -> Wrist(9)
    _draw_segment(ctx, 7, 9, color=(0, 200, 0), thickness=3)    # green

def _drawer_right_forearm(ctx: VizContext):
    # Elbow(8) -> Wrist(10)
    _draw_segment(ctx, 8, 10, color=(0, 200, 0), thickness=3)   # green

def _drawer_full_skeleton(ctx: VizContext):
    for j0, j1 in _SKELETON_EDGES:
        _draw_segment(ctx, j0, j1, color=(0, 255, 255), thickness=3)

# ---- registrations ----
# Highlight upper arm for shoulder-related ROMs (flexion/extension/abduction)
register_drawer("left_shoulder_flexion", _drawer_left_upper_arm)
register_drawer("left_shoulder_extension", _drawer_left_upper_arm)
register_drawer("left_shoulder_abduction", _drawer_left_upper_arm)

register_drawer("right_shoulder_flexion", _drawer_right_upper_arm)
register_drawer("right_shoulder_extension", _drawer_right_upper_arm)
register_drawer("right_shoulder_abduction", _drawer_right_upper_arm)

register_drawer("skeleton", _drawer_full_skeleton)

# If you want, elbows can also highlight the forearm segment:
#register_drawer("left_elbow_flexion", _drawer_left_forearm)
#register_drawer("right_elbow_flexion", _drawer_right_forearm)
