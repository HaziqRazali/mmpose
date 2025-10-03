# utils/drawers3d_segments.py
# 3D counterparts of segment highlight overlays from drawers_segments.py

import numpy as np
from .viz3d import VizContext3D, register_drawer_3d, _make_lineset

# Colors (match 2D intent)
_YELLOW = (1.0, 1.0, 0.0)
_GREEN  = (0.0, 0.78, 0.0)

def _add_segment(ctx: VizContext3D, j0: int, j1: int, color):
    if len(ctx.kpts3d) <= max(j0, j1):
        return
    P = ctx.kpts3d[j0]
    Q = ctx.kpts3d[j1]
    if P is None or Q is None:
        return
    if not (np.isfinite(P).all() and np.isfinite(Q).all()):
        return
    pts = np.stack([P, Q], axis=0)
    ls = _make_lineset(pts, [(0, 1)], color)
    ctx.overlays.append(ls)


# ---- drawers ----

def _drawer_left_upper_arm_3d(ctx: VizContext3D):
    # Shoulder(5) -> Elbow(7)
    _add_segment(ctx, 5, 7, _YELLOW)

def _drawer_right_upper_arm_3d(ctx: VizContext3D):
    # Shoulder(6) -> Elbow(8)
    _add_segment(ctx, 6, 8, _YELLOW)

def _drawer_left_forearm_3d(ctx: VizContext3D):
    # Elbow(7) -> Wrist(9)
    _add_segment(ctx, 7, 9, _GREEN)

def _drawer_right_forearm_3d(ctx: VizContext3D):
    # Elbow(8) -> Wrist(10)
    _add_segment(ctx, 8, 10, _GREEN)


# ---- registrations ----
# Mirror the 2D registrations so behavior remains consistent.

# Shoulder-related ROMs (upper arm highlight)
register_drawer_3d("left_shoulder_flexion",    _drawer_left_upper_arm_3d)
register_drawer_3d("left_shoulder_extension",  _drawer_left_upper_arm_3d)
register_drawer_3d("left_shoulder_abduction",  _drawer_left_upper_arm_3d)

register_drawer_3d("right_shoulder_flexion",   _drawer_right_upper_arm_3d)
register_drawer_3d("right_shoulder_extension", _drawer_right_upper_arm_3d)
register_drawer_3d("right_shoulder_abduction", _drawer_right_upper_arm_3d)

# Elbow ROMs (forearm highlight)
register_drawer_3d("left_elbow_flexion",  _drawer_left_forearm_3d)
register_drawer_3d("right_elbow_flexion", _drawer_right_forearm_3d)
