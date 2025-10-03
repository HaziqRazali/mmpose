# utils/drawers3d_internal_rotation.py
# 3D visualization for left_shoulder_internal_rotation:
# - Draw torso normal arrow anchored at left shoulder
# - Draw forearm arrow (LE -> LW)
# Mirrors the intent of drawers_internal_rotation.py (2D) but in 3D.

import numpy as np
from .viz3d import VizContext3D, register_drawer_3d, _make_lineset

_BLUE  = (0.0, 0.0, 1.0)
_GREEN = (0.0, 0.78, 0.0)

def _vec_ok(v):
    return (v is not None) and np.all(np.isfinite(v))


def _drawer_shoulder_internal_rotation_3d(ctx: VizContext3D):
    # Indices (COCO-WholeBody-like as in your codebase):
    LS = ctx.kpts3d[5]   if len(ctx.kpts3d) > 5  else None
    RS = ctx.kpts3d[6]   if len(ctx.kpts3d) > 6  else None
    LH = ctx.kpts3d[11]  if len(ctx.kpts3d) > 11 else None
    RH = ctx.kpts3d[12]  if len(ctx.kpts3d) > 12 else None
    LE = ctx.kpts3d[7]   if len(ctx.kpts3d) > 7  else None
    LW = ctx.kpts3d[9]   if len(ctx.kpts3d) > 9  else None

    if not all(_vec_ok(p) for p in (LS, RS)) or (not _vec_ok(LE)) or (not _vec_ok(LW)):
        return

    geoms = []

    # Torso plane normal n = cross(RS-LS, RH-LH) if both hips are valid; else fall back cross with whichever hip is present
    if _vec_ok(LH) and _vec_ok(RH):
        v_s = RS - LS
        v_h = RH - LH
        n = np.cross(v_s, v_h)
    else:
        hip = RH if _vec_ok(RH) else (LH if _vec_ok(LH) else None)
        if not _vec_ok(hip):
            return
        n = np.cross(RS - LS, hip - LS)

    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return
    n = n / norm

    # Arrow length based on shoulder width
    shoulder_len = np.linalg.norm(RS - LS)
    L = 0.5 * shoulder_len if np.isfinite(shoulder_len) and (shoulder_len > 1e-6) else 0.2

    # Build torso-normal line (blue): LS -> LS + L*n
    tip = LS + L * n
    pts = np.stack([LS, tip], axis=0)
    geoms.append(_make_lineset(pts, [(0, 1)], _BLUE))

    # Forearm arrow (green): LE -> LW
    if _vec_ok(LE) and _vec_ok(LW):
        pts2 = np.stack([LE, LW], axis=0)
        geoms.append(_make_lineset(pts2, [(0, 1)], _GREEN))

    ctx.overlays.extend(geoms)

# Register at import time
register_drawer_3d("left_shoulder_internal_rotation", _drawer_shoulder_internal_rotation_3d)
