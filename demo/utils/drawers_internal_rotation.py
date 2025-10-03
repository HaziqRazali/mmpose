# drawers_internal_rotation.py
import numpy as np
import cv2
from .viz import VizContext, register_drawer

def _project(P, fx, fy, ox, oy):
    Z = float(P[2]) if P is not None else 0.0
    if not np.isfinite(Z) or abs(Z) < 1e-6:
        Z = 1e-6
    u = ox + fx * (float(P[0]) / Z)
    v = oy + fy * (float(P[1]) / Z)
    return np.array([u, v], dtype=np.float32)

def _median_depth_at(depth, u, v, k=5):
    h, w = depth.shape[:2]
    x = int(round(u)); y = int(round(v))
    r = k // 2
    x0 = max(0, x - r); x1 = min(w, x + r + 1)
    y0 = max(0, y - r); y1 = min(h, y + r + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch)]
    if vals.size < (patch.size * 0.5): return np.nan
    return float(np.median(vals))

def _rgb_to_depth_xy(u_rgb, v_rgb, rw, rh, dw, dh):
    return u_rgb * (dw / float(rw)), v_rgb * (dh / float(rh))

def _backproject(u, v, Z, fx, fy, ox, oy):
    if not np.isfinite(Z) or Z <= 0: return None
    X = (u - ox) * Z / fx
    Y = (v - oy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

def _avg_point(kpts_xy, spec):
    if isinstance(spec, (list, tuple)):
        pts = []
        for s in spec:
            p = _avg_point(kpts_xy, s)
            if p is not None: pts.append(p)
        if not pts: return None
        arr = np.array(pts, dtype=float)
        return arr.mean(axis=0)
    idx = int(spec)
    if idx < 0 or idx >= len(kpts_xy): return None
    return kpts_xy[idx][:2]

def _pt3d_and_2d(kxy, spec, depth_frame, rgb_size, median_k):
    rw, rh = rgb_size
    h_d, w_d = depth_frame['h'], depth_frame['w']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
    pt2 = _avg_point(kxy, spec)
    if pt2 is None: return None, None
    u_d, v_d = _rgb_to_depth_xy(pt2[0], pt2[1], rw, rh, w_d, h_d)
    Z = _median_depth_at(depth_frame['depth'], u_d, v_d, k=median_k)
    if not np.isfinite(Z) or Z <= 0: return None, None
    P3 = _backproject(u_d, v_d, Z, fx, fy, ox, oy)
    return P3, np.array([pt2[0], pt2[1]], dtype=np.float32)

def _drawer_shoulder_internal_rotation(ctx: VizContext):
    if ctx.depth_frame is None:  # need 3D
        return
    fx, fy, ox, oy = ctx.depth_frame['fx'], ctx.depth_frame['fy'], ctx.depth_frame['ox'], ctx.depth_frame['oy']

    LS3, LS2 = _pt3d_and_2d(ctx.kpts_xy, 5, ctx.depth_frame, ctx.rgb_size, ctx.median_k)
    RS3, RS2 = _pt3d_and_2d(ctx.kpts_xy, 6, ctx.depth_frame, ctx.rgb_size, ctx.median_k)
    LH3, _   = _pt3d_and_2d(ctx.kpts_xy, 11, ctx.depth_frame, ctx.rgb_size, ctx.median_k)
    RH3, _   = _pt3d_and_2d(ctx.kpts_xy, 12, ctx.depth_frame, ctx.rgb_size, ctx.median_k)
    LE3, LE2 = _pt3d_and_2d(ctx.kpts_xy, 7, ctx.depth_frame, ctx.rgb_size, ctx.median_k)
    LW3, LW2 = _pt3d_and_2d(ctx.kpts_xy, 9, ctx.depth_frame, ctx.rgb_size, ctx.median_k)
    if any(p is None for p in (LS3, RS3, LE2, LW2)):
        return

    # Torso plane normal
    if (LH3 is not None) and (RH3 is not None):
        v_s = RS3 - LS3
        v_h = RH3 - LH3
        n = np.cross(v_s, v_h)
    else:
        n = np.cross(RS3 - LS3, (RH3 if RH3 is not None else LH3) - LS3)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-6: return
    n = n / n_norm

    # Arrow length based on shoulder width
    shoulder_len = np.linalg.norm(RS3 - LS3)
    L = 0.5 * shoulder_len if np.isfinite(shoulder_len) and shoulder_len > 1e-6 else 0.2

    # Project torso-normal arrow (blue)
    base2 = _project(LS3, fx, fy, ox, oy)
    tip3  = LS3 + L * n
    tip2  = _project(tip3, fx, fy, ox, oy)
    if np.isfinite(base2).all() and np.isfinite(tip2).all():
        cv2.arrowedLine(ctx.frame, (int(base2[0]), int(base2[1])), (int(tip2[0]), int(tip2[1])),
                        (255,0,0), 2, tipLength=0.2)

    # Forearm arrow (green) in 2D for visual reference
    if np.isfinite(LE2).all() and np.isfinite(LW2).all():
        cv2.arrowedLine(ctx.frame, (int(LE2[0]), int(LE2[1])), (int(LW2[0]), int(LW2[1])),
                        (0,200,0), 2, tipLength=0.2)

# Register at import time
register_drawer("left_shoulder_internal_rotation", _drawer_shoulder_internal_rotation)
