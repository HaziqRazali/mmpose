import numpy as np
import math

# ---------- helpers ----------
def _avg_point(kpts_xy, spec):
    """Return avg point of keypoints for int index or list of indices."""
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

# ---------- 2D angles ----------
def angle2d_from_vecpair(kpts_xy, vec_pair):
    """Angle in 2D between two vecs defined by keypoints."""
    #print(vec_pair)
    (P0, P1), (Q0, Q1) = vec_pair
    p0 = _avg_point(kpts_xy, P0); p1 = _avg_point(kpts_xy, P1)
    q0 = _avg_point(kpts_xy, Q0); q1 = _avg_point(kpts_xy, Q1)
    if any(v is None for v in (p0, p1, q0, q1)):
        return None
    v1 = np.array(p1, float) - np.array(p0, float)
    v2 = np.array(q1, float) - np.array(q0, float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    v1 /= n1; v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def angle2d_between_segments_across_frames(k1, k2, j_sh=5, j_el=7):
    """Angle between same 2D segment at t1 vs t2 (shoulder->elbow)."""
    p1 = k1[j_sh][:2]; q1 = k1[j_el][:2]
    p2 = k2[j_sh][:2]; q2 = k2[j_el][:2]
    if not (np.isfinite(p1).all() and np.isfinite(q1).all()
            and np.isfinite(p2).all() and np.isfinite(q2).all()):
        return None
    v1 = q1 - p1; v2 = q2 - p2
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    v1 /= n1; v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))

# ---------- depth utilities ----------
def _median_depth_at(depth: np.ndarray, u: float, v: float, k: int = 5) -> float:
    h, w = depth.shape[:2]
    x = int(round(u)); y = int(round(v))
    r = k // 2
    x0 = max(0, x - r); x1 = min(w, x + r + 1)
    y0 = max(0, y - r); y1 = min(h, y + r + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch)]
    if vals.size < 3:          # was: < 50% of patch; this is more permissive/robust
        return np.nan
    return float(np.median(vals))

def _rgbkpt_to_depth_xy(u_rgb, v_rgb, rgb_w, rgb_h, depth_w, depth_h):

    """Map pixel from RGB space to depth space."""
    # If RGB is 1280 x 720 and depth is 640 x 360, it just scales everything by 0.5 in both axes. Verifiable via --show-3d

    su = depth_w / float(rgb_w); 
    sv = depth_h / float(rgb_h)
    #print(su, depth_w, rgb_w)
    #print(sv, depth_h, rgb_h)
    #sys.exit()
    return u_rgb * su, v_rgb * sv

def _backproject(u, v, Z, fx, fy, ox, oy):
    """Backproject depth pixel (u,v,Z) into 3D camera space."""
    if not np.isfinite(Z) or Z <= 0:
        return None
    X = (u - ox) * Z / fx
    Y = (v - oy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

def _project(P, fx, fy, ox, oy):
    """Project a 3D point P into 2D pixel coords."""
    Z = float(P[2]) if P is not None else 0.0
    if not np.isfinite(Z) or abs(Z) < 1e-6:
        Z = 1e-6
    u = ox + fx * (float(P[0]) / Z)
    v = oy + fy * (float(P[1]) / Z)
    return np.array([u, v], dtype=np.float32)

# ---------- 3D angles ----------
def angle3d_from_vecpair(kpts_xy_rgb, vec_pair, depth_frame, rgb_size, median_k=5):

    if depth_frame is None:
        return None, False

    """3D angle between vecs defined by keypoints, using depth backprojection."""
    h_d, w_d = depth_frame['h'], depth_frame['w']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
    depth = depth_frame['depth']
    rw, rh = rgb_size

    def pt3d(spec):
        pt2 = _avg_point(kpts_xy_rgb, spec)
        if pt2 is None: return None
        u_d, v_d = _rgbkpt_to_depth_xy(pt2[0], pt2[1], rw, rh, w_d, h_d)
        Z = _median_depth_at(depth, u_d, v_d, k=median_k)
        if not np.isfinite(Z) or Z <= 0: return None
        return _backproject(u_d, v_d, Z, fx, fy, ox, oy)

    (P0,P1),(Q0,Q1) = vec_pair
    A,B,C,D = pt3d(P0), pt3d(P1), pt3d(Q0), pt3d(Q1)
    if any(p is None for p in (A,B,C,D)):
        return None, False
    v1, v2 = B-A, D-C
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None, False
    v1 /= n1; v2 /= n2
    dot = float(np.clip(np.dot(v1,v2), -1.0,1.0))
    return math.degrees(math.acos(dot)), True

def angle3d_between_segments_across_frames(k1, k2, d1, d2, rgb_size, median_k=5, j_sh=5, j_el=7):
    """3D inter-frame angle for same segment (e.g. shoulder->elbow)."""
    if d1 is None or d2 is None:
        return None
    h1,w1,fx1,fy1,ox1,oy1 = d1['h'],d1['w'],d1['fx'],d1['fy'],d1['ox'],d1['oy']
    h2,w2,fx2,fy2,ox2,oy2 = d2['h'],d2['w'],d2['fx'],d2['fy'],d2['ox'],d2['oy']
    depth1, depth2 = d1['depth'], d2['depth']
    rw, rh = rgb_size

    def pt3d(k, j, dw, dh, depth, fx, fy, ox, oy):
        pt2 = k[j][:2]
        if not np.isfinite(pt2).all(): return None
        u,v = _rgbkpt_to_depth_xy(pt2[0],pt2[1],rw,rh,dw,dh)
        Z = _median_depth_at(depth,u,v,k=median_k)
        if not (np.isfinite(Z) and Z>0): return None
        return _backproject(u,v,Z,fx,fy,ox,oy)

    SH1, EL1 = pt3d(k1,j_sh,w1,h1,depth1,fx1,fy1,ox1,oy1), pt3d(k1,j_el,w1,h1,depth1,fx1,fy1,ox1,oy1)
    SH2, EL2 = pt3d(k2,j_sh,w2,h2,depth2,fx2,fy2,ox2,oy2), pt3d(k2,j_el,w2,h2,depth2,fx2,fy2,ox2,oy2)
    if any(p is None for p in (SH1, EL1, SH2, EL2)): 
        return None
    v1, v2 = EL1-SH1, EL2-SH2
    n1,n2 = np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6: return None
    v1/=n1; v2/=n2
    dot=float(np.clip(np.dot(v1,v2),-1.0,1.0))
    return math.degrees(math.acos(dot))

def angle3d_internal_rotation_left_simple(kpts_xy_rgb, depth_frame, rgb_size, median_k=5):
    """
    Approx internal/external rotation of LEFT shoulder.
    Torso normal (placeholder = [0,0,1]) vs forearm vector.
    """
    if depth_frame is None:
        return None
    h_d, w_d = depth_frame['h'], depth_frame['w']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
    depth = depth_frame['depth']
    rw, rh = rgb_size

    def pt3d(spec):
        pt2 = _avg_point(kpts_xy_rgb, spec)
        if pt2 is None: return None
        u_d,v_d = _rgbkpt_to_depth_xy(pt2[0],pt2[1],rw,rh,w_d,h_d)
        Z = _median_depth_at(depth,u_d,v_d,k=median_k)
        if not np.isfinite(Z) or Z <= 0: return None
        return _backproject(u_d,v_d,Z,fx,fy,ox,oy)

    LS, RS, LH, RH = pt3d(5), pt3d(6), pt3d(11), pt3d(12)
    LE, LW = pt3d(7), pt3d(9)
    if any(p is None for p in (LS,RS,LE,LW)): return None

    # Simplified: use global z-axis
    n = np.array([0,0,1],dtype=np.float32)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-6: return None
    n /= n_norm

    F = LW-LE
    f_norm = np.linalg.norm(F)
    if f_norm<1e-6: return None
    F /= f_norm

    dot=float(np.clip(np.dot(n,F),-1.0,1.0))
    return math.degrees(math.acos(dot))
