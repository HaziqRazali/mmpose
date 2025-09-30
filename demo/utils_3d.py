import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional, Tuple
import numpy as np

def extract_intrinsics_from_depth(depth_img):
    """Return intrinsics dict from a RealSense depth frame, or None if unavailable."""
    try:
        depth_intrin = depth_img.profile.as_video_stream_profile().get_intrinsics()
        return {
            "fx": depth_intrin.fx,
            "fy": depth_intrin.fy,
            "ppx": depth_intrin.ppx,
            "ppy": depth_intrin.ppy,
            "depth_scale": 0.001  # RealSense default meters
        }
    except Exception as e:
        print(f"[!] Cannot extract intrinsics for 3D viz: {e}")
        return None


def deproject(depth_val, x, y, intrin):
    fx, fy = intrin["fx"], intrin["fy"]
    ppx, ppy = intrin["ppx"], intrin["ppy"]
    X = (x - ppx) * depth_val / fx
    Y = (y - ppy) * depth_val / fy
    Z = depth_val
    return [X, Y, Z]


def _reduce_depth_window(depth_img, cx, cy, w, ignore_zeros, reducer, dmin, dmax):
    """
    Sample a w×w window around integer center (cx,cy).
    Returns a single scalar depth in meters or np.nan if none valid.
    """
    H, W = depth_img.height, depth_img.width
    r = int(max(0, (w // 2)))
    xs = range(max(0, cx - r), min(W, cx + r + 1))
    ys = range(max(0, cy - r), min(H, cy + r + 1))
    vals = []
    for y in ys:
        for x in xs:
            try:
                z = float(depth_img.get_distance(int(x), int(y)))
            except Exception:
                z = 0.0
            if ignore_zeros and (z <= 0.0):
                continue
            if not np.isfinite(z) or z <= 0.0:
                continue
            if z < dmin or z > dmax:
                continue
            vals.append(z)
    if not vals:
        return np.nan
    arr = np.array(vals, dtype=float)
    if reducer == "mean":
        return float(np.mean(arr))
    # default median
    return float(np.median(arr))


def compute_joint_xyz_for_person(
    person_kpts,
    person_vis,
    depth_img,
    intrin_dict,
    kpt_thr,
    window_size=3,
    reducer="median",
    ignore_zeros=True,
    depth_min=0.1,
    depth_max=4.0,
):
    """
    Compute per-joint XYZ (meters) for one person. NaN where invalid.
    Depth is spatially denoised by reducing a window around each keypoint.
    """
    joint_xyz = []
    H = depth_img.height
    W = depth_img.width

    w = int(max(1, window_size))
    if w % 2 == 0:
        w += 1  # enforce odd

    for (x, y), v in zip(person_kpts, person_vis):
        if v > kpt_thr and 0 <= int(x) < W and 0 <= int(y) < H:
            z = _reduce_depth_window(
                depth_img,
                int(round(x)),
                int(round(y)),
                w=w,
                ignore_zeros=bool(ignore_zeros),
                reducer=str(reducer).lower(),
                dmin=float(depth_min),
                dmax=float(depth_max),
            )
            if np.isfinite(z) and z > 0.0:
                joint_xyz.append(deproject(z, x, y, intrin_dict))
            else:
                joint_xyz.append([np.nan, np.nan, np.nan])
        else:
            joint_xyz.append([np.nan, np.nan, np.nan])
    return np.array(joint_xyz)


def compute_3d_skeletons(keypoints, visibility, depth_img, intrin_dict, kpt_thr,
                         window_size=3, reducer="median", ignore_zeros=True,
                         depth_min=0.1, depth_max=4.0):
    """Vector over people. Returns a list of (J,3) arrays (NaNs where invalid)."""
    out = []
    for person_kpts, person_vis in zip(keypoints, visibility):
        out.append(compute_joint_xyz_for_person(
            person_kpts, person_vis, depth_img, intrin_dict, kpt_thr,
            window_size=window_size, reducer=reducer, ignore_zeros=ignore_zeros,
            depth_min=depth_min, depth_max=depth_max))
    return out


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_3d_skeletons(joint_xyz_list, skeleton, show_kpt_subset, kpt_thr):
    """Matplotlib 3D scatter/lines for all people."""
    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Skeleton")

    for joint_xyz in joint_xyz_list:
        # points
        for j, (x, y, z) in enumerate(joint_xyz):
            if not np.isnan(z) and j in show_kpt_subset:
                ax.scatter(x, y, z, c='green', s=10)

        # links
        for idx1, idx2 in skeleton:
            if idx1 < len(joint_xyz) and idx2 < len(joint_xyz):
                pt1, pt2 = joint_xyz[idx1], joint_xyz[idx2]
                if not np.any(np.isnan(pt1)) and not np.any(np.isnan(pt2)):
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='blue', linewidth=2)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 3.0)
    set_axes_equal(ax)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=-80, azim=-90)
    plt.pause(0.001)


def angle(p_a, p_b, p_c):
    """
    Angle at vertex B formed by segments A-B and C-B (degrees).
    Each of p_a, p_b, p_c: shape (D,) or (K,D) -> averaged to (D,).
    Works for 2D, 3D, ... nD points.
    """
    def _avg_point(p):
        arr = np.asarray(p, dtype=float)
        if arr.ndim == 1:
            return arr
        elif arr.ndim == 2:
            return np.nanmean(arr, axis=0)
        else:
            raise ValueError("Each input must be shape (D,) or (K,D).")

    A = _avg_point(p_a)
    B = _avg_point(p_b)
    C = _avg_point(p_c)

    if np.any(np.isnan(A)) or np.any(np.isnan(B)) or np.any(np.isnan(C)):
        return np.nan

    u = A - B
    v = C - B
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if not (np.isfinite(nu) and np.isfinite(nv)) or nu < 1e-6 or nv < 1e-6:
        return np.nan

    u /= nu
    v /= nv
    cosang = np.clip(np.dot(u, v), -1.0, 1.0)
    raw = np.degrees(np.arccos(cosang))
    return raw


# -------- Vector-pair helpers (no triplets anywhere) --------

def resolve_point(points, spec):
    """
    Fetch a single point from an int id or list[int] spec.
    points: np.ndarray (J,2) or (J,3).
    Returns a (D,) array. NaN-safe mean for lists. np.nan vector if invalid.
    """
    arr = np.asarray(points, dtype=float)
    D = arr.shape[1] if arr.ndim == 2 else 0
    if D not in (2, 3):
        return np.array([np.nan, np.nan, np.nan]) if D == 3 else np.array([np.nan, np.nan])

    def _one(idx):
        try:
            idx = int(idx)
            if idx < 0 or idx >= arr.shape[0]:
                return None
            p = arr[idx]
            if not np.all(np.isfinite(p)):
                return None
            return p
        except Exception:
            return None

    if isinstance(spec, (list, tuple)):
        pts = [p for p in (_one(i) for i in spec) if p is not None]
        if not pts:
            return np.full((D,), np.nan, dtype=float)
        return np.nanmean(np.vstack(pts), axis=0)
    else:
        p = _one(spec)
        if p is None:
            return np.full((D,), np.nan, dtype=float)
        return p


def angle_from_vecpair(points, pair):
    """
    Compute the angle between two segments defined by a vector-pair.
    pair = [[P0,P1],[Q0,Q1]], each an int or list[int].
    points: (J,2) or (J,3).
    Returns degrees in [0,180] or np.nan.
    """
    try:
        (P0, P1), (Q0, Q1) = pair
    except Exception:
        return np.nan

    A = resolve_point(points, P1)  # map to angle(A,B,C) form: B is shared origin
    B = resolve_point(points, P0)
    C = resolve_point(points, Q1)

    if A.shape != B.shape or B.shape != C.shape:
        return np.nan
    return angle(A, B, C)

# =========================
# Offline depth + geometry
# =========================

def depth16_to_meters(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    """
    Convert a 16-bit depth image (uint16) to meters.
    depth_scale is meters per unit (e.g., RealSense typical 0.001).
    Returns float32 meters with NaN for zeros.
    """
    if depth_raw is None:
        return None
    d = depth_raw.astype(np.float32) * float(depth_scale)
    d[d <= 0.0] = np.nan
    return d


def meters_to_depth16(depth_m: np.ndarray, depth_scale: float) -> np.ndarray:
    """
    Convert meters to uint16 depth given meters-per-unit scale.
    Values <=0 or NaN become 0.
    """
    if depth_m is None:
        return None
    x = np.copy(depth_m).astype(np.float32)
    x[~np.isfinite(x)] = 0.0
    x[x <= 0.0] = 0.0
    q = np.round(x / float(depth_scale)).astype(np.uint16)
    return q


def backproject_pixel_to_camera(u: float, v: float, depth_m: float,
                                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Back-project a pixel (u,v) with metric depth to camera coords (meters).
    Intrinsics in pixels. Returns (3,) float32 [X,Y,Z].
    """
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float32)


def project_camera_to_pixel(X: float, Y: float, Z: float,
                            fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float]:
    """
    Project a 3D camera-space point (meters) to pixel coords.
    """
    if not np.isfinite(Z) or Z <= 1e-9:
        return (np.nan, np.nan)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return (float(u), float(v))


def build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Return a 3x3 intrinsics matrix.
    """
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def parse_intrinsics_dict(intrin: dict) -> Tuple[float, float, float, float]:
    """
    Accepts dicts like {'fx':..,'fy':..,'cx':..,'cy':..} or
    RealSense-style {'ppx':..,'ppy':..,'fx':..,'fy':..}. Returns (fx,fy,cx,cy).
    """
    fx = float(intrin.get('fx'))
    fy = float(intrin.get('fy'))
    cx = float(intrin.get('cx', intrin.get('ppx')))
    cy = float(intrin.get('cy', intrin.get('ppy')))
    return fx, fy, cx, cy


def angle_from_vecpair(joints_xyz: np.ndarray, vecpair) -> float:
    """
    Compute angle (deg) between vectors (P1-P0) and (Q1-P0) from joints_xyz (K,3).
    vecpair = ((P0,P1),(Q0,Q1)), where each index may be an int or an iterable of ints
    to be averaged. Returns np.nan if any endpoint invalid.
    """
    if joints_xyz is None or joints_xyz.ndim != 2 or joints_xyz.shape[1] != 3:
        return np.nan

    def _pt(spec):
        if isinstance(spec, (list, tuple)):
            pts = []
            for i in spec:
                j = int(i)
                if j < 0 or j >= joints_xyz.shape[0]:
                    return None
                p = joints_xyz[j, :]
                if not np.all(np.isfinite(p)):
                    return None
                pts.append(p)
            if not pts:
                return None
            return np.nanmean(np.vstack(pts), axis=0)
        j = int(spec)
        if j < 0 or j >= joints_xyz.shape[0]:
            return None
        p = joints_xyz[j, :]
        if not np.all(np.isfinite(p)):
            return None
        return p

    try:
        (P0, P1), (Q0, Q1) = vecpair
    except Exception:
        return np.nan

    B = _pt(P0)
    A = _pt(P1)
    C = _pt(Q1)
    if B is None or A is None or C is None:
        return np.nan

    u = A - B
    v = C - B
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu <= 1e-9 or nv <= 1e-9:
        return np.nan

    u /= nu
    v /= nv
    cosang = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))
