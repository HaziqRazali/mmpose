# utils/pcd.py
# Build dense point clouds (all valid pixels) from iPad depth + RGB.
# Also provides helpers to compute 3D keypoints from 2D kpts using the same depth.

from typing import Optional, Tuple, List, Dict
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    o3d = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

from .geometry import _rgbkpt_to_depth_xy, _median_depth_at, _backproject


def _require_o3d():
    if o3d is None:
        raise ImportError(
            "Open3D is required for utils.pcd but is not available. "
            f"Original import error: {_IMPORT_ERR}"
        )


def _depth_to_xyz_all(depth: np.ndarray,
                      fx: float, fy: float, ox: float, oy: float,
                      max_depth: Optional[float] = None) -> np.ndarray:
    """
    Back-project ALL valid depth pixels (no decimation).
    Returns array [N, 3] in camera coordinates (OpenCV convention).
    """
    h, w = depth.shape[:2]
    # Build a grid of pixel coordinates
    us = np.arange(w, dtype=np.float32)
    vs = np.arange(h, dtype=np.float32)
    U, V = np.meshgrid(us, vs)                           # [h,w]

    Z = depth.astype(np.float32)                         # [h,w]
    if max_depth is not None and np.isfinite(max_depth):
        Z = np.where((Z > 0) & (Z <= max_depth), Z, np.nan)

    valid = np.isfinite(Z) & (Z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    Uv = U[valid]; Vv = V[valid]; Zv = Z[valid]
    X = (Uv - ox) * Zv / fx
    Y = (Vv - oy) * Zv / fy

    XYZ = np.stack([X, Y, Zv], axis=1)                   # [N,3]
    return XYZ


def _rgb_for_depth_pixels(rgb_bgr: np.ndarray,
                          depth_w: int, depth_h: int) -> np.ndarray:
    """
    Map each depth pixel to the corresponding RGB pixel and fetch color.
    Assumes depth and RGB are already spatially aligned via simple scaling
    (same approach used in geometry._rgbkpt_to_depth_xy).
    Input rgb is BGR uint8; returns RGB float in [0,1] for ALL pixels (not just valid).
    """
    # Compute RGB->depth scaling, then invert mapping for color sampling.
    rgb_h, rgb_w = rgb_bgr.shape[:2]
    su = rgb_w / float(depth_w)
    sv = rgb_h / float(depth_h)

    # Build grid in depth space, map to RGB
    us = np.arange(depth_w, dtype=np.float32)
    vs = np.arange(depth_h, dtype=np.float32)
    U, V = np.meshgrid(us, vs)  # [h,w]
    u_rgb = U * su
    v_rgb = V * sv

    # Sample with nearest neighbor
    u_nn = np.clip(np.round(u_rgb).astype(np.int32), 0, rgb_w - 1)
    v_nn = np.clip(np.round(v_rgb).astype(np.int32), 0, rgb_h - 1)

    # BGR -> RGB, normalize to [0,1]
    rgb_samp_bgr = rgb_bgr[v_nn, u_nn, :]                               # [h,w,3]
    rgb_samp_rgb = rgb_samp_bgr[..., ::-1].astype(np.float32) / 255.0   # [h,w,3]
    return rgb_samp_rgb


def build_point_cloud(depth_frame: dict,
                      rgb_frame_bgr: Optional[np.ndarray],
                      rgb_size: Tuple[int, int],
                      max_depth: Optional[float] = None,
                      voxel_size: Optional[float] = None,
                      tint: Optional[Tuple[float, float, float]] = None,
                      transform: Optional[np.ndarray] = None):
    """
    Create an Open3D point cloud from the given depth frame and aligned RGB.
    - depth_frame: dict {depth[h,w], fx,fy,ox,oy, h,w, ...}
    - rgb_frame_bgr: optional BGR image for coloring. If None, grayscale depth colormap is used.
    - rgb_size: (W, H) of the RGB frame used during pose (needed for depth<->rgb mapping parity).
    - max_depth: clip depth beyond this (meters).
    - voxel_size: optional downsample voxel size (meters); affects RENDER ONLY.
    - tint: optional (r,g,b) multiplier applied to colors for visual distinction (e.g., for t2).
    - transform: optional 4x4 pose to apply to the cloud (e.g., to offset t2).
    Returns: open3d.geometry.PointCloud (possibly voxel-downsampled).
    """
    _require_o3d()

    depth = depth_frame['depth']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
    dh, dw = depth_frame['h'], depth_frame['w']

    # XYZ for all valid pixels
    XYZ = _depth_to_xyz_all(depth, fx, fy, ox, oy, max_depth=max_depth)  # [N,3]

    if XYZ.shape[0] == 0:
        # empty cloud
        pcd = o3d.geometry.PointCloud()
        return pcd

    # Colors for all pixels: build a per-pixel color image, then mask valid indices
    if rgb_frame_bgr is not None:
        colors_full = _rgb_for_depth_pixels(rgb_frame_bgr, dw, dh).reshape(-1, 3)  # [h*w, 3]
    else:
        # fallback grayscale based on Z
        d = depth.copy()
        mask = ~np.isfinite(d)
        v = d[~mask]
        if v.size == 0:
            gray = np.zeros((dh * dw, 3), dtype=np.float32)
        else:
            mn = np.nanpercentile(v, 2.0)
            mx = np.nanpercentile(v, 98.0)
            if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                mn, mx = np.nanmin(v), np.nanmax(v)
            d[mask] = mn
            g = np.clip((d - mn) / max(mx - mn, 1e-6), 0, 1).astype(np.float32)
            gray = np.stack([g, g, g], axis=2).reshape(-1, 3)  # [h*w,3]
        colors_full = gray

    # Mask with valid depth (same as XYZ mask); re-derive to match XYZ count
    valid = np.isfinite(depth) & (depth > 0)
    colors = colors_full[valid.reshape(-1), :]  # [N,3]

    # Optional tint (e.g., dim/tint t2 cloud)
    if tint is not None:
        colors = np.clip(colors * np.array(tint, dtype=np.float32)[None, :], 0.0, 1.0)

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XYZ.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # Optional transform (e.g., offset t2 along +X)
    if transform is not None:
        assert transform.shape == (4, 4)
        pcd.transform(transform)

    # Optional voxel downsample (viewer-side)
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return pcd


def keypoints3d_from_kxy(kpts_xy: np.ndarray,
                         depth_frame: dict,
                         rgb_size: Tuple[int, int],
                         median_k: int = 5) -> List[Optional[np.ndarray]]:
    """
    Convert 2D keypoints [K,3] to 3D camera points using the SAME sampling rules
    as geometry.angle3d_* helpers (median patch depth + backproject).
    Returns list of length K with np.array([X,Y,Z], float32) or None.
    """
    rw, rh = rgb_size
    h_d, w_d = depth_frame['h'], depth_frame['w']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
    depth = depth_frame['depth']

    #ys = [k[1] for k in kpts_xy]
    #print("[DBG] Δv RGB-space? max(y)-min(y) =", max(ys)-min(ys))
    out: List[Optional[np.ndarray]] = []
    for j in range(kpts_xy.shape[0]):
        pt2 = kpts_xy[j][:2]
        if not np.isfinite(pt2).all():
            out.append(None); continue
        u_d, v_d = _rgbkpt_to_depth_xy(pt2[0], pt2[1], rw, rh, w_d, h_d)
        Z = _median_depth_at(depth, u_d, v_d, k=median_k)
        if not (np.isfinite(Z) and Z > 0):
            out.append(None); continue
        
        #print("[DBG] depth.shape (h,w) =", depth.shape)
        #print("[DBG] header (w_d,h_d) =", w_d, h_d)
        #print("[DBG] intrinsics fx,fy,ox,oy =", fx, fy, ox, oy)
        P = _backproject(u_d, v_d, Z, fx, fy, ox, oy)
        out.append(P)
    return out


def make_offset_transform(dx: float = 0.25, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    """
    Build a simple 4x4 translation transform to display t2 offset in the same window.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([dx, dy, dz], dtype=np.float64)
    return T
