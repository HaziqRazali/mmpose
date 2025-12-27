# utils/pcd.py
# Build dense point clouds (all valid pixels) from iPad depth + RGB.
# Also provides helpers to compute 3D keypoints from 2D kpts using the same depth.

from typing import Optional, Tuple, List, Dict
import cv2
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


def _depth_to_xyz_all(depth_map: np.ndarray, rgb_frame_bgr: np.ndarray,
                      fx: float, fy: float, ox: float, oy: float, dh: float, dw: float,
                      max_depth: Optional[float] = None) -> np.ndarray:
    
    z = depth_map.astype(np.float32)

    # Generate per-pixel coordinate grid (depth image space)
    x, y = np.meshgrid(np.arange(dw), np.arange(dh))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    if 0:
        print("_depth_to_xyz_all")
        u = int(round(200.60452270507812))  # x coordinate (width)
        v = int(round(316.62603759765625))  # y coordinate (height)
        if 0 <= v < z.shape[0] and 0 <= u < z.shape[1]:
            Z = depth_map[v, u]
            X = (u - ox) * Z / fx
            Y = (v - oy) * Z / fy
            Z = Z / 4
            print(fx, fy, ox, oy)
            print(u, v)
            print(np.array([X, Y, Z]))
        print()

    # Mask out invalid or extreme depth values
    z_far   = 50000
    z_near  = 0
    valid_mask = ~np.isnan(z)  & (z < z_far) & (z > z_near)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]

    # Backproject valid pixels (u,v,Z) → 3D camera coordinates (X,Y,Z)
    x3d = (x - ox) * z / fx
    y3d = (y - oy) * z / fy
    z = z/4
    points = np.stack((x3d, y3d, z), axis=-1)  ## .reshape(-1, 3)

    # Sample RGB color for each valid 3D point
    # Depth map is half-res → RGB is full-res, so use *2 to align
    rgb_frame_rgb = cv2.cvtColor(rgb_frame_bgr, cv2.COLOR_BGR2RGB)
    colors = rgb_frame_rgb[y.astype(np.int32) * 2, x.astype(np.int32) * 2] / 255.0

    return points, colors

def _depth_to_xyz_all_old(depth: np.ndarray,
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
                      transform: Optional[np.ndarray] = None,
                      flip_yz: bool = False):
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

    depth_map = depth_frame['depth']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
    dh, dw = depth_frame['h'], depth_frame['w']

    # XYZ for all valid pixels
    XYZ, colors = _depth_to_xyz_all(depth_map, rgb_frame_bgr, fx, fy, ox, oy, dh, dw, max_depth=max_depth)  # [N,3]

    if XYZ.shape[0] == 0:
        # empty cloud
        pcd = o3d.geometry.PointCloud()
        return pcd

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XYZ.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # if flip_yz:
    #     #import numpy as np
    #     rot_z_180 = np.array([
    #         [-1,  0,  0, 0],   # flip X
    #         [ 0, -1,  0, 0],   # flip Y
    #         [ 0,  0,  1, 0],   # keep Z
    #         [ 0,  0,  0, 1],
    #     ], dtype=float)
    #     pcd.transform(rot_z_180)

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
    
    # Inputs and intrinsics
    rw, rh = rgb_size
    h_d, w_d = depth_frame['h'], depth_frame['w']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']

    depth_map = depth_frame['depth']

    # For each keypoint: map RGB→depth coords, sample median Z, backproject
    out: List[Optional[np.ndarray]] = []
    for j in range(kpts_xy.shape[0]):
        pt2 = kpts_xy[j][:2]
        if not np.isfinite(pt2).all():
            out.append(None); continue
        
        # Map the RGB-space keypoint to depth-space pixel coords (u_d, v_d)
        u_d, v_d = _rgbkpt_to_depth_xy(pt2[0], pt2[1], rw, rh, w_d, h_d)

        # Robust median depth around (u_d, v_d) from the converted depth_map
        Z = _median_depth_at(depth_map, u_d, v_d, k=median_k)
        if not (np.isfinite(Z) and Z > 0 and Z < 50000.0):
            out.append(None); continue

        # Backproject (u_d, v_d, Z) → (X, Y, Z) in camera coordinates
        P = _backproject(u_d, v_d, Z, fx, fy, ox, oy)
        out.append(P)

    if 1:
        print("keypoints3d_from_kxy")
        print(out[5], out[7])
        print()

    return out

# def keypoints3d_from_kxy(kpts_xy: np.ndarray,
#                          depth_frame: dict,
#                          rgb_size: Tuple[int, int],
#                          median_k: int = 5) -> List[Optional[np.ndarray]]:
#     """
#     Convert 2D keypoints [K,3] to 3D camera points using the SAME sampling rules
#     as geometry.angle3d_* helpers (median patch depth + backproject).
#     Returns list of length K with np.array([X,Y,Z], float32) or None.
#     """
#     rw, rh = rgb_size
#     h_d, w_d = depth_frame['h'], depth_frame['w']
#     fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
#     depth = depth_frame['depth']

#     #ys = [k[1] for k in kpts_xy]
#     #print("[DBG] Δv RGB-space? max(y)-min(y) =", max(ys)-min(ys))
#     out: List[Optional[np.ndarray]] = []
#     for j in range(kpts_xy.shape[0]):
#         pt2 = kpts_xy[j][:2]
#         if not np.isfinite(pt2).all():
#             out.append(None); continue
#         u_d, v_d = _rgbkpt_to_depth_xy(pt2[0], pt2[1], rw, rh, w_d, h_d)
#         Z = _median_depth_at(depth, u_d, v_d, k=median_k)
#         if not (np.isfinite(Z) and Z > 0):
#             out.append(None); continue
        
#         #print("[DBG] depth.shape (h,w) =", depth.shape)
#         #print("[DBG] header (w_d,h_d) =", w_d, h_d)
#         #print("[DBG] intrinsics fx,fy,ox,oy =", fx, fy, ox, oy)
#         P = _backproject(u_d, v_d, Z, fx, fy, ox, oy)
#         out.append(P)
#     return out

def make_offset_transform(dx: float = 0.25, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    """
    Build a simple 4x4 translation transform to display t2 offset in the same window.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([dx, dy, dz], dtype=np.float64)
    return T
