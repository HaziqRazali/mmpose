import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_intrinsics_from_depth(depth_img):
    """Return intrinsics dict from a RealSense depth frame, or None if unavailable."""
    try:
        depth_intrin = depth_img.profile.as_video_stream_profile().get_intrinsics()
        return {
            "fx": depth_intrin.fx,
            "fy": depth_intrin.fy,
            "ppx": depth_intrin.ppx,
            "ppy": depth_intrin.ppy,
            "depth_scale": 0.001  # RealSense default
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

def compute_joint_xyz_for_person(person_kpts, person_vis, depth_img, intrin_dict, kpt_thr):
    """Compute per-joint XYZ (meters) for one person. NaN where invalid."""
    joint_xyz = []
    H = depth_img.height
    W = depth_img.width
    for (x, y), v in zip(person_kpts, person_vis):
        if v > kpt_thr and 0 <= int(x) < W and 0 <= int(y) < H:
            depth_val = depth_img.get_distance(int(x), int(y))
            if depth_val and depth_val > 0:
                joint_xyz.append(deproject(depth_val, x, y, intrin_dict))
            else:
                joint_xyz.append([np.nan, np.nan, np.nan])
        else:
            joint_xyz.append([np.nan, np.nan, np.nan])
    return np.array(joint_xyz)


def compute_3d_skeletons(keypoints, visibility, depth_img, intrin_dict, kpt_thr):
    """
    Vector over people. Returns a list of (J,3) arrays (NaNs where invalid).
    keypoints: [N, J, 2], visibility: [N, J]
    """
    out = []
    for person_kpts, person_vis in zip(keypoints, visibility):
        out.append(compute_joint_xyz_for_person(
            person_kpts, person_vis, depth_img, intrin_dict, kpt_thr))
    return out

def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale.'''
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
    fig.clf()  # Clear only the current figure (preserve window)
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
    plt.pause(0.001)  # Refresh the plot without blocking

def angle(p_a, p_b, p_c):
    """
    Angle at vertex B formed by segments A-B and C-B (degrees).

    Each of p_a, p_b, p_c can be:
      - a single point: shape (D,)
      - a list/array of points: shape (K, D) -> averaged (NaN-safe) to (D,)

    Works for 2D, 3D, ... nD points.
    """

    def _avg_point(p):
        arr = np.asarray(p, dtype=float)
        if arr.ndim == 1:
            return arr
        elif arr.ndim == 2:
            # average multiple points; ignores NaNs if present
            return np.nanmean(arr, axis=0)
        else:
            raise ValueError("Each input must be shape (D,) or (K, D).")

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
    raw = np.degrees(np.arccos(cosang))  # 0 = aligned, 180 = opposite
    return raw
