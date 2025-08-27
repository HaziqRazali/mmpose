import os
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils_variables import left_right_body_kpt_ids, skeleton

def deproject(depth_value, x, y, intrinsics):
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]

    X = (x - ppx) * depth_value / fx
    Y = (y - ppy) * depth_value / fy
    Z = depth_value
    return [X, Y, Z]

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="Date in YYYYMMDD format")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp in HHMMSS format")
    args = parser.parse_args()

    date = args.date
    timestamp = args.timestamp

    color_img_path  = f"/home/haziq/mmpose/snapshots/image_{date}_{timestamp}.png"
    depth_img_path  = f"/home/haziq/mmpose/snapshots/depth_{date}_{timestamp}.png"
    intrinsic_path  = f"/home/haziq/mmpose/snapshots/intrinsics_{date}_{timestamp}.json"
    pred_path       = f"/home/haziq/mmpose/snapshots/pred_{date}_{timestamp}.json"

    color_img = cv2.imread(color_img_path)
    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
    pred_data = json.load(open(pred_path, "r"))
    intrinsics = json.load(open(intrinsic_path, "r"))

    depth_scale = intrinsics.get("depth_scale", 0.001)

    # Convert to numpy arrays
    keypoints   = np.array(pred_data["keypoints"])           # shape: [N, J, 2]
    visibility  = np.array(pred_data["keypoints_visible"])   # shape: [N, J]

    print(f"Num Persons: {len(keypoints)}")

    # ================== 2D SKELETON OVERLAY ==================
    # Loop over each person
    for person_kpts, person_vis in zip(keypoints, visibility):
        # Draw keypoints
        for joint_idx, ((x, y), v) in enumerate(zip(person_kpts, person_vis)):
            if v > 0.3 and joint_idx in left_right_body_kpt_ids:
                cv2.circle(color_img, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

        # Draw skeleton connections
        for idx1, idx2 in skeleton:
            if idx1 < len(person_kpts) and idx2 < len(person_kpts):
                x1, y1 = person_kpts[idx1]
                x2, y2 = person_kpts[idx2]
                v1 = person_vis[idx1]
                v2 = person_vis[idx2]

                if v1 > 0.3 and v2 > 0.3:
                    cv2.line(color_img, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0, 0), thickness=2)

    # Show image
    cv2.imshow("Keypoints", color_img)
    cv2.waitKey(0) 

    # ================== 3D SKELETON VISUALIZATION ==================
    plt.ion()  # Interactive mode on
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Skeleton")

    for person_kpts, person_vis in zip(keypoints, visibility):
        joint_xyz = []
        for (x, y), v in zip(person_kpts, person_vis):
            if v > 0.3 and 0 <= int(x) < depth_img.shape[1] and 0 <= int(y) < depth_img.shape[0]:
                depth_val = depth_img[int(y), int(x)] * depth_scale
                if depth_val > 0:
                    joint_xyz.append(deproject(depth_val, x, y, intrinsics))
                    print(v, deproject(depth_val, x, y, intrinsics))
                else:
                    joint_xyz.append([np.nan, np.nan, np.nan])
            else:
                joint_xyz.append([np.nan, np.nan, np.nan])

        joint_xyz = np.array(joint_xyz)

        for j, (x, y, z) in enumerate(joint_xyz):
            if not np.isnan(z) and j in left_right_body_kpt_ids:
                ax.scatter(x, y, z, c='green', s=10)

        for idx1, idx2 in skeleton:
            if idx1 < len(joint_xyz) and idx2 < len(joint_xyz):
                pt1 = joint_xyz[idx1]
                pt2 = joint_xyz[idx2]
                if not np.any(np.isnan(pt1)) and not np.any(np.isnan(pt2)):
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], c='blue', linewidth=2)

        # # Get all valid X and Y for bounding range calculation
        # all_xyz = np.concatenate([
        #     np.array(joint_xyz)[[not np.any(np.isnan(pt)) for pt in joint_xyz]]
        #     for joint_xyz in [np.array([
        #         deproject(depth_img[int(y), int(x)] * depth_scale, x, y, intrinsics)
        #         if v > 0.3 and 0 <= int(x) < depth_img.shape[1] and 0 <= int(y) < depth_img.shape[0] and depth_img[int(y), int(x)] > 0
        #         else [np.nan, np.nan, np.nan]
        #         for (x, y), v in zip(person_kpts, person_vis)
        #     ]) for person_kpts, person_vis in zip(keypoints, visibility)]
        # ], axis=0)

        # valid_xyz = all_xyz[~np.isnan(all_xyz).any(axis=1)]
        # if len(valid_xyz) > 0:
        #     x_vals, y_vals = valid_xyz[:, 0], valid_xyz[:, 1]
        #     max_range = max(np.ptp(x_vals), np.ptp(y_vals)) / 2
        #     mid_x, mid_y = np.mean(x_vals), np.mean(y_vals)

        #     ax.set_xlim(mid_x - max_range, mid_x + max_range)
        #     ax.set_ylim(mid_y - max_range, mid_y + max_range)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0.0, 3.0)
        set_axes_equal(ax)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=-80, azim=-90)
        plt.tight_layout()
        plt.show()

    input("Press Enter to close both windows...")  # Keeps script alive
    cv2.destroyAllWindows()
    plt.close()