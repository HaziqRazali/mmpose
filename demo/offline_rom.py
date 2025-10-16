#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mmdet.apis import init_detector
from mmpose.apis import init_model as init_pose_model
from mmpose.utils import adapt_mmdet_pipeline
from mmengine.registry import init_default_scope

# --- project modules (adjust the package prefix to your layout) ---
from utils.presets import needs_t2, get_vectors_for_preset, draw_policy, DRAW_ALWAYS, DRAW_IF_NO_DRAWER
from utils.ipad_depthio import DepthZip, depth_to_vis
from utils.detectors import run_once, draw_bboxes
from utils.calculators import compute as compute_rom
from utils.viz import annotate_panel, draw_vectors2d, stack_side_by_side, VizContext, draw_for_rom, _CUSTOM_DRAWERS

# Import drawers once so they self-register
import utils.drawers_internal_rotation  # noqa: F401
import utils.drawers_segments           # noqa: F401

# --- 3D point cloud + 3D viz twin ---
from utils.pcd import build_point_cloud, keypoints3d_from_kxy, make_offset_transform
from utils.viz3d import VizContext3D, draw_for_rom_3d, show_in_one_window, make_axis

# Import 3D drawers once so they self-register
import utils.drawers3d_segments          # noqa: F401
import utils.drawers3d_internal_rotation # noqa: F401

# ---------------- Time + video I/O ----------------

def parse_hhmmss_ms(s: str) -> float:
    parts = s.split(':')
    if len(parts) == 3:
        h = int(parts[0]); m = int(parts[1]); sec = float(parts[2])
    elif len(parts) == 2:
        h = 0; m = int(parts[0]); sec = float(parts[1])
    else:
        raise ValueError(f'Bad time format: {s}')
    return h * 3600 + m * 60 + sec


def set_cap_to_time(cap: cv2.VideoCapture, t_sec: float) -> None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)


def read_near_target(cap: cv2.VideoCapture, target_sec: float, fps: float, jitter_tol_sec: float = None):
    if jitter_tol_sec is None:
        jitter_tol_sec = 1.0 / max(fps, 1.0)
    best = None
    best_dt = float('inf')
    # Try a couple reads and keep the closest timestamp
    for _ in range(4):
        ok, frame = cap.read()
        if not ok:
            break
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_sec = pos_ms / 1000.0 if pos_ms > 0 else None
        if pos_sec is None:
            return frame, None
        dt = abs(pos_sec - target_sec)
        if dt < best_dt:
            best = (frame.copy(), pos_sec)
            best_dt = dt
        if dt <= jitter_tol_sec:
            break
    return (best[0] if best else None), (best[1] if best else None)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="ROM angle with optional depth (iPad ZIP).")
    ap.add_argument("det_config")
    ap.add_argument("det_checkpoint")
    ap.add_argument("pose_config")
    ap.add_argument("pose_checkpoint")

    ap.add_argument("--input", required=True, help="RGB video path")
    ap.add_argument("--rom-test", required=True, help="ROM name (see presets.py)")
    ap.add_argument("--t1", required=True, help="HH:MM:SS[.ms]")
    ap.add_argument("--t2", required=False, default="00:00:00.000", help="HH:MM:SS[.ms]")
    ap.add_argument("--depth", default="", help="Optional depth zip path: depth_<id>.zip")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--det-cat-id", type=int, default=0)
    ap.add_argument("--median-k", type=int, default=5)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--debug-boxes", action="store_true", help="Draw all detected person boxes")

    # 3D visualization (Open3D) options
    ap.add_argument("--show-3d", action="store_true", help="Open a 3D viewer with point cloud(s) + ROM overlays.")
    ap.add_argument("--show-3d-both", action="store_true", help="If ROM needs t2 and depth exists, render t1 + t2 in one window.")
    ap.add_argument("--pcd-max-depth", type=float, default=None, help="Clip depth > this (meters) when building point clouds.")
    ap.add_argument("--pcd-voxel", type=float, default=0.0, help="Viewer-side downsample voxel size in meters (0 disables).")
    ap.add_argument("--t2-offset", type=float, default=0.25, help="Translation in +X (meters) to display t2 beside t1 in the same window.")
    args = ap.parse_args()

    rom = args.rom_test
    needs_second = needs_t2(rom)

    # Models
    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)
    init_default_scope('mmdet')
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device)

    # RGB video
    video_path = Path(args.input)
    if not video_path.exists():
        raise SystemExit(f"Input not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur_sec = (frame_count / fps) if frame_count > 0 else None

    # Times
    t1 = parse_hhmmss_ms(args.t1)
    t2 = parse_hhmmss_ms(args.t2) if needs_second else 0.0
    if dur_sec is None or dur_sec <= 0:
        dur_sec = max(t1, t2, 1.0)
    min_step = 1.0 / max(fps, 1.0)
    t1 = min(max(0.0, t1), max(0.0, dur_sec - min_step))
    if needs_second:
        t2 = min(max(0.0, t2), max(0.0, dur_sec - min_step))

    # Depth (optional)
    dz = None
    if args.depth:
        dz_path = Path(args.depth)
        if not dz_path.exists():
            raise SystemExit(f"Depth zip not found: {dz_path}")
        dz = DepthZip(str(dz_path))
        if dz.count == 0:
            dz = None

    # --- t1 RGB
    set_cap_to_time(cap, t1)
    f1, rgb_pos1 = read_near_target(cap, t1, fps)
    if f1 is None:
        raise SystemExit("Failed to grab frame near t1.")
    rgb_h, rgb_w = f1.shape[:2]
    #print(rgb_h, rgb_w)
    #sys.exit()
    
    # --- t1 Depth via index-ratio
    d1 = None; d1_idx = None
    if dz is not None:
        ratio1 = (rgb_pos1 or t1) / max(dur_sec, 1.0)
        d1_idx = int(round(np.clip(ratio1, 0.0, 1.0) * (dz.count - 1)))
        d1 = dz.get_frame(d1_idx)

    # --- t1 det+pose
    k1, bbox1, e1, all_b1, sc1, idx1 = run_once(
        f1, det_model, pose_model, args.det_cat_id, args.score_thr,
        prefer_nearer=False, depth_frame=d1, rgb_size=(rgb_w, rgb_h))
    if e1:
        raise SystemExit(f"t1 failed: {e1}")

    #if args.rom_test == "right_elbow_flexion" and k1 is not None:
    #    # right shoulder = joint 6, right elbow = joint 8
    #    k1[6, 0] = k1[8, 0]

    # --- t2 (only if needed)
    f2 = None; rgb_pos2 = None; d2 = None; d2_idx = None; k2 = None
    if needs_second:
        set_cap_to_time(cap, t2)
        f2, rgb_pos2 = read_near_target(cap, t2, fps)
        if f2 is None:
            raise SystemExit("Failed to grab frame near t2.")
        if dz is not None:
            ratio2 = (rgb_pos2 or t2) / max(dur_sec, 1.0)
            d2_idx = int(round(np.clip(ratio2, 0.0, 1.0) * (dz.count - 1)))
            d2 = dz.get_frame(d2_idx)
        k2, bbox2, e2, all_b2, sc2, idx2 = run_once(
            f2, det_model, pose_model, args.det_cat_id, args.score_thr,
            prefer_nearer=False, depth_frame=d2, rgb_size=(rgb_w, rgb_h))
        if e2:
            raise SystemExit(f"t2 failed: {e2}")

    # --- ROM compute
    result = compute_rom(rom, k1, k2, d1, d2, (rgb_w, rgb_h), median_k=args.median_k)

    ang1 = float(result.ang1_deg) if result.ang1_deg is not None else None
    ang2 = float(result.ang2_deg) if result.ang2_deg is not None else None
    delta = float(result.delta_deg) if result.delta_deg is not None else None
    seg_rot = float(result.segment_rotation_t1_to_t2_deg) if result.segment_rotation_t1_to_t2_deg is not None else None
    mode_used = result.mode_used

    # --- outputs
    # /home/haziq/datasets/telept/data/ipad/20251001-hh rgb_1759295456074
    out_dir = Path(args.out_dir) if args.out_dir else video_path.parent / f"{video_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "file": str(video_path),
        "rom_test": rom,
        "mode_used": mode_used,
        "angle_t1_deg": (round(ang1, 3) if ang1 is not None else None),
        "angle_t2_deg": (round(ang2, 3) if ang2 is not None else None),
        "delta_deg": (round(delta, 3) if delta is not None else None),
        "segment_rotation_t1_to_t2_deg": (round(seg_rot, 3) if seg_rot is not None else None),
        "timestamps": {
            "t1_requested": args.t1,
            "t1_rgb_actual_sec": rgb_pos1,
            "t2_requested": (args.t2 if needs_second else None),
            "t2_rgb_actual_sec": (rgb_pos2 if needs_second else None),
        },
        "fps": fps
    }

    if dz is not None:
        t1_depth_norm = d1['ts_sec_norm'] if d1 is not None else None
        t2_depth_norm = d2['ts_sec_norm'] if (needs_second and d2 is not None) else None
        report.update({
            "depth_sync_mode": "index_ratio",
            "depth_file": str(Path(args.depth)),
            "depth_count": dz.count,
            "rgb_duration_sec": dur_sec,
            "depth_t1_index": d1_idx, "depth_t1_norm_sec": t1_depth_norm,
            "depth_t2_index": (d2_idx if needs_second else None),
            "depth_t2_norm_sec": t2_depth_norm,
            "intrinsics_t1": ({k: float(d1[k]) for k in ('fx','fy','ox','oy')} if d1 else None),
            "intrinsics_t2": ({k: float(d2[k]) for k in ('fx','fy','ox','oy')} if (needs_second and d2) else None),
        })

    # Sanitize timestamps for filenames (replace : and .)
    def safe_time_label(s):
        if s is None:
            return "na"
        return s.replace(":", "-").replace(".", "_")

    # Build output filename with timestamps
    t1_label = safe_time_label(args.t1)
    t2_label = safe_time_label(args.t2) if needs_second else None

    if needs_second:
        json_name = f"{video_path.stem}_{rom}_t1-{t1_label}_t2-{t2_label}.json"
    else:
        json_name = f"{video_path.stem}_{rom}_t1-{t1_label}.json"

    json_path = out_dir / json_name

    # Save JSON report
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- visualization panels
    mode_used = "3D"
    f1_ov = annotate_panel(f1.copy(), rom, mode_used, "t1", ang1 if ang1 is not None else ang2, rgb_pos1)
    f2_ov = None
    if needs_second and f2 is not None:
        f2_ov = annotate_panel(f2.copy(), rom, mode_used, "t2", ang2, rgb_pos2)

    # Optional debug boxes
    if args.debug_boxes:
        f1_ov = draw_bboxes(f1_ov, all_b1, sc1, idx1)
        if needs_second:
            f2_ov = draw_bboxes(f2_ov, all_b2, sc2, idx2)

    # Decide default vec overlay via draw policy + whether a drawer exists
    policy = draw_policy(rom)
    has_drawer = rom in _CUSTOM_DRAWERS
    should_draw_vec = (policy == DRAW_ALWAYS) or (policy == DRAW_IF_NO_DRAWER and not has_drawer)

    vec_pair = None
    if should_draw_vec:
        vecs = get_vectors_for_preset(rom)
        if vecs:
            vec_pair = vecs[0]
            f1_ov = draw_vectors2d(f1_ov, k1, vec_pair, 3)
            if needs_second and f2_ov is not None:
                f2_ov = draw_vectors2d(f2_ov, k2, vec_pair, 3)

    # Per-ROM custom drawers (no-op if none registered)
    ctx1 = VizContext(
        frame=f1_ov, kpts_xy=k1, when_label="t1",
        angle_deg=(ang1 if ang1 is not None else ang2),
        rgb_pos_sec=rgb_pos1, vec_pair=vec_pair,
        depth_frame=d1, rgb_size=(rgb_w, rgb_h), median_k=args.median_k
    )
    draw_for_rom(rom, ctx1)

    if needs_second and f2_ov is not None:
        ctx2 = VizContext(
            frame=f2_ov, kpts_xy=k2, when_label="t2",
            angle_deg=ang2, rgb_pos_sec=rgb_pos2, vec_pair=vec_pair,
            depth_frame=d2, rgb_size=(rgb_w, rgb_h), median_k=args.median_k
        )
        draw_for_rom(rom, ctx2)

    # Compose side-by-side (include depth panels if available)
    if dz is not None:
        d1_vis = depth_to_vis(d1['depth']) if d1 is not None else np.zeros_like(f1_ov)
        if needs_second and f2_ov is not None:
            d2_vis = depth_to_vis(d2['depth']) if d2 is not None else np.zeros_like(f2_ov)
            left = stack_side_by_side(f1_ov, d1_vis)
            right = stack_side_by_side(f2_ov, d2_vis)
            combined = stack_side_by_side(left, right)
        else:
            combined = stack_side_by_side(f1_ov, d1_vis)
    else:
        combined = stack_side_by_side(f1_ov, f2_ov) if (needs_second and f2_ov is not None) else f1_ov

    if args.show:
        title = f"ROM {'t1 vs t2' if needs_second else 't1'} (RGB{' | Depth' if dz is not None else ''})"
        cv2.imshow(title, combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- Save composite panel (JPEG)
    if args.save_frames:
        # Reuse the same safe timestamp helper
        def safe_time_label(s):
            if s is None:
                return "na"
            return s.replace(":", "-").replace(".", "_")

        t1_label = safe_time_label(args.t1)
        t2_label = safe_time_label(args.t2) if needs_second else None

        if needs_second:
            img_name = f"{video_path.stem}_{rom}_t1-{t1_label}_t2-{t2_label}.jpg"
        else:
            img_name = f"{video_path.stem}_{rom}_t1-{t1_label}.jpg"

        out_img = out_dir / img_name
        cv2.imwrite(str(out_img), combined)
        print(f"[INFO] Saved composite frame to: {out_img}")

    #print(json.dumps(report, indent=2))

    # ---------------- 3D visualization (Open3D) ----------------
    if args.show_3d and dz is not None:
        geoms = []

        # Small origin axis (for reference)
        try:
            axis = make_axis(length=0.1)
            geoms.append(axis)
        except Exception as _e:
            axis = None  # Open3D missing, will raise later when building pcd

        # Build t1 point cloud (all pixels), optional voxel downsample via --pcd-voxel
        pcd_t1 = None
        kpts3d_t1 = None
        try:
            pcd_t1 = build_point_cloud(
                d1, f1, (rgb_w, rgb_h),
                max_depth=args.pcd_max_depth,
                voxel_size=(args.pcd_voxel if args.pcd_voxel and args.pcd_voxel > 0 else None),
                tint=None,
                transform=None
            ) if d1 is not None else None

            # 3D keypoints for vec/overlays at t1 (uses same depth sampling rules as geometry.py)
            kpts3d_t1 = keypoints3d_from_kxy(k1, d1, (rgb_w, rgb_h), median_k=args.median_k) if d1 is not None else None

        except ImportError as e:
            raise SystemExit(f"Open3D not available for 3D viz (--show-3d). Error: {e}")

        # Determine if we should draw 3D default vec-pair (mirror 2D policy computed earlier)
        vec_pair_3d = vec_pair if should_draw_vec else None

        # Per-ROM specialized 3D overlays for t1
        ctx3d_t1 = None
        if kpts3d_t1 is not None:
            ctx3d_t1 = VizContext3D(
                rom_name=rom,
                when_label="t1",
                kpts3d=kpts3d_t1,
                vec_pair=vec_pair_3d,
                pcd=pcd_t1,
                median_k=args.median_k
            )
            # Draw default vec-pair in 3D if allowed
            from utils.viz3d import draw_vectors3d
            draw_vectors3d(ctx3d_t1)
            # Specialized 3D drawer (if registered)
            draw_for_rom_3d(rom, ctx3d_t1)

        # Add t1 geometries
        if pcd_t1 is not None:
            geoms.append(pcd_t1)
        if ctx3d_t1 is not None and ctx3d_t1.overlays:
            geoms.extend(ctx3d_t1.overlays)

        # Optionally add t2 cloud + overlays in the same window, offset by +X
        if needs_second and args.show_3d_both and (d2 is not None) and (f2 is not None):
            T = make_offset_transform(dx=args.t2_offset, dy=0.0, dz=0.0)

            try:
                pcd_t2 = build_point_cloud(
                    d2, f2, (rgb_w, rgb_h),
                    max_depth=args.pcd_max_depth,
                    voxel_size=(args.pcd_voxel if args.pcd_voxel and args.pcd_voxel > 0 else None),
                    tint=(0.9, 0.9, 0.9),  # subtle tint so t2 is distinguishable
                    transform=T
                )

                kpts3d_t2 = keypoints3d_from_kxy(k2, d2, (rgb_w, rgb_h), median_k=args.median_k)

            except ImportError as e:
                raise SystemExit(f"Open3D not available for 3D viz (--show-3d). Error: {e}")

            ctx3d_t2 = VizContext3D(
                rom_name=rom,
                when_label="t2",
                kpts3d=kpts3d_t2,
                vec_pair=vec_pair_3d,  # same policy: draw if allowed and no 3D-specialized override
                pcd=pcd_t2,
                median_k=args.median_k
            )
            from utils.viz3d import draw_vectors3d
            draw_vectors3d(ctx3d_t2)
            draw_for_rom_3d(rom, ctx3d_t2)

            geoms.append(pcd_t2)
            if ctx3d_t2.overlays:
                # also offset overlay primitives by T (apply in-place transform when possible)
                # LineSets have points we can transform manually:
                import numpy as _np
                for g in ctx3d_t2.overlays:
                    if hasattr(g, "points"):
                        P = _np.asarray(g.points)
                        P_h = _np.concatenate([P, _np.ones((P.shape[0], 1), dtype=P.dtype)], axis=1)
                        P_t = (T @ P_h.T).T[:, :3]
                        # write back transformed points to this overlay
                        g.points = type(g.points)(P_t)
                geoms.extend(ctx3d_t2.overlays)

        # Finally: show everything in one window
        if geoms:
            show_in_one_window(
                geoms,
                title=f"ROM 3D: {rom} ({'t1+t2' if (needs_second and args.show_3d_both) else 't1'})"
            )

if __name__ == "__main__":
    main()
