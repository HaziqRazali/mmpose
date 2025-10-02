#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROM angle-at-two-timestamps with Depth ZIP using index-ratio sync by default.
Now supports --debug-boxes to visualize all detected person bboxes.

Run:
python demo/topdown_demo_with_mmdet.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth \
  --input /path/to/rgb_xxx.mp4 \
  --rom-test left_shoulder_flexion \
  --t1 00:00:02.000 --t2 00:00:33.000 \
  --depth /path/to/depth_xxx.zip \
  --show --save-frames --debug-boxes
"""

import argparse
import json
import math
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
from mmengine.registry import init_default_scope

from utils_variables import get_vectors_for_preset


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


# ---------------- Depth ZIP reader ----------------

_HEADER_RE = re.compile(rb'^(timestamp|width|height|data_length|fx|fy|ox|oy)\s*:\s*([0-9\.\-eE]+)\s*$', re.M)

class DepthZip:
    """Stream depth frames from a ZIP of frame_*.bin files; normalize timestamps for reporting."""
    def __init__(self, zip_path: str):
        self.zf = zipfile.ZipFile(zip_path, 'r')
        names = [n for n in self.zf.namelist() if n.lower().endswith('.bin')]
        def frame_id(s):
            m = re.search(r'(\d+)', Path(s).stem)
            return int(m.group(1)) if m else -1
        self.entries = sorted(names, key=frame_id)

        self._timestamps = []
        self._sizes = []
        self._intrinsics = []
        for n in self.entries:
            with self.zf.open(n, 'r') as f:
                head = f.read(2048)
            hmap = {}
            for m in _HEADER_RE.finditer(head):
                k = m.group(1).decode('ascii')
                v = float(m.group(2).decode('ascii'))
                hmap[k] = v
            w = int(hmap.get('width', 0)); h = int(hmap.get('height', 0))
            self._timestamps.append(float(hmap.get('timestamp', 0.0)))
            self._sizes.append((h, w))
            self._intrinsics.append({
                'fx': float(hmap.get('fx', 0.0)), 'fy': float(hmap.get('fy', 0.0)),
                'ox': float(hmap.get('ox', 0.0)), 'oy': float(hmap.get('oy', 0.0))
            })

        self.count = len(self.entries)
        self.base_ts = self._timestamps[0] if self.count else 0.0
        # normalized timestamps (seconds since depth recording began)
        self.norm_ts = [(t - self.base_ts) for t in self._timestamps]

    def get_frame(self, index: int):
        """Return dict with depth array and header fields; includes normalized timestamp."""
        name = self.entries[index]
        with self.zf.open(name, 'r') as f:
            raw = f.read()
        m = list(_HEADER_RE.finditer(raw))
        if not m:
            raise ValueError(f'Bad header: {name}')
        header_end = m[-1].end()
        while header_end < len(raw) and raw[header_end:header_end+1] in (b'\r', b'\n', b'\t', b' '):
            header_end += 1
        head = raw[:header_end]; payload = raw[header_end:]

        hmap = {}
        for m2 in _HEADER_RE.finditer(head):
            k = m2.group(1).decode('ascii')
            v = float(m2.group(2).decode('ascii'))
            hmap[k] = v
        w = int(hmap['width']); h = int(hmap['height'])
        dl = int(hmap['data_length'])
        if dl != w*h*4:
            dl = w*h*4
        depth = np.frombuffer(payload[:dl], dtype='<f4', count=w*h).reshape(h, w).astype(np.float32)
        depth[~np.isfinite(depth)] = np.nan
        depth[depth <= 0] = np.nan
        ts_abs = float(hmap['timestamp'])
        ts_norm = ts_abs - self.base_ts
        return {
            'depth': depth, 'ts_sec_abs': ts_abs, 'ts_sec_norm': ts_norm,
            'h': h, 'w': w, 'fx': float(hmap['fx']), 'fy': float(hmap['fy']),
            'ox': float(hmap['ox']), 'oy': float(hmap['oy']),
        }

    @property
    def sizes(self): return self._sizes
    @property
    def intrinsics(self): return self._intrinsics


# ---------------- Geometry + drawing ----------------

def _to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

def _avg_point(kpts_xy, spec):
    if isinstance(spec, (list, tuple)):
        pts = []
        for s in spec:
            p = _avg_point(kpts_xy, s)
            if p is not None: pts.append(p)
        if not pts: return None
        arr = np.array(pts, dtype=float)
        return arr.mean(axis=0)
    else:
        idx = int(spec)
        if idx < 0 or idx >= len(kpts_xy): return None
        return kpts_xy[idx][:2]

def angle2d_from_vecpair(kpts_xy, vec_pair):
    (P0, P1), (Q0, Q1) = vec_pair
    p0 = _avg_point(kpts_xy, P0); p1 = _avg_point(kpts_xy, P1)
    q0 = _avg_point(kpts_xy, Q0); q1 = _avg_point(kpts_xy, Q1)
    if any(v is None for v in (p0, p1, q0, q1)): return None
    v1 = np.array(p1, float) - np.array(p0, float)
    v2 = np.array(q1, float) - np.array(q0, float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None
    v1 /= n1; v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def draw_vectors2d(frame_bgr, kpts_xy, vec_pair, thickness=3):
    (P0, P1), (Q0, Q1) = vec_pair
    p0 = _avg_point(kpts_xy, P0); p1 = _avg_point(kpts_xy, P1)
    q0 = _avg_point(kpts_xy, Q0); q1 = _avg_point(kpts_xy, Q1)
    def ok(pt): return pt is not None and np.isfinite(pt).all()
    if ok(p0) and ok(p1):
        cv2.line(frame_bgr, tuple(np.int32(p0)), tuple(np.int32(p1)), (255,0,0), thickness, cv2.LINE_AA)
        cv2.circle(frame_bgr, tuple(np.int32(p0)), 5, (255,0,0), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, tuple(np.int32(p1)), 5, (255,0,0), -1, cv2.LINE_AA)
    if ok(q0) and ok(q1):
        cv2.line(frame_bgr, tuple(np.int32(q0)), tuple(np.int32(q1)), (0,200,0), thickness, cv2.LINE_AA)
        cv2.circle(frame_bgr, tuple(np.int32(q0)), 5, (0,200,0), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, tuple(np.int32(q1)), 5, (0,200,0), -1, cv2.LINE_AA)
    return frame_bgr

def draw_bboxes(frame, bboxes, scores=None, chosen_idx=None):
    """Draw all person boxes; highlight the chosen one."""
    if bboxes is None:
        return frame
    for i, b in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, b[:4])
        color = (0, 165, 255)        # orange
        if chosen_idx is not None and i == chosen_idx:
            color = (0, 255, 255)    # yellow for chosen
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        if scores is not None and len(scores) > i:
            lbl = f"{i}:{scores[i]:.2f}"
            cv2.putText(frame, lbl, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, lbl, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# -------- Depth sampling + 3D --------

def _median_depth_at(depth: np.ndarray, u: float, v: float, k: int = 5) -> float:
    h, w = depth.shape[:2]
    x = int(round(u)); y = int(round(v))
    r = k // 2
    x0 = max(0, x - r); x1 = min(w, x + r + 1)
    y0 = max(0, y - r); y1 = min(h, y + r + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch)]
    if vals.size < (patch.size * 0.5): return np.nan
    return float(np.median(vals))

def _rgbkpt_to_depth_xy(u_rgb, v_rgb, rgb_w, rgb_h, depth_w, depth_h):
    su = depth_w / float(rgb_w); sv = depth_h / float(rgb_h)
    return u_rgb * su, v_rgb * sv

def _backproject(u, v, Z, fx, fy, ox, oy):
    if not np.isfinite(Z) or Z <= 0: return None
    X = (u - ox) * Z / fx
    Y = (v - oy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

def _project(P, fx, fy, ox, oy):
    """Project a 3D camera-space point to pixel coords."""
    Z = float(P[2]) if P is not None else 0.0
    if not np.isfinite(Z) or abs(Z) < 1e-6:
        Z = 1e-6
    u = ox + fx * (float(P[0]) / Z)
    v = oy + fy * (float(P[1]) / Z)
    return np.array([u, v], dtype=np.float32)

def angle3d_from_vecpair(kpts_xy_rgb, vec_pair, depth_frame, rgb_size, median_k=5):
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
    (P0, P1), (Q0, Q1) = vec_pair
    A = pt3d(P0); B = pt3d(P1); C = pt3d(Q0); D = pt3d(Q1)
    if any(p is None for p in (A,B,C,D)): return None, False
    v1 = B - A; v2 = D - C
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None, False
    v1 /= n1; v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot)), True

def angle2d_between_segments_across_frames(k1, k2, j_sh=5, j_el=7):
    """
    Inter-frame angle between the same 2D segment at t1 vs t2.
    Segment = shoulder (j_sh) -> elbow (j_el).
    """
    p1 = k1[j_sh][:2]; q1 = k1[j_el][:2]
    p2 = k2[j_sh][:2]; q2 = k2[j_el][:2]
    if not (np.isfinite(p1).all() and np.isfinite(q1).all() and np.isfinite(p2).all() and np.isfinite(q2).all()):
        return None
    v1 = q1 - p1; v2 = q2 - p2
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    v1 /= n1; v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def angle3d_between_segments_across_frames(k1, k2, d1, d2, rgb_size, median_k=5, j_sh=5, j_el=7):
    """
    Inter-frame angle between the same 3D segment at t1 vs t2.
    Backprojects shoulder/elbow at each time with depth intrinsics.
    """
    if d1 is None or d2 is None:
        return None

    h1, w1 = d1['h'], d1['w']; fx1, fy1, ox1, oy1 = d1['fx'], d1['fy'], d1['ox'], d1['oy']
    h2, w2 = d2['h'], d2['w']; fx2, fy2, ox2, oy2 = d2['fx'], d2['fy'], d2['ox'], d2['oy']
    depth1 = d1['depth']; depth2 = d2['depth']
    rw, rh = rgb_size

    def to_depth_xy(pt_rgb, dw, dh):
        return _rgbkpt_to_depth_xy(pt_rgb[0], pt_rgb[1], rw, rh, dw, dh)

    # t1 backproject
    sh1 = k1[j_sh][:2]; el1 = k1[j_el][:2]
    if not (np.isfinite(sh1).all() and np.isfinite(el1).all()):
        return None
    u_sh1, v_sh1 = to_depth_xy(sh1, w1, h1); Z_sh1 = _median_depth_at(depth1, u_sh1, v_sh1, k=median_k)
    u_el1, v_el1 = to_depth_xy(el1, w1, h1); Z_el1 = _median_depth_at(depth1, u_el1, v_el1, k=median_k)
    if not (np.isfinite(Z_sh1) and Z_sh1 > 0 and np.isfinite(Z_el1) and Z_el1 > 0):
        return None
    SH1 = _backproject(u_sh1, v_sh1, Z_sh1, fx1, fy1, ox1, oy1)
    EL1 = _backproject(u_el1, v_el1, Z_el1, fx1, fy1, ox1, oy1)
    if SH1 is None or EL1 is None:
        return None

    # t2 backproject
    sh2 = k2[j_sh][:2]; el2 = k2[j_el][:2]
    if not (np.isfinite(sh2).all() and np.isfinite(el2).all()):
        return None
    u_sh2, v_sh2 = to_depth_xy(sh2, w2, h2); Z_sh2 = _median_depth_at(depth2, u_sh2, v_sh2, k=median_k)
    u_el2, v_el2 = to_depth_xy(el2, w2, h2); Z_el2 = _median_depth_at(depth2, u_el2, v_el2, k=median_k)
    if not (np.isfinite(Z_sh2) and Z_sh2 > 0 and np.isfinite(Z_el2) and Z_el2 > 0):
        return None
    SH2 = _backproject(u_sh2, v_sh2, Z_sh2, fx2, fy2, ox2, oy2)
    EL2 = _backproject(u_el2, v_el2, Z_el2, fx2, fy2, ox2, oy2)
    if SH2 is None or EL2 is None:
        return None

    v1 = EL1 - SH1
    v2 = EL2 - SH2
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    v1 /= n1; v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def angle3d_internal_rotation_left_simple(kpts_xy_rgb, depth_frame, rgb_size, median_k=5):
    """
    Computes the internal/external rotation angle of the LEFT shoulder assuming perfect form.
    Steps:
      1) Build trunk plane from both shoulders and both hips, take its normal A.
      2) Build forearm vector F = left_wrist (9) - left_elbow (7).
      3) Return angle between A and F in degrees.
    Requires valid depth_frame (3D). Returns (angle_deg or None).
    """
    if depth_frame is None:
        return None
    h_d, w_d = depth_frame['h'], depth_frame['w']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
    depth = depth_frame['depth']
    rw, rh = rgb_size

    def pt3d(spec):
        pt2 = _avg_point(kpts_xy_rgb, spec)
        if pt2 is None:
            return None
        u_d, v_d = _rgbkpt_to_depth_xy(pt2[0], pt2[1], rw, rh, w_d, h_d)
        Z = _median_depth_at(depth, u_d, v_d, k=median_k)
        if not np.isfinite(Z) or Z <= 0:
            return None
        return _backproject(u_d, v_d, Z, fx, fy, ox, oy)

    LS = pt3d(5); RS = pt3d(6); LH = pt3d(11); RH = pt3d(12)
    LE = pt3d(7); LW = pt3d(9)
    if any(p is None for p in (LS, RS, LE, LW)):
        return None

    # # Trunk plane normal: use both shoulders and both hips when available
    # if LH is not None and RH is not None:
    #     v_s = RS - LS
    #     v_h = RH - LH
    #     n = np.cross(v_s, v_h)
    # else:
    #     # Fallback: plane from (LS, RS, LH) if RH missing (or LS, RS, RH if LH missing)
    #     if LH is None:
    #         if RH is None:
    #             return None
    #         A, B, C = LS, RS, RH
    #     else:
    #         A, B, C = LS, RS, LH
    #     n = np.cross(B - A, C - A)

    n = np.array([0, 0, 1], dtype=np.float32)

    n_norm = np.linalg.norm(n)
    if n_norm < 1e-6:
        return None
    n = n / n_norm

    F = LW - LE
    f_norm = np.linalg.norm(F)
    if f_norm < 1e-6:
        return None
    F = F / f_norm

    dot = float(np.clip(np.dot(n, F), -1.0, 1.0))
    ang = math.degrees(math.acos(dot))
    return ang

def draw_internal_rotation_vecs(kpts_xy_rgb, depth_frame, rgb_size, img,
                                median_k=5, torso_color=(255,0,0), forearm_color=(0,200,0)):
    """
    Draws:
      - Torso-plane normal (3D) projected to the image as an arrow starting at left shoulder.
      - Left forearm arrow (elbow -> wrist) in 2D.
    """
    if depth_frame is None:
        return

    h_d, w_d = depth_frame['h'], depth_frame['w']
    fx, fy, ox, oy = depth_frame['fx'], depth_frame['fy'], depth_frame['ox'], depth_frame['oy']
    depth = depth_frame['depth']
    rw, rh = rgb_size

    def pt3d_and_2d(spec):
        pt2 = _avg_point(kpts_xy_rgb, spec)
        if pt2 is None: return None, None
        u_d, v_d = _rgbkpt_to_depth_xy(pt2[0], pt2[1], rw, rh, w_d, h_d)
        Z = _median_depth_at(depth, u_d, v_d, k=median_k)
        if not np.isfinite(Z) or Z <= 0: return None, None
        P3 = _backproject(u_d, v_d, Z, fx, fy, ox, oy)
        return P3, np.array([pt2[0], pt2[1]], dtype=np.float32)

    LS3, LS2 = pt3d_and_2d(5)
    RS3, RS2 = pt3d_and_2d(6)
    LH3, LH2 = pt3d_and_2d(11)
    RH3, RH2 = pt3d_and_2d(12)
    LE3, LE2 = pt3d_and_2d(7)
    LW3, LW2 = pt3d_and_2d(9)

    # Need these to draw
    if any(p is None for p in (LS3, RS3, LE2, LW2)):
        return

    # Torso plane normal in 3D
    if (LH3 is not None) and (RH3 is not None):
        v_s = RS3 - LS3
        v_h = RH3 - LH3
        n = np.cross(v_s, v_h)
    else:
        # Fallback: plane through (LS, RS, LH) or (LS, RS, RH)
        if LH3 is not None:
            n = np.cross(RS3 - LS3, LH3 - LS3)
        elif RH3 is not None:
            n = np.cross(RS3 - LS3, RH3 - LS3)
        else:
            return
    #n = np.array([0, 0, 1], dtype=np.float32) 

    n_norm = np.linalg.norm(n)
    if n_norm < 1e-6: return
    n = n / n_norm

    # Arrow length: proportional to shoulder width in 3D
    shoulder_len = np.linalg.norm(RS3 - LS3)
    L = 0.5 * shoulder_len if np.isfinite(shoulder_len) and shoulder_len > 1e-6 else 0.2

    # Project torso-normal arrow
    base2 = _project(LS3, fx, fy, ox, oy)
    tip3  = LS3 + L * n
    tip2  = _project(tip3, fx, fy, ox, oy)

    if np.isfinite(base2).all() and np.isfinite(tip2).all():
        cv2.arrowedLine(img,
                        (int(base2[0]), int(base2[1])),
                        (int(tip2[0]),  int(tip2[1])),
                        torso_color, 2, tipLength=0.2)

    # Forearm arrow (2D image-space)
    if np.isfinite(LE2).all() and np.isfinite(LW2).all():
        cv2.arrowedLine(img,
                        (int(LE2[0]), int(LE2[1])),
                        (int(LW2[0]), int(LW2[1])),
                        forearm_color, 2, tipLength=0.2)

def depth_to_vis(depth: np.ndarray):
    d = depth.copy()
    mask = ~np.isfinite(d)
    if np.all(mask): return np.zeros((depth.shape[0], depth.shape[1], 3), np.uint8)
    v = d[~mask]
    mn, mx = float(np.nanpercentile(v, 2.0)), float(np.nanpercentile(v, 98.0))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        mn, mx = np.nanmin(v), np.nanmax(v)
    d[mask] = mn
    d8 = np.clip((d - mn) / max(mx - mn, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    vis[mask] = (0,0,0)
    return vis


# -------- Detection + pose --------

def largest_person_bbox(det_result, det_cat_id=0, score_thr=0.3):
    pred = det_result.pred_instances
    b = _to_np(pred.bboxes); labels = _to_np(pred.labels); scores = _to_np(pred.scores)
    keep = np.logical_and(labels == det_cat_id, scores > score_thr)
    b = b[keep]; s = scores[keep]
    if b.size == 0:
        return None, None, None
    areas = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    idx = int(np.argmax(areas))
    return b, s, idx

def _bbox_center(b):
    x1, y1, x2, y2 = b[:4]
    return (0.5*(x1+x2), 0.5*(y1+y2))

def run_once(frame_bgr, det_model, pose_model, det_cat_id, score_thr,
             prefer_nearer=False, depth_frame=None, rgb_size=None):
    det_result = inference_detector(det_model, frame_bgr)
    kept_bboxes, kept_scores, idx_area = largest_person_bbox(det_result, det_cat_id, score_thr)
    if kept_bboxes is None:
        return None, None, "no-person", None, None, None

    chosen_idx = idx_area             # <- fixed to largest-area
    chosen_bbox = kept_bboxes[chosen_idx]

    pose_results = inference_topdown(pose_model, frame_bgr, bboxes=chosen_bbox[None, :4])
    if not pose_results:
        return None, None, "no-pose", kept_bboxes, kept_scores, chosen_idx
    data = merge_data_samples(pose_results)
    if data is None or not hasattr(data, "pred_instances") or len(data.pred_instances) == 0:
        return None, None, "no-pose", kept_bboxes, kept_scores, chosen_idx
    k = _to_np(data.pred_instances.keypoints)[0]
    return k, chosen_bbox, None, kept_bboxes, kept_scores, chosen_idx

# -------- Overlays --------

def overlay_text(frame, lines, org=(20, 40)):
    y = org[1]
    for line in lines:
        cv2.putText(frame, line, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        y += 28
    return frame

def stack_side_by_side(img_left, img_right):
    h = max(img_left.shape[0], img_right.shape[0])
    def resize_to_h(img, H):
        return cv2.resize(img, (int(round(img.shape[1] * (H / img.shape[0]))), H))
    L = resize_to_h(img_left, h); R = resize_to_h(img_right, h)
    return np.hstack([L, R])


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="ROM angle at two timestamps with Depth ZIP (index-ratio sync)")
    ap.add_argument("det_config")
    ap.add_argument("det_checkpoint")
    ap.add_argument("pose_config")
    ap.add_argument("pose_checkpoint")
    ap.add_argument("--input", required=True, help="RGB video path")
    ap.add_argument("--rom-test", required=True, help="Preset name from utils_variables")
    ap.add_argument("--t1", required=True, help="HH:MM:SS[.ms]")
    ap.add_argument("--t2", required=True, help="HH:MM:SS[.ms]")
    ap.add_argument("--depth", default="", help="Optional depth zip path depth_<id>.zip")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--det-cat-id", type=int, default=0)
    ap.add_argument("--median-k", type=int, default=5)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--debug-boxes", action="store_true", help="Draw all detected person boxes")
    args = ap.parse_args()

    # ROM preset
    vecs = get_vectors_for_preset(args.rom_test)
    if not vecs:
        raise SystemExit(f"No vector-pair found for preset '{args.rom_test}'.")
    vec_pair = vecs[0]

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
    if dur_sec is None or dur_sec <= 0:
        dur_sec = max(parse_hhmmss_ms(args.t1), parse_hhmmss_ms(args.t2), 1.0)

    # Depth
    dz = None
    if args.depth:
        dz_path = Path(args.depth)
        if not dz_path.exists():
            raise SystemExit(f"Depth zip not found: {dz_path}")
        dz = DepthZip(str(dz_path))
        if dz.count == 0:
            dz = None

    # Clamp times
    t1 = parse_hhmmss_ms(args.t1); t2 = parse_hhmmss_ms(args.t2)
    t1 = min(max(0.0, t1), max(0.0, dur_sec - 1.0 / max(fps, 1.0)))
    t2 = min(max(0.0, t2), max(0.0, dur_sec - 1.0 / max(fps, 1.0)))

    # --- t1 RGB
    set_cap_to_time(cap, t1)
    f1, rgb_pos1 = read_near_target(cap, t1, fps)
    if f1 is None:
        raise SystemExit("Failed to grab frame near t1.")
    rgb_h, rgb_w = f1.shape[:2]

    # --- t1 Depth via index-ratio
    d1 = None; d1_idx = None
    if dz is not None:
        ratio1 = (rgb_pos1 or t1) / max(dur_sec, 1.0)
        d1_idx = int(round(np.clip(ratio1, 0.0, 1.0) * (dz.count - 1)))
        d1 = dz.get_frame(d1_idx)

    k1, bbox1, e1, all_b1, sc1, idx1 = run_once(
        f1, det_model, pose_model, args.det_cat_id, args.score_thr,
        prefer_nearer=False, depth_frame=d1, rgb_size=(rgb_w, rgb_h))
    if e1:
        raise SystemExit(f"t1 failed: {e1}")

    # --- t2 RGB
    set_cap_to_time(cap, t2)
    f2, rgb_pos2 = read_near_target(cap, t2, fps)
    if f2 is None:
        raise SystemExit("Failed to grab frame near t2.")

    # --- t2 Depth via index-ratio
    d2 = None; d2_idx = None
    if dz is not None:
        ratio2 = (rgb_pos2 or t2) / max(dur_sec, 1.0)
        d2_idx = int(round(np.clip(ratio2, 0.0, 1.0) * (dz.count - 1)))
        d2 = dz.get_frame(d2_idx)

    k2, bbox2, e2, all_b2, sc2, idx2 = run_once(
        f2, det_model, pose_model, args.det_cat_id, args.score_thr,
        prefer_nearer=False, depth_frame=d2, rgb_size=(rgb_w, rgb_h))
    if e2:
        raise SystemExit(f"t2 failed: {e2}")

    # --- angles / segment rotation
    if args.rom_test == "left_shoulder_flexion":
        
        # Inter-frame rotation of the left upper arm (shoulder->elbow) between t1 and t2
        mode_used = "2D"
        seg_angle = None

        if dz is not None and d1 is not None and d2 is not None:
            seg_angle = angle3d_between_segments_across_frames(
                k1, k2, d1, d2, (rgb_w, rgb_h), median_k=args.median_k, j_sh=5, j_el=7
            )
            if seg_angle is not None:
                mode_used = "3D"

        if seg_angle is None:
            seg_angle = angle2d_between_segments_across_frames(k1, k2, j_sh=5, j_el=7)
            if seg_angle is None:
                raise SystemExit("Inter-frame segment angle failed due to degenerate vectors or missing kpts.")
            mode_used = "2D_fallback" if dz is not None else "2D"

        # For downstream overlay code that expects ang1/ang2, we feed the same
        # single rotation angle to both panels and set delta=0. We also add a
        # dedicated JSON field for clarity.
        ang1 = float(seg_angle)
        ang2 = float(seg_angle)
        delta = 0.0
        interframe_segment_rotation = float(seg_angle)

    elif args.rom_test == "left_shoulder_internal_rotation":
        # Per-frame angle between trunk-plane normal and left forearm (elbow->wrist)
        # Assumes perfect form (no elbow flexion confound); uses 3D only.
        if dz is None or d1 is None or d2 is None:
            raise SystemExit("left_shoulder_internal_rotation requires depth (3D). Provide --depth-zip.")
        a1 = angle3d_internal_rotation_left_simple(k1, d1, (rgb_w, rgb_h), median_k=args.median_k)
        a2 = angle3d_internal_rotation_left_simple(k2, d2, (rgb_w, rgb_h), median_k=args.median_k)
        if a1 is None or a2 is None:
            raise SystemExit("Internal rotation angle failed due to missing/degenerate 3D points.")
        ang1, ang2 = float(a1), float(a2)
        delta = None  # no subtraction/delta for this ROM
        interframe_segment_rotation = None
        mode_used = "3D"

    else:
        # Original per-frame ROM angle (two vectors within each frame, t1 & t2)
        a1_3d = a2_3d = None
        mode_used = "2D"
        if dz is not None and d1 is not None and d2 is not None:
            a1_3d, ok1 = angle3d_from_vecpair(k1, vec_pair, d1, (rgb_w, rgb_h), median_k=args.median_k)
            a2_3d, ok2 = angle3d_from_vecpair(k2, vec_pair, d2, (rgb_w, rgb_h), median_k=args.median_k)
            if (a1_3d is not None) and (a2_3d is not None):
                mode_used = "3D"

        if mode_used == "3D":
            ang1, ang2 = a1_3d, a2_3d
        else:
            a1_2d = angle2d_from_vecpair(k1, vec_pair)
            a2_2d = angle2d_from_vecpair(k2, vec_pair)
            if a1_2d is None or a2_2d is None:
                raise SystemExit("Per-frame angle failed due to degenerate vectors or missing kpts.")
            ang1, ang2 = a1_2d, a2_2d
            mode_used = "2D_fallback" if dz is not None else "2D"

        delta = ang2 - ang1
        interframe_segment_rotation = None

    # --- outputs
    out_dir = Path(args.out_dir) if args.out_dir else video_path.parent / f"{video_path.stem}_rom_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "file": str(video_path),
        "rom_test": args.rom_test,
        "mode_used": mode_used,
        "angle_t1_deg": round(float(ang1), 3),
        "angle_t2_deg": round(float(ang2), 3),
        "delta_deg": (round(float(delta), 3) if delta is not None else None),
        # add this line (will be a number for left_shoulder_flexion, else null):
        "segment_rotation_t1_to_t2_deg": (round(float(interframe_segment_rotation), 3)
                                          if interframe_segment_rotation is not None else None),
        "timestamps": {
            "t1_requested": args.t1, "t1_rgb_actual_sec": rgb_pos1,
            "t2_requested": args.t2, "t2_rgb_actual_sec": rgb_pos2
        },
        "fps": fps
    }

    if dz is not None:
        # normalized depth times for the chosen frames (seconds since depth start)
        t1_depth_norm = d1['ts_sec_norm'] if d1 is not None else None
        t2_depth_norm = d2['ts_sec_norm'] if d2 is not None else None
        report.update({
            "depth_sync_mode": "index_ratio",
            "depth_file": str(Path(args.depth)),
            "depth_count": dz.count,
            "rgb_duration_sec": dur_sec,
            "depth_t1_index": d1_idx, "depth_t1_norm_sec": t1_depth_norm,
            "depth_t2_index": d2_idx, "depth_t2_norm_sec": t2_depth_norm,
            "intrinsics_t1": ({k: float(d1[k]) for k in ('fx','fy','ox','oy')} if d1 else None),
            "intrinsics_t2": ({k: float(d2[k]) for k in ('fx','fy','ox','oy')} if d2 else None),
        })

    json_path = out_dir / f"{video_path.stem}_{args.rom_test}_t1t2.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- visualization
    def annotate_panel(frame_rgb, angle, when_label, rgb_pos_sec):
        lines = [f"{when_label}  Mode: {mode_used}",
                f"ROM: {args.rom_test}",
                f"Angle: {angle:.2f} deg",
                f"RGB actual: {rgb_pos_sec:.3f}s" if rgb_pos_sec is not None else "RGB actual: n/a"]
        return overlay_text(frame_rgb, lines)

    # Show angle only on the second panel for left_shoulder_flexion
    if args.rom_test in {"left_shoulder_flexion", "left_shoulder_extension", "left_shoulder_abduction"}:
        # t1: no angle line, keep ROM text the same
        lines_t1 = [f"t1  Mode: {mode_used}",
                    f"ROM: {args.rom_test}",
                    f"RGB actual: {rgb_pos1:.3f}s" if rgb_pos1 is not None else "RGB actual: n/a"]
        f1_ov = overlay_text(f1.copy(), lines_t1)

        # t2: use the normal annotator so it shows "Angle: ... deg"
        f2_ov = annotate_panel(f2.copy(), ang2, "t2", rgb_pos2)
    else:
        # original behavior for other ROM tests
        f1_ov = annotate_panel(f1.copy(), ang1, "t1", rgb_pos1)
        f2_ov = annotate_panel(f2.copy(), ang2, "t2", rgb_pos2)

    # draw debug boxes before vectors, if requested
    if args.debug_boxes:
        f1_ov = draw_bboxes(f1_ov, all_b1, sc1, idx1)
        f2_ov = draw_bboxes(f2_ov, all_b2, sc2, idx2)

    if args.rom_test == "left_shoulder_internal_rotation":
        # Custom overlay: torso-plane normal (blue) + forearm (green)
        if dz is not None and d1 is not None and d2 is not None:
            draw_internal_rotation_vecs(k1, d1, (rgb_w, rgb_h), f1_ov, median_k=args.median_k,
                                        torso_color=(255,0,0), forearm_color=(0,200,0))
            draw_internal_rotation_vecs(k2, d2, (rgb_w, rgb_h), f2_ov, median_k=args.median_k,
                                        torso_color=(255,0,0), forearm_color=(0,200,0))
        else:
            # No depth available; fall back to generic vec-pair so you still see something.
            f1_ov = draw_vectors2d(f1_ov, k1, vec_pair, 3)
            f2_ov = draw_vectors2d(f2_ov, k2, vec_pair, 3)

    elif args.rom_test in {"left_shoulder_flexion", "left_shoulder_extension", "left_shoulder_abduction"}:
        seg_pair = [[5, 7], [5, 7]]  # draw the same segment; second line overlaps
        f1_ov = draw_vectors2d(f1_ov, k1, seg_pair, 3)
        f2_ov = draw_vectors2d(f2_ov, k2, seg_pair, 3)
    else:
        f1_ov = draw_vectors2d(f1_ov, k1, vec_pair, 3)
        f2_ov = draw_vectors2d(f2_ov, k2, vec_pair, 3)


    if dz is not None:
        d1_vis = depth_to_vis(d1['depth']) if d1 is not None else np.zeros((f1_ov.shape[0], f1_ov.shape[1], 3), np.uint8)
        d2_vis = depth_to_vis(d2['depth']) if d2 is not None else np.zeros((f2_ov.shape[0], f2_ov.shape[1], 3), np.uint8)
        left = stack_side_by_side(f1_ov, d1_vis)
        right = stack_side_by_side(f2_ov, d2_vis)
        combined = stack_side_by_side(left, right)
    else:
        combined = stack_side_by_side(f1_ov, f2_ov)

    if args.show:
        cv2.imshow("ROM t1 vs t2 (RGB | Depth)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save_frames:
        cv2.imwrite(str(out_dir / f"{video_path.stem}_{args.rom_test}_t1t2_side.jpg"), combined)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
