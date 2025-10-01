# utils_rom.py
# AUTO-ROM state + core mechanics (smoothing, motion/hold logic, extrema locking, pairing).
# No voice and no UI/presentation here.

import time
from collections import deque
import numpy as np
import cv2
from utils_3d import angle_from_vecpair
from utils_variables import get_vectors_for_preset

class SessionState:
    def __init__(self):
        # Angle buffers (post-baseline angle)
        self.last_angle = None
        self.angle_series = deque()

        # Which ROM test is currently active
        self.active_rom = None

        # Baseline / zeroing
        self.baseline_deg = None
        self.baseline_set_ts = None
        self.current_raw_angle = np.nan

        # === AUTO-ZERO flags (baseline helper) ===
        self.auto_zero_pending = False
        self.auto_zero_start_time = None
        self.auto_zero_window_sec = 1.0
        self.auto_zero_buffer = []
        self.auto_zero_min_samples = 10
        self.auto_zero_max_std_deg = 999
        self.last_zero_source = None      # 'auto' or 'manual'

        # === AUTO-ROM filters/state ===
        self.filt_angle = None
        self.prev_filt_angle = None
        self.prev_ts = None
        self.vel_dps = 0.0

        # Hold detection
        self.hold_window = deque()        # (ts, filt_angle)
        self.stopped_since_ts = None
        self.moving_counter = 0
        self.stopped_counter = 0
        self.last_move_sign = 0           # -1 / 0 / +1

        # Trial lifecycle
        self.trial_active = False
        self.trial_start_ts = None
        self.trial_armed = False
        self.trial_refractory_until = None

        # One-shot start control
        self.arm_after_baseline = False
        self.first_auto_arm_consumed = True  # set False when auto-baseline is armed

        # Gating to finalize (baseline return)
        self.baseline_hold_start_ts = None

        # Extrema events and best ROM pair
        self.extrema_events = []          # list of {'kind': 'min'|'max', 'ts': float, 'angle': float, 'frame': np.ndarray|None}
        self.pending_capture_kind = None  # 'min'|'max' (attach HUD frame next cycle)
        self.best_pair = None             # {'min': evt, 'max': evt, 'rom': float}

        # Latches
        self.lock_refractory_until = None
        self.after_lock_motion_needed = False
        self.last_locked_kind = None
        self.last_locked_ts = None

        # Output bookkeeping
        self.no_motion_reported = False
        self.rom_save_dir = None
        self.session_root = None
        self.session_stamp = None

        # Optional separate result window name (UI-owned)
        self.result_window_name = None


def reset_auto_rom_state(state: SessionState, keep_trial_ready: bool = False):
    """Clear ROM runtime state (filters, windows, locks, extrema).
    If keep_trial_ready=True, keep trial_active/start_ts unchanged; otherwise clear."""
    state.filt_angle = None
    state.prev_filt_angle = None
    state.prev_ts = None
    state.vel_dps = 0.0

    state.hold_window.clear()
    state.stopped_since_ts = None
    state.moving_counter = 0
    state.stopped_counter = 0
    state.last_move_sign = 0

    if not keep_trial_ready:
        state.trial_active = False
        state.trial_start_ts = None
    state.trial_armed = False
    state.trial_refractory_until = None

    state.baseline_hold_start_ts = None

    state.extrema_events = []
    state.pending_capture_kind = None
    state.best_pair = None

    state.lock_refractory_until = None
    state.after_lock_motion_needed = False
    state.last_locked_kind = None
    state.last_locked_ts = None

    state.no_motion_reported = False
    # keep rom_save_dir/session fields


def maybe_start_trial(args, state: SessionState, t_now: float):
    """Arm → start: requires motion + amplitude gate away from baseline."""
    if state.trial_active or state.baseline_deg is None:
        return
    if not state.trial_armed:
        return

    amp_ok = abs(state.filt_angle or 0.0) >= args.rom_start_amp
    # ~5 frames of movement above 'go' threshold
    if amp_ok and state.moving_counter >= 5:
        state.trial_active = True
        state.trial_start_ts = t_now
        state.extrema_events = []
        state.best_pair = None
        state.no_motion_reported = False
        state.baseline_hold_start_ts = None
        state.trial_armed = False  # single-rep arm
        print("[ROM] Trial started.")


def confirm_hold_and_lock_extremum(args, state: SessionState, t_now: float, frame_bgr):
    """Lock a min/max when stable hold detected. Attaches no frame yet; mark pending capture."""
    # require fresh motion after previous lock
    if state.after_lock_motion_needed and abs(state.vel_dps) < args.rom_v_go:
        return None
    if state.lock_refractory_until and t_now < state.lock_refractory_until:
        return None
    if not state.hold_window:
        return None
    t0 = state.hold_window[0][0]
    if (t_now - t0) < args.rom_hold_sec:
        return None

    vals = np.array([y for (_, y) in state.hold_window], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < max(5, int(0.3 / 0.033)):
        return None
    if np.nanstd(vals) > args.rom_std_max:
        return None

    ang_med = float(np.nanmedian(vals))
    if state.last_move_sign > 0:
        kind = 'max'
    elif state.last_move_sign < 0:
        kind = 'min'
    else:
        return None

    # de-dup within short window if angle barely moved
    if state.last_locked_kind == kind and state.last_locked_ts and (t_now - state.last_locked_ts < 1.2):
        if state.extrema_events:
            prev = state.extrema_events[-1]
            if prev['kind'] == kind and abs(ang_med - prev['angle']) < 3.0:
                return None

    evt = {'kind': kind, 'ts': t_now, 'angle': ang_med, 'frame': None}
    state.extrema_events.append(evt)
    state.pending_capture_kind = kind

    #cv2.putText(frame_bgr, f"{kind.upper()} locked: {ang_med:.1f} deg", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    state.hold_window.clear()
    state.lock_refractory_until = t_now + args.rom_lock_refractory_sec
    state.after_lock_motion_needed = True
    state.last_locked_kind = kind
    state.last_locked_ts = t_now
    print(f"[ROM] {kind.upper()} locked at {ang_med:.2f}°")

    update_best_pair_if_possible(args, state)
    return evt


def update_best_pair_if_possible(args, state: SessionState):
    """Pair the latest min and max (order-invariant) and keep the best ROM so far."""
    best = state.best_pair
    last_min, last_max = None, None
    for evt in state.extrema_events:
        if evt['kind'] == 'min':
            last_min = evt
        elif evt['kind'] == 'max':
            last_max = evt
        if last_min is not None and last_max is not None:
            ang_min = last_min['angle']
            ang_max = last_max['angle']
            rom = ang_max - ang_min
            if rom < 0:  # swap if out of order
                ang_min, ang_max = ang_max, ang_min
                last_min, last_max = last_max, last_min
                rom = ang_max - ang_min
            if rom >= args.rom_min_amplitude:
                if (best is None) or (rom > best['rom']):
                    state.best_pair = best = {'min': last_min, 'max': last_max, 'rom': rom}
            last_min, last_max = None, None


def attach_locked_frame_if_pending(state: SessionState, combined_bgr):
    """Once HUD is drawn, attach the rendered frame to the most recent pending min/max."""
    if state.pending_capture_kind in ('min', 'max') and state.extrema_events:
        for evt in reversed(state.extrema_events):
            if evt['frame'] is None and evt['kind'] == state.pending_capture_kind:
                evt['frame'] = combined_bgr.copy()
                break
        state.pending_capture_kind = None

# ==========================================
# Offline/batch angle helpers (no UI/globals)
# ==========================================

def _resolve_point_2d_with_scores_from_pose(pose: Dict[str, Any], spec, kpt_thr: float = 0.30):
    """
    Resolve a 2D point from a pose dict with keys:
      - 'kpts_xy':  (K,2) float array of xy
      - 'kpt_scores': (K,) float array of confidences in [0,1]
    'spec' can be an int index or an iterable of indices to be averaged.
    Returns np.array([x,y]) with NaNs if unavailable.
    """
    kxy = pose.get('kpts_xy', None)
    ksc = pose.get('kpt_scores', None)
    if kxy is None or ksc is None:
        return np.array([np.nan, np.nan], dtype=float)

    arr = np.asarray(kxy, dtype=float)
    sco = np.asarray(ksc, dtype=float)
    K = arr.shape[0]

    def _one(i):
        try:
            j = int(i)
            if j < 0 or j >= K:
                return None
            if not np.isfinite(sco[j]) or float(sco[j]) < kpt_thr:
                return None
            p = arr[j, :2]
            if not np.all(np.isfinite(p)):
                return None
            return p
        except Exception:
            return None

    if isinstance(spec, (list, tuple)):
        pts = [p for p in (_one(i) for i in spec) if p is not None]
        if not pts:
            return np.array([np.nan, np.nan], dtype=float)
        return np.nanmean(np.vstack(pts), axis=0)

    p = _one(spec)
    if p is None:
        return np.array([np.nan, np.nan], dtype=float)
    return p


def _angle_from_vecpair_auto(pose: Dict[str, Any], vecpair, kpt_thr: float = 0.30):
    """
    Compute angle using 3D if available and valid; else fall back to 2D.

    pose keys used:
      - 'joint_xyz' : (K,3) float meter-space joints (optional)
      - 'kpts_xy'   : (K,2) float pixel-space keypoints (optional)
      - 'kpt_scores': (K,)  float confidences [0,1] (optional)

    vecpair structure (as in your presets): ((P0,P1),(Q0,Q1)), where each is an
    index or an iterable of indices to be averaged.

    Returns (angle_deg or None, source_str '3D'|'2D'|None).
    """
    # Try 3D first
    j3d = pose.get('joint_xyz', None)
    if isinstance(j3d, np.ndarray) and j3d.ndim == 2 and j3d.shape[1] == 3:
        ang3d = angle_from_vecpair(j3d, vecpair)
        if np.isfinite(ang3d):
            return float(ang3d), "3D"

    # Fallback to 2D
    try:
        (P0, P1), (Q0, Q1) = vecpair
    except Exception:
        return None, None

    A = _resolve_point_2d_with_scores_from_pose(pose, P1, kpt_thr)
    B = _resolve_point_2d_with_scores_from_pose(pose, P0, kpt_thr)
    C = _resolve_point_2d_with_scores_from_pose(pose, Q1, kpt_thr)

    if np.any(np.isnan(A)) or np.any(np.isnan(B)) or np.any(np.isnan(C)):
        return None, None

    u = A - B
    v = C - B
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu <= 1e-6 or nv <= 1e-6:
        return None, None

    u /= nu
    v /= nv
    cosang = float(np.clip(np.dot(u, v), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cosang)))
    return ang, "2D"


# ===================================
# Batch/offline ROM evaluation helper
# ===================================

def compute_rom_for_pair(
    pose1: Dict[str, Any],
    pose2: Dict[str, Any],
    test_name: str,
    *,
    last_vals: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[float], Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    """
    Compute ROM between two poses for a given test preset.

    Returns
    -------
    rom_deg : float or None
        Absolute ROM in degrees.
    angles1 : dict
        Single-sided: {"main": a1}
        L/R-sided:    {"L": a1L, "R": a1R}
    angles2 : dict
        Same keys as angles1.
    """
    vecpairs = get_vectors_for_preset(test_name)
    if not vecpairs:
        return None, {}, {}

    # Single-sided
    if len(vecpairs) == 1:
        a1, src1 = _angle_from_vecpair_auto(pose1, vecpairs[0])
        a2, src2 = _angle_from_vecpair_auto(pose2, vecpairs[0])
        rom = None if (a1 is None or a2 is None) else float(abs(a2 - a1))
        if last_vals is not None:
            last_vals.update({"ang1": a1, "src1": src1, "ang2": a2, "src2": src2})
        return rom, {"main": a1}, {"main": a2}

    # Two-sided (L/R)
    a1L, src1L = _angle_from_vecpair_auto(pose1, vecpairs[0])
    a2L, src2L = _angle_from_vecpair_auto(pose2, vecpairs[0])
    a1R, src1R = _angle_from_vecpair_auto(pose1, vecpairs[1])
    a2R, src2R = _angle_from_vecpair_auto(pose2, vecpairs[1])
    #print(f"[DEBUG] test={test_name}  L: {a1L}->{a2L} src=({src1L},{src2L})  R: {a1R}->{a2R} src=({src1R},{src2R})")

    dL = None if (a1L is None or a2L is None) else float(abs(a2L - a1L))
    dR = None if (a1R is None or a2R is None) else float(abs(a2R - a1R))
    rom_candidates = [d for d in (dL, dR) if d is not None]
    rom = None if not rom_candidates else float(max(rom_candidates))

    if last_vals is not None:
        last_vals.update({
            "a1L": a1L, "a2L": a2L, "src1L": src1L, "src2L": src2L,
            "a1R": a1R, "a2R": a2R, "src1R": src1R, "src2R": src2R,
        })

    return rom, {"L": a1L, "R": a1R}, {"L": a2L, "R": a2R}
