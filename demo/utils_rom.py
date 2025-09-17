# utils_rom.py
# AUTO-ROM state + core mechanics (smoothing, motion/hold logic, extrema locking, pairing).
# No voice and no UI/presentation here.

import time
from collections import deque
import numpy as np
import cv2

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
