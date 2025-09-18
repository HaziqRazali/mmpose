# utils_results.py
# Finalization & persistence for ROM trials:
# - Check finalize conditions (baseline return or timeout)
# - Create session/trial dirs
# - Save MIN/MAX/baseline images
# - Write rom_summary.json
# - Reset SessionState
# - Call visualization panel

import os
import time
import json_tricks as json
import numpy as np
import cv2
import mmcv
import mmengine

from utils_visualization import show_and_save_result_panel
from utils_rom import SessionState  # type hint / clarity (module must be accessible)

def finalize_rom_trial(args, state: SessionState, t_now, frame_bgr, combined_bgr=None):
    """
    Returns:
        result (dict) if finalized, else None
    """
    # In DEBUG mode we never persist or finalize anything
    if getattr(args, 'debug', False):
        return None

    # Not active → nothing to do
    if not state.trial_active:
        return None

    # Baseline return condition
    near_base = (state.filt_angle is not None) and (abs(state.filt_angle) <= args.rom_baseline_tol) and (abs(state.vel_dps) < args.rom_v_stop)
    baseline_back = False
    if near_base:
        if state.baseline_hold_start_ts is None:
            state.baseline_hold_start_ts = t_now
        elif (t_now - state.baseline_hold_start_ts) >= args.rom_baseline_hold_sec:
            baseline_back = True
    else:
        state.baseline_hold_start_ts = None

    # Timeout condition
    timeout = bool(state.trial_start_ts and (t_now - state.trial_start_ts >= args.rom_timeout_sec))
    if not (baseline_back or timeout):
        return None

    # Prepare session/trial directories
    if state.session_root is None:
        root = args.output_root if args.output_root else "."
        mmengine.mkdir_or_exist(root)
        state.session_stamp = time.strftime("%Y%m%d_%H%M%S")
        state.session_root = os.path.join(root, state.session_stamp)
        mmengine.mkdir_or_exist(state.session_root)

    trial_stamp = time.strftime("%H%M%S")
    trial_dir = os.path.join(state.session_root, f"{args.rom_test}_{trial_stamp}")
    mmengine.mkdir_or_exist(trial_dir)
    state.rom_save_dir = trial_dir

    # Base result payload
    result = {
        'test_name': args.rom_test,
        'baseline_deg': state.baseline_deg,
        'kpt_thr': args.kpt_thr,
        'hold_sec': args.rom_hold_sec,
        'std_max': args.rom_std_max,
        'v_go': args.rom_v_go,
        'v_stop': args.rom_v_stop,
        'start_amp': args.rom_start_amp,
        'baseline_tol': args.rom_baseline_tol,
        'baseline_hold_sec': args.rom_baseline_hold_sec,
        'min_amplitude': args.rom_min_amplitude,
        'timeout_sec': args.rom_timeout_sec,
        'events': [{'kind': e['kind'], 'ts': e['ts'], 'angle': e['angle']} for e in state.extrema_events],
        'status': 'ok',
    }

    # If only one extremum was found, synthesize the mate at baseline (0 deg)
    if state.best_pair is None and baseline_back:
        last_min = next((e for e in reversed(state.extrema_events) if e['kind'] == 'min'), None)
        last_max = next((e for e in reversed(state.extrema_events) if e['kind'] == 'max'), None)
        if last_min and (not last_max):
            baseline_evt = {'kind': 'max', 'ts': t_now, 'angle': 0.0, 'frame': None}
            rom_val = baseline_evt['angle'] - last_min['angle']
            if rom_val >= args.rom_min_amplitude:
                state.best_pair = {'min': last_min, 'max': baseline_evt, 'rom': rom_val}
        elif last_max and (not last_min):
            baseline_evt = {'kind': 'min', 'ts': t_now, 'angle': 0.0, 'frame': None}
            rom_val = last_max['angle'] - baseline_evt['angle']
            if rom_val >= args.rom_min_amplitude:
                state.best_pair = {'min': baseline_evt, 'max': last_max, 'rom': rom_val}

    # Save images / write result
    if state.best_pair is None:
        if len(state.extrema_events) == 0:
            result['status'] = 'no_motion'
            result['min_deg'] = state.baseline_deg
            result['max_deg'] = state.baseline_deg
            result['rom_deg'] = 0.0
            img_path = os.path.join(state.rom_save_dir, f"BASELINE_{state.baseline_deg if state.baseline_deg is not None else 0.0:.1f}deg.png")
            mmcv.imwrite((combined_bgr if combined_bgr is not None else frame_bgr), img_path)
            result['baseline_image'] = img_path
            print("[ROM] No movement detected; saved baseline image.")
        else:
            result['status'] = 'insufficient_range'
            result['min_deg'] = None
            result['max_deg'] = None
            result['rom_deg'] = 0.0
            print("[ROM] Finalized without a valid min/max pair (insufficient range).")
    else:
        best = state.best_pair
        min_evt, max_evt, rom_val = best['min'], best['max'], best['rom']
        hud_src = combined_bgr if combined_bgr is not None else frame_bgr
        if min_evt.get('frame') is None:
            min_evt['frame'] = hud_src.copy()
        if max_evt.get('frame') is None:
            max_evt['frame'] = hud_src.copy()

        min_img_path = os.path.join(state.rom_save_dir, f"MIN_{min_evt['angle']:.1f}deg.png")
        max_img_path = os.path.join(state.rom_save_dir, f"MAX_{max_evt['angle']:.1f}deg.png")
        mmcv.imwrite(min_evt['frame'], min_img_path)
        mmcv.imwrite(max_evt['frame'], max_img_path)

        result.update({
            'min_deg': float(min_evt['angle']),
            'min_ts': float(min_evt['ts']),
            'min_image': min_img_path,
            'max_deg': float(max_evt['angle']),
            'max_ts': float(max_evt['ts']),
            'max_image': max_img_path,
            'rom_deg': float(rom_val),
        })

        cv2.putText(frame_bgr, f"ROM saved: {rom_val:.1f} deg", (10, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        print(f"[ROM] Saved MIN {min_evt['angle']:.2f}°, MAX {max_evt['angle']:.2f}°, ROM {rom_val:.2f}°")

    # Write summary JSON
    json_path = os.path.join(state.rom_save_dir, "rom_summary.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[ROM] Summary saved: {json_path}")

    # Lifecycle reset (stay disarmed, arm later on next command)
    state.trial_active = False
    state.trial_start_ts = None
    state.extrema_events = []
    state.best_pair = None
    state.pending_capture_kind = None
    state.no_motion_reported = (result['status'] == 'no_motion')
    state.baseline_hold_start_ts = None
    state.lock_refractory_until = None
    state.trial_armed = False
    state.trial_refractory_until = t_now + args.rom_refractory_sec

    # Compose + save + optional popup
    show_and_save_result_panel(args, state, result)
    return result
