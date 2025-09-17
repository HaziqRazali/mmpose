# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from collections import deque

from utils_variables import rom_test
from utils import set_kpt_preset, handle_hotkeys_for_presets

# Optional 3D utilities; falls back to 2D angle if unavailable
try:
    from utils_3d import extract_intrinsics_from_depth, compute_3d_skeletons, angle
except Exception:
    def angle(a, b, c):
        a = np.array(a, dtype=float); b = np.array(b, dtype=float); c = np.array(c, dtype=float)
        va = a - b; vc = c - b
        na = np.linalg.norm(va); nc = np.linalg.norm(vc)
        if na < 1e-6 or nc < 1e-6:
            return np.nan
        cosang = np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

# =========================
# Optional voice support
# =========================
import threading
import queue
import re

try:
    import azure.cognitiveservices.speech as speechsdk
    _HAS_AZURE_SPEECH = True
except (ImportError, ModuleNotFoundError):
    _HAS_AZURE_SPEECH = False

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    import pyrealsense2 as rs
except (ImportError, ModuleNotFoundError):
    rs = None


# =========================
# Session State
# =========================

class SessionState:
    def __init__(self):
        self.last_angle = None
        self.angle_series = deque()
        self.active_rom = None
        self.baseline_deg = None          # baseline offset (None = unzeroed)
        self.baseline_set_ts = None       # when baseline locked
        self.current_raw_angle = np.nan   # latest raw angle each frame

        # === AUTO-ZERO ===========================================
        self.auto_zero_pending = False
        self.auto_zero_start_time = None
        self.auto_zero_window_sec = 1.0
        self.auto_zero_buffer = []
        self.auto_zero_min_samples = 10
        self.auto_zero_max_std_deg = 999
        self.last_zero_source = None      # 'auto' (from preset) | 'manual'
        # ==========================================================

        # === AUTO-ROM ============================================
        # Filtering / velocity
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

        # Trial / segment lifecycle
        self.trial_active = False
        self.trial_start_ts = None
        self.trial_armed = False
        self.trial_refractory_until = None

        # One-shot start control
        self.arm_after_baseline = False   # queue start if said before baseline locked
        self.first_auto_arm_consumed = True  # set to False on baseline lock from preset

        # Baseline return / finalize gating
        self.baseline_hold_start_ts = None

        # Extrema tracking
        self.extrema_events = []          # {'kind':'min'|'max','ts':..,'angle':..,'frame':np.ndarray}
        self.pending_capture_kind = None  # 'min'|'max' to capture after render
        self.best_pair = None             # {'min':event,'max':event,'rom':float}

        # Locks/latches
        self.lock_refractory_until = None
        self.after_lock_motion_needed = False
        self.last_locked_kind = None
        self.last_locked_ts = None

        # Output
        self.no_motion_reported = False
        self.rom_save_dir = None

        # === SESSION-LEVEL OUTPUT ROOT ===
        self.session_root = None          # e.g., "<output_root>/20250915_111404"
        self.session_stamp = None         # e.g., "20250915_111404"

# =========================
# Sparkline (small angle strip)
# =========================

def render_sparkline_strip(width, height, series, t_now, window_sec, label_text=None, gap_sec=0.4):
    strip = np.zeros((height, width, 3), dtype=np.uint8)

    if window_sec <= 0 or len(series) < 2:
        if label_text:
            cv2.putText(strip, label_text, (10, int(0.65 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return strip

    t0 = t_now - float(window_sec)
    filtered = [(max(t0, t), a) for (t, a) in series if (t >= t0 and np.isfinite(a))]
    if len(filtered) < 2:
        if label_text:
            cv2.putText(strip, label_text, (10, int(0.65 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return strip

    ts = [t for (t, _) in filtered]
    ys = [a for (_, a) in filtered]

    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or abs(y_max - y_min) < 1e-9:
        y_min, y_max = (0.0, 1.0) if not np.isfinite(y_min) or not np.isfinite(y_max) else (y_min - 1.0, y_max + 1.0)

    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (32, 32, 32), thickness=-1)
    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (180, 180, 180), thickness=1)

    def map_x(t):
        tx = (t - t0) / max(window_sec, 1e-6)
        tx = 0.0 if not np.isfinite(tx) else min(max(tx, 0.0), 1.0)
        return int(round(tx * (width - 1)))

    def map_y(y):
        ty = (y - y_min) / max((y_max - y_min), 1e-6)
        ty = 0.0 if not np.isfinite(ty) else min(max(ty, 0.0), 1.0)
        return int(round((height - 1) - ty * (height - 1)))

    segments = []
    seg = [(map_x(ts[0]), map_y(ys[0]))]
    for i in range(1, len(ts)):
        if (ts[i] - ts[i - 1]) > gap_sec:
            if len(seg) >= 2:
                segments.append(np.array(seg, dtype=np.int32))
            seg = []
        seg.append((map_x(ts[i]), map_y(ys[i])))
    if len(seg) >= 2:
        segments.append(np.array(seg, dtype=np.int32))

    for poly in segments:
        cv2.polylines(strip, [poly], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.putText(strip, f"[{y_min:.1f}, {y_max:.1f}]", (10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    last_y = ys[-1]
    if np.isfinite(last_y):
        cv2.putText(strip, f"{last_y:.1f} deg", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if label_text:
        txt_w = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(strip, label_text, (width - txt_w - 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return strip


# =========================
# Voice mapping
# =========================

def _build_voice_aliases_from_rom(rom_test_dict):
    """
    Build VOICE_PRESET_ALIASES automatically from rom_test keys.
    For each preset name (e.g., 'right_elbow_flexion'), generate:
      - exact phrase:        r"\bright elbow flexion\b"
      - flexible token path: r"\bright\b.*\belbow\b.*\bflexion\b"
      - side+joint variants: r"\bright\b.*\belbow\b", r"\bright\b.*\belbow\b.*\bflexion\b"
      - special cases:       'full_body' and '133'
      - numeric alias:       'mode {i}' and '{i}' based on ordering
    """
    aliases = {}
    preset_names = list(rom_test_dict.keys())

    for i, name in enumerate(preset_names, start=1):
        toks = name.split('_')
        human = ' '.join(toks)
        patterns = []

        # exact "humanized" phrase
        patterns.append(rf"\b{re.escape(human)}\b")

        # flexible token order with gaps allowed (tokens in order)
        patterns.append(r"\b" + r"\b.*\b".join(map(re.escape, toks)) + r"\b")

        # side + joint (+ first action token) variants
        if toks and toks[0] in ('left', 'right') and len(toks) >= 2:
            side = toks[0]
            joint = toks[1]
            patterns.append(rf"\b{re.escape(side)}\b.*\b{re.escape(joint)}\b")
            if len(toks) >= 3:
                action = toks[2]
                patterns.append(rf"\b{re.escape(side)}\b.*\b{re.escape(joint)}\b.*\b{re.escape(action)}\b")

        # special presets
        if name == 'full_body':
            patterns.extend([
                r"\bfull body\b", r"\bwhole body\b", r"\ball body\b", r"\bbody\b"
            ])
        if name == '133':
            patterns.extend([
                r"\b133\b", r"\bone thirty three\b", r"\bone three three\b"
            ])

        # numeric aliases (ordered by rom_test key order)
        patterns.append(rf"\bmode {i}\b")
        patterns.append(rf"\b{i}\b")

        # de-dup while preserving order
        seen = set()
        dedup = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                dedup.append(p)

        aliases[name] = dedup

    return aliases

VOICE_PRESET_ALIASES = _build_voice_aliases_from_rom(rom_test)

VOICE_COMMAND_PATTERNS = {
    "next": [r"\bnext\b", r"\bforward\b", r"\bgo next\b"],
    "prev": [r"\bprevious\b", r"\bback\b", r"\bgo back\b"],
    "zero": [r"\bzero\b", r"\brezero\b", r"\breset baseline\b", r"\bset baseline\b", r"\bcalibrate\b"],
    "start": [r"\bstart\b", r"\bbegin\b", r"\bgo\b", r"\bstart test\b", r"\brecord\b"]
}

def _match_any(text: str, patterns):
    for p in patterns:
        if re.search(p, text):
            return True
    return False


class VoiceController:
    def __init__(self, subscription_key, region, device_name=None, language="en-US"):
        self.enabled = _HAS_AZURE_SPEECH and bool(subscription_key) and bool(region)
        self.q = queue.Queue()
        self._thread = None
        self._stop = threading.Event()
        self._speech_config = None
        self._audio_config = None
        self._recognizer = None
        self._device_name = device_name
        self._language = language
        self._subscription_key = subscription_key
        self._region = region
        self.last_heard = ""

    def start(self):
        if not self.enabled:
            print("[VOICE] Azure Speech not available. Voice disabled.")
            return
        try:
            self._speech_config = speechsdk.SpeechConfig(subscription=self._subscription_key, region=self._region)
            self._speech_config.speech_recognition_language = self._language
            self._speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText"
            )
            self._audio_config = speechsdk.audio.AudioConfig(device_name=self._device_name) if self._device_name else speechsdk.audio.AudioConfig()
            self._recognizer = speechsdk.SpeechRecognizer(speech_config=self._speech_config, audio_config=self._audio_config)
            self._recognizer.recognizing.connect(self._on_recognizing)
            self._recognizer.recognized.connect(self._on_recognized)
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            print("[VOICE] Listening… say commands like 'full body', 'right elbow', 'shoulder abduction', 'zero', 'start', 'next', 'previous'.")
        except Exception as e:
            print(f"[VOICE] Failed to init speech: {e}")
            self.enabled = False

    def _run_loop(self):
        try:
            self._recognizer.start_continuous_recognition()
            while not self._stop.is_set():
                time.sleep(0.05)
        finally:
            try:
                self._recognizer.stop_continuous_recognition()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _on_recognizing(self, evt):
        text = (evt.result.text or "").strip()
        if text:
            self.last_heard = text
            print(f"[VOICE-INTERIM] {text}")

    def _on_recognized(self, evt):
        text = (evt.result.text or "").strip().lower()
        if not text:
            return
        self.last_heard = text
        print(f"[VOICE-FINAL] {text}")

        for preset_name, patterns in VOICE_PRESET_ALIASES.items():
            if _match_any(text, patterns):
                self.q.put(('preset', preset_name)); return
        if _match_any(text, VOICE_COMMAND_PATTERNS["start"]):
            self.q.put(('start', None)); return
        if _match_any(text, VOICE_COMMAND_PATTERNS["next"]):
            self.q.put(('next', None)); return
        if _match_any(text, VOICE_COMMAND_PATTERNS["prev"]):
            self.q.put(('prev', None)); return
        if _match_any(text, VOICE_COMMAND_PATTERNS["zero"]):
            self.q.put(('zero', None)); return
        if "elbow" in text:
            self.q.put(('preset', "right_elbow_flexion")); return
        if "shoulder" in text or "shoudler" in text:
            self.q.put(('preset', "right_shoulder_abduction")); return
        if "body" in text:
            self.q.put(('preset', "full_body")); return

    def poll(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


# =========================
# Core Processing
# =========================

def _apply_voice_command(args, state: SessionState, cmd_tuple):
    if cmd_tuple is None:
        return
    cmd, arg = cmd_tuple

    if cmd == 'preset':
        if arg in rom_test:
            set_kpt_preset(args, arg)
            if state is not None:
                # Arm auto-zero and reset baseline + ROM state
                state.auto_zero_pending = True
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                state.baseline_deg = None
                state.baseline_set_ts = None
                state.last_angle = None
                state.angle_series.clear()
                _reset_auto_rom_state(state)
                # this preset-triggered zero enables exactly ONE auto-arm after baseline
                state.first_auto_arm_consumed = False
                state.last_zero_source = 'auto'
                print(f"[AUTO-ZERO] Armed for preset '{arg}'. Holding ~{state.auto_zero_window_sec:.1f}s before baseline lock.")
        else:
            print(f"[VOICE] Unknown preset: {arg}")

    elif cmd == 'next':
        set_kpt_preset(args, _cycle_for_voice(args.kpt_preset_name, +1))
        if state is not None:
            state.auto_zero_pending = True
            state.auto_zero_start_time = None
            state.auto_zero_buffer.clear()
            state.baseline_deg = None
            state.baseline_set_ts = None
            state.last_angle = None
            state.angle_series.clear()
            _reset_auto_rom_state(state)
            state.first_auto_arm_consumed = False
            state.last_zero_source = 'auto'
            print(f"[AUTO-ZERO] Armed for next preset '{args.rom_test}'.")

    elif cmd == 'prev':
        set_kpt_preset(args, _cycle_for_voice(args.kpt_preset_name, -1))
        if state is not None:
            state.auto_zero_pending = True
            state.auto_zero_start_time = None
            state.auto_zero_buffer.clear()
            state.baseline_deg = None
            state.baseline_set_ts = None
            state.last_angle = None
            state.angle_series.clear()
            _reset_auto_rom_state(state)
            state.first_auto_arm_consumed = False
            state.last_zero_source = 'auto'
            print(f"[AUTO-ZERO] Armed for previous preset '{args.rom_test}'.")

    elif cmd == 'zero':
        if np.isfinite(state.current_raw_angle):
            state.baseline_deg = float(state.current_raw_angle)
            state.baseline_set_ts = time.time()
            state.angle_series.clear()
            state.last_angle = None
            state.auto_zero_pending = False
            state.auto_zero_start_time = None
            state.auto_zero_buffer.clear()
            _reset_auto_rom_state(state, keep_trial_ready=False)
            state.trial_armed = False               # manual zero does NOT auto-arm
            state.first_auto_arm_consumed = True    # disable first auto-arm
            state.last_zero_source = 'manual'
            print(f"[VOICE] Baseline set to {state.baseline_deg:.2f}°")
        else:
            print("[VOICE] Cannot zero: no valid angle this frame.")

    elif cmd == 'start':
        if state.baseline_deg is None:
            state.arm_after_baseline = True
            print("[ROM] Start queued. Will arm as soon as baseline locks.")
        else:
            state.trial_armed = True
            state.arm_after_baseline = False
            state.first_auto_arm_consumed = True  # once you say start, we consume the auto-first semantics
            print("[ROM] Armed for a single repetition. Move when ready.")


def _cycle_for_voice(name: str, step: int) -> str:
    preset_names = list(rom_test.keys())
    i = preset_names.index(name) if name in preset_names else 0
    return preset_names[(i + step) % len(preset_names)]


def _reset_auto_rom_state(state: SessionState, keep_trial_ready=False):
    state.filt_angle = None
    state.prev_filt_angle = None
    state.prev_ts = None
    state.vel_dps = 0.0

    state.hold_window.clear()
    state.stopped_since_ts = None
    state.moving_counter = 0
    state.stopped_counter = 0
    state.last_move_sign = 0

    # lifecycle
    state.trial_active = False if not keep_trial_ready else state.trial_active
    state.trial_start_ts = None if not keep_trial_ready else state.trial_start_ts
    state.trial_armed = False
    state.trial_refractory_until = None

    # finalize gating
    state.baseline_hold_start_ts = None

    # extrema
    state.extrema_events = []
    state.pending_capture_kind = None
    state.best_pair = None

    # locks/latches
    state.lock_refractory_until = None
    state.after_lock_motion_needed = False
    state.last_locked_kind = None
    state.last_locked_ts = None

    state.no_motion_reported = False
    # keep rom_save_dir for the run


def _maybe_start_trial(args, state: SessionState, t_now):
    """Start trial only when armed (or auto-first right after baseline), moving, and amplitude away from baseline is large enough."""
    if state.trial_active or state.baseline_deg is None:
        return

    # No auto re-arm on refractory expiry; only explicit arm, or first auto after baseline.
    if not state.trial_armed:
        return

    # need sustained motion + amplitude gate
    amp_ok = abs(state.filt_angle or 0.0) >= args.rom_start_amp
    if amp_ok and state.moving_counter >= 5:  # ~5 consecutive frames above v_go
        state.trial_active = True
        state.trial_start_ts = t_now
        state.extrema_events = []
        state.best_pair = None
        state.no_motion_reported = False
        state.baseline_hold_start_ts = None
        state.trial_armed = False  # consume the arm (single rep)
        print("[ROM] Trial started.")


def _confirm_hold_and_lock_extremum(args, state: SessionState, t_now, frame_bgr):
    """If we have a stable hold window, lock a MIN or MAX depending on last movement sign."""
    # Block repeated locks until we see new motion above v_go
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
    if vals.size < max(5, int(0.3/0.033)):
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

    # de-dup same-kind locks within a short window if angle barely changed
    if state.last_locked_kind == kind and state.last_locked_ts and (t_now - state.last_locked_ts < 1.2):
        if state.extrema_events:
            prev = state.extrema_events[-1]
            if prev['kind'] == kind and abs(ang_med - prev['angle']) < 3.0:
                return None

    evt = {'kind': kind, 'ts': t_now, 'angle': ang_med, 'frame': None}
    state.extrema_events.append(evt)
    state.pending_capture_kind = kind

    cv2.putText(frame_bgr, f"{kind.upper()} locked: {ang_med:.1f} deg", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    state.hold_window.clear()
    state.lock_refractory_until = t_now + args.rom_lock_refractory_sec

    # require fresh motion before allowing another lock
    state.after_lock_motion_needed = True
    state.last_locked_kind = kind
    state.last_locked_ts = t_now

    print(f"[ROM] {kind.upper()} locked at {ang_med:.2f}°")

    _update_best_pair_if_possible(args, state)
    return evt


def _update_best_pair_if_possible(args, state: SessionState):
    best = state.best_pair
    last_min = None
    last_max = None
    for evt in state.extrema_events:
        if evt['kind'] == 'min':
            last_min = evt
        elif evt['kind'] == 'max':
            last_max = evt
        if last_min is not None and last_max is not None:
            ang_min = last_min['angle']
            ang_max = last_max['angle']
            rom = ang_max - ang_min
            if rom < 0:
                ang_min, ang_max = ang_max, ang_min
                last_min, last_max = last_max, last_min
                rom = ang_max - ang_min

            if rom >= args.rom_min_amplitude:
                if (best is None) or (rom > best['rom']):
                    state.best_pair = best = {'min': last_min, 'max': last_max, 'rom': rom}

            last_min = None
            last_max = None


def _finalize_if_ready(args, state: SessionState, t_now, frame_bgr, combined_bgr=None):
    """Finalize when the patient returns & holds near baseline, or on timeout.
    combined_bgr: If provided, saved images will include the HUD/sparkline.
    """
    if not state.trial_active:
        return

    # baseline return check
    near_base = (state.filt_angle is not None) and (abs(state.filt_angle) <= args.rom_baseline_tol) and (abs(state.vel_dps) < args.rom_v_stop)
    baseline_back = False
    if near_base:
        if state.baseline_hold_start_ts is None:
            state.baseline_hold_start_ts = t_now
        elif (t_now - state.baseline_hold_start_ts) >= args.rom_baseline_hold_sec:
            baseline_back = True
    else:
        state.baseline_hold_start_ts = None

    # timeout from trial start (patient never returns)
    timeout = False
    if state.trial_start_ts and (t_now - state.trial_start_ts >= args.rom_timeout_sec):
        timeout = True

    if not (baseline_back or timeout):
        return

    # Prepare folder
    # -------------------------
    # SESSION/TRIAL FOLDER SETUP
    # -------------------------
    # 1) ensure a single session folder per run (first finalize defines it)
    if state.session_root is None:
        root = args.output_root if args.output_root else "."
        mmengine.mkdir_or_exist(root)
        state.session_stamp = time.strftime("%Y%m%d_%H%M%S")  # e.g., 20250915_111404
        state.session_root = os.path.join(root, state.session_stamp)
        mmengine.mkdir_or_exist(state.session_root)

    # 2) for every finalize, create a per-trial subfolder:
    #    <session>/<test>_<HHMMSS>
    trial_stamp = time.strftime("%H%M%S")  # e.g., 111404
    trial_dir = os.path.join(state.session_root, f"{args.rom_test}_{trial_stamp}")
    mmengine.mkdir_or_exist(trial_dir)
    state.rom_save_dir = trial_dir

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

    # Synthesize missing mate at baseline if needed
    if state.best_pair is None and baseline_back:
        last_min = next((e for e in reversed(state.extrema_events) if e['kind'] == 'min'), None)
        last_max = next((e for e in reversed(state.extrema_events) if e['kind'] == 'max'), None)

        if last_min and (not last_max):
            # Do NOT attach a frame here; let fallback fill with HUD later
            baseline_evt = {'kind': 'max', 'ts': t_now, 'angle': 0.0, 'frame': None}
            rom_val = baseline_evt['angle'] - last_min['angle']
            if rom_val >= args.rom_min_amplitude:
                state.best_pair = {'min': last_min, 'max': baseline_evt, 'rom': rom_val}
        elif last_max and (not last_min):
            # Do NOT attach a frame here; let fallback fill with HUD later
            baseline_evt = {'kind': 'min', 'ts': t_now, 'angle': 0.0, 'frame': None}
            rom_val = last_max['angle'] - baseline_evt['angle']
            if rom_val >= args.rom_min_amplitude:
                state.best_pair = {'min': baseline_evt, 'max': last_max, 'rom': rom_val}

    if state.best_pair is None:
        # If nothing valid, decide status
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
        min_evt = best['min']
        max_evt = best['max']
        rom_val = best['rom']

        # Use HUD frame if available
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

    json_path = os.path.join(state.rom_save_dir, "rom_summary.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[ROM] Summary saved: {json_path}")

    # finalize lifecycle
    state.trial_active = False
    state.trial_start_ts = None
    state.extrema_events = []
    state.best_pair = None
    state.pending_capture_kind = None
    state.no_motion_reported = (result['status'] == 'no_motion')
    state.baseline_hold_start_ts = None
    state.lock_refractory_until = None

    # stay DISARMED after finalize; no auto re-arm
    state.trial_armed = False
    state.trial_refractory_until = t_now + args.rom_refractory_sec

def process_one_image(args,
                      color_img,
                      depth_img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      state=None,
                      voice: VoiceController = None):

    det_result = inference_detector(detector, color_img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    pose_results = inference_topdown(pose_estimator, color_img, bboxes)
    data_samples = merge_data_samples(pose_results)

    if isinstance(color_img, str):
        color_img = mmcv.imread(color_img, channel_order='rgb')
    elif isinstance(color_img, np.ndarray):
        color_img = mmcv.bgr2rgb(color_img)

    # Reset state if preset changed
    if state is not None:
        if state.active_rom is None:
            state.active_rom = args.rom_test
            state.baseline_deg = None
            state.baseline_set_ts = None
        elif args.rom_test != state.active_rom:
            # finalize ongoing trial first (if any) — here we don't have HUD yet; pass combined_bgr=None
            if args.auto_rom and state.trial_active:
                _finalize_if_ready(args, state, time.time(), mmcv.rgb2bgr(color_img.copy()), combined_bgr=None)
            state.active_rom = args.rom_test
            state.last_angle = None
            state.angle_series.clear()
            state.baseline_deg = None
            state.baseline_set_ts = None
            _reset_auto_rom_state(state)
            state.first_auto_arm_consumed = False   # next baseline lock should auto-arm once
            state.last_zero_source = 'auto'

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            color_img,
            depth_img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            show_kpt_subset=rom_test[args.rom_test],
            skeleton_style=args.skeleton_style,
            show=False,
            wait_time=0,
            kpt_thr=args.kpt_thr)

    frame_rgb = visualizer.get_image()
    frame_bgr = mmcv.rgb2bgr(frame_rgb)

    t_now = time.time()
    pred = data_samples.get('pred_instances', None)

    angle_t = np.nan
    current_ids = rom_test.get(args.rom_test, [])
    is_angle_mode = isinstance(current_ids, (list, tuple)) and len(current_ids) == 3

    # 3D if depth available; else 2D
    if (depth_img is not None) and (pred is not None):
        try:
            intrin_dict = extract_intrinsics_from_depth(depth_img)
        except Exception:
            intrin_dict = None
        if intrin_dict is not None:
            keypoints = getattr(pred, 'transformed_keypoints', None)
            if keypoints is None:
                keypoints = pred.keypoints
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()

            visibility = getattr(pred, 'keypoints_visible', None)
            if visibility is None:
                visibility = np.ones(keypoints.shape[:2])
            elif hasattr(visibility, 'cpu'):
                visibility = visibility.cpu().numpy()

            if keypoints.shape[0] > 0:
                joints_xyz = compute_3d_skeletons(keypoints, visibility, depth_img, intrin_dict, args.kpt_thr)
                joints_xyz = np.array(joints_xyz)
                try:
                    if is_angle_mode:
                        a, b, c = current_ids
                        angle_t = angle(joints_xyz[0, a], joints_xyz[0, b], joints_xyz[0, c])
                except Exception as e:
                    cv2.putText(frame_bgr, f"Angle err: {e}", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    if (depth_img is None) and (pred is not None) and is_angle_mode:
        try:
            keypoints = getattr(pred, 'transformed_keypoints', None)
            if keypoints is None:
                keypoints = pred.keypoints
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()

            visibility = getattr(pred, 'keypoints_visible', None)
            if visibility is None:
                visibility = np.ones(keypoints.shape[:2])
            elif hasattr(visibility, 'cpu'):
                visibility = visibility.cpu().numpy()

            if keypoints.shape[0] > 0:
                a, b, c = current_ids
                k = keypoints[0]
                v = visibility[0]
                if np.all(v[[a, b, c]] >= args.kpt_thr):
                    ax, ay = k[a][:2]
                    bx, by = k[b][:2]
                    cx, cy = k[c][:2]
                    angle_t = angle([ax, ay], [bx, by], [cx, cy])
        except Exception as e:
            cv2.putText(frame_bgr, f"2D angle err: {e}", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # store raw
    state.current_raw_angle = angle_t if np.isfinite(angle_t) else np.nan

    # === AUTO-ZERO: collect for ~1s after voice preset, then lock baseline ===
    if is_angle_mode and state.auto_zero_pending:
        if state.auto_zero_start_time is None and np.isfinite(state.current_raw_angle):
            state.auto_zero_start_time = t_now
            state.auto_zero_buffer = [float(state.current_raw_angle)]
            cv2.putText(frame_bgr, "Auto-zero: capturing...", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
        elif state.auto_zero_start_time is not None:
            if np.isfinite(state.current_raw_angle):
                state.auto_zero_buffer.append(float(state.current_raw_angle))
            cv2.putText(frame_bgr, "Auto-zero: capturing...", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
            if (t_now - state.auto_zero_start_time) >= state.auto_zero_window_sec:
                buf = np.array(state.auto_zero_buffer, dtype=float)
                buf = buf[np.isfinite(buf)]
                if buf.size >= state.auto_zero_min_samples:
                    std_ok = (np.nanstd(buf) <= state.auto_zero_max_std_deg)
                    if std_ok:
                        baseline = float(np.nanmedian(buf))
                        state.baseline_deg = baseline
                        state.baseline_set_ts = t_now
                        state.auto_zero_pending = False
                        state.auto_zero_start_time = None
                        state.auto_zero_buffer.clear()
                        state.angle_series.clear()
                        state.last_angle = None
                        _reset_auto_rom_state(state, keep_trial_ready=False)
                        # --- ARMING RULES ---
                        # 1) If user said "start" earlier, arm now.
                        if state.arm_after_baseline:
                            state.trial_armed = True
                            state.arm_after_baseline = False
                            state.first_auto_arm_consumed = True
                        # 2) Else if this baseline came from preset/auto AND auto-first not yet consumed, arm once.
                        elif (state.last_zero_source == 'auto') and (not state.first_auto_arm_consumed):
                            state.trial_armed = True
                            state.first_auto_arm_consumed = True
                        else:
                            state.trial_armed = False
                        print(f"[AUTO-ZERO] Baseline locked at {baseline:.2f}° for {args.rom_test}")
                    else:
                        state.auto_zero_start_time = t_now
                        state.auto_zero_buffer = []
                        print("[AUTO-ZERO] Too jittery; extending capture window.")
                else:
                    state.auto_zero_start_time = t_now
                    state.auto_zero_buffer = []
                    print("[AUTO-ZERO] Not enough valid samples; extending capture window.")

    # apply baseline
    if np.isfinite(angle_t) and (state.baseline_deg is not None):
        angle_disp = angle_t - state.baseline_deg
    else:
        angle_disp = angle_t

    # === AUTO-ROM: smoothing + velocity + extrema + (NO finalize yet) ========
    if args.auto_rom and is_angle_mode and np.isfinite(angle_disp):
        # EMA smoothing
        if state.filt_angle is None:
            state.filt_angle = float(angle_disp)
            state.prev_filt_angle = float(angle_disp)
            state.prev_ts = t_now
        else:
            alpha = float(args.rom_ema_alpha)
            state.filt_angle = alpha * float(angle_disp) + (1.0 - alpha) * state.filt_angle

            dt = max(t_now - (state.prev_ts or t_now), 1e-6)
            state.vel_dps = (state.filt_angle - (state.prev_filt_angle if state.prev_filt_angle is not None else state.filt_angle)) / dt

            # hysteresis counters
            if abs(state.vel_dps) > args.rom_v_go:
                state.moving_counter += 1
            else:
                state.moving_counter = 0

            if abs(state.vel_dps) < args.rom_v_stop:
                state.stopped_counter += 1
                if state.stopped_since_ts is None:
                    state.stopped_since_ts = t_now
            else:
                state.stopped_counter = 0
                state.stopped_since_ts = None

            # movement sign tracker
            if state.vel_dps > args.rom_v_stop:
                state.last_move_sign = +1
            elif state.vel_dps < -args.rom_v_stop:
                state.last_move_sign = -1
            elif abs(state.vel_dps) > 1.0 and state.last_move_sign == 0:
                state.last_move_sign = 1 if state.vel_dps > 0 else -1

            # hold window when near-stopped
            if abs(state.vel_dps) < args.rom_v_stop:
                state.hold_window.append((t_now, state.filt_angle))
                while state.hold_window and (t_now - state.hold_window[0][0] > args.rom_hold_sec + 0.2):
                    state.hold_window.popleft()
            else:
                state.hold_window.clear()

            # Clear after-lock latch when fresh motion resumes
            if state.after_lock_motion_needed and abs(state.vel_dps) > args.rom_v_go:
                state.after_lock_motion_needed = False

            # gate trial start (armed + velocity + amplitude)
            _maybe_start_trial(args, state, t_now)

            # lock candidate extrema during segment
            if state.trial_active:
                _confirm_hold_and_lock_extremum(args, state, t_now, frame_bgr)

            # DO NOT finalize here; we want HUD frames in saved images
            state.prev_filt_angle = state.filt_angle
            state.prev_ts = t_now

    # =======================  PLOT / HUD  ==================================
    if state is not None and args.plot_seconds > 0 and is_angle_mode and np.isfinite(angle_disp):
        state.last_angle = float(angle_disp)
        state.angle_series.append((t_now, float(angle_disp)))
        cutoff = t_now - float(args.plot_seconds)
        while state.angle_series and state.angle_series[0][0] < cutoff:
            state.angle_series.popleft()

    # HUD
    if is_angle_mode:
        if np.isfinite(angle_disp):
            cv2.putText(frame_bgr, f"Angle: {angle_disp:.1f} deg", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Angle: --.- deg", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 240), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame_bgr, f"ROM: {args.rom_test} (no angle)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 240), 2, cv2.LINE_AA)

    zero_text = "Zeroed" if state.baseline_deg is not None else "Unzeroed"
    arm_text = "Active" if state.trial_active else ("Armed" if state.trial_armed else "Disarmed")
    cv2.putText(frame_bgr, f"Test: {args.rom_test} | "
                        f"{'Zeroed' if state.baseline_deg is not None else 'Unzeroed'} | "
                        f"{arm_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    H, W = frame_bgr.shape[:2]
    sparkline = render_sparkline_strip(
        width=W,
        height=int(args.plot_height),
        series=list(state.angle_series) if (state is not None and is_angle_mode) else [],
        t_now=t_now,
        window_sec=float(args.plot_seconds),
        label_text="Angle history (s)",
        gap_sec=float(args.plot_gap_sec)
    ) if args.plot_seconds > 0 else None

    if sparkline is not None:
        combined_bgr = np.vstack([frame_bgr, sparkline])
    else:
        combined_bgr = frame_bgr

    # attach frame to most recent locked event (after rendering)
    if args.auto_rom and state.pending_capture_kind in ('min', 'max') and state.extrema_events:
        for evt in reversed(state.extrema_events):
            if evt['frame'] is None and evt['kind'] == state.pending_capture_kind:
                evt['frame'] = combined_bgr.copy()
                break
        state.pending_capture_kind = None

    # ============
    # Input: Voice, then Keyboard
    # ============
    if voice and voice.enabled:
        while True:
            cmd = voice.poll()
            if cmd is None:
                break
            _apply_voice_command(args, state, cmd)

    if args.show:
        cv2.imshow("Pose", combined_bgr)
        key = cv2.pollKey() if hasattr(cv2, "pollKey") else cv2.waitKey(1)
        handle_hotkeys_for_presets(args, key)

        # hotkey: b to zero baseline manually
        if key == ord('b'):
            if np.isfinite(state.current_raw_angle):
                state.baseline_deg = float(state.current_raw_angle)
                state.baseline_set_ts = time.time()
                state.angle_series.clear()
                state.last_angle = None
                state.auto_zero_pending = False
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                _reset_auto_rom_state(state, keep_trial_ready=False)
                # manual zero does NOT auto-arm
                state.trial_armed = False
                state.first_auto_arm_consumed = True
                state.last_zero_source = 'manual'
                print(f"[KPT] Baseline set to {state.baseline_deg:.2f}°")
            else:
                print("[KPT] Cannot zero: no valid angle this frame.")

        # optional: s to arm start (subsequent reps)
        if key == ord('s'):
            if state.baseline_deg is None:
                state.arm_after_baseline = True
                print("[ROM] Start queued. Will arm as soon as baseline locks.")
            else:
                state.trial_armed = True
                state.arm_after_baseline = False
                state.first_auto_arm_consumed = True
                print("[ROM] Armed for a single repetition. Move when ready.")

        if args.show_interval > 0:
            time.sleep(args.show_interval)
    else:
        key = -1

    # === FINALIZE AFTER HUD IS DRAWN ===
    if args.auto_rom and is_angle_mode:
        _finalize_if_ready(args, state, t_now, frame_bgr, combined_bgr)

    return data_samples.get('pred_instances', None), key, combined_bgr

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth = param.get_distance(x, y)
        print(f"[mouse_callback] Mouse = ({x}, {y}) → Distance: {depth:.3f} meters")


def main():
    parser = ArgumentParser()
    parser.add_argument('det_config')
    parser.add_argument('det_checkpoint')
    parser.add_argument('pose_config')
    parser.add_argument('pose_checkpoint')
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--output-root', type=str, default='')
    parser.add_argument('--save-predictions', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--det-cat-id', type=int, default=0)
    parser.add_argument('--bbox-thr', type=float, default=0.3)
    parser.add_argument('--nms-thr', type=float, default=0.3)
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--draw-heatmap', action='store_true', default=False)
    parser.add_argument('--show-kpt-idx', action='store_true', default=False)
    parser.add_argument('--skeleton-style', default='mmpose', type=str,
                        choices=['mmpose', 'openpose'])
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--thickness', type=int, default=3)
    parser.add_argument('--show-interval', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--draw-bbox', action='store_true')

    parser.add_argument('--rom_test', default="full_body", type=str)
    parser.add_argument('--show3d', action='store_true')

    parser.add_argument('--plot-seconds', type=float, default=5.0)
    parser.add_argument('--plot-height', type=int, default=80)
    parser.add_argument('--plot-gap-sec', type=float, default=0.4)

    # Voice
    parser.add_argument('--voice', action='store_true', help='Enable voice control via Azure Speech SDK')
    parser.add_argument('--voice-key', type=str, default=os.environ.get("AZURE_SPEECH_KEY", ""))
    parser.add_argument('--voice-region', type=str, default=os.environ.get("AZURE_SPEECH_REGION", ""))
    parser.add_argument('--voice-mic', type=str, default="plughw:CARD=PCH,DEV=0")
    parser.add_argument('--voice-lang', type=str, default='en-US')

    # AUTO-ROM
    parser.add_argument('--auto-rom', action='store_true', help='Auto-detect min/max/ROM, save images + JSON')
    parser.add_argument('--rom-ema-alpha', type=float, default=0.25)
    parser.add_argument('--rom-v-go', type=float, default=12.0)
    parser.add_argument('--rom-v-stop', type=float, default=6.0)
    parser.add_argument('--rom-hold-sec', type=float, default=0.4)
    parser.add_argument('--rom-std-max', type=float, default=2.0)
    parser.add_argument('--rom-min-amplitude', type=float, default=10.0)
    parser.add_argument('--rom-timeout-sec', type=float, default=12.0)
    parser.add_argument('--rom-start-amp', type=float, default=5.0, help='Min |angle| from baseline to start trial (deg)')
    parser.add_argument('--rom-baseline-tol', type=float, default=5.0, help='How close counts as baseline (deg)')
    parser.add_argument('--rom-baseline-hold-sec', type=float, default=0.6, help='Hold near baseline to finalize (s)')
    parser.add_argument('--rom-refractory-sec', type=float, default=1.0, help='Cooldown after finalize before any re-arm (s)')
    parser.add_argument('--rom-lock-refractory-sec', type=float, default=0.8, help='Cooldown after locking MIN/MAX (s)')

    assert has_mmdet, 'Please install mmdet to run the demo.'
    args = parser.parse_args()

    if args.auto_rom:
        assert args.output_root != '', "--auto-rom requires --output-root to save results."

    args.kpt_preset_name = args.rom_test
    set_kpt_preset(args, args.kpt_preset_name)

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_{os.path.splitext(os.path.basename(args.input))[0]}.json'

    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    state = SessionState()

    # Start voice (optional)
    voice = None
    if args.voice:
        voice = VoiceController(
            subscription_key=args.voice_key,
            region=args.voice_region,
            device_name=args.voice_mic,
            language=args.voice_lang
        )
        voice.start()

    # Determine input type
    if args.input == 'webcam':
        input_type = 'webcam'
    elif args.input == 'realsense':
        input_type = 'realsense'
    else:
        mt = mimetypes.guess_type(args.input)[0]
        input_type = mt.split('/')[0] if mt else 'video'

    if input_type == 'image':
        pred_instances, key, combined_bgr = process_one_image(
            args, args.input, None, detector, pose_estimator, visualizer, 0, state, voice)
        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)
        if output_file:
            mmcv.imwrite(combined_bgr, output_file)

    elif input_type == 'realsense':
        if rs is None:
            raise ImportError("pyrealsense2 is required for 'realsense' input mode.")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline_profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = depth_frame

                pred_instances, key, combined_bgr = process_one_image(
                    args, color_image, depth_image, detector, pose_estimator, visualizer, 0.001, state, voice)

                if key == 27:
                    break
        finally:
            pipeline.stop()
            if voice:
                voice.stop()

    elif input_type in ['webcam', 'video']:
        cap = cv2.VideoCapture(0) if args.input == 'webcam' else cv2.VideoCapture(args.input)
        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                frame_idx += 1
                if not success:
                    break

                pred_instances, key, combined_bgr = process_one_image(
                    args, frame, None, detector, pose_estimator, visualizer, 0.001, state, voice)

                if key == 27:
                    break

                if args.save_predictions:
                    pred_instances_list.append(dict(frame_id=frame_idx, instances=split_instances(pred_instances)))

                if output_file:
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        H, W = combined_bgr.shape[:2]
                        video_writer = cv2.VideoWriter(output_file, fourcc, 25, (W, H))
                    video_writer.write(combined_bgr)
        finally:
            if video_writer:
                video_writer.release()
            cap.release()
            if voice:
                voice.stop()

    else:
        args.save_predictions = False
        if voice:
            voice.stop()
        raise ValueError(f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(dict(meta_info=pose_estimator.dataset_meta, instance_info=pred_instances_list), f, indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type_print = input_type.replace('webcam', 'video')
        print_log(f'the output {input_type_print} has been saved at {output_file}', logger='current', level=logging.INFO)


if __name__ == '__main__':
    main()
