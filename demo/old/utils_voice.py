# utils_voice.py
# Voice presets, patterns, Azure Speech wrapper, and command adapter.
# Standalone and importable by topdown_demo_with_mmdet.py

import os
import time
import re
import threading
import queue
import numpy as np

from utils_variables import rom_test
from utils import set_kpt_preset

try:
    import azure.cognitiveservices.speech as speechsdk
    _HAS_AZURE_SPEECH = True
except (ImportError, ModuleNotFoundError):
    _HAS_AZURE_SPEECH = False


# ----------------------------
# Aliases / Patterns
# ----------------------------

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
        seen = set(); dedup = []
        for p in patterns:
            if p not in seen:
                seen.add(p); dedup.append(p)
        aliases[name] = dedup
    return aliases


VOICE_PRESET_ALIASES = _build_voice_aliases_from_rom(rom_test)

VOICE_COMMAND_PATTERNS = {
    #"next":  [r"\bnext\b", r"\bforward\b", r"\bgo next\b"],
    #"prev":  [r"\bprevious\b", r"\bback\b", r"\bgo back\b"],
    #"zero":  [r"\bzero\b", r"\brezero\b", r"\breset baseline\b", r"\bset baseline\b", r"\bcalibrate\b"],
    #"start": [r"\bstart\b", r"\bbegin\b", r"\bgo\b", r"\bstart test\b", r"\brecord\b"],
}

def _match_any(text: str, patterns):
    for p in patterns:
        if re.search(p, text):
            return True
    return False

# ----------------------------
# Voice engine wrapper
# ----------------------------

class VoiceController:
    """Thin Azure Speech SDK wrapper that emits ('preset', name) or ('start'|'next'|'prev'|'zero', None)."""
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

        # Preset names (STRICT-FIRST, then FALLBACK)
        # 1) Strict: require ALL tokens from the preset name to appear as whole words.
        #    e.g., "left shoulder abduction" must contain 'left' 'shoulder' 'abduction'.
        for preset_name in VOICE_PRESET_ALIASES.keys():
            toks = preset_name.split('_')  # ['left','shoulder','abduction']
            if all(re.search(rf"\b{re.escape(tok)}\b", text) for tok in toks):
                self.q.put(('preset', preset_name)); return

        # 2) Fallback: accept looser aliases (side+joint only, flexible gaps, etc.)
        for preset_name, patterns in VOICE_PRESET_ALIASES.items():
            if _match_any(text, patterns):
                self.q.put(('preset', preset_name)); return

        # Generic commands
        if _match_any(text, VOICE_COMMAND_PATTERNS["start"]):
            self.q.put(('start', None)); return
        if _match_any(text, VOICE_COMMAND_PATTERNS["next"]):
            self.q.put(('next', None)); return
        if _match_any(text, VOICE_COMMAND_PATTERNS["prev"]):
            self.q.put(('prev', None)); return
        if _match_any(text, VOICE_COMMAND_PATTERNS["zero"]):
            self.q.put(('zero', None)); return

        # Fallback heuristics
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


# ----------------------------
# App-facing command adapter
# ----------------------------

def cycle_for_voice(name: str, step: int) -> str:
    preset_names = list(rom_test.keys())
    i = preset_names.index(name) if name in preset_names else 0
    return preset_names[(i + step) % len(preset_names)]


def apply_voice_command(args, state, cmd_tuple, reset_auto_rom_state=None):
    """
    Translate voice tuples into state transitions.
    Optionally pass a `reset_auto_rom_state(state, keep_trial_ready=False)` callable from the main app.
    """
    if cmd_tuple is None:
        return
    cmd, arg = cmd_tuple

    # ---------------------------- DEBUG MODE ----------------------------
    if getattr(args, 'debug', False):
        if cmd == 'preset':
            if arg in rom_test:
                set_kpt_preset(args, arg)
                if args.zero and state is not None:
                    state.auto_zero_pending = True
                    state.auto_zero_start_time = None
                    state.auto_zero_buffer.clear()
                    state.baseline_deg = None
                    state.baseline_set_ts = None
                    state.last_angle = None
                    state.angle_series.clear()
                    state.first_auto_arm_consumed = False
                    state.last_zero_source = 'auto'
                else:
                    if state is not None:
                        state.auto_zero_pending = False
                print(f"[VOICE][DEBUG] preset → '{arg}' ({'zeroing on' if args.zero else 'no-zero'})")
            else:
                print(f"[VOICE] Unknown preset: {arg}")
            return

        elif cmd in ('next', 'prev'):
            step = +1 if cmd == 'next' else -1
            set_kpt_preset(args, cycle_for_voice(args.kpt_preset_name, step))
            if args.zero and state is not None:
                state.auto_zero_pending = True
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                state.baseline_deg = None
                state.baseline_set_ts = None
                state.last_angle = None
                state.angle_series.clear()
                state.first_auto_arm_consumed = False
                state.last_zero_source = 'auto'
            else:
                if state is not None:
                    state.auto_zero_pending = False
            print(f"[VOICE][DEBUG] preset → '{args.rom_test}' ({'zeroing on' if args.zero else 'no-zero'})")
            return

        elif cmd == 'zero':
            if not args.zero:
                print("[VOICE][DEBUG] Ignoring 'zero' (no-zero mode).")
                return
            if (state is not None) and np.isfinite(state.current_raw_angle):
                state.baseline_deg = float(state.current_raw_angle)
                state.baseline_set_ts = time.time()
                state.angle_series.clear()
                state.last_angle = None
                state.auto_zero_pending = False
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                # debug: never arm; just update display baseline
                state.trial_armed = False
                state.first_auto_arm_consumed = True
                state.last_zero_source = 'manual'
                print(f"[VOICE][DEBUG] Baseline set to {state.baseline_deg:.2f}°")
            else:
                print("[VOICE][DEBUG] Cannot zero: no valid angle this frame.")
            return

        elif cmd == 'start':
            print("[VOICE][DEBUG] Ignoring 'start' in debug.")
            return

        return  # ignore anything else in debug

    # -------------------------- NON-DEBUG MODE --------------------------
    if cmd == 'preset':
        if arg in rom_test:
            set_kpt_preset(args, arg)
            if state is not None:
                if args.zero:
                    state.auto_zero_pending = True
                    state.auto_zero_start_time = None
                    state.auto_zero_buffer.clear()
                    state.baseline_deg = None
                    state.baseline_set_ts = None
                    state.last_angle = None
                    state.angle_series.clear()
                    if callable(reset_auto_rom_state):
                        reset_auto_rom_state(state)
                    state.first_auto_arm_consumed = False
                    state.last_zero_source = 'auto'
                else:
                    state.auto_zero_pending = False
        else:
            print(f"[VOICE] Unknown preset: {arg}")

    elif cmd in ('next', 'prev'):
        step = +1 if cmd == 'next' else -1
        set_kpt_preset(args, cycle_for_voice(args.kpt_preset_name, step))
        if state is not None:
            if args.zero:
                state.auto_zero_pending = True
                state.auto_zero_start_time = None
                state.auto_zero_buffer.clear()
                state.baseline_deg = None
                state.baseline_set_ts = None
                state.last_angle = None
                state.angle_series.clear()
                if callable(reset_auto_rom_state):
                    reset_auto_rom_state(state)
                state.first_auto_arm_consumed = False
                state.last_zero_source = 'auto'
            else:
                state.auto_zero_pending = False

    elif cmd == 'zero':
        if not args.zero:
            print("[VOICE] Ignoring 'zero' (no-zero mode).")
            return
        if (state is not None) and np.isfinite(state.current_raw_angle):
            state.baseline_deg = float(state.current_raw_angle)
            state.baseline_set_ts = time.time()
            state.angle_series.clear()
            state.last_angle = None
            state.auto_zero_pending = False
            state.auto_zero_start_time = None
            state.auto_zero_buffer.clear()
            if callable(reset_auto_rom_state):
                reset_auto_rom_state(state, keep_trial_ready=False)
            # manual zero does NOT auto-arm
            state.trial_armed = False
            state.first_auto_arm_consumed = True
            state.last_zero_source = 'manual'
            print(f"[VOICE] Baseline set to {state.baseline_deg:.2f}°")
        else:
            print("[VOICE] Cannot zero: no valid angle this frame.")

    # elif cmd == 'start':
    #     # unchanged: allows arming in non-debug
    #     if state.baseline_deg is None and args.zero:
    #         state.arm_after_baseline = True
    #         print("[ROM] Start queued. Will arm as soon as baseline locks.")
    #     else:
    #         state.trial_armed = True
    #         state.arm_after_baseline = False
    #         state.first_auto_arm_consumed = True
    #         print("[ROM] Armed for a single repetition. Move when ready.")

# ----------------------------
# Optional CLI helpers
# ----------------------------

def add_voice_args(parser):
    parser.add_argument('--voice', action='store_true', help='Enable voice control via Azure Speech SDK')
    parser.add_argument('--voice-key', type=str, default=os.environ.get("AZURE_SPEECH_KEY", ""))
    parser.add_argument('--voice-region', type=str, default=os.environ.get("AZURE_SPEECH_REGION", ""))
    parser.add_argument('--voice-mic', type=str, default="plughw:CARD=PCH,DEV=0")
    parser.add_argument('--voice-lang', type=str, default='en-US')
    return parser


def init_voice_from_args(args):
    if not getattr(args, 'voice', False):
        return None
    key = getattr(args, 'voice_key', None) or ""
    reg = getattr(args, 'voice_region', None) or ""
    mic = getattr(args, 'voice_mic', None) or None
    lang = getattr(args, 'voice_lang', None) or 'en-US'
    vc = VoiceController(key, reg, mic, lang)
    vc.start()
    return vc
