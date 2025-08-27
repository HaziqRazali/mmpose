# utils.py (Python 3.8+)
from typing import Dict, List
from utils_variables import rom_test   # single source of truth

PRESET_NAMES = list(rom_test.keys())

def set_kpt_preset(args, name: str) -> None:
    """
    Switch the active ROM preset.

    - args.rom_test becomes the active preset name
    - args.kpt_preset_name is kept for display/back-compat
    """
    if name in rom_test:
        args.kpt_preset_name = name
        args.rom_test = name
        print(f"[KPT] preset -> {name}")
    else:
        print(f"[KPT] unknown preset: {name}")

def _cycle(name: str, step: int) -> str:
    i = PRESET_NAMES.index(name)
    return PRESET_NAMES[(i + step) % len(PRESET_NAMES)]

def handle_hotkeys_for_presets(args, key: int) -> None:
    """Map keys to switch presets at runtime."""
    if key == ord(']'):
        set_kpt_preset(args, _cycle(args.kpt_preset_name, +1))
    elif key == ord('['):
        set_kpt_preset(args, _cycle(args.kpt_preset_name, -1))
    elif key == ord('1'):
        set_kpt_preset(args, 'full_body')
    elif key == ord('2'):
        set_kpt_preset(args, 'right_elbow_flexion')
    elif key == ord('3'):
        set_kpt_preset(args, 'right_shoulder_abduction')
