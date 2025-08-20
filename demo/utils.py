# utils.py (Python 3.8-friendly hints)
from typing import Dict, List
from utils_variables import presets  # only dependency

PRESET_NAMES = list(presets.keys())

def set_kpt_preset(args, name: str) -> None:
    if name in presets:
        args.kpt_preset_name = name
        args.show_kpt_subset = presets[name]
        print(f"[KPT] preset -> {name}")
    else:
        print(f"[KPT] unknown preset: {name}")

def _cycle(name: str, step: int) -> str:
    i = PRESET_NAMES.index(name)
    return PRESET_NAMES[(i + step) % len(PRESET_NAMES)]

def handle_hotkeys_for_presets(args, key: int) -> None:
    if key == ord(']'):
        set_kpt_preset(args, _cycle(args.kpt_preset_name, +1))
    elif key == ord('['):
        set_kpt_preset(args, _cycle(args.kpt_preset_name, -1))
    elif key == ord('1'):
        set_kpt_preset(args, 'full_body')
    elif key == ord('2'):
        set_kpt_preset(args, '133')
