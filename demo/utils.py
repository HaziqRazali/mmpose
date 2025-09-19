from typing import Dict, List, Optional, Tuple
from utils_variables import rom_test, CATEGORY_ORDER, CATEGORY_SIDES  # single source of truth

# ---- Preset registries ----
PRESET_NAMES = list(rom_test.keys())
SIDES = ("left", "right", "both")

# NAV ring includes specials + category names (categories are side-aware)
NAV_ORDER = ["full_body", "133"] + CATEGORY_ORDER


def set_kpt_preset(args, name: str) -> None:
    """
    Switch the active preset for visualization.
    - args.rom_test becomes the active preset name
    - args.kpt_preset_name is kept for display/back-compat
    """
    if name in rom_test:
        args.kpt_preset_name = name
        args.rom_test = name
        print(f"[KPT] preset -> {name}")
    else:
        print(f"[KPT] unknown preset: {name}")


def _cycle_linear(name: str, step: int) -> str:
    """Legacy linear cycle over every rom_test entry (used by [ and ])."""
    i = PRESET_NAMES.index(name)
    return PRESET_NAMES[(i + step) % len(PRESET_NAMES)]


def _infer_category_and_side(preset_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (category, side) where side in {'left','right','both'},
    or (None, None) for non-category presets (e.g., 'full_body', '133').
    """
    if preset_name in ("full_body", "133"):
        return None, None

    for cat, sides in CATEGORY_SIDES.items():
        for side, name in sides.items():
            if name == preset_name:
                return cat, side
    return None, None


def _preset_for(cat: str, side: str) -> Optional[str]:
    """Map (category, side) -> preset name, or None if not available."""
    sides = CATEGORY_SIDES.get(cat, {})
    return sides.get(side)


def _init_nav_state_if_needed(args):
    """Ensure args has category/side navigation state."""
    if not hasattr(args, "curr_category") or not hasattr(args, "curr_side"):
        cat, side = _infer_category_and_side(getattr(args, "rom_test", "full_body"))
        args.curr_category = cat
        args.curr_side = side


def _nav_index_from_current(args) -> int:
    """
    Resolve current position in NAV_ORDER:
      - If on 'full_body' or '133' -> their own token index.
      - If on a category preset (left/right/both) -> index of that category token.
      - Otherwise -> start before 0 (so next/prev behaves sanely).
    """
    cur = getattr(args, "rom_test", "full_body")
    if cur in ("full_body", "133"):
        try:
            return NAV_ORDER.index(cur)
        except ValueError:
            return -1

    cat, _side = _infer_category_and_side(cur)
    if cat in CATEGORY_ORDER:
        return NAV_ORDER.index(cat)  # cat is present in NAV_ORDER
    return -1


def _to_next_in_nav(args, step: int):
    """
    Step through NAV_ORDER with keys 1/2:
      - Includes 'full_body' and '133' plus all categories.
      - When landing on a category, keep the SAME SIDE as current if possible,
        otherwise fall back to 'both'.
      - When landing on 'full_body' or '133', set that preset and clear category/side state.
    """
    _init_nav_state_if_needed(args)

    idx = _nav_index_from_current(args)
    n = len(NAV_ORDER)
    idx = (idx + step) % n
    token = NAV_ORDER[idx]

    if token in ("full_body", "133"):
        # Specials: no side context
        args.curr_category = None
        args.curr_side = None
        set_kpt_preset(args, token)
        return

    # token is a category name
    cat = token
    cur_side = args.curr_side if (args.curr_category in CATEGORY_ORDER and args.curr_side in SIDES) else "both"
    # Keep same side if available; else fall back to 'both'
    preset = _preset_for(cat, cur_side) or _preset_for(cat, "both")
    side_used = cur_side if _preset_for(cat, cur_side) else "both"

    args.curr_category = cat
    args.curr_side = side_used
    set_kpt_preset(args, preset)


def _cycle_side(args):
    """
    Within the current category, cycle Left → Right → Both → Left…
    If currently on 'full_body' or '133', jump to the FIRST category at 'both'.
    """
    _init_nav_state_if_needed(args)

    if args.curr_category not in CATEGORY_ORDER:
        # From full_body/133 → jump to first category BOTH
        args.curr_category = CATEGORY_ORDER[0]
        args.curr_side = "both"
        set_kpt_preset(args, _preset_for(args.curr_category, args.curr_side))
        return

    # Existing category; compute next side
    side = args.curr_side if args.curr_side in SIDES else "both"
    next_idx = (SIDES.index(side) + 1) % len(SIDES)
    next_side = SIDES[next_idx]

    # If next_side not available, fallback to 'both'
    preset = _preset_for(args.curr_category, next_side) or _preset_for(args.curr_category, "both")
    args.curr_side = next_side if _preset_for(args.curr_category, next_side) else "both"
    set_kpt_preset(args, preset)


def handle_hotkeys_for_presets(args, key: int) -> None:
    """
    Hotkeys:
      - '1': next in NAV ring (full_body → 133 → categories)
      - '2': prev in NAV ring
      - '3': cycle side within current category: Left → Right → Both → Left…
    """

    # New NAV ring
    if key == ord('1'):
        _to_next_in_nav(args, -1)
    elif key == ord('2'):
        _to_next_in_nav(args, +1)
    elif key == ord('3'):
        _cycle_side(args)