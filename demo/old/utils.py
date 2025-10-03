from typing import Dict, List, Optional, Tuple
from utils_variables import rom_test, CATEGORY_ORDER, CATEGORY_SIDES  # single source of truth
import os
from pathlib import Path
import json
import cv2
import numpy as np

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

# =========================
# Offline RGB/D batch utils
# =========================

_IMAGE_EXTS = (".png", ".jpg", ".jpeg")

def parse_hhmmss_to_seconds(ts: str) -> float:
    """
    Convert 'HH:MM:SS' or 'HH:MM:SS.mmm' to seconds (float).
    Raises ValueError on malformed input.
    """
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad timestamp '{ts}'")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return float(h * 3600 + m * 60) + s


def frame_index_from_ts(ts: str, fps: float) -> int:
    """
    Map timestamp string to zero-based frame index using floor(seconds*fps).
    """
    return int(max(0, int(parse_hhmmss_to_seconds(ts) * float(fps))))



def _find_image_with_index(root: Path, idx: int) -> Optional[Path]:
    """
    Look for {root}/{idx:06d}{.png|.jpg|.jpeg}.
    Returns Path or None.
    """
    stem = f"{idx:06d}"
    for ext in _IMAGE_EXTS:
        p = root / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _imread_color(path: Path):
    """
    BGR uint8 image or None.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def _imread_depth(path: Path):
    """
    Read depth as-is. Supports 16-bit PNGs or 32F EXRs if present.
    Returns ndarray or None.
    """
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return depth


class RGBDFrameLoader:
    """
    Minimal offline loader for numbered RGB and optional depth frames.

    Expected layout:
      rgb_dir/000001.png (or .jpg/.jpeg)
      depth_dir/000001.png  [optional]

    Notes:
      - No timing file required; batch mode maps HH:MM:SS -> index via fps.
      - Depth is returned as raw ndarray (no scaling). Caller handles units.
    """

    def __init__(self, rgb_dir: str, depth_dir: Optional[str] = None, fps: float = 30.0):
        self.rgb_dir = Path(rgb_dir) if rgb_dir else None
        self.depth_dir = Path(depth_dir) if depth_dir else None
        if not self.rgb_dir or not self.rgb_dir.is_dir():
            raise NotADirectoryError(f"rgb_dir invalid: {rgb_dir}")
        if self.depth_dir and not self.depth_dir.is_dir():
            # Allow missing depth_dir if user supplied None
            raise NotADirectoryError(f"depth_dir invalid: {depth_dir}")
        self.fps = float(fps)

    def get_by_index(self, idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[str]]:
        """
        Returns:
          rgb_img, depth_img, rgb_path_str, depth_path_str
        Any missing element is returned as None.
        """
        rgb_path = _find_image_with_index(self.rgb_dir, idx)
        rgb_img = _imread_color(rgb_path) if rgb_path else None

        depth_img, depth_path_str = None, None
        if self.depth_dir is not None:
            depth_path = _find_image_with_index(self.depth_dir, idx)
            if depth_path:
                depth_img = _imread_depth(depth_path)
                depth_path_str = str(depth_path)

        return rgb_img, depth_img, (str(rgb_path) if rgb_path else None), depth_path_str

    def get_by_timestamp(self, ts: str) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[str]]:
        """
        Map HH:MM:SS(.ms) -> frame index, then load that index.
        Returns:
          idx, rgb_img, depth_img, rgb_path_str, depth_path_str
        """
        idx = frame_index_from_ts(ts, self.fps)
        rgb, depth, rp, dp = self.get_by_index(idx)
        return idx, rgb, depth, rp, dp
