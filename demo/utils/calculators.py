# calculators.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable, List

from .presets import get_vectors_for_preset
from .geometry import (
    angle2d_from_vecpair,
    angle3d_from_vecpair,
    angle2d_between_segments_across_frames,
    angle3d_between_segments_across_frames,
    angle3d_internal_rotation_left_simple,
)

# =========================
# Result container
# =========================

@dataclass
class RomResult:
    mode_used: str                                   # "3D" | "2D" | "2D_fallback"
    ang1_deg: Optional[float]                        # angle at t1 (if applicable)
    ang2_deg: Optional[float]                        # angle at t2 (if applicable)
    delta_deg: Optional[float]                       # ang2 - ang1 (None if not meaningful)
    segment_rotation_t1_to_t2_deg: Optional[float]   # only for inter-frame segment rotation ROMs


# =========================
# Vec-pair resolver (allows per-ROM endpoint overrides later if needed)
# =========================

VECPAIR_OVERRIDE: Dict[str, List] = {
    # Example to override a single ROM’s vec endpoints without touching presets:
    # "left_elbow_flexion": [[7, 5], [7, 9]],
}

def resolve_vecpair(rom_name: str) -> Optional[List]:
    """
    Returns a single vec_pair [[P0,P1],[Q0,Q1]] if available, else None.
    Prefers overrides, falls back to presets.
    """
    if rom_name in VECPAIR_OVERRIDE:
        return VECPAIR_OVERRIDE[rom_name]
    vecs = get_vectors_for_preset(rom_name)
    if not vecs:
        raise ValueError(f"No vec-pair for ROM '{rom_name}' and no specialized calculator.")
    vec_pair = vecs[0]
    if not (isinstance(vec_pair, (list, tuple)) and len(vec_pair) == 2):
        return None
    return vec_pair

# =========================
# Generic per-frame vec-pair ROM
# =========================

def _per_frame_vecpair(rom_name: str,
                       k1, k2,
                       d1, d2,
                       rgb_size: Tuple[int, int],
                       median_k: int = 5) -> RomResult:
    vec_pair = resolve_vecpair(rom_name)
    if vec_pair is None:
        raise ValueError(f"No vec-pair available for ROM '{rom_name}'")

    # Try t1 in 3D
    mode_used = "2D"
    a1_3d = None
    if d1 is not None:
        a1_3d, ok1 = angle3d_from_vecpair(k1, vec_pair, d1, rgb_size, median_k=median_k)

    # Try t2 in 3D only if k2/d2 exist
    a2_3d = None
    if k2 is not None and d2 is not None:
        a2_3d, ok2 = angle3d_from_vecpair(k2, vec_pair, d2, rgb_size, median_k=median_k)

    if a1_3d is not None and (k2 is None or a2_3d is not None):
        # 3D for all frames we actually have
        ang1 = float(a1_3d)
        ang2 = float(a2_3d) if k2 is not None else None
        mode_used = "3D"
    else:
        # 2D fallback(s)
        a1_2d = angle2d_from_vecpair(k1, vec_pair)
        if a1_2d is None:
            raise ValueError(f"{rom_name}: 2D angle at t1 failed (degenerate/missing keypoints).")
        ang1 = float(a1_2d)

        if k2 is not None:
            a2_2d = angle2d_from_vecpair(k2, vec_pair)
            if a2_2d is None:
                raise ValueError(f"{rom_name}: 2D angle at t2 failed (degenerate/missing keypoints).")
            ang2 = float(a2_2d)
        else:
            ang2 = None

        # Label as 2D_fallback only if depth was available but unusable
        had_depth = (d1 is not None) and ((k2 is None) or (d2 is not None))
        mode_used = "2D_fallback" if had_depth else "2D"

    delta = (ang2 - ang1) if (ang2 is not None) else None
    return RomResult(mode_used=mode_used, ang1_deg=ang1, ang2_deg=ang2, delta_deg=delta,
                     segment_rotation_t1_to_t2_deg=None)

# =========================
# Special ROMs
# =========================

def _left_shoulder_flexion(k1, k2, d1, d2, rgb_size: Tuple[int, int], median_k: int = 5) -> RomResult:
    """
    Inter-frame rotation of the LEFT upper arm (segment 5->7) from t1 to t2.
    - Prefer 3D with depth; fallback to 2D segment rotation if needed.
    - For downstream viz that expects ang1/ang2, we duplicate a single angle into both and set delta=0.
    - Also expose the single rotation in segment_rotation_t1_to_t2_deg.
    """
    mode_used = "2D"
    seg_angle = None

    if (d1 is not None) and (d2 is not None):
        seg_angle = angle3d_between_segments_across_frames(
            k1, k2, d1, d2, rgb_size, median_k=median_k, j_sh=5, j_el=7
        )
        if seg_angle is not None:
            mode_used = "3D"

    if seg_angle is None:
        seg_angle = angle2d_between_segments_across_frames(k1, k2, j_sh=5, j_el=7)
        if seg_angle is None:
            raise ValueError("left_shoulder_flexion: inter-frame segment rotation failed (degenerate/missing keypoints).")
        mode_used = "2D_fallback" if (d1 is not None and d2 is not None) else "2D"

    ang = float(seg_angle)
    return RomResult(
        mode_used=mode_used,
        ang1_deg=ang,                         # duplicated for panel display
        ang2_deg=ang,
        delta_deg=0.0,                        # not meaningful for this ROM
        segment_rotation_t1_to_t2_deg=ang,    # dedicated field
    )

def _left_shoulder_internal_rotation(k1, k2, d1, d2, rgb_size: Tuple[int, int], median_k: int = 5) -> RomResult:
    
    if d1 is None:
        raise ValueError("left_shoulder_internal_rotation requires depth at t1.")
    a1 = angle3d_internal_rotation_left_simple(k1, d1, rgb_size, median_k=median_k)
    if a1 is None:
        raise ValueError("left_shoulder_internal_rotation t1 failed (missing/degenerate 3D points).")

    ang2 = None
    if k2 is not None and d2 is not None:
        a2 = angle3d_internal_rotation_left_simple(k2, d2, rgb_size, median_k=median_k)
        if a2 is None:
            raise ValueError("left_shoulder_internal_rotation t2 failed (missing/degenerate 3D points).")
        ang2 = float(a2)

    return RomResult(
        mode_used="3D",
        ang1_deg=float(a1),
        ang2_deg=ang2,
        delta_deg=(ang2 - float(a1)) if ang2 is not None else None,
        segment_rotation_t1_to_t2_deg=None,
    )

# =========================
# Registry
# =========================

# Signature of per-ROM functions placed in _REGISTRY:
#   fn(k1, k2, d1, d2, rgb_size, median_k) -> RomResult
RomFn = Callable[[any, any, Optional[dict], Optional[dict], Tuple[int, int], int], RomResult]

_REGISTRY: Dict[str, RomFn] = {
    # Special behaviors
    "left_shoulder_flexion": _left_shoulder_flexion,
    "left_shoulder_internal_rotation": _left_shoulder_internal_rotation,

    # You can add more special cases here as needed:
    # "right_shoulder_flexion": _right_shoulder_flexion_custom,
    # ...
}

# Any ROM name not listed above will fall back to the generic per-frame vec-pair behavior.
def compute(rom_name: str,
            k1, k2=None,
            d1: Optional[dict]=None, d2: Optional[dict]=None,
            rgb_size: Tuple[int, int]=(640, 480),
            median_k: int = 5) -> RomResult:
    #print("11111")
    fn = _REGISTRY.get(rom_name, None)
    if fn is not None:
        return fn(k1, k2, d1, d2, rgb_size, median_k)
    # default
    return _per_frame_vecpair(rom_name, k1, k2, d1, d2, rgb_size, median_k)

