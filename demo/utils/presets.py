# presets.py
"""
Single source of truth for ROM presets/metadata used by both calculators and viz.

Design:
- DATA ONLY (plus tiny accessors). No geometry/math here.
- Calculators: generic ROMs use get_vectors_for_preset(); specialized ROMs ignore it.
- Viz: default vec overlay obeys DRAW_POLICY and whether a drawer is registered.

Conventions:
- Vec-pair = [[P0, P1], [Q0, Q1]]
  where each endpoint can be:
    - int  (single joint index), or
    - list/tuple of ints (average of those joints)
  (Matches your _avg_point() logic.)
"""

from typing import Dict, List, Optional, Tuple, Union

# -------------------------------
# Types (for readability only)
# -------------------------------
VecEndpoint = Union[int, List[int], Tuple[int, ...]]
VecPair = List[List[VecEndpoint]]     # [[P0, P1], [Q0, Q1]]
VecPairList = List[VecPair]           # allow multiple vec-pairs per ROM (usually 1)

# -------------------------------
# Draw policy (controls default vec overlay)
# -------------------------------
DRAW_ALWAYS = "always"            # always draw default vec overlay
DRAW_NEVER = "never"              # never draw default vec overlay (drawer-only)
DRAW_IF_NO_DRAWER = "if_no_drawer"  # draw only if no drawer registered (default)

# -------------------------------
# Canonical vec-pairs per ROM
# (list-of-vec-pairs so you can expand later if needed)
# -------------------------------
VECPAIR: Dict[str, VecPairList] = {
    # --- Elbows (upper arm vs forearm) ---
    "left_elbow_flexion":  [[[7, 5], [7, 9]]],
    "right_elbow_flexion": [[[8, 6], [8, 10]]],

    # --- Knees (thigh vs shank) ---
    "left_knee_flexion":   [[[13, 11], [13, 15]]],
    "right_knee_flexion":  [[[14, 12], [14, 16]]],

    # --- Ankles (shank vs foot); tweak endpoints to your convention ---
    "left_ankle_dorsiflexion":  [[[13, 15], [15, 17]]],
    "right_ankle_dorsiflexion": [[[14, 16], [16, 18]]],

    # --- Shoulders (example abduction/extension vs torso axis; adjust as needed) ---
    "left_shoulder_abduction":  [[[5, 7], [5, 11]]],
    "right_shoulder_abduction": [[[6, 8], [6, 12]]],
    "left_shoulder_extension":  [[[5, 7], [5, 11]]],
    "right_shoulder_extension": [[[6, 8], [6, 12]]],

    # NOTE:
    # - We intentionally omit a vec-pair for:
    #     * left_shoulder_internal_rotation  -> special 3D calc + custom drawer
    #     * left_shoulder_flexion            -> inter-frame segment rotation; viz via drawer
}

# -------------------------------
# ROMs that inherently need two timestamps
# (e.g., inter-frame rotation ROMs). Others default to single-frame.
# -------------------------------
REQUIRES_T2: Dict[str, bool] = {
    "left_shoulder_flexion": True,
    "left_shoulder_extension": True,
    "left_shoulder_abduction": True,
    # Add more if you introduce right_shoulder_flexion, etc.
    # "right_shoulder_flexion": True,
    # You can also mark abduction/extension True if you always compare t1 vs t2:
    # "left_shoulder_abduction": True,
    # "right_shoulder_abduction": True,
    # "left_shoulder_extension": True,
    # "right_shoulder_extension": True,
}

# -------------------------------
# Viz draw policy per ROM
# (controls the default vec overlay only; drawers still run if registered)
# -------------------------------
DRAW_POLICY: Dict[str, str] = {
    # Internal rotation should be drawer-only (torso-normal + forearm arrows):
    "left_shoulder_internal_rotation": DRAW_NEVER,

    # Example: force vecs even if a drawer exists
    # "left_elbow_flexion": DRAW_ALWAYS,
}

# -------------------------------
# Optional: per-ROM joint subsets for viz (keep if you use it; otherwise leave empty)
# -------------------------------
_SHOW_KPT_SUBSET: Dict[str, List[int]] = {
    # Example subsets (purely for UI/highlighting):
    # "left_elbow_flexion": [5, 7, 9],
    # "right_elbow_flexion": [6, 8, 10],
}

# -------------------------------
# Accessors (tiny, stable API)
# -------------------------------

def get_vectors_for_preset(name: str) -> VecPairList:
    """
    Return a list of vec-pairs for a ROM; empty list means:
      - no default vectors for this ROM (likely a special calc/drawer handles it).
    """
    return VECPAIR.get(name, [])

def needs_t2(name: str) -> bool:
    """Whether this ROM should use a second timestamp by default."""
    return REQUIRES_T2.get(name, False)

def draw_policy(name: str) -> str:
    """
    Return one of DRAW_ALWAYS, DRAW_NEVER, DRAW_IF_NO_DRAWER.
    Defaults to DRAW_IF_NO_DRAWER if not specified.
    """
    return DRAW_POLICY.get(name, DRAW_IF_NO_DRAWER)

def get_show_kpt_subset(name: str) -> Optional[List[int]]:
    """Optional: which joints to emphasize for viz (if you use this feature)."""
    return _SHOW_KPT_SUBSET.get(name)

def list_roms() -> List[str]:
    """
    Utility: list all ROM keys known to presets (union of VECPAIR and other maps).
    """
    names = set(VECPAIR.keys()) | set(REQUIRES_T2.keys()) | set(DRAW_POLICY.keys()) | set(_SHOW_KPT_SUBSET.keys())
    return sorted(names)

def register_vecpair(rom_name: str, vec_pair: VecPair, replace: bool = False) -> None:
    """
    Dynamically add/override a vec-pair at runtime.
    - If replace=False: append as an additional vec-pair for that ROM.
    - If replace=True: replace any existing vec-pairs for that ROM.
    """
    if replace or rom_name not in VECPAIR:
        VECPAIR[rom_name] = [vec_pair]
    else:
        VECPAIR[rom_name].append(vec_pair)
