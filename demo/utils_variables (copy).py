import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "mmpose/configs/_base_/datasets/"))
import coco_wholebody as cw  # reference only

# joint names to id mapping
# https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py

# ---------- Special visualization subsets ----------

full_body = [
    # Upper Body
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    # Lower Body
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    # Left Hand
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    # Right Hand
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132
]

# ---------- Vector-pair ROM presets ONLY (triplets fully removed) ----------
# Value form:
#  - vector pair: [[P0,P1],[Q0,Q1]], each point is int id or list[int] to average
# Special keys:
#  - "full_body" -> sentinel
#  - "133" -> list(range(133)) for viz only

_rom_single_vectors = {
    "full_body": "full_body",
    "133": [i for i in range(0, 133)],

    # elbow
    "left_elbow_flexion":   [[7, 5], [7, 9]],
    "right_elbow_flexion":  [[8, 6], [8, 10]],

    # shoulder flexion/abduction use spine proxy [[shoulder, wrist], [mid-shoulders, mid-hips]]
    "left_shoulder_flexion":        [[5, 7], [5, 11]],
    "right_shoulder_flexion":       [[6, 8], [[5, 6], [11, 12]]],
    "left_shoulder_abduction":      [[5, 7], [[5, 6], [11, 12]]],
    "right_shoulder_abduction":     [[6, 8], [[5, 6], [11, 12]]],
    "left_shoulder_extension":      [[5, 7], [5, 11]],
    "right_shoulder_extension":      [[5, 7], [5, 11]],

    # shoulder rotation placeholders
    "left_shoulder_external_rotation":  [[7, 5], [7, 9]],
    "right_shoulder_external_rotation": [[8, 6], [8, 10]],
    "left_shoulder_internal_rotation":  [[7, 9], [7, 5]],
    "right_shoulder_internal_rotation": [[8, 10], [8, 6]],

    # knee
    "left_knee_flexion":     [[13, 11], [13, 15]],
    "right_knee_flexion":    [[14, 12], [14, 16]],
    "left_knee_extension":   [[13, 11], [13, 15]],
    "right_knee_extension":  [[14, 12], [14, 16]],

    # ankle (toe average)
    "left_ankle_dorsiflexion":     [[15, 13], [15, [17, 18]]],
    "right_ankle_dorsiflexion":    [[16, 14], [16, [20, 21]]],
    "left_ankle_plantarflexion":   [[15, 13], [15, [17, 18]]],
    "right_ankle_plantarflexion":  [[16, 14], [16, [20, 21]]],
}

CATEGORY_SIDES = {
    "elbow_flexion": {"left":"left_elbow_flexion","right":"right_elbow_flexion","both":"elbow_flexion"},
    "shoulder_flexion": {"left":"left_shoulder_flexion","right":"right_shoulder_flexion","both":"shoulder_flexion"},
    "shoulder_abduction": {"left":"left_shoulder_abduction","right":"right_shoulder_abduction","both":"shoulder_abduction"},
    "shoulder_external_rotation": {"left":"left_shoulder_external_rotation","right":"right_shoulder_external_rotation","both":"shoulder_external_rotation"},
    "shoulder_internal_rotation": {"left":"left_shoulder_internal_rotation","right":"right_shoulder_internal_rotation","both":"shoulder_internal_rotation"},
    "knee_flexion": {"left":"left_knee_flexion","right":"right_knee_flexion","both":"knee_flexion"},
    "knee_extension": {"left":"left_knee_extension","right":"right_knee_extension","both":"knee_extension"},
    "ankle_dorsiflexion": {"left":"left_ankle_dorsiflexion","right":"right_ankle_dorsiflexion","both":"ankle_dorsiflexion"},
    "ankle_plantarflexion": {"left":"left_ankle_plantarflexion","right":"right_ankle_plantarflexion","both":"ankle_plantarflexion"},
}

CATEGORY_ORDER = [
    "elbow_flexion",
    "shoulder_flexion",
    "shoulder_abduction",
    "shoulder_external_rotation",
    "shoulder_internal_rotation",
    "knee_flexion",
    "knee_extension",
    "ankle_dorsiflexion",
    "ankle_plantarflexion",
]

def get_vectors_for_preset(name):
    """Return a list of vector-pairs for the preset (handles 'both' by concatenation)."""
    if name in ("full_body", "133"):
        return []
    if name in _rom_single_vectors:
        vec = _rom_single_vectors[name]
        if isinstance(vec, list) and len(vec) == 2 and all(isinstance(v, (list, tuple)) for v in vec):
            return [vec]
    for _cat, sides in CATEGORY_SIDES.items():
        if sides.get("both") == name:
            left_name, right_name = sides["left"], sides["right"]
            return get_vectors_for_preset(left_name) + get_vectors_for_preset(right_name)
    return []

def get_show_kpt_subset(name):
    """Subset of joint IDs to render for the current preset, derived from vector-pairs only."""
    if name == "full_body":
        return full_body
    if name == "133":
        return [i for i in range(0, 133)]
    ids = set()
    def _add(p):
        if isinstance(p, (list, tuple)):
            for q in p: _add(q)
        else:
            try: ids.add(int(p))
            except Exception: pass
    vecs = get_vectors_for_preset(name)
    for vec in vecs:
        if isinstance(vec, (list, tuple)) and len(vec) == 2:
            (P0,P1),(Q0,Q1) = vec
            _add(P0); _add(P1); _add(Q0); _add(Q1)
    return sorted(ids)

# Export for demo: name -> kpt subset
def _all_preset_names():
    names = set(_rom_single_vectors.keys())
    for _cat, sides in CATEGORY_SIDES.items():
        names.add(sides["both"])
    return sorted(names)

rom_test = {}
for _name in _all_preset_names():
    try:
        rom_test[_name] = get_show_kpt_subset(_name)
    except Exception:
        rom_test[_name] = []

# ==============================
# Offline RGB/D batch defaults
# ==============================

# Default FPS used to convert HH:MM:SS(.ms) timestamps to frame indices
OFFLINE_DEFAULT_FPS: float = 30.0

# Default parent directory for batch outputs if a caller does not supply one.
# Callers may still pass an explicit --output-root or similar to override.
OFFLINE_DEFAULT_OUTPUT_ROOT: str = "outputs"

# Prefix for auto-created batch session folders, e.g., "batch_20250930_181530"
OFFLINE_SESSION_PREFIX: str = "batch_"

# Max display width when showing compare panels interactively during batch runs.
# Only used by UIs that choose to read it; safe to ignore elsewhere.
OFFLINE_PANEL_MAX_WIDTH: int = 1400
