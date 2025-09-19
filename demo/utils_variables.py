import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "mmpose/configs/_base_/datasets/"))
import coco_wholebody as cw

# joint names to id mapping
# https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py

full_body = [

    # Upper Body
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

    # Lower Body
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,

    # Left Hand (Root + Fingers)
    91, 92, 93, 94, 95,
    96, 97, 98, 99,
    100, 101, 102, 103,
    104, 105, 106, 107,
    108, 109, 110, 111,

    # Right Hand (Root + Fingers)
    112, 113, 114, 115, 116,
    117, 118, 119, 120,
    121, 122, 123, 124,
    125, 126, 127, 128,
    129, 130, 131, 132
]

def _flatten_unique(seq):
    """Flatten nested lists/tuples and keep unique ints in stable order."""
    out = []
    seen = set()
    def _walk(x):
        if isinstance(x, (list, tuple)):
            for xi in x:
                _walk(xi)
        else:
            try:
                v = int(x)
            except Exception:
                return
            if v not in seen:
                seen.add(v)
                out.append(v)
    _walk(seq)
    return out

# Base (single-side) presets
_rom_single = {
    
    # no rom computation
    "full_body": full_body,
    "133": [i for i in range(0, 133)],

    # elbow
    "left_elbow_flexion":   [5, 7, 9],
    "right_elbow_flexion":  [6, 8, 10],

    # shoulder
    "left_shoulder_flexion":        [11, 5, 7],
    "right_shoulder_flexion":       [8, 6, 12],
    "left_shoulder_abduction":      [11, 5, 7],
    "right_shoulder_abduction":     [8, 6, 12],
    "left_shoulder_external_rotation":  [7, 5, 9],
    "right_shoulder_external_rotation": [8, 6, 10],
    "left_shoulder_internal_rotation":  [7, 5, 9],
    "right_shoulder_internal_rotation": [8, 6, 10],

    # knee
    "left_knee_flexion":     [11, 13, 15],
    "right_knee_flexion":    [12, 14, 16],
    "left_knee_extension":   [11, 13, 15],
    "right_knee_extension":  [12, 14, 16],

    # ankle (note nested third element for toes)
    "left_ankle_dorsiflexion":     [13, 15, [17, 18]],
    "right_ankle_dorsiflexion":    [14, 16, [20, 21]],
    "left_ankle_plantarflexion":   [13, 15, [17, 18]],
    "right_ankle_plantarflexion":  [14, 16, [20, 21]],
}

# Build grouped (Both) presets as Left∪Right unions for visualization
def _group_union(left_name, right_name):
    return _flatten_unique(_rom_single[left_name] + _rom_single[right_name])

rom_groups = {
    "elbow_flexion":                 _group_union("left_elbow_flexion", "right_elbow_flexion"),
    "shoulder_flexion":              _group_union("left_shoulder_flexion", "right_shoulder_flexion"),
    "shoulder_abduction":            _group_union("left_shoulder_abduction", "right_shoulder_abduction"),
    "shoulder_external_rotation":    _group_union("left_shoulder_external_rotation", "right_shoulder_external_rotation"),
    "shoulder_internal_rotation":    _group_union("left_shoulder_internal_rotation", "right_shoulder_internal_rotation"),
    "knee_flexion":                  _group_union("left_knee_flexion", "right_knee_flexion"),
    "knee_extension":                _group_union("left_knee_extension", "right_knee_extension"),
    "ankle_dorsiflexion":            _group_union("left_ankle_dorsiflexion", "right_ankle_dorsiflexion"),
    "ankle_plantarflexion":          _group_union("left_ankle_plantarflexion", "right_ankle_plantarflexion"),
}

# Public: single + group
rom_test = {**_rom_single, **rom_groups}

# Ordered categories for cycling
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

# Map category -> {left, right, both} names
CATEGORY_SIDES = {
    "elbow_flexion": {
        "left":  "left_elbow_flexion",
        "right": "right_elbow_flexion",
        "both":  "elbow_flexion",
    },
    "shoulder_flexion": {
        "left":  "left_shoulder_flexion",
        "right": "right_shoulder_flexion",
        "both":  "shoulder_flexion",
    },
    "shoulder_abduction": {
        "left":  "left_shoulder_abduction",
        "right": "right_shoulder_abduction",
        "both":  "shoulder_abduction",
    },
    "shoulder_external_rotation": {
        "left":  "left_shoulder_external_rotation",
        "right": "right_shoulder_external_rotation",
        "both":  "shoulder_external_rotation",
    },
    "shoulder_internal_rotation": {
        "left":  "left_shoulder_internal_rotation",
        "right": "right_shoulder_internal_rotation",
        "both":  "shoulder_internal_rotation",
    },
    "knee_flexion": {
        "left":  "left_knee_flexion",
        "right": "right_knee_flexion",
        "both":  "knee_flexion",
    },
    "knee_extension": {
        "left":  "left_knee_extension",
        "right": "right_knee_extension",
        "both":  "knee_extension",
    },
    "ankle_dorsiflexion": {
        "left":  "left_ankle_dorsiflexion",
        "right": "right_ankle_dorsiflexion",
        "both":  "ankle_dorsiflexion",
    },
    "ankle_plantarflexion": {
        "left":  "left_ankle_plantarflexion",
        "right": "right_ankle_plantarflexion",
        "both":  "ankle_plantarflexion",
    },
}
