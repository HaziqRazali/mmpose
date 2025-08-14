
# https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py

import sys
sys.path.append("/home/haziq/mmpose/configs/_base_/datasets/")
import coco_wholebody as cw


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

lower_left_body = [
        11, 13, 15, 17, 18, 19  # left_hip, left_knee, left_ankle, left_big_toe, left_small_toe, left_heel
    ]

left_body_kpt_ids = [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19,
            91, 92, 93, 94, 95,
            96, 97, 98, 99,
            100, 101, 102, 103,
            104, 105, 106, 107,
            108, 109, 110, 111
        ]

right_body_kpt_ids = [
            2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22,
            112, 113, 114, 115, 116,
            117, 118, 119, 120,
            121, 122, 123, 124,
            125, 126, 127, 128,
            129, 130, 131, 132
        ]

right_lower_body_kpt_ids = [
    12, 14, 16, 20, 21, 22  # right_hip, right_knee, right_ankle, right_big_toe, right_small_toe, right_heel
]

presets = {
    "full_body": full_body,
    "133": [i for i in range(0,133)],
}

# key in specific test and the app computes the ROM
rom_test = {


    "left_elbow_flexion":   [5,7,9,True],
    "right_elbow_flexion":  [6,8,9,True],


}