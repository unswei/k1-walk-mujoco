from __future__ import annotations

LEG_FUNCTIONAL_JOINTS = (
    "hip_yaw",
    "hip_roll",
    "hip_pitch",
    "knee_pitch",
    "ankle_pitch",
    "ankle_roll",
)

JOINT_MAP_LEFT = {
    "hip_yaw": "Left_Hip_Yaw",
    "hip_roll": "Left_Hip_Roll",
    "hip_pitch": "Left_Hip_Pitch",
    "knee_pitch": "Left_Knee_Pitch",
    "ankle_pitch": "Left_Ankle_Pitch",
    "ankle_roll": "Left_Ankle_Roll",
}

JOINT_MAP_RIGHT = {
    "hip_yaw": "Right_Hip_Yaw",
    "hip_roll": "Right_Hip_Roll",
    "hip_pitch": "Right_Hip_Pitch",
    "knee_pitch": "Right_Knee_Pitch",
    "ankle_pitch": "Right_Ankle_Pitch",
    "ankle_roll": "Right_Ankle_Roll",
}

# Sign used to map functional mirrored motion into model joint sign conventions.
MIRROR_SIGN = {
    "Left_Hip_Yaw": 1,
    "Left_Hip_Roll": 1,
    "Left_Hip_Pitch": 1,
    "Left_Knee_Pitch": 1,
    "Left_Ankle_Pitch": 1,
    "Left_Ankle_Roll": 1,
    "Right_Hip_Yaw": 1,
    "Right_Hip_Roll": 1,
    "Right_Hip_Pitch": 1,
    "Right_Knee_Pitch": 1,
    "Right_Ankle_Pitch": 1,
    "Right_Ankle_Roll": 1,
}
