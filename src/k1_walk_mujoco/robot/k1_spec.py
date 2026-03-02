from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from k1_walk_mujoco.robot.k1_defaults import load_robot_pd_config

_CFG = load_robot_pd_config()

ALL_JOINTS = tuple(_CFG["all_joints"])
CONTROLLED_JOINTS = tuple(_CFG["controlled_joints"])
NOMINAL_QPOS = dict(_CFG["nominal_qpos"])
JOINT_LIMITS = {k: tuple(v) for k, v in _CFG["joint_limits"].items()}
EFFORT_LIMITS = dict(_CFG["effort_limits"])
ACTION_SCALE_RAD = float(_CFG["action_scale_rad"])
PD_GAINS = dict(_CFG["pd_gains"])


@dataclass(frozen=True)
class ControlledJointArrays:
    names: tuple[str, ...]
    q_nominal: np.ndarray
    q_low: np.ndarray
    q_high: np.ndarray
    effort: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


def controlled_joint_arrays() -> ControlledJointArrays:
    names = CONTROLLED_JOINTS
    q_nominal = np.array([NOMINAL_QPOS[j] for j in names], dtype=np.float64)
    q_low = np.array([JOINT_LIMITS[j][0] for j in names], dtype=np.float64)
    q_high = np.array([JOINT_LIMITS[j][1] for j in names], dtype=np.float64)
    effort = np.array([EFFORT_LIMITS[j] for j in names], dtype=np.float64)
    kp = np.array([PD_GAINS[j]["kp"] for j in names], dtype=np.float64)
    kd = np.array([PD_GAINS[j]["kd"] for j in names], dtype=np.float64)
    return ControlledJointArrays(
        names=names,
        q_nominal=q_nominal,
        q_low=q_low,
        q_high=q_high,
        effort=effort,
        kp=kp,
        kd=kd,
    )
