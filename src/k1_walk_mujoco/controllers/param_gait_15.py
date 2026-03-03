from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from k1_walk_mujoco.robot.k1_mapping import JOINT_MAP_LEFT, JOINT_MAP_RIGHT, MIRROR_SIGN
from k1_walk_mujoco.robot.k1_spec import CONTROLLED_JOINTS, JOINT_LIMITS, NOMINAL_QPOS

PARAMETER_NAMES: tuple[str, ...] = (
    "f",
    "hip_pitch_offset",
    "hip_pitch_amp",
    "knee_offset",
    "knee_amp",
    "ankle_offset",
    "ankle_amp",
    "knee_phase",
    "ankle_phase",
    "hip_roll_offset",
    "hip_roll_amp",
    "roll_phase",
    "ankle_roll_amp",
    "ankle_roll_phase",
    "swing_knee_boost_amp",
)

PHASE_PARAMS = (
    "knee_phase",
    "ankle_phase",
    "roll_phase",
    "ankle_roll_phase",
)

DEFAULT_FILTER_ALPHA = 0.35
_TWO_PI = 2.0 * np.pi


def _joint_span(joint_name: str, joint_limits: Mapping[str, tuple[float, float]]) -> float:
    q_min, q_max = joint_limits[joint_name]
    return float(q_max - q_min)


def _offset_interval(
    *,
    joint_name: str,
    nominal_qpos: Mapping[str, float],
    joint_limits: Mapping[str, tuple[float, float]],
    ratio: float,
) -> tuple[float, float]:
    q0 = float(nominal_qpos[joint_name])
    span = _joint_span(joint_name, joint_limits)
    return q0 - ratio * span, q0 + ratio * span


def _intersect_intervals(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return max(a[0], b[0]), min(a[1], b[1])


def _shared_offset_bounds(
    *,
    functional_joint: str,
    nominal_qpos: Mapping[str, float],
    joint_limits: Mapping[str, tuple[float, float]],
    joint_map_left: Mapping[str, str],
    joint_map_right: Mapping[str, str],
    ratio: float,
) -> tuple[float, float]:
    left_joint = joint_map_left[functional_joint]
    right_joint = joint_map_right[functional_joint]
    left_interval = _offset_interval(
        joint_name=left_joint,
        nominal_qpos=nominal_qpos,
        joint_limits=joint_limits,
        ratio=ratio,
    )
    right_interval = _offset_interval(
        joint_name=right_joint,
        nominal_qpos=nominal_qpos,
        joint_limits=joint_limits,
        ratio=ratio,
    )
    lo, hi = _intersect_intervals(left_interval, right_interval)
    if lo <= hi:
        return lo, hi
    # Fallback for asymmetric limits; keep a safe, shared window.
    left_q0 = float(nominal_qpos[left_joint])
    right_q0 = float(nominal_qpos[right_joint])
    min_span = min(_joint_span(left_joint, joint_limits), _joint_span(right_joint, joint_limits))
    center = 0.5 * (left_q0 + right_q0)
    return center - ratio * min_span, center + ratio * min_span


def _shared_amp_upper(
    *,
    functional_joint: str,
    joint_limits: Mapping[str, tuple[float, float]],
    joint_map_left: Mapping[str, str],
    joint_map_right: Mapping[str, str],
    ratio: float,
) -> float:
    left_joint = joint_map_left[functional_joint]
    right_joint = joint_map_right[functional_joint]
    span = min(_joint_span(left_joint, joint_limits), _joint_span(right_joint, joint_limits))
    return ratio * span


def compute_param_bounds(
    *,
    joint_limits: Mapping[str, tuple[float, float]] | None = None,
    nominal_qpos: Mapping[str, float] | None = None,
    joint_map_left: Mapping[str, str] | None = None,
    joint_map_right: Mapping[str, str] | None = None,
) -> dict[str, tuple[float, float]]:
    jl = joint_limits if joint_limits is not None else JOINT_LIMITS
    q0 = nominal_qpos if nominal_qpos is not None else NOMINAL_QPOS
    left = joint_map_left if joint_map_left is not None else JOINT_MAP_LEFT
    right = joint_map_right if joint_map_right is not None else JOINT_MAP_RIGHT

    roll_joint_left = left["hip_roll"]
    ankle_roll_joint_left = left["ankle_roll"]
    knee_joint_left = left["knee_pitch"]

    r_roll = _joint_span(roll_joint_left, jl)
    r_ankle_roll = _joint_span(ankle_roll_joint_left, jl)
    r_knee = _joint_span(knee_joint_left, jl)

    bounds: dict[str, tuple[float, float]] = {
        "f": (0.6, 2.0),
        "hip_pitch_offset": _shared_offset_bounds(
            functional_joint="hip_pitch",
            nominal_qpos=q0,
            joint_limits=jl,
            joint_map_left=left,
            joint_map_right=right,
            ratio=0.20,
        ),
        "knee_offset": _shared_offset_bounds(
            functional_joint="knee_pitch",
            nominal_qpos=q0,
            joint_limits=jl,
            joint_map_left=left,
            joint_map_right=right,
            ratio=0.20,
        ),
        "ankle_offset": _shared_offset_bounds(
            functional_joint="ankle_pitch",
            nominal_qpos=q0,
            joint_limits=jl,
            joint_map_left=left,
            joint_map_right=right,
            ratio=0.20,
        ),
        "hip_pitch_amp": (
            0.0,
            _shared_amp_upper(
                functional_joint="hip_pitch",
                joint_limits=jl,
                joint_map_left=left,
                joint_map_right=right,
                ratio=0.25,
            ),
        ),
        "knee_amp": (
            0.0,
            _shared_amp_upper(
                functional_joint="knee_pitch",
                joint_limits=jl,
                joint_map_left=left,
                joint_map_right=right,
                ratio=0.25,
            ),
        ),
        "ankle_amp": (
            0.0,
            _shared_amp_upper(
                functional_joint="ankle_pitch",
                joint_limits=jl,
                joint_map_left=left,
                joint_map_right=right,
                ratio=0.25,
            ),
        ),
        "hip_roll_offset": (-0.05 * r_roll, 0.05 * r_roll),
        "hip_roll_amp": (0.0, 0.15 * r_roll),
        "ankle_roll_amp": (0.0, 0.15 * r_ankle_roll),
        "swing_knee_boost_amp": (0.0, 0.30 * r_knee),
    }

    for name in PHASE_PARAMS:
        bounds[name] = (-np.pi, np.pi)
    return bounds


def default_seed_params(
    *,
    nominal_qpos: Mapping[str, float] | None = None,
    joint_map_left: Mapping[str, str] | None = None,
) -> dict[str, float]:
    q0 = nominal_qpos if nominal_qpos is not None else NOMINAL_QPOS
    left = joint_map_left if joint_map_left is not None else JOINT_MAP_LEFT
    seed = {
        "f": 1.2,
        "hip_pitch_offset": float(q0[left["hip_pitch"]]) - 0.10,
        "hip_pitch_amp": 0.20,
        "knee_offset": float(q0[left["knee_pitch"]]) + 0.20,
        "knee_amp": 0.25,
        "ankle_offset": float(q0[left["ankle_pitch"]]) - 0.10,
        "ankle_amp": 0.20,
        "knee_phase": 0.6,
        "ankle_phase": -0.2,
        "hip_roll_offset": float(q0[left["hip_roll"]]),
        "hip_roll_amp": 0.06,
        "roll_phase": 0.0,
        "ankle_roll_amp": 0.04,
        "ankle_roll_phase": 0.0,
        "swing_knee_boost_amp": 0.10,
    }
    return seed


def clamp_params(
    params: Mapping[str, float],
    *,
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in PARAMETER_NAMES:
        if name not in params:
            raise KeyError(f"Missing gait parameter: {name}")
        lo, hi = bounds[name]
        out[name] = float(np.clip(float(params[name]), lo, hi))
    return out


def params_dict_to_vector(params: Mapping[str, float]) -> np.ndarray:
    return np.array([float(params[name]) for name in PARAMETER_NAMES], dtype=np.float64)


def params_vector_to_dict(values: Sequence[float]) -> dict[str, float]:
    if len(values) != len(PARAMETER_NAMES):
        raise ValueError(f"Expected {len(PARAMETER_NAMES)} parameters, got {len(values)}")
    return {
        name: float(v)
        for name, v in zip(PARAMETER_NAMES, values, strict=True)
    }


class ParamGait15:
    def __init__(
        self,
        *,
        params: Mapping[str, float],
        joint_map_left: Mapping[str, str] | None = None,
        joint_map_right: Mapping[str, str] | None = None,
        mirror_sign: Mapping[str, int] | None = None,
        joint_limits: Mapping[str, tuple[float, float]] | None = None,
        nominal_qpos: Mapping[str, float] | None = None,
        controlled_joints: Sequence[str] | None = None,
        filter_alpha: float = DEFAULT_FILTER_ALPHA,
    ) -> None:
        self.joint_map_left = dict(JOINT_MAP_LEFT if joint_map_left is None else joint_map_left)
        self.joint_map_right = dict(JOINT_MAP_RIGHT if joint_map_right is None else joint_map_right)
        self.mirror_sign = dict(MIRROR_SIGN if mirror_sign is None else mirror_sign)
        self.joint_limits = dict(JOINT_LIMITS if joint_limits is None else joint_limits)
        self.nominal_qpos = dict(NOMINAL_QPOS if nominal_qpos is None else nominal_qpos)
        self.controlled_joints = tuple(CONTROLLED_JOINTS if controlled_joints is None else controlled_joints)

        self.bounds = compute_param_bounds(
            joint_limits=self.joint_limits,
            nominal_qpos=self.nominal_qpos,
            joint_map_left=self.joint_map_left,
            joint_map_right=self.joint_map_right,
        )
        self.params = clamp_params(params, bounds=self.bounds)
        self.filter_alpha = float(np.clip(filter_alpha, 0.0, 1.0))

        self._q_low = np.array([self.joint_limits[name][0] for name in self.controlled_joints])
        self._q_high = np.array([self.joint_limits[name][1] for name in self.controlled_joints])
        self._q_nominal = np.array([self.nominal_qpos[name] for name in self.controlled_joints])
        self._joint_to_index = {name: i for i, name in enumerate(self.controlled_joints)}

        self._time_s = 0.0
        self._q_filt = self._q_nominal.copy()

    def reset(self) -> None:
        self._time_s = 0.0
        self._q_filt = self._q_nominal.copy()

    def set_params(self, params: Mapping[str, float]) -> None:
        self.params = clamp_params(params, bounds=self.bounds)

    def set_params_vector(self, values: Sequence[float]) -> None:
        self.set_params(params_vector_to_dict(values))

    def params_vector(self) -> np.ndarray:
        return params_dict_to_vector(self.params)

    def _swing(self, phi: float) -> float:
        s = np.sin(phi)
        return float(max(0.0, s) ** 2)

    def _leg_canonical_targets(self, phi_leg: float) -> dict[str, float]:
        p = self.params
        hip_pitch = p["hip_pitch_offset"] + p["hip_pitch_amp"] * np.sin(phi_leg)
        knee_pitch = (
            p["knee_offset"]
            + p["knee_amp"] * np.sin(phi_leg + p["knee_phase"])
            + p["swing_knee_boost_amp"] * self._swing(phi_leg)
        )
        ankle_pitch = p["ankle_offset"] + p["ankle_amp"] * np.sin(phi_leg + p["ankle_phase"])

        roll_raw = p["hip_roll_offset"] + p["hip_roll_amp"] * np.sin(phi_leg + p["roll_phase"])
        ankle_roll_raw = p["ankle_roll_amp"] * np.sin(phi_leg + p["ankle_roll_phase"])

        return {
            "hip_yaw": 0.0,
            "hip_pitch": float(hip_pitch),
            "knee_pitch": float(knee_pitch),
            "ankle_pitch": float(ankle_pitch),
            "hip_roll": float(roll_raw),
            "ankle_roll": float(ankle_roll_raw),
        }

    def _apply_leg_targets(
        self,
        *,
        out_by_joint: dict[str, float],
        leg: str,
        phi_leg: float,
    ) -> None:
        canonical = self._leg_canonical_targets(phi_leg)
        mapping = self.joint_map_left if leg == "left" else self.joint_map_right
        leg_roll_sign = 1.0 if leg == "left" else -1.0

        for func_name, actual_name in mapping.items():
            q0 = float(self.nominal_qpos[actual_name])

            if func_name == "hip_yaw":
                q_target = q0
            elif func_name in {"hip_roll", "ankle_roll"}:
                mirrored = leg_roll_sign * canonical[func_name]
                sign = float(self.mirror_sign.get(actual_name, 1))
                q_target = q0 + sign * mirrored
            else:
                sign = float(self.mirror_sign.get(actual_name, 1))
                delta = canonical[func_name] - q0
                q_target = q0 + sign * delta

            q_lo, q_hi = self.joint_limits[actual_name]
            out_by_joint[actual_name] = float(np.clip(q_target, q_lo, q_hi))

    def step(self, t: float, dt: float) -> np.ndarray:
        self._time_s = float(t)

        p = self.params
        phi = _TWO_PI * p["f"] * float(t)
        phi_left = phi
        phi_right = phi + np.pi

        q_target_by_joint: dict[str, float] = {}
        self._apply_leg_targets(out_by_joint=q_target_by_joint, leg="left", phi_leg=phi_left)
        self._apply_leg_targets(out_by_joint=q_target_by_joint, leg="right", phi_leg=phi_right)

        q_des = self._q_nominal.copy()
        for name, value in q_target_by_joint.items():
            idx = self._joint_to_index.get(name)
            if idx is None:
                continue
            q_des[idx] = value

        q_des = np.clip(q_des, self._q_low, self._q_high)
        alpha = self.filter_alpha
        self._q_filt = (1.0 - alpha) * self._q_filt + alpha * q_des
        self._q_filt = np.clip(self._q_filt, self._q_low, self._q_high)

        self._time_s = float(t + dt)
        return self._q_filt.copy()

    def step_dict(self, t: float, dt: float) -> dict[str, float]:
        q = self.step(t=t, dt=dt)
        return {name: float(q[i]) for i, name in enumerate(self.controlled_joints)}

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        joint_map_left: Mapping[str, str] | None = None,
        joint_map_right: Mapping[str, str] | None = None,
        mirror_sign: Mapping[str, int] | None = None,
        joint_limits: Mapping[str, tuple[float, float]] | None = None,
        nominal_qpos: Mapping[str, float] | None = None,
        controlled_joints: Sequence[str] | None = None,
    ) -> "ParamGait15":
        q0 = nominal_qpos if nominal_qpos is not None else NOMINAL_QPOS
        left = joint_map_left if joint_map_left is not None else JOINT_MAP_LEFT
        bounds = compute_param_bounds(
            joint_limits=joint_limits,
            nominal_qpos=q0,
            joint_map_left=left,
            joint_map_right=joint_map_right,
        )
        defaults = default_seed_params(nominal_qpos=q0, joint_map_left=left)

        raw_params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
        if not isinstance(raw_params, dict):
            raise TypeError("Expected `params` mapping in gait config")

        merged = dict(defaults)
        for name in PARAMETER_NAMES:
            if name in raw_params:
                merged[name] = float(raw_params[name])

        params = clamp_params(merged, bounds=bounds)
        alpha = float(cfg.get("filter_alpha", DEFAULT_FILTER_ALPHA)) if isinstance(cfg, dict) else DEFAULT_FILTER_ALPHA

        return cls(
            params=params,
            joint_map_left=joint_map_left,
            joint_map_right=joint_map_right,
            mirror_sign=mirror_sign,
            joint_limits=joint_limits,
            nominal_qpos=nominal_qpos,
            controlled_joints=controlled_joints,
            filter_alpha=alpha,
        )
