from __future__ import annotations

import numpy as np

from k1_walk_mujoco.controllers.param_gait_15 import (
    PARAMETER_NAMES,
    ParamGait15,
    clamp_params,
    compute_param_bounds,
    default_seed_params,
)
from k1_walk_mujoco.robot.k1_spec import CONTROLLED_JOINTS, JOINT_LIMITS, NOMINAL_QPOS


def test_default_seed_is_clamped_into_bounds() -> None:
    bounds = compute_param_bounds()
    seed = default_seed_params()
    clamped = clamp_params(seed, bounds=bounds)

    assert set(clamped.keys()) == set(PARAMETER_NAMES)
    for name in PARAMETER_NAMES:
        lo, hi = bounds[name]
        assert lo <= clamped[name] <= hi


def test_controller_step_output_shape_and_limits() -> None:
    bounds = compute_param_bounds()
    params = clamp_params(default_seed_params(), bounds=bounds)
    controller = ParamGait15(params=params)

    for i in range(50):
        t = i * 0.02
        q_des = controller.step(t=t, dt=0.02)
        assert q_des.shape == (len(CONTROLLED_JOINTS),)
        assert np.isfinite(q_des).all()

        for idx, joint_name in enumerate(CONTROLLED_JOINTS):
            q_min, q_max = JOINT_LIMITS[joint_name]
            assert q_min <= q_des[idx] <= q_max


def test_hip_yaw_held_at_nominal() -> None:
    bounds = compute_param_bounds()
    params = clamp_params(default_seed_params(), bounds=bounds)
    controller = ParamGait15(params=params)

    q_des = controller.step(t=0.35, dt=0.02)
    left_idx = CONTROLLED_JOINTS.index("Left_Hip_Yaw")
    right_idx = CONTROLLED_JOINTS.index("Right_Hip_Yaw")

    assert np.isclose(q_des[left_idx], NOMINAL_QPOS["Left_Hip_Yaw"])
    assert np.isclose(q_des[right_idx], NOMINAL_QPOS["Right_Hip_Yaw"])


def test_from_config_clamps_out_of_range_values() -> None:
    controller = ParamGait15.from_config(
        {
            "filter_alpha": 2.0,
            "params": {
                "f": 9.0,
                "hip_pitch_amp": 9.0,
                "knee_phase": 9.0,
            },
        }
    )

    bounds = controller.bounds
    assert controller.filter_alpha == 1.0
    assert bounds["f"][0] <= controller.params["f"] <= bounds["f"][1]
    assert bounds["hip_pitch_amp"][0] <= controller.params["hip_pitch_amp"] <= bounds["hip_pitch_amp"][1]
    assert bounds["knee_phase"][0] <= controller.params["knee_phase"] <= bounds["knee_phase"][1]
