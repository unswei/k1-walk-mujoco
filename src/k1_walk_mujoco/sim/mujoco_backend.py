from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from k1_walk_mujoco.robot.k1_spec import CONTROLLED_JOINTS, NOMINAL_QPOS


@dataclass
class State:
    base_pos: np.ndarray
    base_quat: np.ndarray
    base_ang_vel: np.ndarray
    base_lin_vel: np.ndarray
    joint_qpos: np.ndarray
    joint_qvel: np.ndarray


class MujocoBackend:
    """Low-level MuJoCo backend using actuator `data.ctrl` torque commands."""

    def __init__(self, mjcf_path: Path, controlled_joints: tuple[str, ...] = CONTROLLED_JOINTS):
        self.model, self.data = load_model(mjcf_path)
        self.controlled_joints = controlled_joints
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.controlled_joints
        ]
        self._joint_qpos_idx = [int(self.model.jnt_qposadr[j]) for j in self._joint_ids]
        self._joint_qvel_idx = [int(self.model.jnt_dofadr[j]) for j in self._joint_ids]
        self._actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.controlled_joints
        ]
        self._free_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint"
        )
        self._base_body_id = int(self.model.jnt_bodyid[self._free_joint_id])

    def reset(
        self,
        rng: np.random.Generator,
        qpos_noise_std: float,
        qvel_noise_std: float,
    ) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0

        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                qpos_idx = int(self.model.jnt_qposadr[j])
                qvel_idx = int(self.model.jnt_dofadr[j])
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                self.data.qpos[qpos_idx] = NOMINAL_QPOS.get(joint_name, self.data.qpos[qpos_idx])
                self.data.qpos[qpos_idx] += rng.normal(0.0, qpos_noise_std)
                self.data.qvel[qvel_idx] += rng.normal(0.0, qvel_noise_std)

        mujoco.mj_forward(self.model, self.data)

    def step(self, tau: np.ndarray, n_substeps: int) -> None:
        self.data.ctrl[:] = 0.0
        tau = np.asarray(tau, dtype=np.float64)
        for actuator_idx, torque in zip(self._actuator_ids, tau):
            self.data.ctrl[actuator_idx] = torque
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def get_state(self) -> State:
        qpos = self.data.qpos
        base_pos = qpos[0:3].copy()
        base_quat = qpos[3:7].copy()

        body_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._base_body_id,
            body_vel,
            0,
        )
        base_ang_vel = body_vel[0:3].copy()
        base_lin_vel = body_vel[3:6].copy()

        joint_qpos = self.data.qpos[self._joint_qpos_idx].copy()
        joint_qvel = self.data.qvel[self._joint_qvel_idx].copy()

        return State(
            base_pos=base_pos,
            base_quat=base_quat,
            base_ang_vel=base_ang_vel,
            base_lin_vel=base_lin_vel,
            joint_qpos=joint_qpos,
            joint_qvel=joint_qvel,
        )


def load_model(mjcf_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    return model, data
