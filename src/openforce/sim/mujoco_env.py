"""MuJoCo simulation environment for robot arms."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

# Default 6-DOF arm MJCF for quick prototyping
_DEFAULT_ARM_XML = """
<mujoco model="6dof_arm">
  <option gravity="0 0 -9.81" timestep="0.002">
    <flag contact="enable"/>
  </option>

  <default>
    <joint damping="0.5" armature="0.1"/>
    <geom contype="1" conaffinity="1" condim="3" rgba="0.8 0.6 0.2 1" margin="0.001"/>
  </default>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.01" rgba="0.9 0.9 0.9 1" contype="1" conaffinity="1"/>

    <body name="link0" pos="0 0 0.1">
      <geom type="cylinder" size="0.06 0.05" rgba="0.3 0.3 0.3 1"/>
      <joint name="joint0" type="hinge" axis="0 0 1" range="-3.14 3.14"/>

      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.4" rgba="0.8 0.2 0.2 1"/>
        <joint name="joint1" type="hinge" axis="0 1 0" range="-2.0 2.0"/>

        <body name="link2" pos="0 0 0.4">
          <geom type="capsule" size="0.035" fromto="0 0 0 0 0 0.35" rgba="0.2 0.8 0.2 1"/>
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.5 2.5"/>

          <body name="link3" pos="0 0 0.35">
            <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint3" type="hinge" axis="0 0 1" range="-3.14 3.14"/>

            <body name="link4" pos="0 0 0.2">
              <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.8 0.8 0.2 1"/>
              <joint name="joint4" type="hinge" axis="0 1 0" range="-2.0 2.0"/>

              <body name="link5" pos="0 0 0.15">
                <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1" rgba="0.8 0.2 0.8 1"/>
                <joint name="joint5" type="hinge" axis="0 0 1" range="-3.14 3.14"/>

                <body name="ee" pos="0 0 0.1">
                  <site name="ee_site" pos="0 0 0" size="0.02" rgba="1 0 0 1"/>
                  <geom type="sphere" size="0.03" rgba="1 0.5 0 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor0" joint="joint0" ctrlrange="-100 100" ctrllimited="true"/>
    <motor name="motor1" joint="joint1" ctrlrange="-100 100" ctrllimited="true"/>
    <motor name="motor2" joint="joint2" ctrlrange="-100 100" ctrllimited="true"/>
    <motor name="motor3" joint="joint3" ctrlrange="-50 50" ctrllimited="true"/>
    <motor name="motor4" joint="joint4" ctrlrange="-50 50" ctrllimited="true"/>
    <motor name="motor5" joint="joint5" ctrlrange="-50 50" ctrllimited="true"/>
  </actuator>

  <sensor>
    <framepos name="ee_pos" objtype="site" objname="ee_site"/>
  </sensor>
</mujoco>
"""


class MuJoCoArmEnv:
    """MuJoCo simulation environment wrapping a robot arm model.

    Args:
        xml_path: Path to MJCF XML or URDF file. If None, uses built-in 6-DOF arm.
        dt: Override model timestep (None = use model default).
        ee_site_name: Name of the end-effector site in the model.
            Used by ``get_ee_position`` and ``get_jacobian``.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        dt: float | None = None,
        ee_site_name: str = "ee_site",
    ) -> None:
        if xml_path is not None:
            self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        else:
            self._model = mujoco.MjModel.from_xml_string(_DEFAULT_ARM_XML)

        if dt is not None:
            self._model.opt.timestep = dt

        self._data = mujoco.MjData(self._model)
        self._ee_site_name = ee_site_name
        mujoco.mj_forward(self._model, self._data)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def n_joints(self) -> int:
        return self._model.nv

    @property
    def dt(self) -> float:
        return self._model.opt.timestep

    @property
    def time(self) -> float:
        return self._data.time

    def reset(self, q0: NDArray[np.floating] | None = None) -> None:
        """Reset simulation to initial state or given configuration."""
        mujoco.mj_resetData(self._model, self._data)
        if q0 is not None:
            self._data.qpos[:] = q0
        mujoco.mj_forward(self._model, self._data)

    def step(self, tau: NDArray[np.floating] | None = None) -> None:
        """Advance simulation by one timestep.

        Args:
            tau: Joint torque commands. If None, uses current ctrl values.
        """
        if tau is not None:
            self._data.ctrl[:] = tau
        mujoco.mj_step(self._model, self._data)

    @property
    def ee_site_name(self) -> str:
        """Name of the end-effector site."""
        return self._ee_site_name

    def get_ee_position(self) -> NDArray[np.floating]:
        """Get end-effector Cartesian position."""
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, self._ee_site_name
        )
        return self._data.site_xpos[site_id].copy()

    def get_jacobian(self) -> NDArray[np.floating]:
        """Get full end-effector Jacobian (6 x nv).

        Returns:
            Jacobian matrix, shape (6, nv). Top 3 rows = translational,
            bottom 3 rows = rotational.
        """
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, self._ee_site_name
        )
        nv = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, site_id)
        return np.vstack([jacp, jacr])

    def apply_external_force(
        self,
        body_name: str,
        force: NDArray[np.floating],
        torque: NDArray[np.floating] | None = None,
    ) -> None:
        """Apply an external force/torque to a body (in world frame).

        Args:
            body_name: Name of the MuJoCo body.
            force: Force vector [fx, fy, fz].
            torque: Torque vector [tx, ty, tz]. Defaults to zero.
        """
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self._data.xfrc_applied[body_id, :3] = force
        if torque is not None:
            self._data.xfrc_applied[body_id, 3:] = torque

    def clear_external_forces(self) -> None:
        """Clear all externally applied forces."""
        self._data.xfrc_applied[:] = 0.0
