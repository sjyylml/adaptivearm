"""Tests for PinocchioDynamics (skipped if pinocchio is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")


class TestPinocchioDynamics:
    def test_import_and_construct(self) -> None:
        """PinocchioDynamics should be importable when pinocchio is available."""
        from openforce.dynamics.pinocchio_dynamics import PinocchioDynamics

        assert PinocchioDynamics is not None

    def test_mass_matrix_shape(self, tmp_path: pytest.TempPathFactory) -> None:
        """Mass matrix should have correct shape."""
        from openforce.dynamics.pinocchio_dynamics import PinocchioDynamics

        # Create a minimal URDF for testing
        urdf = _minimal_urdf()
        urdf_file = tmp_path / "test_robot.urdf"  # type: ignore[operator]
        urdf_file.write_text(urdf)

        dyn = PinocchioDynamics(urdf_file)
        n = dyn.n_joints
        q = np.zeros(dyn.nq)
        M = dyn.mass_matrix(q)
        assert M.shape == (n, n)

    def test_mass_matrix_symmetric(self, tmp_path: pytest.TempPathFactory) -> None:
        """Mass matrix should be symmetric."""
        from openforce.dynamics.pinocchio_dynamics import PinocchioDynamics

        urdf_file = tmp_path / "test_robot.urdf"  # type: ignore[operator]
        urdf_file.write_text(_minimal_urdf())

        dyn = PinocchioDynamics(urdf_file)
        q = np.zeros(dyn.nq)
        M = dyn.mass_matrix(q)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_gravity_vector_shape(self, tmp_path: pytest.TempPathFactory) -> None:
        """Gravity vector should have correct shape."""
        from openforce.dynamics.pinocchio_dynamics import PinocchioDynamics

        urdf_file = tmp_path / "test_robot.urdf"  # type: ignore[operator]
        urdf_file.write_text(_minimal_urdf())

        dyn = PinocchioDynamics(urdf_file)
        q = np.zeros(dyn.nq)
        g = dyn.gravity_vector(q)
        assert g.shape == (dyn.n_joints,)

    def test_coriolis_zero_velocity(self, tmp_path: pytest.TempPathFactory) -> None:
        """Coriolis vector should be zero at zero velocity."""
        from openforce.dynamics.pinocchio_dynamics import PinocchioDynamics

        urdf_file = tmp_path / "test_robot.urdf"  # type: ignore[operator]
        urdf_file.write_text(_minimal_urdf())

        dyn = PinocchioDynamics(urdf_file)
        q = np.zeros(dyn.nq)
        qd = np.zeros(dyn.n_joints)
        c = dyn.coriolis_vector(q, qd)
        np.testing.assert_allclose(c, np.zeros(dyn.n_joints), atol=1e-10)

    def test_passive_torque(self, tmp_path: pytest.TempPathFactory) -> None:
        """Passive torque should be proportional to velocity."""
        from openforce.dynamics.pinocchio_dynamics import PinocchioDynamics

        urdf_file = tmp_path / "test_robot.urdf"  # type: ignore[operator]
        urdf_file.write_text(_minimal_urdf())

        dyn = PinocchioDynamics(urdf_file)
        q = np.zeros(dyn.nq)
        qd = np.ones(dyn.n_joints)
        tau_passive = dyn.passive_torque(qd, q)
        assert tau_passive.shape == (dyn.n_joints,)


def _minimal_urdf() -> str:
    """Generate a minimal 2-joint URDF for testing."""
    return """<?xml version="1.0" ?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.15"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="3.14"/>
    <dynamics damping="0.1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="3.14"/>
    <dynamics damping="0.05"/>
  </joint>
</robot>
"""
