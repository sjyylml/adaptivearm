"""Tests for the robot model registry and model-based adapter loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adaptivearm.adapters.sim import SimAdapter
from adaptivearm.models import (
    RobotModelInfo,
    get_model,
    list_models,
    register_model,
)
from adaptivearm.sim.mujoco_env import MuJoCoArmEnv


class TestModelRegistry:
    def test_default_registered(self) -> None:
        """The default_6dof model should be registered on import."""
        info = get_model("default_6dof")
        assert info.name == "default_6dof"
        assert info.n_joints == 6
        assert info.ee_site_name == "ee_site"
        assert info.ee_body_name == "ee"

    def test_list_models(self) -> None:
        """list_models should include default_6dof."""
        names = list_models()
        assert "default_6dof" in names

    def test_get_model_path_exists(self) -> None:
        """The model file should exist on disk."""
        info = get_model("default_6dof")
        assert info.model_path.exists()
        assert info.model_path.suffix == ".xml"

    def test_get_unknown_raises(self) -> None:
        """Looking up an unregistered name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent_robot")

    def test_register_custom(self, tmp_path: Path) -> None:
        """Registering a custom model should make it retrievable."""
        # Create a minimal MJCF for testing
        xml_content = """
        <mujoco>
          <worldbody>
            <body name="link0">
              <joint name="j0" type="hinge" axis="0 0 1"/>
              <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3"/>
              <body name="ee" pos="0 0 0.3">
                <site name="tip" pos="0 0 0" size="0.01"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor joint="j0" ctrlrange="-10 10"/>
          </actuator>
        </mujoco>
        """
        model_file = tmp_path / "test_robot.xml"
        model_file.write_text(xml_content)

        register_model(RobotModelInfo(
            name="_test_custom_robot",
            model_path=model_file,
            n_joints=1,
            ee_site_name="tip",
            ee_body_name="ee",
            description="Test robot",
        ))

        info = get_model("_test_custom_robot")
        assert info.name == "_test_custom_robot"
        assert info.n_joints == 1
        assert info.ee_site_name == "tip"
        assert "_test_custom_robot" in list_models()


class TestMuJoCoArmEnvEeSite:
    def test_default_ee_site(self) -> None:
        """Default ee_site_name should be 'ee_site'."""
        env = MuJoCoArmEnv()
        assert env.ee_site_name == "ee_site"
        # Should work without error
        pos = env.get_ee_position()
        assert pos.shape == (3,)

    def test_custom_ee_site(self, tmp_path: Path) -> None:
        """Custom ee_site_name should be used for Jacobian and position."""
        xml = """
        <mujoco>
          <worldbody>
            <body name="link0">
              <joint name="j0" type="hinge" axis="0 0 1"/>
              <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3"/>
              <body name="tip_body" pos="0 0 0.3">
                <site name="my_custom_site" pos="0 0 0" size="0.01"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor joint="j0" ctrlrange="-10 10"/>
          </actuator>
        </mujoco>
        """
        f = tmp_path / "custom.xml"
        f.write_text(xml)

        env = MuJoCoArmEnv(xml_path=f, ee_site_name="my_custom_site")
        assert env.ee_site_name == "my_custom_site"
        pos = env.get_ee_position()
        assert pos.shape == (3,)
        jac = env.get_jacobian()
        assert jac.shape == (6, 1)


class TestSimAdapterModelName:
    def test_model_name_default(self) -> None:
        """SimAdapter(model_name='default_6dof') should load the default arm."""
        adapter = SimAdapter(model_name="default_6dof")
        assert adapter.n_joints == 6
        state = adapter.reset()
        assert state.q.shape == (6,)

    def test_backward_compat_no_args(self) -> None:
        """SimAdapter() with no args should still work."""
        adapter = SimAdapter()
        assert adapter.n_joints == 6

    def test_model_name_unknown_raises(self) -> None:
        """SimAdapter with unknown model_name should raise KeyError."""
        with pytest.raises(KeyError):
            SimAdapter(model_name="nonexistent")

    def test_dynamics_work_with_model_name(self) -> None:
        """Dynamics computation should work with model loaded by name."""
        from adaptivearm.dynamics import MuJoCoDynamics

        adapter = SimAdapter(model_name="default_6dof")
        dynamics = MuJoCoDynamics(adapter.env.model)

        q0 = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
        adapter.reset(q0)

        state = adapter.get_state()
        g = dynamics.gravity_vector(state.q)
        M = dynamics.mass_matrix(state.q)

        # Gravity vector should be non-zero for a non-upright config
        assert np.linalg.norm(g) > 0.1
        # Mass matrix should be positive definite
        assert np.all(np.linalg.eigvalsh(M) > 0)
        # Should be able to send torque without error
        adapter.send_torque(g)
