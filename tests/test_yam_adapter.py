"""Tests for the YAM adapter (offline — no hardware required)."""

from __future__ import annotations

import numpy as np
import pytest

from openforce.models import get_model, list_models


class TestYAMModel:
    def test_yam_registered(self) -> None:
        """The yam model should be registered on import."""
        assert "yam" in list_models()

    def test_yam_model_info(self) -> None:
        """YAM model info should have correct metadata."""
        info = get_model("yam")
        assert info.name == "yam"
        assert info.n_joints == 6
        assert info.ee_site_name == "tcp_site"
        assert info.ee_body_name == "link_6"

    def test_yam_model_path_exists(self) -> None:
        """The YAM MJCF file should exist."""
        info = get_model("yam")
        assert info.model_path.exists()
        assert info.model_path.suffix == ".xml"

    def test_yam_mujoco_loadable(self) -> None:
        """The YAM MJCF should load in MuJoCo without error."""
        import mujoco

        info = get_model("yam")
        model = mujoco.MjModel.from_xml_path(str(info.model_path))
        assert model.nv == 6

    def test_yam_dynamics(self) -> None:
        """MuJoCoDynamics should compute valid M, g for the YAM model."""
        import mujoco

        from openforce.dynamics import MuJoCoDynamics

        info = get_model("yam")
        model = mujoco.MjModel.from_xml_path(str(info.model_path))
        dynamics = MuJoCoDynamics(model)

        q = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        g = dynamics.gravity_vector(q)
        M = dynamics.mass_matrix(q)

        assert g.shape == (6,)
        assert M.shape == (6, 6)
        assert np.linalg.norm(g) > 0.1
        assert np.all(np.linalg.eigvalsh(M) > 0)

    def test_yam_jacobian(self) -> None:
        """Jacobian computation should work with the YAM model."""
        from openforce.sim.mujoco_env import MuJoCoArmEnv

        info = get_model("yam")
        env = MuJoCoArmEnv(
            xml_path=info.model_path,
            ee_site_name=info.ee_site_name,
        )
        assert env.n_joints == 6
        pos = env.get_ee_position()
        assert pos.shape == (3,)
        jac = env.get_jacobian()
        assert jac.shape == (6, 6)


class TestYAMConfig:
    def test_default_config(self) -> None:
        """Default YAMConfig should have sensible values."""
        i2rt = pytest.importorskip("i2rt")  # noqa: F841
        from openforce.adapters.yam import YAMConfig

        config = YAMConfig()
        assert config.channel == "can0"
        assert config.n_joints == 6
        assert config.dt == 0.004
        assert config.zero_gravity_mode is True
        assert config.max_torque.shape == (6,)

    def test_custom_config(self) -> None:
        """Custom config values should be respected."""
        i2rt = pytest.importorskip("i2rt")  # noqa: F841
        from openforce.adapters.yam import YAMConfig

        config = YAMConfig(channel="can1", dt=0.002, torque_scale=0.5)
        assert config.channel == "can1"
        assert config.dt == 0.002
        assert config.torque_scale == 0.5


class TestYAMImportGuard:
    def test_import_without_i2rt(self) -> None:
        """YAMAdapter should be importable even without i2rt installed.

        The __init__.py uses try/except so the module loads cleanly.
        """
        import openforce.adapters.yam as yam_mod

        # __all__ will be empty if i2rt is not installed,
        # or contain YAMAdapter/YAMConfig if it is.
        assert isinstance(yam_mod.__all__, list)


class TestYAMObserverIntegration:
    """Test that observers can be constructed with YAM dynamics."""

    def test_momentum_observer_with_yam_dynamics(self) -> None:
        """MomentumObserver should work with YAM model dynamics."""
        import mujoco

        from openforce.dynamics import MuJoCoDynamics
        from openforce.estimation import MomentumObserver

        info = get_model("yam")
        model = mujoco.MjModel.from_xml_path(str(info.model_path))
        dynamics = MuJoCoDynamics(model)

        observer = MomentumObserver(
            dynamics=dynamics,
            n_joints=6,
            dt=0.004,
            gains=np.full(6, 20.0),
        )
        observer.reset()

    def test_ekf_observer_with_yam_dynamics(self) -> None:
        """EKFObserver should work with YAM model dynamics."""
        import mujoco

        from openforce.dynamics import MuJoCoDynamics
        from openforce.estimation import EKFObserver

        info = get_model("yam")
        model = mujoco.MjModel.from_xml_path(str(info.model_path))
        dynamics = MuJoCoDynamics(model)

        observer = EKFObserver(
            dynamics=dynamics,
            n_joints=6,
            dt=0.004,
        )
        observer.reset()
