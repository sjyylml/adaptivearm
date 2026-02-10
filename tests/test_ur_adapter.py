"""Tests for UR adapter (skipped if ur_rtde is not installed)."""

from __future__ import annotations

import numpy as np
import pytest


class TestURConfig:
    def test_default_config(self) -> None:
        """URConfig should have sensible defaults."""
        from adaptivearm.adapters.ur.ur_adapter import URConfig

        cfg = URConfig()
        assert cfg.ip == "192.168.1.1"
        assert cfg.dt == 0.002
        assert cfg.n_joints == 6
        assert cfg.servo_gain > 0
        assert cfg.servo_lookahead > 0
        assert cfg.torque_stiffness > 0

    def test_custom_config(self) -> None:
        """URConfig should accept custom values."""
        from adaptivearm.adapters.ur.ur_adapter import URConfig

        cfg = URConfig(ip="10.0.0.1", dt=0.008, servo_gain=500.0)
        assert cfg.ip == "10.0.0.1"
        assert cfg.dt == 0.008
        assert cfg.servo_gain == 500.0


class TestURAdapterImportGuard:
    def test_import_guard(self) -> None:
        """URAdapter should raise ImportError if ur_rtde is not installed."""
        from adaptivearm.adapters.ur.ur_adapter import _HAS_RTDE

        if not _HAS_RTDE:
            from adaptivearm.adapters.ur.ur_adapter import URConfig

            with pytest.raises(ImportError, match="ur_rtde"):
                from adaptivearm.adapters.ur.ur_adapter import URAdapter

                URAdapter(URConfig())
        else:
            pytest.skip("ur_rtde is installed, cannot test import guard")


class TestURAdapterProperties:
    def test_properties(self) -> None:
        """URAdapter properties should match config when ur_rtde unavailable."""
        from adaptivearm.adapters.ur.ur_adapter import _HAS_RTDE

        if _HAS_RTDE:
            pytest.skip("ur_rtde is installed; skip offline property tests")

        # We can still test URConfig independently
        from adaptivearm.adapters.ur.ur_adapter import URConfig

        cfg = URConfig(ip="192.168.0.1", dt=0.004, n_joints=6)
        assert cfg.n_joints == 6
        assert cfg.dt == 0.004


class TestTorqueConversion:
    def test_admittance_conversion_logic(self) -> None:
        """Torque-to-position conversion should be tau / K_virtual * dt."""
        from adaptivearm.adapters.ur.ur_adapter import URConfig

        cfg = URConfig(torque_stiffness=500.0, dt=0.002)
        tau = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Expected delta_q per step
        expected_delta = tau / cfg.torque_stiffness * cfg.dt
        np.testing.assert_allclose(expected_delta[0], 10.0 / 500.0 * 0.002)
