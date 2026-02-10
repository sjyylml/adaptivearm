"""Tests for WebMonitor (data logging only, no HTTP server)."""

from __future__ import annotations

import numpy as np
import pytest

from openforce.core.robot_state import RobotState
from openforce.core.types import ControlOutput, ObserverOutput
from openforce.monitoring.web_monitor import (
    MonitorSnapshot,
    WebMonitor,
    WebMonitorConfig,
)


@pytest.fixture
def monitor():
    """Create a WebMonitor without starting the server."""
    return WebMonitor(WebMonitorConfig(buffer_size=100))


def _make_state(n_joints: int = 6, timestamp: float = 0.0) -> RobotState:
    """Create a simple robot state for testing."""
    return RobotState(
        q=np.ones(n_joints) * 0.1,
        qd=np.ones(n_joints) * 0.01,
        tau_motor=np.ones(n_joints) * 0.5,
        timestamp=timestamp,
    )


class TestWebMonitor:
    def test_construct(self, monitor: WebMonitor) -> None:
        """WebMonitor should be constructible."""
        assert monitor is not None

    def test_log_state_only(self, monitor: WebMonitor) -> None:
        """Logging with state only should work."""
        state = _make_state(timestamp=0.1)
        monitor.log(state)

        data = monitor.get_data()
        assert len(data) == 1
        assert data[0]["timestamp"] == 0.1
        assert len(data[0]["q"]) == 6

    def test_log_with_observer_output(self, monitor: WebMonitor) -> None:
        """Logging with observer output should include tau_ext."""
        state = _make_state()
        obs_output = ObserverOutput(
            tau_ext=np.ones(6) * 0.2,
            timestamp=0.0,
        )
        monitor.log(state, observer_output=obs_output)

        data = monitor.get_data()
        assert len(data) == 1
        assert "tau_ext" in data[0]
        assert len(data[0]["tau_ext"]) == 6

    def test_log_with_control_output(self, monitor: WebMonitor) -> None:
        """Logging with control output should include tau_cmd."""
        state = _make_state()
        ctrl_output = ControlOutput(tau_cmd=np.ones(6) * 0.3)
        monitor.log(state, control_output=ctrl_output)

        data = monitor.get_data()
        assert len(data) == 1
        assert "tau_cmd" in data[0]

    def test_log_with_extra(self, monitor: WebMonitor) -> None:
        """Logging with extra data should include it."""
        state = _make_state()
        monitor.log(state, extra={"my_metric": 42.0})

        data = monitor.get_data()
        assert data[0]["extra"]["my_metric"] == 42.0

    def test_get_latest(self, monitor: WebMonitor) -> None:
        """get_latest should return the most recent snapshot."""
        for i in range(5):
            monitor.log(_make_state(timestamp=float(i)))

        latest = monitor.get_latest()
        assert latest is not None
        assert latest.timestamp == 4.0

    def test_get_latest_empty(self, monitor: WebMonitor) -> None:
        """get_latest on empty buffer should return None."""
        assert monitor.get_latest() is None

    def test_get_data_last_n(self, monitor: WebMonitor) -> None:
        """get_data with last_n should return only last N items."""
        for i in range(10):
            monitor.log(_make_state(timestamp=float(i)))

        data = monitor.get_data(last_n=3)
        assert len(data) == 3
        assert data[0]["timestamp"] == 7.0
        assert data[2]["timestamp"] == 9.0

    def test_buffer_limit(self, monitor: WebMonitor) -> None:
        """Buffer should not exceed configured size."""
        for i in range(150):
            monitor.log(_make_state(timestamp=float(i)))

        data = monitor.get_data()
        assert len(data) == 100  # buffer_size=100

    def test_clear(self, monitor: WebMonitor) -> None:
        """Clear should empty the buffer."""
        for i in range(5):
            monitor.log(_make_state(timestamp=float(i)))

        assert len(monitor.get_data()) == 5
        monitor.clear()
        assert len(monitor.get_data()) == 0

    def test_snapshot_to_dict(self) -> None:
        """MonitorSnapshot.to_dict should produce serializable output."""
        snapshot = MonitorSnapshot(
            timestamp=1.0,
            q=np.array([0.1, 0.2]),
            qd=np.array([0.01, 0.02]),
            tau_motor=np.array([0.5, 0.6]),
            tau_ext=np.array([0.1, 0.2]),
            tau_cmd=np.array([0.3, 0.4]),
            extra={"key": "value"},
        )
        d = snapshot.to_dict()
        assert d["timestamp"] == 1.0
        assert d["q"] == [0.1, 0.2]
        assert d["tau_ext"] == [0.1, 0.2]
        assert d["tau_cmd"] == [0.3, 0.4]
        assert d["extra"]["key"] == "value"

    def test_config_defaults(self) -> None:
        """WebMonitorConfig defaults should be reasonable."""
        config = WebMonitorConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.buffer_size == 5000
