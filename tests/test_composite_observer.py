"""Tests for the composite/fusion observer."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from adaptivearm.core.robot_state import RobotState
from adaptivearm.core.types import ObserverOutput
from adaptivearm.estimation.base_observer import BaseObserver
from adaptivearm.estimation.composite_observer import (
    CompositeObserver,
    FusionStrategy,
)


class MockObserver(BaseObserver):
    """Simple mock observer returning a fixed output."""

    def __init__(self, tau_ext: np.ndarray) -> None:
        self._tau_ext = tau_ext.copy()
        self._reset_count = 0

    def reset(self) -> None:
        self._reset_count += 1

    def update(self, state: RobotState) -> ObserverOutput:
        return ObserverOutput(tau_ext=self._tau_ext.copy(), timestamp=state.timestamp)


def _make_state(n: int = 6) -> RobotState:
    """Create a dummy RobotState."""
    return RobotState(
        q=np.zeros(n),
        qd=np.zeros(n),
        tau_motor=np.zeros(n),
        timestamp=0.1,
    )


class TestCompositeObserver:
    def test_weighted_average_equal(self) -> None:
        """Equal weights should average the outputs."""
        obs1 = MockObserver(np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        obs2 = MockObserver(np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        composite = CompositeObserver(
            [obs1, obs2], strategy=FusionStrategy.WEIGHTED_AVERAGE
        )
        output = composite.update(_make_state())
        assert_allclose(output.tau_ext[0], 3.0)

    def test_weighted_average_custom_weights(self) -> None:
        """Custom weights should produce weighted average."""
        obs1 = MockObserver(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        obs2 = MockObserver(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        composite = CompositeObserver(
            [obs1, obs2],
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
            weights=np.array([0.8, 0.2]),
        )
        output = composite.update(_make_state())
        assert_allclose(output.tau_ext[0], 8.0)

    def test_max_norm_strategy(self) -> None:
        """MAX_NORM should select observer with largest residual."""
        obs1 = MockObserver(np.ones(6) * 1.0)
        obs2 = MockObserver(np.ones(6) * 5.0)

        composite = CompositeObserver(
            [obs1, obs2], strategy=FusionStrategy.MAX_NORM
        )
        output = composite.update(_make_state())
        assert_allclose(output.tau_ext, np.ones(6) * 5.0)

    def test_min_norm_strategy(self) -> None:
        """MIN_NORM should select observer with smallest residual."""
        obs1 = MockObserver(np.ones(6) * 1.0)
        obs2 = MockObserver(np.ones(6) * 5.0)

        composite = CompositeObserver(
            [obs1, obs2], strategy=FusionStrategy.MIN_NORM
        )
        output = composite.update(_make_state())
        assert_allclose(output.tau_ext, np.ones(6) * 1.0)

    def test_custom_callable(self) -> None:
        """Custom callable fusion should be used."""
        obs1 = MockObserver(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        obs2 = MockObserver(np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        def sum_fusion(outputs: list[ObserverOutput]) -> ObserverOutput:
            tau_sum = sum(o.tau_ext for o in outputs)
            return ObserverOutput(
                tau_ext=np.asarray(tau_sum), timestamp=outputs[0].timestamp
            )

        composite = CompositeObserver([obs1, obs2], strategy=sum_fusion)
        output = composite.update(_make_state())
        assert_allclose(output.tau_ext[0], 3.0)

    def test_reset_resets_all(self) -> None:
        """Reset should propagate to all sub-observers."""
        obs1 = MockObserver(np.zeros(6))
        obs2 = MockObserver(np.zeros(6))

        composite = CompositeObserver([obs1, obs2])
        composite.reset()
        assert obs1._reset_count == 1
        assert obs2._reset_count == 1

    def test_nested_composite(self) -> None:
        """CompositeObserver should support nesting."""
        obs1 = MockObserver(np.ones(6) * 2.0)
        obs2 = MockObserver(np.ones(6) * 4.0)
        obs3 = MockObserver(np.ones(6) * 6.0)

        inner = CompositeObserver(
            [obs1, obs2], strategy=FusionStrategy.WEIGHTED_AVERAGE
        )
        outer = CompositeObserver(
            [inner, obs3], strategy=FusionStrategy.WEIGHTED_AVERAGE
        )
        output = outer.update(_make_state())
        # inner avg = 3.0, outer avg of (3.0, 6.0) = 4.5
        assert_allclose(output.tau_ext, np.ones(6) * 4.5)

    def test_single_observer(self) -> None:
        """Single observer should pass through."""
        obs = MockObserver(np.ones(6) * 7.0)
        composite = CompositeObserver([obs])
        output = composite.update(_make_state())
        assert_allclose(output.tau_ext, np.ones(6) * 7.0)

    def test_empty_raises(self) -> None:
        """Empty observer list should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="at least one"):
            CompositeObserver([])
