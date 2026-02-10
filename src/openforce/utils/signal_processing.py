"""Signal processing utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class LowPassFilter:
    """First-order IIR low-pass filter (exponential moving average).

    Args:
        cutoff_freq: Cutoff frequency in Hz. If <= 0, filter is bypassed.
        dt: Sampling period in seconds.
        n_channels: Number of independent channels to filter.
    """

    def __init__(self, cutoff_freq: float, dt: float, n_channels: int) -> None:
        self._bypass = cutoff_freq <= 0
        if not self._bypass:
            omega = 2.0 * np.pi * cutoff_freq
            self._alpha = (omega * dt) / (1.0 + omega * dt)
        else:
            self._alpha = 1.0
        self._y: NDArray[np.floating] = np.zeros(n_channels)
        self._initialized = False

    def reset(self) -> None:
        """Reset filter state."""
        self._y[:] = 0.0
        self._initialized = False

    def __call__(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply one filter step.

        Args:
            x: Input signal, shape (n_channels,).

        Returns:
            Filtered output, shape (n_channels,).
        """
        if self._bypass:
            return np.asarray(x, dtype=np.float64)
        if not self._initialized:
            self._y = np.asarray(x, dtype=np.float64).copy()
            self._initialized = True
            return self._y.copy()
        self._y = self._alpha * x + (1.0 - self._alpha) * self._y
        return self._y.copy()
