"""Tests for math utilities."""

import numpy as np
from numpy.testing import assert_allclose

from adaptivearm.utils.math_utils import pseudoinverse, skew_symmetric, wrap_angle


class TestSkewSymmetric:
    def test_basic(self) -> None:
        S = skew_symmetric([1.0, 2.0, 3.0])
        assert S.shape == (3, 3)
        # Skew-symmetric: S = -S^T
        assert_allclose(S, -S.T)

    def test_cross_product(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert_allclose(skew_symmetric(a) @ b, np.cross(a, b))


class TestPseudoinverse:
    def test_identity(self) -> None:
        J = np.eye(3)
        J_pinv = pseudoinverse(J, damping=1e-10)
        assert_allclose(J_pinv, np.eye(3), atol=1e-6)

    def test_rectangular(self) -> None:
        rng = np.random.default_rng(42)
        J = rng.standard_normal((3, 6))
        J_pinv = pseudoinverse(J, damping=1e-8)
        # J * J_pinv ≈ I for full-row-rank J
        assert_allclose(J @ J_pinv, np.eye(3), atol=1e-4)

    def test_singular_with_damping(self) -> None:
        J = np.zeros((3, 3))
        J[0, 0] = 1.0
        # Should not raise; damping prevents singular inverse
        J_pinv = pseudoinverse(J, damping=0.01)
        assert J_pinv.shape == (3, 3)
        assert np.all(np.isfinite(J_pinv))


class TestWrapAngle:
    def test_within_range(self) -> None:
        assert_allclose(wrap_angle(0.5), 0.5, atol=1e-10)

    def test_wrap_positive(self) -> None:
        assert_allclose(wrap_angle(np.pi + 0.1), -np.pi + 0.1, atol=1e-10)

    def test_wrap_negative(self) -> None:
        assert_allclose(wrap_angle(-np.pi - 0.1), np.pi - 0.1, atol=1e-10)

    def test_array(self) -> None:
        angles = np.array([0.0, 2 * np.pi, -2 * np.pi])
        wrapped = wrap_angle(angles)
        assert_allclose(wrapped, [0.0, 0.0, 0.0], atol=1e-10)
