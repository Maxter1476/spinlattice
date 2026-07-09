import numpy as np
import pytest

from spinlattice import (
    ONSAGER_TC,
    binder_cumulant,
    blocked_error,
    integrated_autocorrelation_time,
    specific_heat,
    susceptibility,
)


def test_onsager_constant():
    assert pytest.approx(2.269185, abs=1e-6) == ONSAGER_TC


def test_binder_fully_ordered():
    """For m identically +-1, <m^4> = <m^2>^2 = 1, so U4 = 2/3."""
    m = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
    assert binder_cumulant(m) == pytest.approx(2.0 / 3.0)


def test_binder_gaussian_is_near_zero():
    """For a centered Gaussian, <m^4> = 3 <m^2>^2, so U4 -> 0."""
    m = np.random.default_rng(0).normal(size=200_000)
    assert binder_cumulant(m) == pytest.approx(0.0, abs=0.02)


def test_autocorrelation_iid_is_half():
    x = np.random.default_rng(1).normal(size=50_000)
    assert integrated_autocorrelation_time(x) == pytest.approx(0.5, abs=0.05)


def test_autocorrelation_ar1_matches_theory():
    """AR(1) with coefficient a has tau_int = (1 + a) / (2 (1 - a))."""
    rng = np.random.default_rng(2)
    a = 0.9
    n = 400_000
    x = np.empty(n)
    x[0] = 0.0
    noise = rng.normal(size=n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    expected = (1 + a) / (2 * (1 - a))  # 9.5
    assert integrated_autocorrelation_time(x) == pytest.approx(expected, rel=0.15)


def test_blocked_error_iid_matches_naive():
    x = np.random.default_rng(3).normal(size=16_384)
    naive = x.std(ddof=1) / np.sqrt(len(x))
    assert blocked_error(x) == pytest.approx(naive, rel=0.5)


def test_susceptibility_and_specific_heat_nonnegative():
    m = np.random.default_rng(4).uniform(-1, 1, size=1000)
    e = np.random.default_rng(5).uniform(-2, 0, size=1000)
    assert susceptibility(m, n_spins=100, beta=0.4) >= 0.0
    assert specific_heat(e, n_spins=100, beta=0.4) >= 0.0
