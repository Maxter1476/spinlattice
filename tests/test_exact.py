"""Validation against exactly solvable cases.

Small periodic lattices have enumerable state spaces, so sampled mean
energies must agree with the analytic Boltzmann averages within statistical
error. This pins down the sampled distribution itself — not just internal
consistency:

- 2x2 (16 states): closed form, checked against enumeration, then against
  Glauber checkerboard dynamics and the Wolff algorithm.
- 4x4 (65,536 states): full vectorized enumeration, checked against the
  default checkerboard Metropolis rule.
- The checkerboard Metropolis rule on 2x2 is *non-ergodic* (stripe states
  unreachable) and must be rejected — the bug this guards against produced a
  ~13% energy bias.
"""

import numpy as np
import pytest

from spinlattice import (
    IsingLattice,
    exact_energy_2x2,
    metropolis_sweep,
    run_metropolis,
    run_wolff,
)


def exact_energy_enumeration(size: int, beta: float) -> float:
    """Mean energy per spin by enumerating all 2^(size^2) states, vectorized."""
    n = size * size
    bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
    spins = (2 * bits - 1).reshape(-1, size, size)
    bonds = spins * (np.roll(spins, 1, axis=1) + np.roll(spins, 1, axis=2))
    energies = -bonds.sum(axis=(1, 2)).astype(float)
    # subtract E_min before exponentiating for numerical stability
    weights = np.exp(-beta * (energies - energies.min()))
    return float((energies * weights).sum() / weights.sum() / n)


@pytest.mark.parametrize("beta", [0.2, 0.4407, 0.8])
def test_closed_form_matches_enumeration(beta):
    assert exact_energy_2x2(beta) == pytest.approx(exact_energy_enumeration(2, beta), abs=1e-12)


@pytest.mark.parametrize("beta", [0.3, 0.6])
def test_glauber_reproduces_exact_2x2_energy(beta):
    lattice = IsingLattice(2, rng=np.random.default_rng(123))
    result = run_metropolis(lattice, beta, n_sweeps=80_000, thermalization=2_000, rule="glauber")
    assert result["energy"].mean() == pytest.approx(exact_energy_2x2(beta), abs=0.03)


@pytest.mark.parametrize("beta", [0.3, 0.6])
def test_wolff_reproduces_exact_2x2_energy(beta):
    lattice = IsingLattice(2, rng=np.random.default_rng(321))
    result = run_wolff(lattice, beta, n_steps=120_000, thermalization=2_000)
    assert result["energy"].mean() == pytest.approx(exact_energy_2x2(beta), abs=0.03)


def test_metropolis_reproduces_exact_4x4_energy():
    beta = 0.4
    exact = exact_energy_enumeration(4, beta)
    lattice = IsingLattice(4, rng=np.random.default_rng(7))
    result = run_metropolis(lattice, beta, n_sweeps=60_000, thermalization=2_000)
    assert result["energy"].mean() == pytest.approx(exact, abs=0.03)


def test_checkerboard_metropolis_rejected_on_2x2():
    """Regression guard: on 2x2 every spin of a stripe state has dE = 0, so the
    deterministic Metropolis 'always flip when dE <= 0' flips whole sublattices
    at once; stripe states become unreachable and the chain samples the wrong
    distribution. The library must refuse rather than silently return ~13%
    biased energies."""
    lattice = IsingLattice(2, rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="non-ergodic"):
        metropolis_sweep(lattice, beta=0.4)
