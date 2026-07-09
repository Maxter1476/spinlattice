"""Single-spin-flip dynamics with checkerboard vectorization.

A full sweep updates the two checkerboard sublattices in turn. Within one
sublattice no two sites are neighbours, so all of its spins can be updated
simultaneously with vectorized NumPy while per-site detailed balance is
preserved — orders of magnitude faster than a Python site loop.

Two acceptance rules are provided:

- ``"metropolis"``: flip with probability min(1, exp(-beta dE)). Fastest
  mixing, but the deterministic "always flip when dE <= 0" step interacts
  badly with simultaneous updates on the degenerate 2x2 lattice: there every
  site of a stripe configuration has dE = 0, whole sublattices flip
  deterministically, stripe states form a closed class the chain can never
  enter, and ergodicity is lost. The code therefore rejects L = 2 under this
  rule (the regression test demonstrates the failure explicitly).
- ``"glauber"``: heat-bath rule, flip with probability 1 / (1 + exp(beta dE)).
  Every per-site outcome has probability strictly inside (0, 1), so the chain
  is ergodic on every lattice size — this is the standard choice in
  GPU-checkerboard Ising codes.
"""

from __future__ import annotations

import numpy as np

from .lattice import IsingLattice

__all__ = ["metropolis_sweep", "run_metropolis"]

_RULES = ("metropolis", "glauber")


def _sublattice_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.indices((size, size))
    black = (ii + jj) % 2 == 0
    return black, ~black


def metropolis_sweep(lattice: IsingLattice, beta: float, *, rule: str = "metropolis") -> None:
    """Perform one full checkerboard sweep (every spin offered one flip).

    Parameters
    ----------
    lattice:
        Configuration to update in place.
    beta:
        Inverse temperature 1/T in units of 1/J.
    rule:
        ``"metropolis"`` (default) or ``"glauber"`` — see module docstring.
    """
    if rule not in _RULES:
        raise ValueError(f"unknown acceptance rule {rule!r}, expected one of {_RULES}")
    if rule == "metropolis" and lattice.size == 2:
        raise ValueError(
            "checkerboard Metropolis is non-ergodic on the 2x2 lattice "
            "(stripe states are unreachable); use rule='glauber'"
        )
    for mask in _sublattice_masks(lattice.size):
        neigh = lattice.neighbour_sum()
        # Energy change if spin s flips: dE = 2 s (J * sum_neigh + h)
        delta_e = 2.0 * lattice.spins * (lattice.coupling * neigh + lattice.field)
        if rule == "metropolis":
            p_flip = np.exp(-beta * np.maximum(delta_e, 0.0))
        else:
            p_flip = 1.0 / (1.0 + np.exp(beta * delta_e))
        flip = mask & (lattice.rng.random(lattice.spins.shape) < p_flip)
        lattice.spins[flip] *= -1


def run_metropolis(
    lattice: IsingLattice,
    beta: float,
    n_sweeps: int,
    *,
    thermalization: int = 0,
    measure_every: int = 1,
    rule: str = "metropolis",
) -> dict[str, np.ndarray]:
    """Run single-spin-flip dynamics and record observables.

    Returns a dict with per-measurement arrays ``energy`` (per spin) and
    ``magnetization`` (per spin).
    """
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be >= 1")
    for _ in range(thermalization):
        metropolis_sweep(lattice, beta, rule=rule)

    energies: list[float] = []
    mags: list[float] = []
    for sweep in range(n_sweeps):
        metropolis_sweep(lattice, beta, rule=rule)
        if sweep % measure_every == 0:
            energies.append(lattice.energy() / lattice.n_spins)
            mags.append(lattice.magnetization())
    return {"energy": np.array(energies), "magnetization": np.array(mags)}
