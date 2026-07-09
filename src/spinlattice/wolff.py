"""Wolff cluster algorithm.

Near the critical temperature, single-spin-flip dynamics suffer critical
slowing down: the autocorrelation time diverges as tau ~ L^z with z ≈ 2.17.
The Wolff algorithm grows and flips whole spin clusters with bond probability
p = 1 - exp(-2 beta J), reducing z to ≈ 0.25 and making near-critical
measurements feasible on lattices where Metropolis alone would decorrelate
far too slowly.

Only valid at zero external field (cluster flips are not detailed-balanced
against a field term).
"""

from __future__ import annotations

import numpy as np

from .lattice import IsingLattice

__all__ = ["wolff_step", "run_wolff"]


def wolff_step(lattice: IsingLattice, beta: float) -> int:
    """Grow one Wolff cluster from a random seed site and flip it.

    Returns the number of spins flipped.
    """
    if lattice.field != 0.0:
        raise ValueError("Wolff algorithm requires zero external field")

    size = lattice.size
    spins = lattice.spins
    p_add = 1.0 - np.exp(-2.0 * beta * lattice.coupling)

    seed = (int(lattice.rng.integers(size)), int(lattice.rng.integers(size)))
    cluster_spin = spins[seed]
    in_cluster = np.zeros((size, size), dtype=bool)
    in_cluster[seed] = True
    frontier = [seed]

    while frontier:
        i, j = frontier.pop()
        for ni, nj in ((i - 1, j), ((i + 1) % size, j), (i, j - 1), (i, (j + 1) % size)):
            ni %= size
            nj %= size
            if (
                not in_cluster[ni, nj]
                and spins[ni, nj] == cluster_spin
                and lattice.rng.random() < p_add
            ):
                in_cluster[ni, nj] = True
                frontier.append((ni, nj))

    spins[in_cluster] *= -1
    return int(in_cluster.sum())


def run_wolff(
    lattice: IsingLattice,
    beta: float,
    n_steps: int,
    *,
    thermalization: int = 0,
    measure_every: int = 1,
) -> dict[str, np.ndarray]:
    """Run Wolff dynamics and record observables (per spin).

    Also returns ``cluster_size`` — mean flipped-cluster fraction is itself a
    useful diagnostic (it approaches the percolation fraction near T_c).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    for _ in range(thermalization):
        wolff_step(lattice, beta)

    energies: list[float] = []
    mags: list[float] = []
    clusters: list[int] = []
    for step in range(n_steps):
        clusters.append(wolff_step(lattice, beta))
        if step % measure_every == 0:
            energies.append(lattice.energy() / lattice.n_spins)
            mags.append(lattice.magnetization())
    return {
        "energy": np.array(energies),
        "magnetization": np.array(mags),
        "cluster_size": np.array(clusters),
    }
