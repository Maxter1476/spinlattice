"""Temperature sweeps and critical-temperature estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import IsingLattice
from .observables import (
    binder_cumulant,
    blocked_error,
    specific_heat,
    susceptibility,
)
from .wolff import run_wolff

__all__ = ["SweepPoint", "temperature_sweep", "estimate_tc_binder"]


@dataclass(frozen=True)
class SweepPoint:
    """Observables measured at a single temperature."""

    temperature: float
    energy: float
    energy_err: float
    abs_magnetization: float
    magnetization_err: float
    susceptibility: float
    specific_heat: float
    binder: float


def temperature_sweep(
    size: int,
    temperatures: np.ndarray,
    *,
    n_steps: int = 2000,
    thermalization: int = 500,
    seed: int | None = None,
) -> list[SweepPoint]:
    """Measure observables across ``temperatures`` using Wolff dynamics.

    Temperatures are visited from high to low with the final configuration
    carried over (annealing), so each point starts near equilibrium.
    """
    rng = np.random.default_rng(seed)
    lattice = IsingLattice(size, rng=rng)
    points: list[SweepPoint] = []
    for temp in sorted(np.asarray(temperatures, dtype=float), reverse=True):
        beta = 1.0 / temp
        result = run_wolff(lattice, beta, n_steps, thermalization=thermalization)
        e, m = result["energy"], result["magnetization"]
        points.append(
            SweepPoint(
                temperature=temp,
                energy=float(e.mean()),
                energy_err=blocked_error(e),
                abs_magnetization=float(np.abs(m).mean()),
                magnetization_err=blocked_error(np.abs(m)),
                susceptibility=susceptibility(m, lattice.n_spins, beta),
                specific_heat=specific_heat(e, lattice.n_spins, beta),
                binder=binder_cumulant(m),
            )
        )
    return sorted(points, key=lambda p: p.temperature)


def estimate_tc_binder(
    sizes: tuple[int, ...],
    temperatures: np.ndarray,
    *,
    n_steps: int = 4000,
    thermalization: int = 1000,
    seed: int | None = None,
) -> tuple[float, dict[int, list[SweepPoint]]]:
    """Estimate T_c from the crossing of Binder cumulants of two+ sizes.

    Returns the estimate and the per-size sweep data. The crossing is located
    as the temperature minimizing the spread of U_4 across sizes (on the
    sampled grid), which converges to the true crossing as the grid refines.
    """
    if len(sizes) < 2:
        raise ValueError("need at least two lattice sizes for a Binder crossing")
    temperatures = np.asarray(temperatures, dtype=float)
    sweeps = {
        size: temperature_sweep(
            size,
            temperatures,
            n_steps=n_steps,
            thermalization=thermalization,
            seed=None if seed is None else seed + size,
        )
        for size in sizes
    }
    spreads = []
    for idx in range(len(temperatures)):
        u4 = [sweeps[size][idx].binder for size in sizes]
        spreads.append(max(u4) - min(u4))
    sorted_temps = sorted(temperatures)
    tc_estimate = float(sorted_temps[int(np.argmin(spreads))])
    return tc_estimate, sweeps
