"""spinlattice — 2D Ising model Monte Carlo, validated against exact results."""

from .lattice import IsingLattice
from .metropolis import metropolis_sweep, run_metropolis
from .observables import (
    ONSAGER_TC,
    binder_cumulant,
    blocked_error,
    exact_energy_2x2,
    integrated_autocorrelation_time,
    specific_heat,
    susceptibility,
)
from .sweep import SweepPoint, estimate_tc_binder, temperature_sweep
from .wolff import run_wolff, wolff_step

__all__ = [
    "ONSAGER_TC",
    "IsingLattice",
    "SweepPoint",
    "binder_cumulant",
    "blocked_error",
    "estimate_tc_binder",
    "exact_energy_2x2",
    "integrated_autocorrelation_time",
    "metropolis_sweep",
    "run_metropolis",
    "run_wolff",
    "specific_heat",
    "susceptibility",
    "temperature_sweep",
    "wolff_step",
]

__version__ = "0.1.0"
