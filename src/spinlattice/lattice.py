"""Square-lattice Ising model state and energetics.

The model is the classic 2D Ising Hamiltonian with periodic boundaries:

    H = -J * sum_<ij> s_i s_j - h * sum_i s_i,   s_i in {-1, +1}

Energies are reported in units of J; temperatures in units of J/k_B.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

__all__ = ["IsingLattice"]


@dataclass
class IsingLattice:
    """Spin configuration on an L x L periodic square lattice.

    Parameters
    ----------
    size:
        Linear dimension L. The lattice has ``L * L`` spins.
    coupling:
        Nearest-neighbour coupling J (J > 0 is ferromagnetic).
    field:
        External field h in units of J.
    rng:
        NumPy random generator; pass a seeded generator for reproducibility.
    """

    size: int
    coupling: float = 1.0
    field: float = 0.0
    rng: np.random.Generator = dataclass_field(default_factory=np.random.default_rng)
    spins: np.ndarray = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError(f"lattice size must be >= 2, got {self.size}")
        self.spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(self.size, self.size))

    @property
    def n_spins(self) -> int:
        """Total number of spins L^2."""
        return self.size * self.size

    def neighbour_sum(self) -> np.ndarray:
        """Sum of the four nearest neighbours at every site (periodic)."""
        s = self.spins
        return (
            np.roll(s, 1, axis=0)
            + np.roll(s, -1, axis=0)
            + np.roll(s, 1, axis=1)
            + np.roll(s, -1, axis=1)
        )

    def energy(self) -> float:
        """Total energy of the configuration, in units of J.

        Each bond is counted once (the roll trick sums two of the four
        neighbours per site, covering every bond exactly once).
        """
        s = self.spins
        bonds = s * (np.roll(s, 1, axis=0) + np.roll(s, 1, axis=1))
        return float(-self.coupling * bonds.sum() - self.field * s.sum())

    def magnetization(self) -> float:
        """Magnetization per spin, in [-1, 1]."""
        return float(self.spins.sum()) / self.n_spins

    def set_uniform(self, value: int = 1) -> None:
        """Reset every spin to ``value`` (a cold start)."""
        if value not in (-1, 1):
            raise ValueError("spin value must be -1 or +1")
        self.spins.fill(value)

    def randomize(self) -> None:
        """Re-draw every spin uniformly at random (a hot start)."""
        self.spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(self.size, self.size))
