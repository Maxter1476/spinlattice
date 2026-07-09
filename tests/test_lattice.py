import numpy as np
import pytest

from spinlattice import IsingLattice


def brute_force_energy(lattice: IsingLattice) -> float:
    """O(N^2)-style reference energy: loop every site, count each bond once."""
    s = lattice.spins
    size = lattice.size
    total = 0.0
    for i in range(size):
        for j in range(size):
            total -= lattice.coupling * s[i, j] * (s[(i + 1) % size, j] + s[i, (j + 1) % size])
            total -= lattice.field * s[i, j]
    return total


def test_uniform_energy():
    lattice = IsingLattice(8, rng=np.random.default_rng(0))
    lattice.set_uniform(1)
    # 2 bonds per site, all aligned: E = -2 J N
    assert lattice.energy() == pytest.approx(-2.0 * lattice.n_spins)
    assert lattice.magnetization() == pytest.approx(1.0)


def test_energy_matches_brute_force():
    rng = np.random.default_rng(42)
    for size in (2, 3, 5, 8):
        lattice = IsingLattice(size, coupling=1.3, field=0.4, rng=rng)
        assert lattice.energy() == pytest.approx(brute_force_energy(lattice))


def test_neighbour_sum_range():
    lattice = IsingLattice(16, rng=np.random.default_rng(1))
    neigh = lattice.neighbour_sum()
    assert neigh.min() >= -4 and neigh.max() <= 4
    assert neigh.shape == (16, 16)


def test_invalid_size_rejected():
    with pytest.raises(ValueError):
        IsingLattice(1)


def test_seeded_runs_reproducible():
    a = IsingLattice(12, rng=np.random.default_rng(7))
    b = IsingLattice(12, rng=np.random.default_rng(7))
    assert np.array_equal(a.spins, b.spins)
