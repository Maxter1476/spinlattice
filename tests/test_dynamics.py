import numpy as np
import pytest

from spinlattice import IsingLattice, run_metropolis, run_wolff, wolff_step


def test_metropolis_and_wolff_agree_off_critical():
    """Both algorithms sample the same distribution: <E> must agree at T=3."""
    beta = 1.0 / 3.0
    lat_m = IsingLattice(8, rng=np.random.default_rng(5))
    lat_w = IsingLattice(8, rng=np.random.default_rng(6))
    e_metro = run_metropolis(lat_m, beta, 8_000, thermalization=1_000)["energy"].mean()
    e_wolff = run_wolff(lat_w, beta, 8_000, thermalization=1_000)["energy"].mean()
    assert e_metro == pytest.approx(e_wolff, abs=0.03)


def test_low_temperature_orders():
    """Deep below T_c the system magnetizes almost fully."""
    lattice = IsingLattice(16, rng=np.random.default_rng(9))
    result = run_metropolis(lattice, beta=1.0, n_sweeps=500, thermalization=1_500)
    assert np.abs(result["magnetization"]).mean() > 0.95


def test_high_temperature_disorders():
    """Far above T_c magnetization fluctuates around zero."""
    lattice = IsingLattice(16, rng=np.random.default_rng(10))
    result = run_metropolis(lattice, beta=0.1, n_sweeps=2_000, thermalization=200)
    assert abs(result["magnetization"].mean()) < 0.05


def test_wolff_rejects_external_field():
    lattice = IsingLattice(8, field=0.5, rng=np.random.default_rng(11))
    with pytest.raises(ValueError):
        wolff_step(lattice, beta=0.4)


def test_wolff_cluster_bounds():
    lattice = IsingLattice(8, rng=np.random.default_rng(12))
    flipped = wolff_step(lattice, beta=0.44)
    assert 1 <= flipped <= lattice.n_spins


def test_energy_conserved_bookkeeping():
    """After any dynamics, energy() still matches a recomputation from scratch."""
    lattice = IsingLattice(10, rng=np.random.default_rng(13))
    run_metropolis(lattice, beta=0.5, n_sweeps=50)
    s = lattice.spins
    recomputed = float(-(s * (np.roll(s, 1, 0) + np.roll(s, 1, 1))).sum())
    assert lattice.energy() == pytest.approx(recomputed)
