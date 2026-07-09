"""Finite-size scaling: the Binder-cumulant crossing must land near Onsager's
exact T_c = 2.269185. This is the headline physics validation of the package."""

import numpy as np

from spinlattice import ONSAGER_TC, estimate_tc_binder


def test_binder_crossing_near_onsager_tc():
    temps = np.arange(2.10, 2.45, 0.05)
    tc, sweeps = estimate_tc_binder(
        (8, 16), temps, n_steps=3_000, thermalization=800, seed=2026
    )
    assert abs(tc - ONSAGER_TC) < 0.11, f"Binder crossing {tc:.3f} vs Onsager {ONSAGER_TC:.3f}"
    # Sanity on the sweep data itself: magnetization decreases with T for the
    # larger lattice across the critical window.
    m16 = [p.abs_magnetization for p in sweeps[16]]
    assert m16[0] > m16[-1]
