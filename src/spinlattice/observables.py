"""Thermodynamic observables, error bars, and exact reference results."""

from __future__ import annotations

import numpy as np

__all__ = [
    "ONSAGER_TC",
    "binder_cumulant",
    "blocked_error",
    "exact_energy_2x2",
    "integrated_autocorrelation_time",
    "specific_heat",
    "susceptibility",
]

#: Onsager's exact critical temperature of the 2D square-lattice Ising model,
#: T_c = 2 / ln(1 + sqrt(2)) ≈ 2.269185 in units of J/k_B.
ONSAGER_TC: float = 2.0 / np.log(1.0 + np.sqrt(2.0))


def susceptibility(magnetizations: np.ndarray, n_spins: int, beta: float) -> float:
    """Magnetic susceptibility per spin, chi = beta N (<m^2> - <|m|>^2).

    Uses <|m|> rather than <m>: on a finite lattice below T_c the symmetry is
    not spontaneously broken over long runs, so <m> would average to zero and
    overestimate chi.
    """
    m = np.asarray(magnetizations, dtype=float)
    return float(beta * n_spins * (np.mean(m**2) - np.mean(np.abs(m)) ** 2))


def specific_heat(energies_per_spin: np.ndarray, n_spins: int, beta: float) -> float:
    """Specific heat per spin, C = beta^2 N (<e^2> - <e>^2)."""
    e = np.asarray(energies_per_spin, dtype=float)
    return float(beta**2 * n_spins * np.var(e))


def binder_cumulant(magnetizations: np.ndarray) -> float:
    """Binder cumulant U_4 = 1 - <m^4> / (3 <m^2>^2).

    U_4 curves for different lattice sizes cross at T_c, giving a
    finite-size-scaling estimate of the critical temperature that is far less
    sensitive to L than the susceptibility peak.
    """
    m = np.asarray(magnetizations, dtype=float)
    m2 = np.mean(m**2)
    if m2 == 0.0:
        return 0.0
    return float(1.0 - np.mean(m**4) / (3.0 * m2**2))


def integrated_autocorrelation_time(series: np.ndarray, *, c_window: float = 6.0) -> float:
    """Integrated autocorrelation time with the Sokal self-consistent window.

    tau_int = 1/2 + sum_t rho(t), summed up to the first window W satisfying
    W >= c_window * tau_int(W). Returns 0.5 for an uncorrelated series.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 2 or np.var(x) == 0.0:
        return 0.5
    x = x - x.mean()
    acf = np.correlate(x, x, mode="full")[n - 1 :]
    acf = acf / acf[0]
    tau = 0.5
    for t in range(1, n):
        tau += acf[t]
        if t >= c_window * tau:
            break
    return float(max(tau, 0.5))


def blocked_error(series: np.ndarray, n_blocks: int = 16) -> float:
    """Standard error of the mean by blocking, robust to autocorrelation."""
    x = np.asarray(series, dtype=float)
    if len(x) < n_blocks:
        return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    usable = (len(x) // n_blocks) * n_blocks
    blocks = x[:usable].reshape(n_blocks, -1).mean(axis=1)
    return float(np.std(blocks, ddof=1) / np.sqrt(n_blocks))


def exact_energy_2x2(beta: float, coupling: float = 1.0) -> float:
    """Exact mean energy per spin of the periodic 2x2 Ising lattice.

    On a 2x2 periodic lattice each site's up/down (and left/right) neighbours
    coincide, so every distinct pair is connected by a double bond: 8 bond
    terms in total. The spectrum is E = -8J (all aligned, x2), E = +8J (the
    two checkerboards), and E = 0 (the remaining 12 states), giving

        Z = 2 e^{8 beta J} + 12 + 2 e^{-8 beta J}
        <E> = -16 J (e^{8 beta J} - e^{-8 beta J}) / Z

    i.e. per spin <E>/4. The test suite cross-checks this closed form against
    a brute-force enumeration of all 16 states, then uses it as ground truth
    for the samplers.
    """
    k = 8.0 * beta * coupling
    z = 2.0 * np.exp(k) + 12.0 + 2.0 * np.exp(-k)
    mean_e = -16.0 * coupling * (np.exp(k) - np.exp(-k)) / z
    return float(mean_e / 4.0)
