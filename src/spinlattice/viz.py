"""Figure generation: sweep curves, spin snapshots, and quench animations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .lattice import IsingLattice
from .metropolis import metropolis_sweep
from .observables import ONSAGER_TC
from .sweep import SweepPoint

__all__ = ["plot_sweep", "plot_snapshots", "save_quench_gif"]


def plot_sweep(points: list[SweepPoint], path: str | Path, *, size: int | None = None) -> Path:
    """Four-panel plot of energy, |m|, susceptibility, and specific heat."""
    temps = [p.temperature for p in points]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    panels = [
        ("energy", "energy_err", r"$\langle E \rangle$ per spin"),
        ("abs_magnetization", "magnetization_err", r"$\langle |m| \rangle$"),
        ("susceptibility", None, r"$\chi$ per spin"),
        ("specific_heat", None, r"$C$ per spin"),
    ]
    for ax, (attr, err_attr, label) in zip(axes.flat, panels, strict=True):
        values = [getattr(p, attr) for p in points]
        errs = [getattr(p, err_attr) for p in points] if err_attr else None
        ax.errorbar(temps, values, yerr=errs, fmt="o-", ms=3, lw=1, capsize=2)
        ax.axvline(ONSAGER_TC, color="crimson", ls="--", lw=1, label=r"Onsager $T_c$")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"$T$  $[J/k_B]$")
    title = "2D Ising model — Wolff cluster Monte Carlo"
    if size is not None:
        title += f"  (L = {size})"
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_snapshots(
    size: int,
    temperatures: tuple[float, ...],
    path: str | Path,
    *,
    n_sweeps: int = 400,
    seed: int | None = None,
) -> Path:
    """Equilibrated spin configurations at several temperatures, side by side."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(temperatures), figsize=(3 * len(temperatures), 3.2))
    for ax, temp in zip(np.atleast_1d(axes), temperatures, strict=True):
        lattice = IsingLattice(size, rng=rng)
        for _ in range(n_sweeps):
            metropolis_sweep(lattice, 1.0 / temp)
        ax.imshow(lattice.spins, cmap="coolwarm", interpolation="nearest")
        ax.set_title(f"T = {temp:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Equilibrated configurations, L = {size} (Onsager $T_c \\approx {ONSAGER_TC:.3f}$)"
    )
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_quench_gif(
    size: int,
    temperature: float,
    path: str | Path,
    *,
    n_frames: int = 60,
    sweeps_per_frame: int = 5,
    seed: int | None = None,
) -> Path:
    """Animate domain coarsening after a quench from T = infinity."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    rng = np.random.default_rng(seed)
    lattice = IsingLattice(size, rng=rng)
    beta = 1.0 / temperature

    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(lattice.spins, cmap="coolwarm", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Quench to T = {temperature:.2f}, L = {size}")

    def update(_frame: int):
        for _ in range(sweeps_per_frame):
            metropolis_sweep(lattice, beta)
        image.set_data(lattice.spins)
        return (image,)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)
    path = Path(path)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
    return path
