"""Command-line interface: sweep, snapshots, quench GIF, and benchmark."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .lattice import IsingLattice
from .metropolis import metropolis_sweep
from .observables import ONSAGER_TC
from .sweep import temperature_sweep
from .viz import plot_snapshots, plot_sweep, save_quench_gif
from .wolff import wolff_step

__all__ = ["main"]


def _cmd_sweep(args: argparse.Namespace) -> None:
    temps = np.linspace(args.tmin, args.tmax, args.points)
    points = temperature_sweep(
        args.size, temps, n_steps=args.steps, thermalization=args.therm, seed=args.seed
    )
    print(f"{'T':>7}  {'<E>/N':>9}  {'<|m|>':>7}  {'chi':>9}  {'C':>7}  {'U4':>6}")
    for p in points:
        print(
            f"{p.temperature:7.3f}  {p.energy:9.4f}  {p.abs_magnetization:7.4f}"
            f"  {p.susceptibility:9.3f}  {p.specific_heat:7.3f}  {p.binder:6.3f}"
        )
    if args.plot:
        out = plot_sweep(points, args.plot, size=args.size)
        print(f"wrote {out}")


def _cmd_snapshots(args: argparse.Namespace) -> None:
    temps = (1.5, round(ONSAGER_TC, 3), 3.5) if args.temps is None else tuple(args.temps)
    out = plot_snapshots(args.size, temps, args.out, seed=args.seed)
    print(f"wrote {out}")


def _cmd_quench(args: argparse.Namespace) -> None:
    out = save_quench_gif(args.size, args.temperature, args.out, seed=args.seed)
    print(f"wrote {out}")


def _cmd_bench(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    lattice = IsingLattice(args.size, rng=rng)
    beta = 1.0 / ONSAGER_TC

    start = time.perf_counter()
    for _ in range(args.sweeps):
        metropolis_sweep(lattice, beta)
    dt = time.perf_counter() - start
    rate = args.sweeps * lattice.n_spins / dt
    print(f"Metropolis: {args.sweeps} sweeps of {args.size}x{args.size} in {dt:.2f}s "
          f"({rate / 1e6:.1f}M spin-updates/s)")

    start = time.perf_counter()
    flipped = sum(wolff_step(lattice, beta) for _ in range(args.sweeps))
    dt = time.perf_counter() - start
    print(f"Wolff: {args.sweeps} clusters in {dt:.2f}s "
          f"(mean cluster {flipped / args.sweeps:.0f} spins)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spinlattice", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sweep", help="temperature sweep with Wolff dynamics")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--tmin", type=float, default=1.6)
    p.add_argument("--tmax", type=float, default=3.2)
    p.add_argument("--points", type=int, default=17)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--therm", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--plot", type=str, default=None, help="write a 4-panel PNG here")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("snapshots", help="equilibrated configuration snapshots")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--temps", type=float, nargs="+", default=None)
    p.add_argument("--out", type=str, default="snapshots.png")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_cmd_snapshots)

    p = sub.add_parser("quench", help="animated GIF of a quench from T = infinity")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--out", type=str, default="quench.gif")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_cmd_quench)

    p = sub.add_parser("bench", help="spin-updates/second benchmark")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--sweeps", type=int, default=200)
    p.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
