#!/usr/bin/env python3
"""
rerun_all.py — rerun all sweeps

  * common lambda grid per gamma (same for all N) 
  * 50 000 realizations per network
  * First minor Ns to get usable data faster.
  * If results csv is skips

Escribe en  results

USAGE
---
    python rerun_all.py                 

Needs functions_2.py same directory and numba (works without numba but ~50-100x slower).
"""

import os
import sys
import time
import argparse
import numpy as np

import functions_2 as fct
from run_sweep import common_grid   # unic grid (higher lambda)

SIZES = [10_000, 30_000, 50_000, 100_000, 300_000, 500_000, 1_000_000]
GAMMAS = [3.5, 2.5]


def run_one(gamma, N, n_real, networks_dir, results_dir, coverage, seed, force):
    out_csv = os.path.join(results_dir, f"sweep_g{gamma}_N{N}.csv")
    if os.path.exists(out_csv) and not force:
        print(f"  SKIP (ya existe): {os.path.basename(out_csv)}")
        return None

    paths = fct.replica_paths(networks_dir, gamma=gamma, N=N)
    if not paths:
        print(f"  [AVISO] sin redes para gamma={gamma}, N={N} en '{networks_dir}'. "
              f"Salto.")
        return None

    os.makedirs(results_dir, exist_ok=True)
    grid = common_grid(gamma, N=N)
    print(f"  {len(paths)} réplicas · {n_real} real/red · "
          f"{len(grid)} lambdas en [{grid.min():.3f}, {grid.max():.3f}] · "
          f"numba={fct.HAVE_NUMBA}")
    t0 = time.time()
    fct.gillespie_sweep(
        paths, grid, gamma=gamma, N=N,
        delta=1.0, coverage_threshold=coverage,
        n_real=n_real, out_csv=out_csv, seed=seed, verbose=False)
    dt = time.time() - t0
    print(f"  -> {os.path.basename(out_csv)}  ({dt/60:.1f} min)")
    return dt


def main():
    p = argparse.ArgumentParser(description="Rerun secuencial de todos los sweeps.")
    p.add_argument("--n_real", type=int, default=50_000,
                   help="realizaciones POR RED (def. 50000, igual que run_sweep.py; "
                        "5 replicas -> 250k total/lambda en N<=1e5)")
    p.add_argument("--networks", default="networks")
    p.add_argument("--results", default="results_3",
                   help="carpeta de salida (def. results_3, no toca results_2)")
    p.add_argument("--coverage", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--only-gamma", type=float, default=None,
                   help="correr solo este gamma (3.5 o 2.5)")
    p.add_argument("--max-N", type=int, default=None,
                   help="saltar tamaños mayores que este (p.ej. 300000)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    gammas = [args.only_gamma] if args.only_gamma else GAMMAS
    gammas = [int(g) if g == int(g) else round(g, 1) for g in gammas]
    sizes = [N for N in SIZES if (args.max_N is None or N <= args.max_N)]

    jobs = [(g, N) for g in gammas for N in sizes]   # menor->mayor N por gamma
    print("=" * 66)
    print(f"RERUN  ·  {len(jobs)} sweeps  ·  {args.n_real} real/red  ->  "
          f"'{args.results}/'")
    print("=" * 66)

    t_start = time.time()
    done = 0
    for g, N in jobs:
        print(f"\n[{done+1}/{len(jobs)}] gamma={g}  N={N}")
        run_one(g, N, n_real=args.n_real, networks_dir=args.networks,
                results_dir=args.results, coverage=args.coverage,
                seed=args.seed, force=args.force)
        done += 1
        print(f"    (transcurrido total: {(time.time()-t_start)/60:.1f} min)")

    print("\n" + "=" * 66)
    print(f"MADE. Total {(time.time()-t_start)/60:.1f} min. "
          f"CSVs in '{args.results}/'.")
    print("Go to analysis notebook")


if __name__ == "__main__":
    main()
