#!/usr/bin/env python3
"""
generate_networks.py — generates the configuration-model networks P(k) ~ k^-gamma
with k_min = 4 and STRUCTURAL CUTOFF k_max = sqrt(N) (Mata et al., PRE 91, 052117),
and saves them as CSV files in the 'networks/' folder.

USAGE
-----
    python generate_networks.py             

Options (with their default values):
    --gammas 3.5 2.5          exponents to generate
    --sizes  10000 ... 1e6    sizes N (by default the 7 required by the assignment)
    --networks networks       output folder
    --base-seed 12345         base seed (reproducible)
    --force                   regenerate even if the CSV already exists

Number of replicas per size (defined in functions_2.n_replicas):
    N <= 100000 -> 5,   N <= 500000 -> 3,   N = 1000000 -> 2

Requirements: numpy, networkx.  (numba is only needed for the simulation, not for generation.)
"""


import os
import sys
import time
import argparse
import numpy as np

import functions_2 as fct


DEFAULT_GAMMAS = [3.5, 2.5]
DEFAULT_SIZES = [10_000, 30_000, 50_000, 100_000, 300_000, 500_000, 1_000_000]


def generate(gammas, sizes, out_dir, base_seed, force, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    # aviso si ya hay redes (probablemente las viejas SIN cutoff)
    existing = [f for f in os.listdir(out_dir) if f.startswith("net_") and f.endswith(".csv")]
    if existing and not force:
        print(f"[AVISO] '{out_dir}' ya contiene {len(existing)} redes. Se SALTAN las")
        print("        existentes. Si son las viejas (sin cutoff k_max=sqrt(N)),")
        print("        vuelve a lanzar con --force para regenerarlas.\n")

    paths, t0 = [], time.time()
    for gamma in gammas:
        for N in sizes:
            kmax = fct.structural_cutoff(N)          # k_max = sqrt(N)
            for rep in range(fct.n_replicas(N)):
                path = os.path.join(out_dir, fct.net_filename(gamma, N, rep))
                paths.append(path)
                if os.path.exists(path) and not force:
                    if verbose:
                        print(f"  SKIP (existe): {os.path.basename(path)}")
                    continue

                seed = fct.make_seed(gamma, N, rep, base_seed)
                G, deg = fct.generate_network(N, gamma, kmin=4, kmax=kmax,
                                              seed=seed, save_path=path)
                if verbose:
                    k2 = float((deg.astype(float) ** 2).mean())
                    print(f"  g={gamma} N={N:>8} r={rep}: "
                          f"E={G.number_of_edges():>9}  "
                          f"<k>={2*G.number_of_edges()/N:5.2f}  "
                          f"<k^2>={k2:8.1f}  "
                          f"kmax_real={int(deg.max()):>4} (cutoff {kmax})  "
                          f"#(k=4)={int((deg == 4).sum()):>6}  "
                          f"-> {os.path.basename(path)}")

    dt = time.time() - t0
    print(f"\nListo: {len(paths)} redes en '{out_dir}/'  ({dt/60:.1f} min).")
    return paths


def main():
    p = argparse.ArgumentParser(
        description="Genera las redes P(k)~k^-gamma con k_max=sqrt(N).")
    p.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS,
                   help="exponentes gamma (def. 3.5 2.5)")
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                   help="tamanos N (def. los 7 de la tarea)")
    p.add_argument("--networks", default="networks",
                   help="carpeta de salida (def. networks)")
    p.add_argument("--base-seed", type=int, default=fct.BASE_SEED,
                   dest="base_seed", help=f"semilla base (def. {fct.BASE_SEED})")
    p.add_argument("--force", action="store_true",
                   help="regenerar aunque el CSV ya exista (usar la 1a vez)")
    args = p.parse_args()

    # normalizar gamma a 1 decimal para que case con los nombres de fichero
    gammas = [round(g, 1) for g in args.gammas]
    gammas = [int(g) if g == int(g) else g for g in gammas]

    print(f"gammas={gammas}  sizes={args.sizes}  ->  '{args.networks}/'  "
          f"(force={args.force})\n")
    generate(gammas, args.sizes, args.networks, args.base_seed, args.force)


if __name__ == "__main__":
    main()
