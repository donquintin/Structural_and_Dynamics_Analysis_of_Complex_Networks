#!/usr/bin/env python3
"""
run_sweep.py — corre el sweep del método del lifespan para UNA red (un (gamma, N))
y escribe el CSV en results_2/.

Pensado para el cluster: se le pasa gamma y N por línea de comandos.

USO
---
    python run_sweep.py GAMMA N [opciones]

Ejemplos:
    python run_sweep.py 3.5 1000000
    python run_sweep.py 2.5 500000 --n_real 20000
    python run_sweep.py 3.5 10000 --n_real 2000 --results results_test

Opciones (con sus valores por defecto):
    --n_real 20000        realizaciones por réplica de red
    --networks networks   carpeta con los CSV de las redes (net_g{g}_N{N}_r{r}.csv)
    --results  results_2  carpeta de salida para el sweep
    --coverage 0.5        umbral de cobertura para declarar brote endémico
    --fine-npoints 30     nº de lambdas en el barrido fino
    --seed 20             semilla base
    --force               re-correr aunque el CSV ya exista (por defecto: resumable)

REQUISITOS
----------
    * functions_2.py debe estar en la misma carpeta (o en el PYTHONPATH).
    * Paquetes: numpy, networkx, numba (opcional pero MUY recomendado: sin numba
      funciona igual pero ~50-100x más lento). scipy/matplotlib NO hacen falta
      aquí (solo para el análisis posterior).

SALIDA
------
    results_2/sweep_g{gamma}_N{N}.csv   con columnas:
       gamma, N, lambda, n_networks, n_real_per_net, n_real_total,
       P_end, P_end_std, tau_mean, tau_std, tau2_mean, tau2_std
"""

import os
import sys
import time
import argparse
import numpy as np

import functions_2 as fct     # motor acelerado (njit) + segundo momento tau2


# --- REJILLA LAMBDA COMUN POR GAMMA  (CAMBIO CLAVE) -------------------------
# ANTES: cada N localizaba el pico con una fase "coarse" y centraba ahi una
# ventana fina. Como <tau> es un observable debil y ruidoso, el pico coarse
# variaba de un N a otro -> cada N acababa con un RANGO DE LAMBDA DISTINTO.
# Eso rompe el finite-size scaling: lambda_p(N) salia zig-zag (panel b), las
# alturas se median sobre soportes distintos (panel c) y los colapsos (e,f)
# eran imposibles porque las curvas no compartian eje x.
#
# AHORA: una sola rejilla FIJA por gamma, IDENTICA para todos los N. Asi
# lambda_p(N), las alturas de pico y los colapsos son directamente comparables.
# La rejilla cubre desde por debajo de lambda_c hasta bien dentro de la fase
# endemica (para que P_end suba y el colapso de P_end tenga senal).
# N_GRID = nº de lambdas por defecto (antes 31). Mas puntos -> pico de chi y
# colapsos mas limpios. El limite INFERIOR baja respecto a la version anterior
# porque en results_3 el pico de chi se salia por el borde de abajo para los N
# grandes (g=3.5 llegaba a 0.080, g=2.5 a 0.005 = el propio borde). Al bajar el
# limite inferior capturamos el pico tambien para N=1e6. El limite SUPERIOR baja
# un poco (menos realizaciones endemicas caras -> mas rapido), pero sigue por
# encima del pico para que P_end tenga senal.
# TOPE de lambda dependiente de N: para los N grandes el coste esta dominado por
# las realizaciones ENDEMICAS de la parte alta de lambda (cada una crece hasta el
# 50% de cobertura ~ O(N) eventos). Como el pico de chi (-> lambda_c, 1/nu) y su
# colapso viven en la zona baja de lambda (~0.05-0.11), recortar el tope de 0.160
# a 0.120 para N>=LARGE_N abarata muchisimo los jobs grandes SIN tocar la zona
# util. Los puntos que quedan COINCIDEN con los de la rejilla pequena (mismo
# linspace, solo se elimina la cola), asi que el FSS/colapsos siguen compartiendo
# eje x. Solo se pierde la cola de P_end a lambda alta (pequena para N grande).
N_GRID  = 41
LARGE_N = 100_000
LAMBDA_LO     = {3.5: 0.050, 2.5: 0.001}
LAMBDA_HI     = {3.5: 0.160, 2.5: 0.050}   # N <  LARGE_N
LAMBDA_HI_BIG = {3.5: 0.120, 2.5: 0.030}   # N >= LARGE_N: recorta la cola endemica
                                           # (g=2.5 tope 0.05 estaba ~5x lambda_c
                                           #  -> tambien caro; picos de chi < 0.023)

# Overrides puntuales (gamma, N) -> tope de lambda. Para gamma=2.5 el pico de
# <tau> caia en el borde del rango barrido: N=1e4 lo tiene por ENCIMA de 0.05
# (rejilla completa demasiado corta por arriba para el N mas pequeno), y N=1e5
# justo en el recorte 0.03. Subimos el tope solo en esos dos casos.
LAMBDA_HI_OVERRIDE = {
    (2.5, 10_000):  0.070,
    (2.5, 100_000): 0.040,
}

# Rejilla "pequena" de referencia (compatibilidad; = grid para N < LARGE_N).
COMMON_GRID = {g: np.linspace(LAMBDA_LO[g], LAMBDA_HI[g], N_GRID)
               for g in LAMBDA_LO}


def _upper_limit(gamma, N):
    if N is not None and (gamma, N) in LAMBDA_HI_OVERRIDE:
        return LAMBDA_HI_OVERRIDE[(gamma, N)]
    if N is not None and N >= LARGE_N:
        return LAMBDA_HI_BIG[gamma]
    return LAMBDA_HI[gamma]


def common_grid(gamma, N=None, npoints=None):
    """Rejilla lambda comun para este gamma. El tope de lambda depende de N
    (recorte de la cola endemica cara para N grandes, y overrides puntuales);
    el PASO es siempre el mismo, asi que los puntos coinciden entre tamanos y el
    FSS/colapsos comparten eje x en la zona util."""
    if gamma not in LAMBDA_LO:
        return np.linspace(0.01, 0.40, npoints or N_GRID)
    lo   = LAMBDA_LO[gamma]
    step = (LAMBDA_HI[gamma] - lo) / ((npoints or N_GRID) - 1)
    hi   = _upper_limit(gamma, N)
    k    = int(np.floor((hi - lo) / step + 1e-9))   # nº de pasos hasta <= hi
    return lo + step * np.arange(k + 1)


def run_one(gamma, N, n_real, networks_dir, results_dir, coverage,
            fine_npoints, seed, force, verbose=True):
    out_csv = os.path.join(results_dir, f"sweep_g{gamma}_N{N}.csv")
    if os.path.exists(out_csv) and not force:
        print(f"SKIP (ya existe): {out_csv}")
        return out_csv

    paths = fct.replica_paths(networks_dir, gamma=gamma, N=N)
    if not paths:
        print(f"[ERROR] no hay redes para gamma={gamma}, N={N} en '{networks_dir}'.")
        print("        Se esperan ficheros tipo net_g{gamma}_N{N}_r{rep}.csv")
        sys.exit(2)

    os.makedirs(results_dir, exist_ok=True)
    grid = common_grid(gamma, N=N, npoints=fine_npoints)
    print(f"=== gamma={gamma}  N={N}:  {len(paths)} réplicas ===")
    print(f"    n_real/réplica={n_real}  coverage={coverage}  numba={fct.HAVE_NUMBA}")
    print(f"    rejilla COMUN: {len(grid)} lambdas en "
          f"[{grid.min():.4f}, {grid.max():.4f}]")

    t0 = time.time()
    fct.gillespie_sweep(
        paths, grid, gamma=gamma, N=N,
        delta=1.0, coverage_threshold=coverage,
        n_real=n_real, out_csv=out_csv, seed=seed, verbose=verbose)
    print(f"    -> escrito {out_csv}  (total {(time.time()-t0)/60:.1f} min)")
    return out_csv


def main():
    p = argparse.ArgumentParser(
        description="Sweep del método del lifespan para una red (gamma, N).")
    p.add_argument("gamma", type=float, help="exponente gamma (p.ej. 3.5 o 2.5)")
    p.add_argument("N", type=int, help="tamaño de la red (nº de nodos)")
    p.add_argument("--n_real", type=int, default=50000,
                   help="realizaciones por réplica (def. 50000; la cola de "
                        "<tau^2>/chi estaba submuestreada con 20000)")
    p.add_argument("--networks", default="networks",
                   help="carpeta de redes (def. networks)")
    p.add_argument("--results", default="results_2",
                   help="carpeta de salida (def. results_2)")
    p.add_argument("--coverage", type=float, default=0.5,
                   help="umbral de cobertura endémica (def. 0.5)")
    p.add_argument("--fine-npoints", type=int, default=N_GRID, dest="fine_npoints",
                   help=f"nº de lambdas en la rejilla COMUN (def. {N_GRID})")
    p.add_argument("--seed", type=int, default=20, help="semilla base (def. 20)")
    p.add_argument("--force", action="store_true",
                   help="re-correr aunque el CSV ya exista")
    p.add_argument("--quiet", action="store_true", help="menos prints por lambda")
    args = p.parse_args()

    # normalizar gamma a 1 decimal (3.5 / 2.5) para que case con los nombres de fichero
    gamma = round(args.gamma, 1)
    gamma = int(gamma) if gamma == int(gamma) else gamma

    run_one(gamma, args.N, n_real=args.n_real,
            networks_dir=args.networks, results_dir=args.results,
            coverage=args.coverage, fine_npoints=args.fine_npoints,
            seed=args.seed, force=args.force, verbose=not args.quiet)


if __name__ == "__main__":
    main()
