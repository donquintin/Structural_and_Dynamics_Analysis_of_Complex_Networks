"""
functions_2.py — FAST drop-in replacement for functions.py.

Same public API as functions.py (network generation, CSV save/load, the
lifespan Gillespie sweep, and the Part-3 analysis), so existing notebooks work
by changing a single import line:

    import functions_2 as fct        # instead of: import functions as fct

What is different (only the simulation core):

  * The Gillespie engine is JIT-compiled with numba (@njit). If numba is not
    installed it falls back transparently to pure Python (same results, slow).
  * The active-edge bookkeeping uses an ARRAY-based inverse index (the
    "twin" half-edge map) instead of a Python dict — the structure described in
    Section IV of the assignment. This is what makes the inner loop njit-able.
  * Per-realization state is allocated ONCE per network and reset on-touch
    (only the visited nodes/edges are cleared between realizations), instead of
    reallocating N-sized arrays every realization. This is the biggest win for
    large N.
  * Each sweep ALSO records the second moment of the lifespan, <tau^2>
    (columns tau2_mean, tau2_std). These are ADDED columns; everything that the
    old CSVs had is preserved, so old analysis code keeps working and the new
    six-panel analysis (lifespan_analysis.py) gets what it needs.

To install numba (in your sis_env):   pip install numba
"""

import os
import csv
import numpy as np
import networkx as nx

# --- numba: use it if present, otherwise no-op decorator (pure-Python fallback)
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(f):
            return f
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap


# ======================================================================
#  PART 1 — NETWORK GENERATION   (identical behaviour to functions.py)
# ======================================================================
GAMMAS = [3.5, 2.5]
SIZES = [10_000, 30_000, 50_000, 100_000, 300_000, 500_000, 1_000_000]
BASE_SEED = 12345


def n_replicas(N):
    if N <= 100_000:
        return 5
    elif N <= 500_000:
        return 3
    return 2


def net_filename(gamma, N, rep):
    return f"net_g{gamma}_N{N}_r{rep}.csv"


def make_seed(gamma, N, rep, base_seed=BASE_SEED):
    return hash((base_seed, gamma, N, rep)) & 0xFFFFFFFF


def structural_cutoff(N, factor=1.0):
    """Structural cutoff k_max = factor * sqrt(N) (Mata et al., PRE 91, 052117).
    Caps hub degree so <k^2> stays controlled (crucial for gamma < 3)."""
    return max(5, int(factor * np.sqrt(N)))


def sample_powerlaw_degrees(N, gamma, kmin=4, kmax=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if kmax is None:                       # structural cutoff k_max = sqrt(N)
        kmax = structural_cutoff(N)
    kmax = max(kmax, kmin + 1)
    ks = np.arange(kmin, kmax + 1)         # was np.arange(kmin, N): no cutoff -> giant hubs
    weights = ks.astype(float) ** (-gamma)
    weights /= weights.sum()
    degrees = rng.choice(ks, size=N, p=weights)
    if degrees.sum() % 2 != 0:
        degrees[rng.integers(N)] += 1
    return degrees


def build_clean_graph(degrees, seed=None):
    G = nx.configuration_model(degrees.tolist(), seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def generate_network(N, gamma, kmin=4, kmax=None, seed=None, save_path=None):
    rng = np.random.default_rng(seed)
    degrees = sample_powerlaw_degrees(N, gamma, kmin=kmin, kmax=kmax, rng=rng)
    nx_seed = None if seed is None else int(seed) & 0x7FFFFFFF
    G = build_clean_graph(degrees, seed=nx_seed)
    if save_path is not None:
        save_graph_csv(G, save_path)
    deg = np.array([d for _, d in sorted(G.degree())])
    return G, deg


def generate_all_networks(out_dir="networks", base_seed=BASE_SEED,
                          gammas=None, sizes=None, force=False, verbose=True):
    gammas = gammas or GAMMAS
    sizes = sizes or SIZES
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for gamma in gammas:
        for N in sizes:
            for rep in range(n_replicas(N)):
                path = os.path.join(out_dir, net_filename(gamma, N, rep))
                paths.append(path)
                if os.path.exists(path) and not force:
                    if verbose:
                        print(f"SKIP (exists): {os.path.basename(path)}")
                    continue
                seed = make_seed(gamma, N, rep, base_seed)
                G, deg = generate_network(N, gamma, kmin=4, seed=seed,
                                          save_path=path)
                if verbose:
                    print(f"gamma={gamma} N={N} rep={rep}: "
                          f"E={G.number_of_edges()} "
                          f"<k>={2*G.number_of_edges()/N:.3f} "
                          f"kmax={deg.max()} "
                          f"#(k=4)={int((deg==4).sum())} "
                          f"-> {os.path.basename(path)}")
    return paths


def save_graph_csv(G, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    N, E = G.number_of_nodes(), G.number_of_edges()
    with open(path, "w", newline="") as f:
        f.write(f"# N={N} E={E}\n")
        w = csv.writer(f)
        w.writerow(["source", "target"])
        for u, v in G.edges():
            w.writerow([u, v])


def load_edges_csv(path):
    N = None
    with open(path) as f:
        first = f.readline()
        if first.startswith("#"):
            for tok in first[1:].split():
                if tok.startswith("N="):
                    N = int(tok[2:])
    edges = np.loadtxt(path, delimiter=",",
                       skiprows=2 if N is not None else 1, dtype=np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)
    if N is None:
        N = int(edges.max()) + 1
    return edges, N


# ---- CSR adjacency + twin (inverse) index ----------------------------
@njit(cache=True)
def _fill_csr_twin(edges, neighbors, twin, cursor):
    """Fill CSR neighbour array and the half-edge twin map (njit)."""
    E = edges.shape[0]
    for e in range(E):
        u = edges[e, 0]
        v = edges[e, 1]
        ju = cursor[u]
        neighbors[ju] = v
        cursor[u] += 1
        jv = cursor[v]
        neighbors[jv] = u
        cursor[v] += 1
        twin[ju] = jv
        twin[jv] = ju


def _edges_to_csr_full(edges, N):
    """CSR + twin map. Returns (neighbors, offsets, deg, twin)."""
    edges = np.asarray(edges, dtype=np.int64)
    E = len(edges)
    deg = np.zeros(N, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(deg)
    neighbors = np.empty(2 * E, dtype=np.int64)
    twin = np.empty(2 * E, dtype=np.int64)
    cursor = offsets[:-1].copy()
    _fill_csr_twin(edges, neighbors, twin, cursor)
    return neighbors, offsets, deg, twin


def edges_to_csr(edges, N):
    """Compat wrapper: returns (neighbors, offsets, deg) like functions.py."""
    nb, off, deg, _twin = _edges_to_csr_full(edges, N)
    return nb, off, deg


def load_network_csr(path):
    """Compat: returns (neighbors, offsets, N, deg) like functions.py."""
    edges, N = load_edges_csv(path)
    nb, off, deg, _twin = _edges_to_csr_full(edges, N)
    return nb, off, N, deg


def _load_csr_full(path):
    """Returns (neighbors, offsets, N, deg, twin) — used by the sweep."""
    edges, N = load_edges_csv(path)
    nb, off, deg, twin = _edges_to_csr_full(edges, N)
    return nb, off, N, deg, twin


def min_degree_nodes_csr(deg, kmin=4):
    return np.where(deg == kmin)[0]


def replica_paths(out_dir, gamma, N):
    paths = []
    for rep in range(n_replicas(N)):
        p = os.path.join(out_dir, net_filename(gamma, N, rep))
        if os.path.exists(p):
            paths.append(p)
    return paths


# ======================================================================
#  PART 2 — GILLESPIE SIS ENGINE  (njit, array inverse index, reset-on-touch)
# ======================================================================
#
# Active edges are stored as DIRECTED half-edge indices (positions in the CSR
# `neighbors` array). For a half-edge index j owned by node u with neighbours[j]=w:
#   * twin[j] is the opposite half-edge (owned by w, pointing to u).
#   * pos_of[j] = position of half-edge j inside act_edges[0:n_act], or -1.
# An edge (u infected -> w susceptible) is active iff its half-edge j (u->w) is
# in act_edges. Removal uses swap-with-last + the pos_of inverse index -> O(1).
# ----------------------------------------------------------------------


@njit(cache=True)
def _infect(node, neighbors, offsets, twin, state, inf_list, inf_pos,
            ever_inf, touched, act_edges, pos_of,
            n_inf, n_act, coverage, touched_n):
    state[node] = 1
    inf_list[n_inf] = node
    inf_pos[node] = n_inf
    n_inf += 1
    if ever_inf[node] == 0:
        ever_inf[node] = 1
        coverage += 1
        touched[touched_n] = node
        touched_n += 1
    for j in range(offsets[node], offsets[node + 1]):
        w = neighbors[j]
        if state[w] == 0:
            # node(I) -- w(S): half-edge j (node->w) becomes active
            act_edges[n_act] = j
            pos_of[j] = n_act
            n_act += 1
        else:
            # w(I) -- node: half-edge twin[j] (w->node) stops being active
            tj = twin[j]
            idx = pos_of[tj]
            last = n_act - 1
            if idx != last:
                le = act_edges[last]
                act_edges[idx] = le
                pos_of[le] = idx
            pos_of[tj] = -1
            n_act -= 1
    return n_inf, n_act, coverage, touched_n


@njit(cache=True)
def _recover(node, neighbors, offsets, twin, state, inf_list, inf_pos,
             act_edges, pos_of, n_inf, n_act):
    p = inf_pos[node]
    last = n_inf - 1
    if p != last:
        moved = inf_list[last]
        inf_list[p] = moved
        inf_pos[moved] = p
    inf_pos[node] = -1
    n_inf -= 1
    state[node] = 0
    for j in range(offsets[node], offsets[node + 1]):
        w = neighbors[j]
        if state[w] == 1:
            # w(I) -- node(S): half-edge twin[j] (w->node) becomes active
            tj = twin[j]
            act_edges[n_act] = tj
            pos_of[tj] = n_act
            n_act += 1
        else:
            # node(was I)->w(S): half-edge j was active, remove it
            idx = pos_of[j]
            last2 = n_act - 1
            if idx != last2:
                le = act_edges[last2]
                act_edges[idx] = le
                pos_of[le] = idx
            pos_of[j] = -1
            n_act -= 1
    return n_inf, n_act


@njit(cache=True)
def _run_many(neighbors, offsets, twin, N, candidates, lam, delta,
              threshold_count, n_real, seed,
              state, inf_list, inf_pos, ever_inf, touched, act_edges, pos_of):
    """Run n_real outbreaks; return (n_endemic, tau_sum, tau2_sum, tau_n)."""
    np.random.seed(seed)
    ncand = len(candidates)
    n_endemic = 0
    tau_sum = 0.0
    tau2_sum = 0.0
    tau_n = 0

    for _r in range(n_real):
        n_inf = 0
        n_act = 0
        coverage = 0
        touched_n = 0

        seed_node = candidates[np.random.randint(ncand)]
        n_inf, n_act, coverage, touched_n = _infect(
            seed_node, neighbors, offsets, twin, state, inf_list, inf_pos,
            ever_inf, touched, act_edges, pos_of, n_inf, n_act, coverage,
            touched_n)

        t = 0.0
        endemic = False
        while n_inf > 0:
            rate_inf = lam * n_act
            rate_rec = delta * n_inf
            lam_tot = rate_inf + rate_rec
            t += -np.log(np.random.random()) / lam_tot
            if np.random.random() * lam_tot < rate_inf:
                e = np.random.randint(n_act)
                v_sus = neighbors[act_edges[e]]
                n_inf, n_act, coverage, touched_n = _infect(
                    v_sus, neighbors, offsets, twin, state, inf_list, inf_pos,
                    ever_inf, touched, act_edges, pos_of, n_inf, n_act,
                    coverage, touched_n)
            else:
                k = np.random.randint(n_inf)
                node = inf_list[k]
                n_inf, n_act = _recover(
                    node, neighbors, offsets, twin, state, inf_list, inf_pos,
                    act_edges, pos_of, n_inf, n_act)
            if coverage >= threshold_count:
                endemic = True
                break

        if endemic:
            n_endemic += 1
        else:
            tau_sum += t
            tau2_sum += t * t
            tau_n += 1

        # ---- reset-on-touch (only visited nodes / still-active edges) ----
        for i in range(touched_n):
            nd = touched[i]
            state[nd] = 0
            ever_inf[nd] = 0
            inf_pos[nd] = -1
        for e in range(n_act):
            pos_of[act_edges[e]] = -1

    return n_endemic, tau_sum, tau2_sum, tau_n


def _alloc_work(N, two_E):
    """Preallocate (and pre-clear) the per-network work arrays."""
    return {
        "state": np.zeros(N, dtype=np.int8),
        "inf_list": np.empty(N, dtype=np.int64),
        "inf_pos": np.full(N, -1, dtype=np.int64),
        "ever_inf": np.zeros(N, dtype=np.uint8),
        "touched": np.empty(N, dtype=np.int64),
        "act_edges": np.empty(two_E, dtype=np.int64),
        "pos_of": np.full(two_E, -1, dtype=np.int64),
    }


def measure_point(neighbors, offsets, twin, N, candidates, lam, delta,
                  coverage_threshold, n_real, seed, work=None):
    """
    Run n_real outbreaks at fixed lambda on ONE network and aggregate.
    Returns dict with P_end, tau_mean, tau2_mean, n_real, n_endemic, tau_n.
    """
    if work is None:
        work = _alloc_work(N, len(neighbors))
    threshold_count = int(np.ceil(coverage_threshold * N))
    n_endemic, tau_sum, tau2_sum, tau_n = _run_many(
        neighbors, offsets, twin, N, candidates, lam, delta,
        threshold_count, n_real, int(seed) & 0x7FFFFFFF,
        work["state"], work["inf_list"], work["inf_pos"], work["ever_inf"],
        work["touched"], work["act_edges"], work["pos_of"])
    return {
        "P_end": n_endemic / n_real,
        "tau_mean": (tau_sum / tau_n) if tau_n > 0 else np.nan,
        "tau2_mean": (tau2_sum / tau_n) if tau_n > 0 else np.nan,
        "n_real": n_real,
        "n_endemic": n_endemic,
        "tau_n": tau_n,
    }


def run_realization(neighbors, offsets, twin, N, candidates, lam, delta,
                    coverage_threshold, seed):
    """Single outbreak (mainly for testing). Returns (endemic, tau)."""
    m = measure_point(neighbors, offsets, twin, N, candidates, lam, delta,
                      coverage_threshold, 1, seed)
    endemic = (m["n_endemic"] == 1)
    tau = np.nan if endemic else m["tau_mean"]
    return endemic, tau


def _seed_for(seed, li, ri):
    return (int(seed) * 1_000_003 + li * 9176 + ri * 131 + 1) & 0x7FFFFFFF


def gillespie_sweep(network_paths, lambdas, gamma, N,
                    delta=1.0, coverage_threshold=0.5, n_real=10_000,
                    out_csv=None, seed=0, verbose=True):
    """
    Full lifespan sweep for ONE (gamma, N). Same signature & CSV layout as
    functions.gillespie_sweep, plus two extra columns: tau2_mean, tau2_std.
    """
    nets = []
    for p in network_paths:
        nb, off, Ncheck, deg, twin = _load_csr_full(p)
        cand = min_degree_nodes_csr(deg, kmin=4)
        if len(cand) == 0:
            raise ValueError(f"No degree-4 nodes in {p}; cannot seed outbreaks.")
        work = _alloc_work(Ncheck, len(nb))
        nets.append((nb, off, twin, Ncheck, cand, work))

    results = []
    for li, lam in enumerate(lambdas):
        per_Pend, per_tau, per_tau2 = [], [], []
        for ri, (nb, off, twin, Nc, cand, work) in enumerate(nets):
            m = measure_point(nb, off, twin, Nc, cand, lam, delta,
                              coverage_threshold, n_real,
                              _seed_for(seed, li, ri), work=work)
            per_Pend.append(m["P_end"])
            if not np.isnan(m["tau_mean"]):
                per_tau.append(m["tau_mean"])
            if not np.isnan(m["tau2_mean"]):
                per_tau2.append(m["tau2_mean"])

        Pend_mean = float(np.mean(per_Pend))
        Pend_std = float(np.std(per_Pend))
        tau_mean = float(np.mean(per_tau)) if per_tau else np.nan
        tau_std = float(np.std(per_tau)) if per_tau else np.nan
        tau2_mean = float(np.mean(per_tau2)) if per_tau2 else np.nan
        tau2_std = float(np.std(per_tau2)) if per_tau2 else np.nan

        results.append({
            "gamma": gamma, "N": N, "lambda": float(lam),
            "n_networks": len(nets),
            "n_real_per_net": n_real,
            "n_real_total": n_real * len(nets),
            "P_end": Pend_mean, "P_end_std": Pend_std,
            "tau_mean": tau_mean, "tau_std": tau_std,
            "tau2_mean": tau2_mean, "tau2_std": tau2_std,
        })
        if verbose:
            print(f"  lambda={lam:.4f}  P_end={Pend_mean:.4f}  "
                  f"tau={tau_mean:.3f}  tau2={tau2_mean:.3f}  (nets={len(nets)})")

    if out_csv is not None:
        write_results_csv(results, out_csv)
    return results


def write_results_csv(results, out_csv):
    if not results:
        return
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fields = ["gamma", "N", "lambda", "n_networks", "n_real_per_net",
              "n_real_total", "P_end", "P_end_std", "tau_mean", "tau_std",
              "tau2_mean", "tau2_std"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow(row)


# ======================================================================
#  PART 3 — ANALYSIS  (verbatim from functions.py; reads the same CSVs)
# ======================================================================
def load_sweep_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise ValueError(f"Empty sweep file: {path}")
    gamma = float(rows[0]["gamma"])
    N = int(float(rows[0]["N"]))
    lam = np.array([float(r["lambda"]) for r in rows])
    P_end = np.array([float(r["P_end"]) for r in rows])
    P_std = np.array([float(r.get("P_end_std", "nan") or "nan") for r in rows])
    tau = np.array([float(r["tau_mean"]) for r in rows])
    tau_std = np.array([float(r.get("tau_std", "nan") or "nan") for r in rows])
    order = np.argsort(lam)
    return {"gamma": gamma, "N": N,
            "lam": lam[order], "P_end": P_end[order], "P_end_std": P_std[order],
            "tau": tau[order], "tau_std": tau_std[order]}


def load_all_sweeps(results_dir, gamma, sizes=None):
    sizes = sizes or SIZES
    data = []
    for N in sizes:
        path = os.path.join(results_dir, f"sweep_g{gamma}_N{N}.csv")
        if os.path.exists(path):
            data.append(load_sweep_csv(path))
        else:
            print(f"  [load] missing: {os.path.basename(path)} (skipped)")
    data.sort(key=lambda d: d["N"])
    return data


def locate_tau_peak(lam, tau, window=5):
    good = ~np.isnan(tau)
    lam, tau = lam[good], tau[good]
    if len(lam) == 0:
        return np.nan, np.nan
    i = int(np.argmax(tau))
    lo = max(0, i - window // 2)
    hi = min(len(lam), lo + window)
    lo = max(0, hi - window)
    xs, ys = lam[lo:hi], tau[lo:hi]
    if len(xs) >= 3:
        a, b, c = np.polyfit(xs, ys, 2)
        if a < 0:
            lam_p = -b / (2 * a)
            tau_p = a * lam_p**2 + b * lam_p + c
            if xs.min() <= lam_p <= xs.max():
                return float(lam_p), float(tau_p)
    return float(lam[i]), float(tau[i])


def fit_lambda_c(data, verbose=True):
    from scipy.optimize import curve_fit
    Ns = np.array([d["N"] for d in data], dtype=float)
    lam_peak = np.empty(len(data))
    tau_peak = np.empty(len(data))
    for k, d in enumerate(data):
        lam_peak[k], tau_peak[k] = locate_tau_peak(d["lam"], d["tau"])
    if verbose:
        print("\n[Step 3] Peak of <tau> per size:")
        print(f"  {'N':>10}  {'lambda_p':>10}  {'tau_peak':>10}")
        for N, lp, tp in zip(Ns, lam_peak, tau_peak):
            print(f"  {int(N):>10}  {lp:>10.5f}  {tp:>10.4f}")

    def model(N, lam_c, a, inv_nu):
        return lam_c + a * N ** (-inv_nu)

    # 1/nu bounded to a PHYSICAL window (was [0.05, 3.0], which let a noisy
    # lam_p(N) drive 1/nu to absurd values and wreck the data collapse).
    lo_b = [0.0, 0.0, 0.2]
    hi_b = [lam_peak.max() * 1.5 + 1e-6, 10.0, 1.2]
    p0 = [max(0.0, lam_peak.min() * 0.8),
          max(1e-3, lam_peak.max() - lam_peak.min()), 0.5]
    try:
        popt, pcov = curve_fit(model, Ns, lam_peak, p0=p0,
                               bounds=(lo_b, hi_b), maxfev=20000)
        lam_c, a, inv_nu = popt
        perr = np.sqrt(np.diag(pcov))
        for name, val, lo, hi in [("1/nu", inv_nu, lo_b[2], hi_b[2]),
                                  ("lambda_c", lam_c, lo_b[0], hi_b[0])]:
            if abs(val - lo) < 1e-3 or abs(val - hi) < 1e-3:
                print(f"  [fit warning] {name} hit a bound ({val:.4f}); "
                      f"results likely unreliable -- need more sizes / "
                      f"realizations, especially large N.")
    except Exception as e:
        print(f"  [fit warning] curve_fit failed ({e}); using fallback.")
        lam_c, a, inv_nu = lam_peak.min(), 0.0, np.nan
        perr = [np.nan, np.nan, np.nan]
    if verbose:
        print(f"\n  Fit lambda_p(N) = lambda_c + a*N^(-1/nu):")
        print(f"    lambda_c = {lam_c:.5f} +/- {perr[0]:.5f}")
        print(f"    1/nu     = {inv_nu:.4f} +/- {perr[2]:.4f}")
        print(f"    a        = {a:.4f}")
    return {"lambda_c": float(lam_c), "inv_nu": float(inv_nu), "a": float(a),
            "lambda_c_err": float(perr[0]), "inv_nu_err": float(perr[2]),
            "N": Ns, "lam_peak": lam_peak, "tau_peak": tau_peak}


def fit_gamma1_over_nu(step3, verbose=True):
    Ns = step3["N"]
    heights = step3["tau_peak"]
    logN = np.log(Ns)
    logH = np.log(heights)
    slope, intercept = np.polyfit(logN, logH, 1)
    if verbose:
        print("\n[Step 4] Peak height of <tau> vs N (log-log):")
        print(f"  {'N':>10}  {'tau_peak':>10}")
        for N, h in zip(Ns, heights):
            print(f"  {int(N):>10}  {h:>10.4f}")
        print(f"\n  Fit <tau>_peak ~ N^(gamma1/nu):")
        print(f"    gamma1/nu = {slope:.4f}")
    return {"gamma1_over_nu": float(slope), "intercept": float(intercept),
            "N": Ns, "tau_peak": heights}


def pend_at_lambda_c(data, lambda_c, verbose=True):
    Ns = np.array([d["N"] for d in data], dtype=float)
    pend_c = np.empty(len(data))
    for k, d in enumerate(data):
        pend_c[k] = np.interp(lambda_c, d["lam"], d["P_end"])
    mask = pend_c > 0
    logN = np.log(Ns[mask])
    logP = np.log(pend_c[mask])
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(logN, logP, 1)
    else:
        slope, intercept = np.nan, np.nan
    beta_over_nu = -slope
    if verbose:
        print(f"\n[Step 5] P_end at lambda_c = {lambda_c:.5f} vs N (log-log):")
        print(f"  {'N':>10}  {'P_end(lc)':>10}")
        for N, p in zip(Ns, pend_c):
            print(f"  {int(N):>10}  {p:>10.5f}")
        print(f"\n  Fit P_end(lambda_c) ~ N^(-beta/nu):")
        print(f"    beta/nu = {beta_over_nu:.4f}")
    return {"beta_over_nu": float(beta_over_nu), "intercept": float(intercept),
            "N": Ns, "pend_c": pend_c}


def analyze_gamma(results_dir, gamma, sizes=None, make_plots=True,
                  fig_dir="figures", verbose=True):
    print("=" * 60)
    print(f"ANALYSIS  gamma = {gamma}")
    print("=" * 60)
    data = load_all_sweeps(results_dir, gamma, sizes=sizes)
    if len(data) < 2:
        raise ValueError(
            f"Need >=2 sizes to fit scaling for gamma={gamma}; "
            f"found {len(data)}. Run more sizes first.")
    step3 = fit_lambda_c(data, verbose=verbose)
    step4 = fit_gamma1_over_nu(step3, verbose=verbose)
    step5 = pend_at_lambda_c(data, step3["lambda_c"], verbose=verbose)
    summary = {
        "gamma": gamma,
        "lambda_c": step3["lambda_c"], "inv_nu": step3["inv_nu"],
        "gamma1_over_nu": step4["gamma1_over_nu"],
        "beta_over_nu": step5["beta_over_nu"],
        "data": data, "step3": step3, "step4": step4, "step5": step5,
    }
    if verbose:
        print("\n" + "-" * 60)
        print(f"SUMMARY  gamma={gamma}")
        print(f"  lambda_c   = {summary['lambda_c']:.5f}")
        print(f"  1/nu       = {summary['inv_nu']:.4f}")
        print(f"  gamma1/nu  = {summary['gamma1_over_nu']:.4f}")
        print(f"  beta/nu    = {summary['beta_over_nu']:.4f}")
        print("-" * 60)
    if make_plots:
        plot_all(summary, fig_dir=fig_dir)
    return summary


def plot_all(summary, fig_dir="figures"):
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    g = summary["gamma"]
    data = summary["data"]
    lam_c = summary["lambda_c"]
    inv_nu = summary["inv_nu"]
    g1_nu = summary["gamma1_over_nu"]
    b_nu = summary["beta_over_nu"]
    saved = []

    def _save(fig, name):
        path = os.path.join(fig_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    for d in data:
        ax.plot(d["lam"], d["tau"], marker="o", ms=3, label=f"N={d['N']:.0e}")
    ax.axvline(lam_c, ls="--", c="k", lw=1, label=f"$\\lambda_c$={lam_c:.4f}")
    ax.set_xlabel("$\\lambda$"); ax.set_ylabel(r"$\langle\tau\rangle$")
    ax.set_title(f"Lifespan, $\\gamma$={g}"); ax.legend(fontsize=7)
    _save(fig, f"tau_vs_lambda_g{g}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for d in data:
        ax.plot(d["lam"], d["P_end"], marker="o", ms=3, label=f"N={d['N']:.0e}")
    ax.axvline(lam_c, ls="--", c="k", lw=1)
    ax.set_xlabel("$\\lambda$"); ax.set_ylabel("$P_{end}$")
    ax.set_title(f"Order parameter, $\\gamma$={g}"); ax.legend(fontsize=7)
    _save(fig, f"Pend_vs_lambda_g{g}.png")

    s3 = summary["step3"]
    fig, ax = plt.subplots(figsize=(6, 4))
    Ns = s3["N"]
    ax.plot(Ns ** (-inv_nu), s3["lam_peak"], "o", label="data")
    xx = np.linspace(0, (Ns ** (-inv_nu)).max() * 1.05, 100)
    ax.plot(xx, lam_c + s3["a"] * xx, "-",
            label=f"fit: $\\lambda_c$={lam_c:.4f}, 1/$\\nu$={inv_nu:.3f}")
    ax.set_xlabel("$N^{-1/\\nu}$"); ax.set_ylabel("$\\lambda_p(N)$")
    ax.set_title(f"Step 3: peak position, $\\gamma$={g}"); ax.legend(fontsize=8)
    _save(fig, f"lambda_p_fit_g{g}.png")

    s4 = summary["step4"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(s4["N"], s4["tau_peak"], "o", label="data")
    fitline = np.exp(s4["intercept"]) * s4["N"] ** g1_nu
    ax.loglog(s4["N"], fitline, "-",
              label=f"slope $\\gamma_1/\\nu$={g1_nu:.3f}")
    ax.set_xlabel("N"); ax.set_ylabel(r"$\langle\tau\rangle_{peak}$")
    ax.set_title(f"Step 4: peak height, $\\gamma$={g}"); ax.legend(fontsize=8)
    _save(fig, f"peak_height_g{g}.png")

    s5 = summary["step5"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(s5["N"], s5["pend_c"], "o", label="data")
    fitline = np.exp(s5["intercept"]) * s5["N"] ** (-b_nu)
    ax.loglog(s5["N"], fitline, "-", label=f"slope $-\\beta/\\nu$={-b_nu:.3f}")
    ax.set_xlabel("N"); ax.set_ylabel("$P_{end}(\\lambda_c)$")
    ax.set_title(f"Step 5: order parameter, $\\gamma$={g}"); ax.legend(fontsize=8)
    _save(fig, f"Pend_at_lc_g{g}.png")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    for d in data:
        N = d["N"]
        x = (d["lam"] - lam_c) * N ** inv_nu
        axL.plot(x, d["tau"] * N ** (-g1_nu), marker="o", ms=3,
                 label=f"N={N:.0e}")
        axR.plot(x, d["P_end"] * N ** (b_nu), marker="o", ms=3,
                 label=f"N={N:.0e}")
    axL.set_xlabel("$(\\lambda-\\lambda_c)\\,N^{1/\\nu}$")
    axL.set_ylabel(r"$\langle\tau\rangle\,N^{-\gamma_1/\nu}$")
    axL.set_title("Collapse: lifespan"); axL.legend(fontsize=7)
    axR.set_xlabel("$(\\lambda-\\lambda_c)\\,N^{1/\\nu}$")
    axR.set_ylabel("$P_{end}\\,N^{\\beta/\\nu}$")
    axR.set_title("Collapse: order parameter"); axR.legend(fontsize=7)
    fig.suptitle(f"Step 6: data collapse, $\\gamma$={g}")
    _save(fig, f"collapse_g{g}.png")

    print(f"\n[plots] saved {len(saved)} figures to '{fig_dir}/'")
    for p in saved:
        print(f"  {p}")
    return saved
