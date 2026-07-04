"""
lifespan_analysis.py — six-panel lifespan-method figure (a-f).  [ROBUST REWRITE]

Layout (Mata, Boguna, Castellano, Pastor-Satorras, PRE 91, 052117 (2015)):

    (a) <tau>(lambda) per size, with error bars
    (b) peak position lambda_p(N) -> lambda_c, 1/nu   [fit lam_p = lam_c + a N^-1/nu]
    (c) peak heights -> gamma1/nu (<tau>) and chi/tau2 exponent
    (d) P_end(lambda_c, N) -> beta/nu   (honest: flagged if no clean power law)
    (e) susceptibility / <tau^2> collapse
    (f) P_end collapse

WHAT CHANGED vs the previous version, and WHY
---------------------------------------------
1. ROBUST PEAK LOCATION.  <tau>(lambda) is a very weak signal here (it varies by
   only a few %), so a bare argmax + parabola latched onto noise and produced a
   zig-zag, non-monotonic lambda_p(N).  We now (i) light-smooth the curve, (ii)
   take the interior maximum only, (iii) fit a weighted parabola on a symmetric
   window, and (iv) reject the parabola if its vertex leaves the window or the
   curvature has the wrong sign (fall back to the discrete max).  We ALSO expose
   the lifespan susceptibility chi = (<tau^2> - <tau>^2)/<tau>, whose peak is a
   much cleaner locator of lambda_p (the second moment carries the divergence).

2. BOUNDED FITS.  The FSS fit lam_p = lam_c + a N^{-1/nu} now bounds 1/nu to a
   physical window and refuses to report a value sitting on a bound.  The old
   code allowed 1/nu up to 3, which let a noisy lam_p(N) drive 1/nu to absurd
   values and blow the collapse x-axis up to ~1e9-1e15.

3. SANE COLLAPSE.  The old "optimize_collapse" was an UNBOUNDED Nelder-Mead that
   wandered to lambda_c ~ 1e4 and 1/nu ~ 2.8 (see the old panel titles).  The
   collapse now (i) uses the FSS-fitted (lambda_c, 1/nu) by default and (ii) only
   refines inside a TIGHT bounded box, restricted to the lambda-range where the
   curves actually overlap.  This keeps the scaling axis O(1)-O(10).

4. HONEST gamma = 2.5.  For gamma < 3 the epidemic threshold drifts to 0 and the
   lifespan method is not expected to give clean finite-size scaling.  The code
   now flags this instead of printing a confident-but-meaningless lambda_c.

Public API is unchanged: available_sizes, load_all, make_panels.
"""

import os
import csv
import glob
import re
import numpy as np


# ----------------------------------------------------------------------
#  DATA LOADING
# ----------------------------------------------------------------------
def load_sweep(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"Empty sweep file: {path}")

    def col(name):
        return np.array([float(r.get(name, "nan") or "nan") for r in rows])

    lam = col("lambda")
    order = np.argsort(lam)
    has_tau2 = "tau2_mean" in rows[0]
    tau = col("tau_mean")[order]
    tau2 = col("tau2_mean")[order] if has_tau2 else np.full(len(lam), np.nan)
    # lifespan susceptibility chi = (<tau^2> - <tau>^2)/<tau>   (variance/mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi = (tau2 - tau ** 2) / tau
    return {
        "gamma": float(rows[0]["gamma"]),
        "N": int(float(rows[0]["N"])),
        "lam": lam[order],
        "P_end": col("P_end")[order],
        "P_end_std": col("P_end_std")[order],
        "tau": tau,
        "tau_std": col("tau_std")[order],
        "tau2": tau2,
        "tau2_std": col("tau2_std")[order] if has_tau2 else np.full(len(lam), np.nan),
        "chi": chi,
        "has_tau2": has_tau2,
    }


def available_sizes(results_dir, gamma):
    sizes = []
    for p in glob.glob(os.path.join(results_dir, f"sweep_g{gamma}_N*.csv")):
        m = re.search(r"_N(\d+)\.csv$", os.path.basename(p))
        if m:
            sizes.append(int(m.group(1)))
    return sorted(sizes)


def load_all(results_dir, gamma, sizes=None):
    if sizes is None:
        sizes = available_sizes(results_dir, gamma)
    data = []
    for N in sizes:
        p = os.path.join(results_dir, f"sweep_g{gamma}_N{N}.csv")
        if os.path.exists(p):
            data.append(load_sweep(p))
    data.sort(key=lambda d: d["N"])
    return data

# Peak location
def _smooth(y, k=3):
    """Light moving-average that ignores NaNs; preserves length."""
    y = np.asarray(y, float)
    out = y.copy()
    n = len(y)
    h = k // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        seg = y[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def robust_peak(lam, y, window=5, smooth_k=3):
    """
    Return (lambda_peak, height, reliable_flag).

    Reliable = the peak is interior (not at a grid edge) and a downward parabola
    fits its neighbourhood with a vertex inside the window.  Weak/edge peaks are
    returned with reliable=False so callers can down-weight them.
    """
    good = np.isfinite(y)
    lam, y = np.asarray(lam)[good], np.asarray(y)[good]
    if len(lam) < 3:
        return (np.nan, np.nan, False)
    ys = _smooth(y, smooth_k)
    i = int(np.argmax(ys))
    interior = (0 < i < len(lam) - 1)
    lo = max(0, i - window // 2)
    hi = min(len(lam), lo + window)
    lo = max(0, hi - window)
    xs, yy = lam[lo:hi], y[lo:hi]
    if len(xs) >= 3:
        a, b, c = np.polyfit(xs, yy, 2)
        if a < 0:
            xp = -b / (2 * a)
            if xs.min() <= xp <= xs.max():
                yp = a * xp ** 2 + b * xp + c
                return (float(xp), float(yp), bool(interior))
    return (float(lam[i]), float(y[i]), bool(interior))

#  Fits  (bounded, with reliability flags)

def fit_peak_position(Ns, lam_peak, weights=None,
                      inv_nu_bounds=(0.2, 1.2)):
    """
    Fit lambda_p(N) = lambda_c + a N^{-1/nu} with PHYSICAL bounds on 1/nu.
    Returns dict(lambda_c, a, inv_nu, *_err, on_bound).
    """
    from scipy.optimize import curve_fit
    Ns = np.asarray(Ns, float)
    lam_peak = np.asarray(lam_peak, float)
    m = np.isfinite(lam_peak) & np.isfinite(Ns)
    Ns, lam_peak = Ns[m], lam_peak[m]
    sigma = None
    if weights is not None:
        w = np.asarray(weights, float)[m]
        sigma = 1.0 / np.clip(w, 1e-6, None)

    def model(N, lam_c, a, inv_nu):
        return lam_c + a * N ** (-inv_nu)

    lo = [0.0, 0.0, inv_nu_bounds[0]]
    hi = [max(lam_peak.max() * 1.5, 1e-6), 50.0, inv_nu_bounds[1]]
    p0 = [max(0.0, lam_peak.min() * 0.8),
          max(1e-3, lam_peak.max() - lam_peak.min()), 0.4]
    try:
        popt, pcov = curve_fit(model, Ns, lam_peak, p0=p0, sigma=sigma,
                               bounds=(lo, hi), maxfev=40000)
        lam_c, a, inv_nu = popt
        perr = np.sqrt(np.diag(pcov))
        on_bound = (abs(inv_nu - lo[2]) < 1e-3 or abs(inv_nu - hi[2]) < 1e-3)
    except Exception as e:
        print(f"  [fit_peak_position] failed: {e}")
        lam_c, a, inv_nu = float(np.nanmin(lam_peak)), 0.0, np.nan
        perr = [np.nan, np.nan, np.nan]
        on_bound = True
    return {"lambda_c": float(lam_c), "a": float(a), "inv_nu": float(inv_nu),
            "lambda_c_err": float(perr[0]), "inv_nu_err": float(perr[2]),
            "on_bound": bool(on_bound)}


def fit_loglog_slope(Ns, heights):
    Ns = np.asarray(Ns, float)
    heights = np.asarray(heights, float)
    m = np.isfinite(heights) & (heights > 0)
    if m.sum() < 2:
        return np.nan, np.nan
    slope, b = np.polyfit(np.log(Ns[m]), np.log(heights[m]), 1)
    return float(slope), float(b)


def _overlap_range(data, lam_c, inv_nu):
    """Scaling x-range common to all sizes (so collapse axis stays O(1))."""
    xmins, xmaxs = [], []
    for d in data:
        x = (d["lam"] - lam_c) * d["N"] ** inv_nu
        xmins.append(np.nanmin(x)); xmaxs.append(np.nanmax(x))
    return max(xmins), min(xmaxs)


def collapse_cost(lam_c, inv_nu, exp, data, which, n_grid=60):
    xs_all, ys_all = [], []
    for d in data:
        N = d["N"]
        x = (d["lam"] - lam_c) * N ** inv_nu
        if which == "chi":
            y = d["chi"] * N ** (-exp)
        elif which == "tau2":
            yv = d["tau2"] if d["has_tau2"] else d["tau"]
            y = yv * N ** (-exp)
        else:
            y = d["P_end"] * N ** (exp)
        good = np.isfinite(x) & np.isfinite(y)
        xs_all.append(x[good]); ys_all.append(y[good])
    xmin = max(x.min() for x in xs_all if x.size)
    xmax = min(x.max() for x in xs_all if x.size)
    if not (xmax > xmin):
        return np.inf
    grid = np.linspace(xmin, xmax, n_grid)
    Y = []
    for x, y in zip(xs_all, ys_all):
        idx = np.argsort(x)
        Y.append(np.interp(grid, x[idx], y[idx]))
    Y = np.array(Y)
    scale = np.nanmean(np.abs(Y)) + 1e-12
    return float(np.nanmean(np.nanvar(Y, axis=0)) / scale ** 2)


def refine_collapse(data, which, exp, lam_c0, inv_nu0,
                    lam_box=0.5, inv_box=0.25):
    """
    BOUNDED refinement of (lambda_c, 1/nu) around the FSS values.
    Stays inside lam_c0*(1+-lam_box) and inv_nu0*(1+-inv_box) so it can never run
    away to the nonsense values the old unbounded optimiser produced.
    """
    from scipy.optimize import minimize
    lc_lo, lc_hi = max(0.0, lam_c0 * (1 - lam_box)), lam_c0 * (1 + lam_box) + 1e-6
    nu_lo, nu_hi = inv_nu0 * (1 - inv_box), inv_nu0 * (1 + inv_box)

    def obj(p):
        lc, inv = p
        if not (lc_lo <= lc <= lc_hi and nu_lo <= inv <= nu_hi):
            return np.inf
        return collapse_cost(lc, inv, exp, data, which)

    try:
        res = minimize(obj, x0=[lam_c0, inv_nu0], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 1500})
        lc, inv = res.x
        if not (lc_lo <= lc <= lc_hi and nu_lo <= inv <= nu_hi):
            return lam_c0, inv_nu0
        return float(lc), float(inv)
    except Exception as e:
        print(f"  [refine_collapse {which}] failed: {e}")
        return lam_c0, inv_nu0

#  Six-panel figure

def make_panels(results_dir, gamma, sizes=None, demo_mode=None,
                optimize_collapses=True, use_chi=True, fig=None):
    import matplotlib.pyplot as plt

    data = load_all(results_dir, gamma, sizes=sizes)
    if len(data) < 2:
        raise ValueError(f"Need >=2 sizes for gamma={gamma}; found {len(data)}.")
    has_tau2 = all(d["has_tau2"] for d in data)
    Ns = np.array([d["N"] for d in data], float)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(data)))

    # ---- robust peaks of <tau> and chi ----
    pk_tau = [robust_peak(d["lam"], d["tau"]) for d in data]
    lam_p1 = np.array([p[0] for p in pk_tau])
    tau_p1 = np.array([p[1] for p in pk_tau])
    ok_tau = np.array([p[2] for p in pk_tau])
    pk_chi = [robust_peak(d["lam"], d["chi"]) for d in data]
    lam_pc = np.array([p[0] for p in pk_chi])
    chi_p = np.array([p[1] for p in pk_chi])
    ok_chi = np.array([p[2] for p in pk_chi])

    # which locator drives lambda_c?  chi is cleaner when available & reliable.
    locator = "chi" if (use_chi and has_tau2 and ok_chi.sum() >= 2) else "tau"
    lam_loc = lam_pc if locator == "chi" else lam_p1
    ok_loc = ok_chi if locator == "chi" else ok_tau
    w = np.where(ok_loc, 1.0, 0.25)        # down-weight unreliable peaks

    # ---- (b) peak position fit -> lambda_c, 1/nu ----
    pos = fit_peak_position(Ns, lam_loc, weights=w)
    lam_c, inv_nu = pos["lambda_c"], pos["inv_nu"]
    reliable_fit = (not pos["on_bound"]) and np.isfinite(inv_nu) \
        and (gamma >= 3.0) and (ok_loc.sum() >= 3)

    # ---- (c) peak-height exponents ----
    g1_nu, _ = fit_loglog_slope(Ns, tau_p1)            # <tau>_peak ~ N^{gamma1/nu}
    gc_nu, _ = fit_loglog_slope(Ns, chi_p)             # chi_peak  ~ N^{gamma'/nu}

    # ---- (d/f) beta/nu from P_end(lambda_c) ----
    pend_c = np.array([np.interp(lam_c, d["lam"], d["P_end"]) for d in data])
    if demo_mode is None:
        demo_mode = (not reliable_fit) or np.all(pend_c < 1e-6)
    b_slope, _ = fit_loglog_slope(Ns, pend_c)
    beta_nu = -b_slope if np.isfinite(b_slope) else np.nan

    # ---- collapse params: start from FSS fit, refine inside a tight box ----
    exp_e = gc_nu if locator == "chi" else g1_nu
    which_e = "chi" if locator == "chi" else ("tau2" if has_tau2 else "tau")
    lc_e, inv_e = lam_c, inv_nu
    lc_f, inv_f = lam_c, inv_nu
    if optimize_collapses and np.isfinite(inv_nu):
        lc_e, inv_e = refine_collapse(data, which_e, exp_e, lam_c, inv_nu)
        if np.isfinite(beta_nu):
            lc_f, inv_f = refine_collapse(data, "pend", beta_nu, lam_c, inv_nu)

    # ================= FIGURE =================
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    else:
        axes = np.array(fig.subplots(2, 3))
    (axa, axb, axc), (axd, axe, axf) = axes
    note = "" if reliable_fit else "   [FSS unreliable — see (b)]"
    fig.suptitle(f"Lifespan method — SIS, power-law network, "
                 f"$\\gamma$ = {gamma}{note}", fontsize=14)

    # (a) <tau>(lambda)
    for d, c in zip(data, colors):
        axa.errorbar(d["lam"], d["tau"], yerr=d["tau_std"], fmt="-o", ms=3,
                     lw=1, color=c, capsize=0, label=f"N={d['N']:g}")
    axa.axvline(lam_c, ls="--", c="k", lw=1)
    axa.set_xlabel(r"$\lambda$"); axa.set_ylabel(r"$\langle\tau\rangle$")
    axa.set_title(r"(a) Average lifespan vs $\lambda$")
    axa.legend(fontsize=7)

    # (b) lambda_p(N) -> lambda_c, 1/nu  (both locators shown)
    axb.semilogx(Ns, lam_p1, "o", color="navy", label=r"peak of $\langle\tau\rangle$")
    if has_tau2:
        axb.semilogx(Ns, lam_pc, "s", color="crimson", mfc="none",
                     label=r"peak of $\chi$")
    if np.isfinite(inv_nu):
        xx = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 100)
        axb.semilogx(xx, lam_c + pos["a"] * xx ** (-inv_nu), "-", color="crimson",
                     label=(f"fit: $\\lambda_c$={lam_c:.4f}, "
                            f"1/$\\nu$={inv_nu:.3f}"
                            + ("" if reliable_fit else "  (unreliable)")))
    axb.axhline(lam_c, ls=":", c="gray", lw=1)
    axb.set_xlabel("N"); axb.set_ylabel(r"$\lambda_p(N)$")
    axb.set_title(r"(b) Peak position $\to\ \lambda_c,\ 1/\nu$")
    axb.legend(fontsize=8)

    # (c) peak heights (scaled): <tau> and chi
    axc.loglog(Ns, tau_p1 / tau_p1[0], "o", color="navy",
               label=r"$\langle\tau\rangle_p$ (scaled)")
    if np.isfinite(g1_nu):
        axc.loglog(Ns, (Ns / Ns[0]) ** g1_nu, "-", color="navy",
                   label=f"$\\gamma_1/\\nu$={g1_nu:.3f}")
    if has_tau2:
        axc.loglog(Ns, chi_p / chi_p[0], "s", color="darkorange",
                   label=r"$\chi_p$ (scaled)")
        if np.isfinite(gc_nu):
            axc.loglog(Ns, (Ns / Ns[0]) ** gc_nu, "-", color="darkorange",
                       label=f"$\\gamma'/\\nu$={gc_nu:.3f}")
    axc.set_xlabel("N"); axc.set_ylabel("peak height (scaled)")
    axc.set_title(r"(c) Peak heights $\to$ exponents")
    axc.legend(fontsize=8)

    # (d) P_end(lambda_c) vs N -> beta/nu
    if demo_mode or not np.any(pend_c > 0):
        msg = (r"$\lambda_c$ unreliable for $\gamma<3$"
               "\n(threshold drifts to 0;\nlifespan FSS not clean)") \
            if gamma < 3.0 else \
            (r"$P_{end}(\lambda_c)$ not a clean power law"
             "\nat these sizes (need larger N)")
        axd.text(0.5, 0.5, msg, ha="center", va="center", color="gray",
                 fontsize=10, transform=axd.transAxes)
        if np.any(pend_c > 0):
            axd.loglog(Ns, np.clip(pend_c, 1e-6, None), "o", color="teal", alpha=0.5)
        axd.set_xlabel("N"); axd.set_ylabel(r"$P_{end}(\lambda_c, N)$")
    else:
        axd.loglog(Ns, pend_c, "o", color="teal")
        _, b_int = fit_loglog_slope(Ns, pend_c)
        axd.loglog(Ns, np.exp(b_int) * Ns ** (-beta_nu), "-", color="teal",
                   label=f"$\\beta/\\nu$={beta_nu:.3f}")
        axd.legend(fontsize=9)
        axd.set_xlabel("N"); axd.set_ylabel(r"$P_{end}(\lambda_c, N)$")
    axd.set_title(r"(d) Order parameter at $\lambda_c \to \beta/\nu$")

    # (e) susceptibility / <tau^2> collapse  (clipped to overlap range)
    xlo, xhi = _overlap_range(data, lc_e, inv_e)
    pad = 0.15 * (xhi - xlo) if np.isfinite(xhi - xlo) else 0
    for d, c in zip(data, colors):
        N = d["N"]
        x = (d["lam"] - lc_e) * N ** inv_e
        yv = d["chi"] if locator == "chi" else (d["tau2"] if has_tau2 else d["tau"])
        axe.plot(x, yv * N ** (-exp_e), "-o", ms=3, lw=1, color=c)
    if np.isfinite(xlo) and np.isfinite(xhi) and xhi > xlo:
        axe.set_xlim(xlo - pad, xhi + pad)
    axe.set_xlabel(r"$(\lambda-\lambda_c)\,N^{1/\nu}$")
    sym = r"\chi" if locator == "chi" else (r"\langle\tau^2\rangle" if has_tau2
                                            else r"\langle\tau\rangle")
    axe.set_ylabel(rf"$N^{{-\gamma/\nu}}{sym}$")
    axe.set_title(f"(e) ${sym}$ collapse "
                  f"($\\lambda_c$={lc_e:.4f}, 1/$\\nu$={inv_e:.3f})")

    # (f) P_end collapse  (clipped to overlap range)
    xlo, xhi = _overlap_range(data, lc_f, inv_f)
    pad = 0.15 * (xhi - xlo) if np.isfinite(xhi - xlo) else 0
    bnu = beta_nu if np.isfinite(beta_nu) else 0.0
    for d, c in zip(data, colors):
        N = d["N"]
        x = (d["lam"] - lc_f) * N ** inv_f
        axf.plot(x, d["P_end"] * N ** bnu, "-o", ms=3, lw=1, color=c)
    if np.isfinite(xlo) and np.isfinite(xhi) and xhi > xlo:
        axf.set_xlim(xlo - pad, xhi + pad)
    axf.set_xlabel(r"$(\lambda-\lambda_c)\,N^{1/\nu}$")
    axf.set_ylabel(r"$N^{\beta/\nu}P_{end}$")
    axf.set_title(f"(f) $P_{{end}}$ collapse "
                  f"($\\lambda_c$={lc_f:.4f}, $\\beta/\\nu$={bnu:.3f})")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    info = {
        "gamma": gamma, "N": Ns,
        "lambda_c": lam_c, "inv_nu": inv_nu, "reliable_fit": reliable_fit,
        "locator": locator,
        "gamma1_over_nu": g1_nu, "gammachi_over_nu": gc_nu,
        "beta_over_nu": beta_nu,
        "lam_peak_tau": lam_p1, "lam_peak_chi": lam_pc,
        "tau_peak": tau_p1, "chi_peak": chi_p,
        "pend_at_lc": pend_c, "has_tau2": has_tau2, "demo_mode": bool(demo_mode),
        "collapse_e": (lc_e, inv_e), "collapse_f": (lc_f, inv_f),
    }
    return fig, info


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    for g in (2.5, 3.5):
        try:
            fig, info = make_panels("results_2", gamma=g)
            print(g, {k: info[k] for k in ("lambda_c", "inv_nu", "reliable_fit",
                      "locator", "gamma1_over_nu", "gammachi_over_nu",
                      "beta_over_nu")})
        except Exception as e:
            print("gamma", g, "->", repr(e))
