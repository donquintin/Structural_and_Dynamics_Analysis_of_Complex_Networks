# === Figuras dobles (gamma=3.5 | gamma=2.5) para el .tex, guardadas en figures_4/ ===
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import lifespan_analysis as la

GAMMAS      = [3.5, 2.5]          # orden de columnas (izq -> der)
RESULTS_DIR = "results_4"
FIG_DIR     = "figures_4"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- analisis por gamma (una vez) ----------
def analyze(results_dir, gamma):
    data = la.load_all(results_dir, gamma)
    Ns   = np.array([d["N"] for d in data], float)
    pk_t = [la.robust_peak(d["lam"], d["tau"]) for d in data]
    pk_c = [la.robust_peak(d["lam"], d["chi"]) for d in data]
    lam_pt = np.array([p[0] for p in pk_t]); tau_p = np.array([p[1] for p in pk_t])
    lam_pc = np.array([p[0] for p in pk_c]); chi_p = np.array([p[1] for p in pk_c])
    ok_c   = np.array([p[2] for p in pk_c])
    fit  = la.fit_peak_position(Ns, lam_pc, weights=np.where(ok_c, 1.0, 0.25))
    lc, inv_nu = fit["lambda_c"], fit["inv_nu"]
    g1_nu, _ = la.fit_loglog_slope(Ns, tau_p)
    gc_nu, _ = la.fit_loglog_slope(Ns, chi_p)
    pend_pc  = np.array([np.interp(lp, d["lam"], d["P_end"]) for d, lp in zip(data, lam_pc)])
    b_slope, b_int = la.fit_loglog_slope(Ns, pend_pc)
    beta_nu  = -b_slope if np.isfinite(b_slope) else 0.0
    return dict(data=data, Ns=Ns, lam_pt=lam_pt, tau_p=tau_p, lam_pc=lam_pc, chi_p=chi_p,
                fit=fit, lc=lc, inv_nu=inv_nu, g1_nu=g1_nu, gc_nu=gc_nu,
                pend_pc=pend_pc, beta_nu=beta_nu, b_int=b_int)

def _colors(n): return plt.cm.viridis(np.linspace(0, 0.9, n))

# ---------- plotters (uno por tipo de figura) ----------
def p_tau(ax, A, g):
    for d, c in zip(A["data"], _colors(len(A["data"]))):
        ax.errorbar(d["lam"], d["tau"], yerr=d["tau_std"], fmt="-o", ms=3, lw=1,
                    color=c, capsize=0, label=f"N={d['N']:g}")
    ax.axvline(A["lc"], ls="--", c="k", lw=1)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$\langle\tau\rangle$")
    ax.set_title(rf"$\gamma={g}$"); ax.legend(fontsize=7)

def p_lamp(ax, A, g):
    Ns = A["Ns"]
    ax.semilogx(Ns, A["lam_pt"], "o", color="navy", label=r"peak $\langle\tau\rangle$")
    ax.semilogx(Ns, A["lam_pc"], "s", color="crimson", mfc="none", label=r"peak $\chi$")
    if np.isfinite(A["inv_nu"]):
        xx = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 100)
        ax.semilogx(xx, A["lc"] + A["fit"]["a"] * xx ** (-A["inv_nu"]), "-", color="crimson",
                    label=rf"$\lambda_c$={A['lc']:.4f}, $1/\nu$={A['inv_nu']:.3f}")
    ax.axhline(A["lc"], ls=":", c="gray", lw=1)
    ax.set_xlabel("N"); ax.set_ylabel(r"$\lambda_p(N)$")
    ax.set_title(rf"$\gamma={g}$"); ax.legend(fontsize=8)

def p_height(ax, A, g):
    Ns = A["Ns"]
    ax.loglog(Ns, A["tau_p"] / A["tau_p"][0], "o", color="navy", label=r"$\langle\tau\rangle_p$")
    if np.isfinite(A["g1_nu"]):
        ax.loglog(Ns, (Ns / Ns[0]) ** A["g1_nu"], "-", color="navy",
                  label=rf"$\gamma_1/\nu$={A['g1_nu']:.3f}")
    ax.loglog(Ns, A["chi_p"] / A["chi_p"][0], "s", color="darkorange", label=r"$\chi_p$")
    if np.isfinite(A["gc_nu"]):
        ax.loglog(Ns, (Ns / Ns[0]) ** A["gc_nu"], "-", color="darkorange",
                  label=rf"$\gamma'/\nu$={A['gc_nu']:.3f}")
    ax.set_xlabel("N"); ax.set_ylabel("peak height (scaled)")
    ax.set_title(rf"$\gamma={g}$"); ax.legend(fontsize=8)

def p_pend_lc(ax, A, g):
    Ns = A["Ns"]; pend = A["pend_pc"]; m = pend > 0
    reliable = (g >= 3.0) and np.all(pend > 0)
    ax.loglog(Ns[m], pend[m], "o", color="teal", ms=7)
    if m.sum() >= 2 and np.isfinite(A["beta_nu"]):
        xx = np.array([Ns[m].min(), Ns[m].max()])
        ax.loglog(xx, np.exp(A["b_int"]) * xx ** (-A["beta_nu"]), "-", color="teal",
                  label=rf"$\beta/\nu$={A['beta_nu']:.3f}" + ("" if reliable else " (no fiable)"))
    ax.set_xlabel("N"); ax.set_ylabel(r"$P_{end}(\lambda_p(N),N)$")
    ax.set_title(rf"$\gamma={g}$" + ("" if reliable else r" [$\gamma<3$]")); ax.legend(fontsize=9)

def p_pend_lam(ax, A, g):
    for d, c in zip(A["data"], _colors(len(A["data"]))):
        ax.plot(d["lam"], d["P_end"], "-o", ms=3, lw=1, color=c, label=f"N={d['N']:g}")
    ax.axvline(A["lc"], ls="--", c="k", lw=1, label=rf"$\lambda_c$={A['lc']:.4f}")
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$P_{end}(\lambda,N)$")
    ax.set_title(rf"$\gamma={g}$"); ax.legend(fontsize=7)

def p_chi_collapse(ax, A, g):
    lc, inv, expo = A["lc"], A["inv_nu"], A["gc_nu"]
    for d, c in zip(A["data"], _colors(len(A["data"]))):
        N = d["N"]
        ax.plot((d["lam"] - lc) * N ** inv, d["chi"] * N ** (-expo),
                "-o", ms=3, lw=1, color=c, label=f"N={N:g}")
    # limite derecho = final REAL de los datos (para no cortar los picos, que en
    # gamma=2.5 caen en el borde); recorte solo de la cola izquierda dispersa.
    xmax_data = max(((np.asarray(d["lam"]) - lc) * d["N"] ** inv).max() for d in A["data"])
    try:
        xlo, xhi = la._overlap_range(A["data"], lc, inv)
    except Exception:
        xlo = np.nan
    if np.isfinite(xlo) and xmax_data > xlo:
        span = xmax_data - xlo
        ax.set_xlim(xlo - 0.08 * span, xmax_data + 0.10 * span)
    ax.set_xlabel(r"$(\lambda-\lambda_c)N^{1/\nu}$"); ax.set_ylabel(r"$N^{-\gamma/\nu}\chi$")
    ax.set_title(rf"$\gamma={g}$ ($\lambda_c$={lc:.4f}, $1/\nu$={inv:.3f})"); ax.legend(fontsize=7)

def _pend_cost(bnu, data, lc, inv_nu, pmin=1e-3, ngrid=40):
    xs, ys = [], []
    for d in data:
        lam, pe, N = np.asarray(d["lam"]), np.asarray(d["P_end"]), d["N"]
        m = pe > pmin
        if m.sum() < 3: continue
        x = (lam[m] - lc) * N ** inv_nu; y = pe[m] * N ** bnu
        o = np.argsort(x); xs.append(x[o]); ys.append(y[o])
    if len(xs) < 2: return 1e9
    xmin = max(x[0] for x in xs); xmax = min(x[-1] for x in xs)
    if xmax <= xmin: return 1e9
    grid = np.linspace(xmin, xmax, ngrid)
    M = np.array([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    return float(np.mean(M.var(axis=0) / (np.abs(M.mean(axis=0)) + 1e-12) ** 2))

def p_pend_collapse(ax, A, g):
    lc, inv = A["lc"], A["inv_nu"]; PMIN = 1e-3
    nsig = sum((np.asarray(d["P_end"]) > PMIN).sum() >= 3 for d in A["data"])
    if nsig >= 2:
        r = minimize_scalar(_pend_cost, bounds=(0.0, 3.0), args=(A["data"], lc, inv),
                            method="bounded"); bnu = float(r.x)
    else:
        bnu = 0.0
    for d, c in zip(A["data"], _colors(len(A["data"]))):
        lam, pe, N = np.asarray(d["lam"]), np.asarray(d["P_end"]), d["N"]
        m = pe > PMIN
        if m.sum() < 1: continue
        ax.plot((lam[m] - lc) * N ** inv, pe[m] * N ** bnu, "-o", ms=3, lw=1, color=c, label=f"N={N:g}")
    ax.set_xlabel(r"$(\lambda-\lambda_c)N^{1/\nu}$"); ax.set_ylabel(r"$N^{\beta/\nu}P_{end}$")
    tag = "" if nsig >= 2 else " [poca senal]"
    ax.set_title(rf"$\gamma={g}$ ($\beta/\nu$={bnu:.3f}){tag}"); ax.legend(fontsize=7)

# ---------- driver: una figura doble por tipo ----------
FIGURES = [
    ("tau_vs_lambda",  p_tau,           "Average lifespan vs $\\lambda$"),
    ("lambda_p_fit",   p_lamp,          "Peak position $\\to\\ \\lambda_c,\\ 1/\\nu$"),
    ("peak_height",    p_height,        "Peak heights $\\to$ exponents"),
    ("Pend_at_lc",     p_pend_lc,       "Order parameter at pseudocritical $\\to\\ \\beta/\\nu$"),
    ("Pend_vs_lambda", p_pend_lam,      "Order parameter $P_{end}$ vs $\\lambda$"),
    ("chi_collapse",   p_chi_collapse,  "$\\chi$ data collapse"),
    ("Pend_collapse",  p_pend_collapse, "$P_{end}$ data collapse ($P_{end}>0$)"),
]

A = {g: analyze(RESULTS_DIR, g) for g in GAMMAS}
for name, fn, sup in FIGURES:
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(6.4 * len(GAMMAS), 4.8))
    for ax, g in zip(np.atleast_1d(axes), GAMMAS):
        fn(ax, A[g], g)
    fig.suptitle(sup, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("guardada:", out)
print("\nExponentes:")
for g in GAMMAS:
    a = A[g]
    print(f"  gamma={g}: lambda_c={a['lc']:.4f} 1/nu={a['inv_nu']:.3f} "
          f"gamma1/nu={a['g1_nu']:.3f} gammachi/nu={a['gc_nu']:.3f} beta/nu={a['beta_nu']:.3f}")
