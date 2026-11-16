"""The unit-excursion (Delta phi = M_Pl) branch as a function of N, with
second-order observables and the quantities needed for the reheating map.

Run:  python src/branch.py            (~10-15 min on 2 cores; writes data/branch_grid.json)

For every model and every N on a grid, the critical parameter (alpha_crit or
mu_crit) is found with the exact-background engine of exact_background.py,
and on that trajectory we record
  eps1, eps2, eps3              Hubble-flow parameters at the pivot (eps2, eps3 analytic
                                from the equation of motion, no finite differences)
  ns1, r1                       first-order observables (as in the manuscript)
  ns2, r2                       second-order observables (Stewart-Lyth / Schwarz et al.)
  v_star, v_end                 V/V0 at the pivot and at eps_H = 1 (for the reheating map)

Second order:  n_s - 1 = -2e1 - e2 - 2e1^2 - (2C+3) e1 e2 - C e2 e3
               r = 16 e1 (1 + C e2),    C = gamma_E + ln 2 - 2
[D. J. Schwarz, C. A. Terrero-Escalante, A. A. Garcia, PLB 517 (2001) 243;
 J. Martin, C. Ringeval, V. Vennin, Phys. Dark Univ. 5-6 (2014) 75]
"""
import os, sys, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import brentq
import exact_background as eb

C_SL = np.euler_gamma + np.log(2.0) - 2.0
HERE = os.path.dirname(os.path.abspath(__file__))
GRID_FILE = os.path.join(HERE, "..", "data", "branch_grid.json")
N_GRID = list(range(44, 81, 2))
MODELS = ["E", "H8", "H12", "H20"]
P_OF = {"H8": 8, "H12": 12, "H20": 20}
MU_GUESS_60 = {"H8": 1.6547, "H12": 2.162, "H20": 3.2252}   # Table 4


# ---------------------------------------------------- second derivatives V''/V0
def d2V_emodel(phi, alpha):
    k = np.sqrt(2.0 / (3.0 * alpha)); u = np.exp(-k * phi)
    return 2.0 * k * k * u * (2.0 * u - 1.0)

def d2V_hilltop(phi, mu, p):
    x = phi / mu
    return -p * (p - 1) * x ** (p - 2) / mu ** 2


# ---------------------------------------------------- trajectories (same starts as eb)
def emodel_sol(alpha, N):
    phi0 = np.sqrt(3 * alpha / 2) * np.log(4 * (N + 15) / (3 * alpha))
    for _ in range(60):
        sol, Nend = eb.trajectory(eb.V_emodel, phi0, (alpha,))
        if Nend > N + 10.0:
            return sol, Nend
        phi0 *= 1.15
    raise RuntimeError("E-model: not enough e-folds")

def hilltop_sol(mu, p, N):
    x0 = eb._sr_hilltop_start(mu, p, N + 15.0)
    return eb.trajectory(eb.V_hilltop, x0 * mu, (mu, p))


def flow(sol, Nstar, Vfun, d2V, args):
    """eps1, eps2, eps3 at Nstar, from the exact solution and the EOM."""
    phi, y = sol.sol(Nstar)                  # y = phi'
    V, dV = Vfun(phi, *args); V2 = d2V(phi, *args)
    L, L1 = dV / V, V2 / V - (dV / V) ** 2   # V'/V and its derivative
    e1 = 0.5 * y * y
    y2 = -(3.0 - e1) * (y + L)               # phi''
    e1p = y * y2
    y3 = e1p * (y + L) - (3.0 - e1) * (y2 + L1 * y)
    e2 = 2.0 * y2 / y
    e3 = (y3 / y2 - y2 / y)                  # d ln eps2 / dN
    return e1, e2, e3

def observables2(e1, e2, e3):
    ns1, r1 = 1 - 2 * e1 - e2, 16 * e1
    ns2 = ns1 - 2 * e1 ** 2 - (2 * C_SL + 3) * e1 * e2 - C_SL * e2 * e3
    r2 = 16 * e1 * (1 + C_SL * e2)
    return ns1, r1, ns2, r2


def point(model, par, N):
    """All branch quantities for model at parameter par and e-fold number N."""
    if model == "E":
        sol, Nend = emodel_sol(par, N); Vf, d2, args = eb.V_emodel, d2V_emodel, (par,)
    else:
        p = P_OF[model]
        sol, Nend = hilltop_sol(par, p, N); Vf, d2, args = eb.V_hilltop, d2V_hilltop, (par, p)
    Ns = Nend - N
    e1, e2, e3 = flow(sol, Ns, Vf, d2, args)
    ns1, r1, ns2, r2 = observables2(e1, e2, e3)
    phis, phie = sol.sol(Ns)[0], sol.sol(Nend)[0]
    return dict(model=model, N=N, par=par, eps1=e1, eps2=e2, eps3=e3,
                ns1=ns1, r1=r1, ns2=ns2, r2=r2,
                v_star=Vf(phis, *args)[0], v_end=Vf(phie, *args)[0],
                phi_star=phis, phi_end=phie, sol=sol, Nend=Nend)


def excursion(model, par, N):
    d = (eb.emodel_exact(par, N) if model == "E"
         else eb.hilltop_exact(par, P_OF[model], N))
    return d["dphi"]

def crit_par(model, N, guess):
    f = lambda q: excursion(model, q, N) - 1.0
    lo, hi = 0.93 * guess, 1.07 * guess
    while f(lo) > 0: lo *= 0.93
    while f(hi) < 0: hi *= 1.07
    return brentq(f, lo, hi, xtol=1e-10 * guess)


def _guess(model, N):
    if model == "E":                      # alpha_crit ~ 0.0117 (N/60)^-0.33
        return 0.011704 * (N / 60.0) ** -0.33
    return MU_GUESS_60[model] * (N / 60.0) ** -0.10

def branch_point(args):
    model, N = args
    q = crit_par(model, float(N), _guess(model, N))
    d = point(model, q, float(N))
    d.pop("sol"); d.pop("Nend")
    return {k: (float(v) if not isinstance(v, str) else v) for k, v in d.items()}


def load_grid():
    with open(GRID_FILE) as f:
        return json.load(f)

def spline(grid, model, key):
    from scipy.interpolate import CubicSpline
    pts = sorted((d["N"], d[key]) for d in grid if d["model"] == model)
    N, y = map(np.array, zip(*pts))
    return CubicSpline(N, y)

def n_crit(grid, model, edge, order=1):
    s = spline(grid, model, "ns1" if order == 1 else "ns2")
    Ns = [d["N"] for d in grid if d["model"] == model]
    return brentq(lambda x: s(x) - edge, min(Ns), max(Ns))


if __name__ == "__main__":
    jobs = [(m, N) for m in MODELS for N in N_GRID]
    with Pool(2) as pool:
        out = pool.map(branch_point, jobs, chunksize=1)
    with open(GRID_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print("written", os.path.normpath(GRID_FILE))
    for m in MODELS:
        print(m, " N_crit(0.9675): first order %.2f, second order %.2f"
              % (n_crit(out, m, 0.9675, 1), n_crit(out, m, 0.9675, 2)))
