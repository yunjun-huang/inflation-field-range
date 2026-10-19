"""Reproduce the manuscript numbers that exact_background.py does not print.

Run:  python src/reproduce_tables.py        (a few minutes)

Every block prints the computed values and compares them with the values
quoted in the manuscript (to the precision printed there). The last line is
either "ALL MATCH" or a list of the entries that differ.
All quantities: exact background (src/exact_background.py), first-order
pivot observables, N = e-folds remaining at the pivot.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import exact_background as eb

NS_EDGE = 0.9675          # P-ACT-LB 2-sigma lower edge: 0.9743 - 2*0.0034
bad = []


def chk(label, val, exp, tol):
    ok = abs(val - exp) <= tol
    if not ok:
        bad.append(f"{label}: computed {val:.6g}, manuscript {exp}")
    return "ok " if ok else "DIFF"


def rel(x):               # tolerance for a 4-significant-figure number
    return 5.1e-4 * abs(x)


# ----------------------------------------------------------------------
print("=== Table: E-model at N=60, scan of alpha ===")
print("  alpha   phi_end  phi_*    n_s      r            dphi")
PAPER_A = {0.01: (0.1564, 1.0973, 0.9661, 3.485e-5, 0.941),
           0.10: (0.3509, 2.5827, 0.9664, 3.413e-4, 2.232),
           1.00: (0.6146, 5.4236, 0.9674, 3.079e-3, 4.809),
           10.0: (0.8319, 9.3929, 0.9694, 1.990e-2, 8.561)}
for a, (pe, ps, ns, r, dp) in PAPER_A.items():
    d = eb.emodel_exact(a, 60.0)
    f = [chk(f"alpha={a} phi_end", d['phi_end'], pe, 5.1e-5),
         chk(f"alpha={a} phi_*", d['phi_star'], ps, 5.1e-5),
         chk(f"alpha={a} n_s", d['ns'], ns, 5.1e-5),
         chk(f"alpha={a} r", d['r'], r, rel(r)),
         chk(f"alpha={a} dphi", d['dphi'], dp, 5.1e-4)]
    print(f"  {a:5.2f}  {d['phi_end']:.4f}  {d['phi_star']:.4f}  {d['ns']:.4f}  "
          f"{d['r']:.3e}  {d['dphi']:.3f}   {' '.join(f)}")

# ----------------------------------------------------------------------
print("\n=== Table: n_s of the dphi = M_Pl E-model vs N ===")
PAPER_N = {55: 0.9630, 58: 0.9649, 60: 0.9661, 62: 0.9672,
           63: 0.9677, 65: 0.9687, 70: 0.9710}
for N, exp in PAPER_N.items():
    ns = eb.emodel_exact(eb.alpha_crit_exact(float(N)), float(N))['ns']
    print(f"  N={N}  n_s={ns:.4f}  (manuscript {exp:.4f})  "
          f"{chk(f'N={N} n_s', ns, exp, 5.1e-5)}   "
          f"{'inside' if ns >= NS_EDGE else 'outside'}")

# ----------------------------------------------------------------------
print("\n=== Table: swampland diagnostics at the dphi = M_Pl points ===")


def eta_E(phi, alpha):
    k = np.sqrt(2.0 / (3.0 * alpha)); u = np.exp(-k * phi)
    return 2.0 * k * k * u * (2.0 * u - 1.0) / (1.0 - u) ** 2


def grad_E(phi, alpha):               # |V'|/V
    k = np.sqrt(2.0 / (3.0 * alpha)); u = np.exp(-k * phi)
    return 2.0 * k * u / (1.0 - u)


def eta_H(phi, mu, p):
    x = phi / mu
    return -p * (p - 1) * x ** (p - 2) / (mu ** 2 * (1.0 - x ** p))


PAPER_S = {('E', 50): (0.00281, 0.02064, 0.9592), ('E', 60): (0.00226, 0.01713, 0.9661),
           ('E', 70): (0.00188, 0.01464, 0.9710),
           ('H', 50): (0.00188, 0.02409, 0.9525), ('H', 60): (0.00148, 0.02000, 0.9605),
           ('H', 70): (0.00121, 0.01709, 0.9662)}
for (m, N), (c0, e0, n0) in PAPER_S.items():
    if m == 'E':
        ac = eb.alpha_crit_exact(float(N)); d = eb.emodel_exact(ac, float(N))
        eta = eta_E(d['phi_star'], ac); grad = grad_E(d['phi_star'], ac); name = 'E-model'
    else:
        mc = eb.mu_crit_exact(8, float(N), 2.0119); d = eb.hilltop_exact(mc, 8, float(N))
        eta = eta_H(d['phi_star'], mc, 8)
        x = d['phi_star'] / mc; grad = 8 * x ** 7 / (mc * (1 - x ** 8)); name = 'hilltop p=8'
    c = np.sqrt(d['r'] / 8.0)
    f = [chk(f"{name} N={N} c", c, c0, 5.1e-6),
         chk(f"{name} N={N} |eta|", abs(eta), e0, 5.1e-6),
         chk(f"{name} N={N} n_s", d['ns'], n0, 5.1e-5)]
    print(f"  {name:11s} N={N}  c=sqrt(r/8)={c:.5f}  |eta_V|={abs(eta):.5f}  "
          f"n_s={d['ns']:.4f}   direct |V'|/V={grad:.6f}   {' '.join(f)}")

# ----------------------------------------------------------------------
print("\n=== Table: minimum excursion inside the n_s interval, hilltop, N=60 ===")
PAPER_M = {8: (21.10, 0.6947, 0.9839, 0.2892, 6.10),
           12: (24.87, 0.7654, 0.9869, 0.2215, 5.51),
           20: (32.14, 0.8378, 0.9904, 0.1526, 4.90)}
for p, (mu0, xs0, xe0, fr0, dp0) in PAPER_M.items():
    mu = brentq(lambda m: eb.hilltop_exact(m, p, 60.0)['ns'] - NS_EDGE,
                8.0, 120.0, xtol=1e-6)
    d = eb.hilltop_exact(mu, p, 60.0)
    xs, xe = d['phi_star'] / mu, d['phi_end'] / mu
    f = [chk(f"p={p} mu", mu, mu0, 5.1e-3), chk(f"p={p} x_*", xs, xs0, 5.1e-5),
         chk(f"p={p} x_end", xe, xe0, 5.1e-5), chk(f"p={p} dphi/mu", d['dphi'] / mu, fr0, 5.1e-5),
         chk(f"p={p} dphi", d['dphi'], dp0, 5.1e-3)]
    print(f"  p={p:2d}  mu={mu:6.2f}  x_*={xs:.4f}  x_end={xe:.4f}  dphi/mu={d['dphi']/mu:.4f}  "
          f"dphi={d['dphi']:.2f}  n_s={d['ns']:.4f}   {' '.join(f)}")


# ----------------------------------------------------------------------
print("\n=== Where is the excursion accumulated? (E-model, alpha_crit, N=60) ===")


def emodel_sol(alpha, N):
    phi0 = np.sqrt(3 * alpha / 2) * np.log(4 * (N + 15) / (3 * alpha))
    for _ in range(60):
        sol, Nend = eb.trajectory(eb.V_emodel, phi0, (alpha,))
        if Nend > N + 10.0:
            break
        phi0 *= 1.15
    return sol, Nend


ac = eb.alpha_crit_exact(60.0)
sol, Nend = emodel_sol(ac, 60.0)
speed = lambda t: abs(sol.sol(t)[1])
tot = quad(speed, Nend - 60.0, Nend, limit=400)[0]
PAPER_F = {3: 57, 5: 65}
for n in (1, 3, 5, 10):
    part = quad(speed, Nend - n, Nend, limit=400)[0]
    pc = 100 * part / tot
    tag = chk(f"last {n} e-folds %", pc, PAPER_F[n], 0.5) if n in PAPER_F else "   "
    print(f"  last {n:2d} e-folds: {pc:5.1f}% of dphi {tag}")
d1 = eb.emodel_exact(1.0, 60.0)
print(f"  alpha=1: 16*eps_H at pivot = {d1['r']:.3e}; growth to 16 = x{16/d1['r']:.0f}")

# ----------------------------------------------------------------------
print("\n=== Extra check (not in the table list): exact engine, hilltop small-mu limit ===")
print("  n_s(Lynker-Schimmrigk Eq.10) vs exact engine at mu = 0.1")
for p in (8, 12, 20):
    for N in (50.0, 60.0, 70.0):
        try:
            d = eb.hilltop_exact(0.1, p, N)
            form = 1.0 - 2.0 * (p - 1) / ((p - 2) * N)
            print(f"  p={p:2d} N={N:.0f}  exact n_s={d['ns']:.5f}  formula={form:.5f}  "
                  f"rel.diff={abs(d['ns']/form-1):.1e}")
        except Exception as e:
            print(f"  p={p:2d} N={N:.0f}  failed: {e}")

# ----------------------------------------------------------------------
print()
if bad:
    print("DIFFERENCES FOUND:")
    for b in bad:
        print("  -", b)
else:
    print("ALL MATCH: every number checked above agrees with the manuscript.")
