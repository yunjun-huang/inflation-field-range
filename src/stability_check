"""Numerical stability check (Appendix B of the manuscript).

Re-runs one E-model and one hilltop critical point with tightened solver
settings. exact_background.py is NOT modified: its solve_ivp call is wrapped
so that rtol / atol / max_step can be overridden.

Run:  python src/stability_check.py
"""
import os, sys, functools
sys.path.insert(0, os.path.dirname(__file__))
import exact_background as eb

_orig_solve_ivp = eb.solve_ivp

def set_solver(rtol=None, atol=None, max_step=None):
    """Override solver settings used inside exact_background."""
    @functools.wraps(_orig_solve_ivp)
    def wrapped(*a, **kw):
        if rtol is not None: kw['rtol'] = rtol
        if atol is not None: kw['atol'] = atol
        if max_step is not None: kw['max_step'] = max_step
        return _orig_solve_ivp(*a, **kw)
    eb.solve_ivp = wrapped

SETTINGS = [
    ("baseline  (rtol=1e-11, atol=1e-13, max_step=0.05)",  dict()),
    ("tightened (rtol=1e-13, atol=1e-15, max_step=0.05)",  dict(rtol=1e-13, atol=1e-15)),
    ("fine step (rtol=1e-11, atol=1e-13, max_step=0.01)",  dict(max_step=0.01)),
]

N = 60.0
# critical parameters found ONCE with baseline settings, then held fixed,
# so that the table shows the change of dphi, ns, r at the same model.
alpha_c = eb.alpha_crit_exact(N)
mu_c = eb.mu_crit_exact(8, N, 2.0119)
print(f"alpha_crit(N=60) = {alpha_c:.10f}")
print(f"mu_crit(p=8,N=60) = {mu_c:.10f}\n")

rows = []
for label, kw in SETTINGS:
    eb.solve_ivp = _orig_solve_ivp
    set_solver(**kw)
    e = eb.emodel_exact(alpha_c, N)
    h = eb.hilltop_exact(mu_c, 8, N)
    rows.append((label, e, h))

for name, idx in (("E-model, N=60, alpha_crit", 1), ("Hilltop p=8, N=60, mu_crit", 2)):
    print(name)
    base = rows[0][idx]
    for row in rows:
        d = row[idx]
        print(f"  {row[0]}")
        print(f"     dphi={d['dphi']:.10f}  ns={d['ns']:.10f}  r={d['r']:.6e}"
              f"   |d(dphi)|={abs(d['dphi']-base['dphi']):.1e}"
              f"  |d(ns)|={abs(d['ns']-base['ns']):.1e}"
              f"  |d(r)/r|={abs(d['r']/base['r']-1):.1e}")
    print()

# eps_2 is a finite difference of the dense output (h = 1e-5): vary h too
print("Sensitivity of n_s to the finite-difference step h in eps_2:")
import numpy as np
eb.solve_ivp = _orig_solve_ivp
for h_fd in (1e-4, 1e-5, 1e-6):
    def obs(sol, N_at, h=h_fd):
        phi, dphi = sol.sol(N_at)
        e1 = 0.5*dphi*dphi
        d2 = (sol.sol(N_at+h)[1] - sol.sol(N_at-h)[1])/(2*h)
        return phi, 1.0 - 2.0*e1 - 2.0*d2/dphi, 16.0*e1
    orig = eb.observables; eb.observables = obs
    e = eb.emodel_exact(alpha_c, N); hh = eb.hilltop_exact(mu_c, 8, N)
    eb.observables = orig
    print(f"  h={h_fd:.0e}: E-model ns={e['ns']:.10f}   hilltop ns={hh['ns']:.10f}")