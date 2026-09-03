import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# ---------- E-model ----------
def k_of(alpha):
    return np.sqrt(2.0/(3.0*alpha))

def V_e(phi, alpha):
    return (1.0 - np.exp(-k_of(alpha)*phi))**2

def eps_e(phi, alpha):
    k = k_of(alpha); u = np.exp(-k*phi)
    return 2.0*k*k*u*u/(1.0-u)**2

def eta_e(phi, alpha):
    # V''/V computed analytically
    k = k_of(alpha); u = np.exp(-k*phi)
    # V = (1-u)^2, dV/dphi = 2k u (1-u), d2V/dphi2 = 2k^2 u (2u-1)
    return 2.0*k*k*u*(2.0*u-1.0)/(1.0-u)**2

def phi_end_e(alpha, eps_target=1.0):
    f = lambda p: eps_e(p, alpha) - eps_target
    lo, hi = 1e-8, 1e-8
    hi = 1e-6
    while f(hi) > 0:
        hi *= 1.5
        if hi > 500: raise RuntimeError
    return brentq(f, 1e-12, hi, xtol=1e-14, rtol=1e-15)

def N_e(phi, phi_end, alpha):
    g = lambda p: 1.0/np.sqrt(2.0*eps_e(p, alpha))
    return quad(g, phi_end, phi, limit=400, epsabs=1e-10, epsrel=1e-9)[0]

def phi_star_e(N, alpha, phi_end):
    hi = phi_end + 0.5
    while N_e(hi, phi_end, alpha) < N:
        hi += 0.5
        if hi > 400: raise RuntimeError
    return brentq(lambda p: N_e(p, phi_end, alpha) - N, phi_end, hi, xtol=1e-12)

def analyse_e(alpha, N, eps_end=1.0):
    pe = phi_end_e(alpha, eps_end)
    ps = phi_star_e(N, alpha, pe)
    e, h = eps_e(ps, alpha), eta_e(ps, alpha)
    return dict(phi_end=pe, phi_star=ps, ns=1-6*e+2*h, r=16*e, dphi=ps-pe)

def alpha_crit(N, eps_end=1.0):
    f = lambda a: analyse_e(a, N, eps_end)['dphi'] - 1.0
    return brentq(f, 1e-4, 5.0, xtol=1e-12)

print("=== Table 1 reproduction (N=60) ===")
for a in (0.01, 0.10, 1.00, 10.0):
    d = analyse_e(a, 60)
    print(f" a={a:6.2f} phi_end={d['phi_end']:.4f} phi_*={d['phi_star']:.4f} "
          f"ns={d['ns']:.4f} r={d['r']:.4e} dphi={d['dphi']:.3f}")

print("\n=== Table 3 reproduction (E-model critical points) ===")
for N in (50, 60, 70):
    ac = alpha_crit(N)
    d = analyse_e(ac, N)
    print(f" N={N} a_crit={ac:.6f} ns={d['ns']:.4f} r={d['r']:.4e}  "
          f"| 12a/N^2={12*ac/N**2:.4e}  ratio={d['r']/(12*ac/N**2):.4f}")

print("\n=== Where is the excursion accumulated? (at alpha_crit, N=60) ===")
ac = alpha_crit(60); pe = phi_end_e(ac)
tot = analyse_e(ac, 60)['dphi']
for n in (1, 3, 5, 10, 20):
    p_n = phi_star_e(n, ac, pe)
    print(f"  last {n:2d} e-folds: dphi={p_n-pe:.4f}  = {100*(p_n-pe)/tot:5.1f}% of total")
print("  --- same diagnostic at alpha=1 (the example used in Sec 5.2) ---")
pe1 = phi_end_e(1.0); tot1 = analyse_e(1.0, 60)['dphi']
for n in (1, 3, 5, 10):
    p_n = phi_star_e(n, 1.0, pe1)
    print(f"  last {n:2d} e-folds: dphi={p_n-pe1:.4f}  = {100*(p_n-pe1)/tot1:5.1f}% of total")

print("\n=== Sensitivity to the end-of-inflation condition (N=60) ===")
print("  eps_end   alpha_crit    r_crit        ns        (dphi held = 1)")
for ee in (1.0, 0.5, 0.2, 0.1):
    ac2 = alpha_crit(60, ee); d2 = analyse_e(ac2, 60, ee)
    print(f"  {ee:5.2f}   {ac2:.6f}   {d2['r']:.4e}   {d2['ns']:.4f}")
print("  ... and dphi at FIXED alpha=0.017612, varying only where we stop:")
for ee in (1.0, 0.5, 0.2, 0.1):
    print(f"  eps_end={ee:4.2f} -> dphi={analyse_e(0.017612, 60, ee)['dphi']:.4f}")
