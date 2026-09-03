import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

def eps_h(x, mu, p):
    return 0.5*p*p*x**(2*p-2)/(mu*mu*(1.0-x**p)**2)
def eta_h(x, mu, p):
    return -p*(p-1)*x**(p-2)/(mu*mu*(1.0-x**p))
def x_end(mu, p):
    return brentq(lambda x: eps_h(x, mu, p)-1.0, 1e-12, 1.0-1e-14, xtol=1e-15)
def N_h(x, xe, mu, p):
    g = lambda y: mu/np.sqrt(2.0*eps_h(y, mu, p))
    return quad(g, x, xe, limit=400, epsabs=1e-10, epsrel=1e-9)[0]
def x_star(N, mu, p):
    xe = x_end(mu, p); x = xe
    while N_h(x, xe, mu, p) < N:
        x *= 0.98
        if x < 1e-14: raise RuntimeError
    return brentq(lambda y: N_h(y, xe, mu, p)-N, x, xe, xtol=1e-14)
def analyse_h(mu, N, p):
    xe = x_end(mu, p); xs = x_star(N, mu, p)
    e, h = eps_h(xs, mu, p), eta_h(xs, mu, p)
    return dict(xs=xs, xe=xe, ns=1-6*e+2*h, r=16*e, dphi=mu*(xe-xs),
                eta_end=eta_h(xe, mu, p))
def mu_crit(N, p):
    return brentq(lambda m: analyse_h(m, N, p)['dphi']-1.0, 0.3, 30.0, xtol=1e-12)

print("=== Table 4 reproduction ===")
for p in (8, 12, 20):
    for N in (50, 60, 70):
        mc = mu_crit(N, p); d = analyse_h(mc, N, p)
        print(f" p={p:2d} N={N} mu_crit={mc:.4f} ns={d['ns']:.4f} r={d['r']:.4e}"
              f"   |eta_V| at phi_end = {abs(d['eta_end']):.2f}")

print("\n=== Hilltop: does |eta_V| reach 1 before eps_V does?  (p=8, N=60) ===")
mc = mu_crit(60, 8); xe = x_end(mc, 8)
for x in (0.5*xe, 0.7*xe, 0.9*xe, 0.99*xe, xe):
    print(f"  x/x_end={x/xe:5.2f}  eps_V={eps_h(x,mc,8):8.4f}  |eta_V|={abs(eta_h(x,mc,8)):8.4f}")

print("\n=== Hilltop with inflation ending at |eta_V|=1 instead of eps_V=1 ===")
def x_end_eta(mu, p):
    return brentq(lambda x: abs(eta_h(x, mu, p))-1.0, 1e-12, 1.0-1e-14, xtol=1e-15)
def analyse_h2(mu, N, p):
    xe = x_end_eta(mu, p); xes = xe
    def Nn(x): 
        return quad(lambda y: mu/np.sqrt(2.0*eps_h(y,mu,p)), x, xes, limit=400,
                    epsabs=1e-10, epsrel=1e-9)[0]
    x = xes
    while Nn(x) < N:
        x *= 0.98
        if x < 1e-14: raise RuntimeError
    xs = brentq(lambda y: Nn(y)-N, x, xes, xtol=1e-14)
    e, h = eps_h(xs,mu,p), eta_h(xs,mu,p)
    return dict(ns=1-6*e+2*h, r=16*e, dphi=mu*(xes-xs), xe=xes)
print("   p  N   eps_V=1 -> mu_c, r_c      |  |eta_V|=1 -> mu_c, r_c     shift")
for p in (8, 12, 20):
    N = 60
    m1 = mu_crit(N,p); d1 = analyse_h(m1,N,p)
    m2 = brentq(lambda m: analyse_h2(m,N,p)['dphi']-1.0, 0.3, 30.0, xtol=1e-10)
    d2 = analyse_h2(m2,N,p)
    print(f"  {p:2d} {N}   mu={m1:6.3f} r={d1['r']:.3e}  |  mu={m2:6.3f} r={d2['r']:.3e}"
          f"   r x{d2['r']/d1['r']:.2f}   ns {d1['ns']:.4f}->{d2['ns']:.4f}")
