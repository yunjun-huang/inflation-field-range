import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

# ============ potentials: return V, dV/dphi ============
def V_emodel(phi, alpha):
    k = np.sqrt(2.0/(3.0*alpha)); u = np.exp(-k*phi)
    return (1.0-u)**2, 2.0*k*u*(1.0-u)

def V_hilltop(phi, mu, p):
    x = phi/mu
    return 1.0 - x**p, -p*x**(p-1)/mu

# ============ exact background ============
def trajectory(Vfun, phi_init, args, Nmax=400.0):
    """Integrate phi'' + (3-eps)(phi' + V_phi/V) = 0 from the slow-roll
    attractor until eps_H = 1."""
    V, dV = Vfun(phi_init, *args)
    dphi0 = -dV/V                      # slow-roll attractor initial condition

    def rhs(N, y):
        phi, dphi = y
        V, dV = Vfun(phi, *args)
        eps = 0.5*dphi*dphi
        return [dphi, -(3.0-eps)*(dphi + dV/V)]

    def stop(N, y):
        return 0.5*y[1]*y[1] - 1.0     # eps_H = 1
    stop.terminal = True; stop.direction = 1

    sol = solve_ivp(rhs, [0.0, Nmax], [phi_init, dphi0], events=stop,
                    rtol=1e-11, atol=1e-13, dense_output=True, max_step=0.05)
    if not sol.t_events[0].size:
        raise RuntimeError("inflation did not end")
    return sol, sol.t_events[0][0]

def observables(sol, N_at):
    """eps_1, eps_2 -> n_s, r at e-fold time N_at."""
    phi, dphi = sol.sol(N_at)
    e1 = 0.5*dphi*dphi
    h = 1e-5
    d2 = (sol.sol(N_at+h)[1] - sol.sol(N_at-h)[1])/(2*h)
    e2 = 2.0*d2/dphi                       # dln(eps_1)/dN
    return phi, 1.0 - 2.0*e1 - e2, 16.0*e1

def analyse_exact(Vfun, args, phi_init, N):
    sol, Nend = trajectory(Vfun, phi_init, args)
    if Nend < N + 1.0:
        raise RuntimeError(f"only {Nend:.1f} e-folds available, need {N}")
    Nstar = Nend - N
    phi_s, ns, r = observables(sol, Nstar)
    phi_e = sol.sol(Nend)[0]
    # exact excursion = integral of |dphi/dN|
    dphi = quad(lambda n: abs(sol.sol(n)[1]), Nstar, Nend, limit=400)[0]
    return dict(phi_star=phi_s, phi_end=phi_e, ns=ns, r=r, dphi=dphi, Nend=Nend)

# ============ E-model ============
def emodel_exact(alpha, N):
    # start on the attractor, far enough back to bank at least N+10 e-folds
    phi0 = np.sqrt(3*alpha/2)*np.log(4*(N+15)/(3*alpha))
    for _ in range(60):
        _, Nend = trajectory(V_emodel, phi0, (alpha,))
        if Nend > N + 10.0:
            break
        phi0 *= 1.15
    return analyse_exact(V_emodel, (alpha,), phi0, N)

def alpha_crit_exact(N):
    return brentq(lambda a: emodel_exact(a, N)['dphi']-1.0, 5e-3, 2.0, xtol=1e-10)

# ============ hilltop ============
def _sr_hilltop_start(mu, p, Ntarget):
    """Slow-roll estimate of x that is Ntarget e-folds from the end.
    Used only to pick a safe starting point for the exact integration."""
    eps = lambda x: 0.5*p*p*x**(2*p-2)/(mu*mu*(1.0-x**p)**2)
    xe = brentq(lambda x: eps(x)-1.0, 1e-12, 1.0-1e-12)
    Nof = lambda x: quad(lambda y: mu/np.sqrt(2.0*eps(y)), x, xe, limit=400)[0]
    x = xe
    while Nof(x) < Ntarget:
        x *= 0.97
        if x < 1e-16: break
    return x

def hilltop_exact(mu, p, N):
    x0 = _sr_hilltop_start(mu, p, N + 15.0)
    return analyse_exact(V_hilltop, (mu, p), x0*mu, N)

def mu_crit_exact(p, N, mu_sr):
    return brentq(lambda m: hilltop_exact(m, p, N)['dphi']-1.0,
                  0.4*mu_sr, 3.0*mu_sr, xtol=1e-9)


# =====================================================================
#  Validation + results.  Run:  python exact_background.py
# =====================================================================
if __name__ == "__main__":
    print("=== Validation (i): chaotic m^2 phi^2, exact vs slow roll ===")
    sol, Nend = trajectory(lambda p: (p*p, 2.0*p), 18.0, ())
    phi_s, ns, r = observables(sol, Nend - 60.0)
    print(f"  phi_end : slow roll sqrt(2)={np.sqrt(2):.4f}   exact={sol.sol(Nend)[0]:.4f}"
          f"   (literature ~1.0)")
    print(f"  at N=60 : ns  slow roll {1 - 8/(4*60+2):.4f}      exact={ns:.4f}")
    print(f"            r   slow roll {32/242:.4f}     exact={r:.4f}")

    SR_E = {50: (0.018956, 9.0252e-05), 60: (0.017612, 5.8325e-05),
            70: (0.016591, 4.0411e-05)}
    SR_H = {(8,50):(2.0566,4.4564e-05),(8,60):(2.0119,2.7563e-05),(8,70):(1.9770,1.8406e-05),
            (12,50):(2.6993,5.9942e-05),(12,60):(2.6270,3.7717e-05),(12,70):(2.5705,2.5556e-05),
            (20,50):(4.0430,7.3120e-05),(20,60):(3.9176,4.6520e-05),(20,70):(3.8195,3.1816e-05)}
    CT = {50: 1.626e-3, 60: 9.784e-4, 70: 6.211e-4}   # constant-tilt bound, Yang Eq.(22)

    print("\n=== E-model: critical point, slow roll vs exact ===")
    print("   N   alpha(SR)  alpha(ex)   r(SR)       r(ex)      ratio  ns(ex)  vs const-tilt")
    for N in (50, 60, 70):
        ac = alpha_crit_exact(N); d = emodel_exact(ac, N); a0, r0 = SR_E[N]
        print(f"  {N}  {a0:.6f}  {ac:.6f}  {r0:.4e}  {d['r']:.4e}  {d['r']/r0:.3f}"
              f"  {d['ns']:.4f}   {CT[N]/d['r']:5.1f}")

    print("\n=== Hilltop: critical point, slow roll vs exact ===")
    print("  p   N   mu(SR)   mu(ex)    r(SR)       r(ex)      ratio  ns(ex)  vs const-tilt")
    for p in (8, 12, 20):
        for N in (50, 60, 70):
            m0, r0 = SR_H[(p, N)]
            mc = mu_crit_exact(p, N, m0); d = hilltop_exact(mc, p, N)
            print(f"  {p:2d} {N}  {m0:7.4f}  {mc:7.4f}  {r0:.4e}  {d['r']:.4e}"
                  f"  {d['r']/r0:.3f}  {d['ns']:.4f}   {CT[N]/d['r']:5.1f}")

    print("\n=== N_crit (exact), 2-sigma edge ns = 0.9675 ===")
    f_E = lambda N: emodel_exact(alpha_crit_exact(N), N)['ns'] - 0.9675
    print(f"  E-model       N_crit = {brentq(f_E, 55, 90, xtol=1e-4):.2f}   (slow roll: 61.37)")
    for p, guess, old in ((8, 2.0119, 71.43), (12, 2.6270, 67.38), (20, 3.9176, 64.67)):
        f_H = lambda N: hilltop_exact(mu_crit_exact(p, N, guess), p, N)['ns'] - 0.9675
        print(f"  hilltop p={p:2d}   N_crit = {brentq(f_H, 55, 110, xtol=1e-3):.2f}"
              f"   (slow roll: {old})")
