"""Fig. 2: where the excursion is accumulated, at the N = 60 critical points.
(a) |phi(n) - phi_end| against e-folds n before the end: exact background,
    slow roll at the same parameter (end at eps_V = 1), constant-tilt ansatz
    with the exact pivot r_* and delta = 1 - n_s.
(b) 16 eps_H = 8 phi'^2 along the exact trajectories vs r_* e^{delta(N-n)}.
Also prints the accumulation fractions quoted in the text."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src')); sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import exact_background as eb
import branch as br
from style import setup, save, FULL, COLORS, LABELS, GREY

N = 60.0
n = np.linspace(0.0, N, 1201)


def curves(model, par, Vf, args):
    d = br.point(model, par, N)
    sol, Nend = d['sol'], d['Nend']
    phi = sol.sol(Nend - n)[0]; dph = np.abs(sol.sol(Nend - n)[1])
    D = np.abs(phi - sol.sol(Nend)[0])
    # slow roll at the same parameter: end at eps_V = 1, dphi/dn = +V'/V
    L = lambda f: Vf(f, *args)[1] / Vf(f, *args)[0]
    lo, hi = (1e-4, 5.0) if model == "E" else (0.3 * args[0], 0.999999 * args[0])
    fe = brentq(lambda f: 0.5 * L(f) ** 2 - 1.0, lo, hi)
    s = solve_ivp(lambda t, y: [L(y[0])], (0, N), [fe], t_eval=n, rtol=1e-11, atol=1e-13)
    Dsr = np.abs(s.y[0] - fe)
    # constant tilt with matched pivot r_* and delta
    rs, dl = d['r1'], 1.0 - d['ns1']
    Dct = np.sqrt(rs / 8) * (2 / dl) * (np.exp(dl * N / 2) - np.exp(dl * (N - n) / 2))
    return D, Dsr, Dct, 8 * dph ** 2, rs * np.exp(dl * (N - n)), d


a_c = eb.alpha_crit_exact(N)
mu_c = eb.mu_crit_exact(8, N, 2.0119)
res = {"E": curves("E", a_c, eb.V_emodel, (a_c,)),
       "H8": curves("H8", mu_c, eb.V_hilltop, (mu_c, 8))}

for m, (D, Dsr, Dct, *_rest) in res.items():
    f = [np.interp(k, n, D) / D[-1] for k in (3, 5, 10)]
    print(f"{m}: Dphi={D[-1]:.6f}; last 3/5/10 e-folds: {f[0]:.1%} / {f[1]:.1%} / {f[2]:.1%};"
          f"  slow roll Dphi={Dsr[-1]:.4f};  constant tilt Dphi={Dct[-1]:.4f}"
          f"  -> (1/Dct)^2 = {1/Dct[-1]**2:.1f}")

setup()
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.6))
for m in ("E", "H8"):
    D, Dsr, Dct, h, hct, _ = res[m]; c = COLORS[m]
    ax[0].plot(n, D, color=c, label=("E-model" if m == "E" else r"hilltop $p=8$") + ": exact")
    ax[0].plot(n, Dsr, color=c, ls='--', lw=1.0, label="slow roll")
    ax[0].plot(n, Dct, color=c, ls=':', lw=1.0, label="constant tilt")
    ax[1].semilogy(n, h, color=c, label=LABELS[m] + r": $16\epsilon_H$")
    ax[1].semilogy(n, hct, color=c, ls=':', lw=1.0)
ax[0].axhline(1.0, color=GREY, ls='-.', lw=0.6)
ax[0].axvspan(0, 5, color='0.92', zorder=0, lw=0)
ax[0].text(2.5, 0.95, "last 5\ne-folds", ha='center', va='top', fontsize=6, color=GREY)
ax[0].set_xlim(0, N); ax[0].set_ylim(0, 1.06)
ax[0].set_xlabel(r"e-folds before the end of inflation, $n$")
ax[0].set_ylabel(r"$|\phi(n)-\phi_{\rm end}|\,/\,M_{\rm Pl}$")
ax[0].legend(frameon=False, ncol=2, loc='center right', bbox_to_anchor=(1.0, 0.43),
             columnspacing=0.8, handlelength=1.7, fontsize=6.2)
ax[0].set_title("(a)", loc='left', fontsize=8.5)
ax[1].axhline(16, color=GREY, ls='-.', lw=0.6)
ax[1].text(57, 21, r"$\epsilon_H=1$", fontsize=7, color=GREY, ha='right')
ax[1].plot([], [], color='k', ls=':', lw=1.0, label=r"constant tilt $r_*e^{\delta(N-n)}$")
ax[1].set_xlim(0, N); ax[1].set_ylim(5e-6, 60)
ax[1].set_xlabel(r"e-folds before the end of inflation, $n$")
ax[1].set_ylabel(r"$16\,\epsilon_H = 8\,(d\phi/dN)^2$")
ax[1].legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 0.86))
ax[1].set_title("(b)", loc='left', fontsize=8.5)
fig.tight_layout()
save(fig, os.path.join(HERE, 'fig2_excursion_vs_efolds.pdf'))
