"""Fig 2: the sub-Planckian point enters the ACT window at N_crit.

Trajectories are now obtained from the exact background equation of motion.
The shaded band marks the excluded region N < N_crit; the dotted line marks
the conventional upper end N = 60, which is a separate statement.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from exact_background import emodel_exact, alpha_crit_exact
from slowroll import NS_ACT_LOW

N_CONVENTIONAL = 60.0


def ns_at_critical_alpha_exact(N):
    """n_s of the dphi = M_Pl model, exact background."""
    return emodel_exact(alpha_crit_exact(N), N)['ns']


# 精确积分较慢：每个点都要解一次 ODE 求根，点数比慢滚版本少
Ns = np.arange(50.0, 72.1, 1.0)
ns_vals = np.array([ns_at_critical_alpha_exact(N) for N in Ns])

N_CRIT = brentq(lambda N: ns_at_critical_alpha_exact(N) - NS_ACT_LOW,
                55.0, 90.0, xtol=1e-4)
print(f"N_crit = {N_CRIT:.2f}")

fig, ax = plt.subplots(figsize=(6.0, 4.5))

# 排除区：N < N_crit
ax.axvspan(50.0, N_CRIT, color='0.92', zorder=0)

# 常规 e-folds 上限是另一件事，单独用点线标出
ax.axvline(N_CONVENTIONAL, color='0.55', ls=':', lw=1.0, zorder=1)
ax.text(N_CONVENTIONAL - 0.3, 0.9620, 'conventional\nupper end',
        ha='right', va='bottom', fontsize=8, color='0.35')

ax.plot(Ns, ns_vals, color='k', lw=1.8,
        label=r'$n_s$ of the model with $\Delta\phi = M_{\rm Pl}$')
ax.axhline(NS_ACT_LOW, color='0.4', ls='--', lw=1.2)
ax.text(50.5, NS_ACT_LOW + 0.0004, r'P-ACT-LB  $2\sigma$ lower edge',
        fontsize=8, color='0.3')

ax.plot(N_CRIT, NS_ACT_LOW, 'o', ms=5, color='k', zorder=5)
ax.annotate(rf'$N_{{\rm crit}} = {N_CRIT:.2f}$',
            xy=(N_CRIT, NS_ACT_LOW), xytext=(65.0, 0.9635),
            fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3'))

ax.text(52.0, 0.9700, 'excluded', fontsize=9, color='0.3')
ax.text(68.5, 0.9690, 'allowed', fontsize=9, color='0.3')

ax.set_xlim(50.0, 72.0)
ax.set_ylim(0.9580, 0.9730)
ax.set_xlabel(r'number of e-folds  $N$')
ax.set_ylabel(r'$n_s$  at  $\Delta\phi = M_{\rm Pl}$')
ax.legend(frameon=False, loc='lower right', fontsize=8)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'fig2_ncrit.pdf')
fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=200)
print(f"saved: {out}")