"""Fig 2: the sub-Planckian point enters the ACT window at N_crit."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from slowroll import ns_at_critical_alpha, NS_ACT_LOW

N_CRIT = 61.37

Ns = np.linspace(50.0, 72.0, 45)
ns_vals = np.array([ns_at_critical_alpha(N) for N in Ns])

fig, ax = plt.subplots(figsize=(6.0, 4.5))

# 常用 e-folds 区间
ax.axvspan(50.0, 60.0, color='0.92', zorder=0)
ax.text(55.0, 0.9622, 'conventional\nrange', ha='center', va='bottom',
        fontsize=8, color='0.35')

ax.plot(Ns, ns_vals, color='k', lw=1.8,
        label=r'$n_s$ of the model with $\Delta\phi = M_{\rm Pl}$')
ax.axhline(NS_ACT_LOW, color='0.4', ls='--', lw=1.2)
ax.text(50.5, NS_ACT_LOW + 0.0004, r'P-ACT-LB  $2\sigma$ lower edge',
        fontsize=8, color='0.3')

ax.plot(N_CRIT, NS_ACT_LOW, 'o', ms=5, color='k', zorder=5)
ax.annotate(rf'$N_{{\rm crit}} = {N_CRIT}$',
            xy=(N_CRIT, NS_ACT_LOW), xytext=(64.0, 0.9640),
            fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3'))

ax.text(52.0, 0.9700, 'excluded', fontsize=9, color='0.3')
ax.text(67.0, 0.9660, 'allowed', fontsize=9, color='0.3')

ax.set_xlim(50.0, 72.0)
ax.set_ylim(0.9615, 0.9730)
ax.set_xlabel(r'number of e-folds  $N$')
ax.set_ylabel(r'$n_s$  at  $\Delta\phi = M_{\rm Pl}$')
ax.legend(frameon=False, loc='lower right', fontsize=8)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'fig2_ncrit.pdf')
fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=200)
print(f"saved: {out}")