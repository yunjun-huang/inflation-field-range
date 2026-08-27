"""Fig 1: field excursion vs tensor-to-scalar ratio, alpha-attractor E-model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from slowroll import V_emodel, analyse

N_TARGET = 60.0
SIGMA_R_NEXTGEN = 1e-3          # CMB-S4 / LiteBIRD 目标灵敏度

# 扫 alpha，收集 (r, dphi)
alphas = np.geomspace(1e-3, 30.0, 60)
rs, dphis = [], []
for a in alphas:
    res = analyse(V_emodel(a), N_TARGET, 1e-6, 15.0 * np.sqrt(1.5 * a))
    rs.append(res['r'])
    dphis.append(res['dphi'])

fig, ax = plt.subplots(figsize=(6.0, 4.5))

ax.loglog(rs, dphis, color='k', lw=1.8, label=r'$\alpha$-attractor (E-model)')
ax.axhline(1.0, color='0.4', ls='--', lw=1.2)
ax.axvline(SIGMA_R_NEXTGEN, color='0.4', ls=':', lw=1.2)

ax.text(2e-6, 1.15, r'$\Delta\phi = M_{\rm Pl}$', fontsize=9, color='0.3')
ax.text(1.25e-3, 0.13, r'CMB-S4 / LiteBIRD  $\sigma(r)$',
        fontsize=9, color='0.3', rotation=90, va='bottom')

ax.set_xlim(1e-6, 1e-1)
ax.set_ylim(0.1, 20.0)
ax.set_xlabel(r'tensor-to-scalar ratio  $r$')
ax.set_ylabel(r'field excursion  $\Delta\phi \, / \, M_{\rm Pl}$')
ax.set_title(rf'$N = {N_TARGET:.0f}$', fontsize=10)
ax.legend(frameon=False, loc='upper left', fontsize=9)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'fig1_r_vs_dphi.pdf')
fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=200)
print(f"saved: {out}")