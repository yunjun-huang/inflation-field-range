"""Fig 1: field excursion vs tensor-to-scalar ratio, alpha-attractor E-model.

Trajectories are now obtained from the exact background equation of motion
(src/exact_background.py) rather than the slow-roll approximation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from exact_background import (emodel_exact, hilltop_exact,
                              alpha_crit_exact, mu_crit_exact)

N_TARGET = 60.0
SIGMA_R_NEXTGEN = 1e-3          # CMB-S4 / LiteBIRD 目标灵敏度
P_HILL = 8

# 扫 alpha，收集 (r, dphi)。精确积分较慢，点数比慢滚版本少
alphas = np.geomspace(3e-4, 30.0, 30)
rs, dphis = [], []
for a in alphas:
    try:
        res = emodel_exact(a, N_TARGET)
    except Exception:
        continue
    rs.append(res['r'])
    dphis.append(res['dphi'])

# hilltop，扫 mu（固定 p）
mus = np.geomspace(0.6, 25.0, 30)
rs_h, dphis_h = [], []
for m in mus:
    try:
        res = hilltop_exact(m, P_HILL, N_TARGET)
    except Exception:
        continue
    rs_h.append(res['r'])
    dphis_h.append(res['dphi'])

# 临界点由求根给出，不再写死
r_crit_E = emodel_exact(alpha_crit_exact(N_TARGET), N_TARGET)['r']
r_crit_H = hilltop_exact(mu_crit_exact(P_HILL, N_TARGET, 2.0119),
                         P_HILL, N_TARGET)['r']
print(f"r_crit: E-model {r_crit_E:.4e}, hilltop p={P_HILL} {r_crit_H:.4e}")

fig, ax = plt.subplots(figsize=(6.0, 4.5))

ax.loglog(rs, dphis, color='k', lw=1.8, label=r'$\alpha$-attractor (E-model)')
ax.loglog(rs_h, dphis_h, color='k', lw=1.5, ls='--',
          label=rf'hilltop ($p={P_HILL}$)')
ax.axhspan(1.0, 20.0, color='0.92', zorder=0)
ax.axhline(1.0, color='0.4', ls='--', lw=1.2)
ax.axvline(SIGMA_R_NEXTGEN, color='0.4', ls=':', lw=1.2)

ax.text(2e-6, 1.15, r'$\Delta\phi = M_{\rm Pl}$', fontsize=9, color='0.3')
ax.text(1.25e-3, 12.0, r'CMB-S4 / LiteBIRD  $\sigma(r)$',
        fontsize=8, color='0.3', rotation=90, va='top')

ax.plot(r_crit_E, 1.0, 'o', ms=5, color='k', zorder=5)
ax.plot(r_crit_H, 1.0, 's', ms=5, color='k', mfc='white', zorder=5)
ax.annotate(rf'$r_{{\rm crit}} = {r_crit_E*1e5:.1f} \times 10^{{-5}}$',
            xy=(r_crit_E, 1.0), xytext=(1.5e-6, 2.5),
            fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3'))

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