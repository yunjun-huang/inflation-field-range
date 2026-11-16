"""Fig. 1: field excursion against r at N = 60, with the current bound and
next-generation sensitivity marked (exact background)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src')); sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
from exact_background import emodel_exact, hilltop_exact, alpha_crit_exact, mu_crit_exact
from style import setup, save, COL, COLORS, LABELS, GREY

N_TARGET = 60.0
R_BK18 = 0.036               # BK18 95% upper limit
SIGMA_LO, SIGMA_HI = 1e-3, 2e-3   # round benchmark / LiteBIRD target sigma(r) (68%)
P_HILL = 8

rs, dphis = [], []
for a in np.geomspace(3e-4, 30.0, 34):
    try:
        d = emodel_exact(a, N_TARGET); rs.append(d['r']); dphis.append(d['dphi'])
    except Exception:
        pass
rs_h, dphis_h = [], []
for m in np.geomspace(0.6, 25.0, 34):
    try:
        d = hilltop_exact(m, P_HILL, N_TARGET); rs_h.append(d['r']); dphis_h.append(d['dphi'])
    except Exception:
        pass
r_crit_E = emodel_exact(alpha_crit_exact(N_TARGET), N_TARGET)['r']
r_crit_H = hilltop_exact(mu_crit_exact(P_HILL, N_TARGET, 2.0119), P_HILL, N_TARGET)['r']
print(f"r_crit: E-model {r_crit_E:.4e}, hilltop p={P_HILL} {r_crit_H:.4e}")

setup()
fig, ax = plt.subplots(figsize=(COL, 2.75))
ax.axhspan(1.0, 30.0, color='0.93', zorder=0, lw=0)
ax.axvspan(R_BK18, 1.0, color='0.80', zorder=0, lw=0)
ax.axvspan(SIGMA_LO, SIGMA_HI, color='#f5d6a8', zorder=0, lw=0)
ax.axhline(1.0, color=GREY, ls='--', lw=0.8)
ax.loglog(rs, dphis, color=COLORS["E"], label=LABELS["E"])
ax.loglog(rs_h, dphis_h, color=COLORS["H8"], ls='--', label=LABELS["H8"])
ax.plot(r_crit_E, 1.0, 'o', ms=4, color=COLORS["E"], zorder=5)
ax.plot(r_crit_H, 1.0, 's', ms=4, mfc='white', color=COLORS["H8"], zorder=5)

ax.text(2.2e-4, 1.15, r'super-Planckian', fontsize=6.5, color=GREY)
ax.text(2.2e-4, 0.85, r'sub-Planckian', fontsize=6.5, color=GREY, va='top')
ax.text(0.046, 0.14, 'excluded\n(BK18)', fontsize=6, color='0.25', rotation=90, va='bottom')
ax.text(1.05e-3, 0.14, r'$\sigma(r)$ targets', fontsize=6, color='0.25', rotation=90, va='bottom')
ax.annotate(r'$r_{\rm crit}$', xy=(r_crit_E, 1.0), xytext=(9e-6, 0.35), fontsize=7,
            arrowprops=dict(arrowstyle='->', lw=0.6, color=GREY))
ax.annotate('', xy=(r_crit_H, 1.0), xytext=(1.05e-5, 0.42),
            arrowprops=dict(arrowstyle='->', lw=0.6, color=GREY))

ax.set_xlim(1e-6, 0.1); ax.set_ylim(0.1, 20.0)
ax.set_xlabel(r'tensor-to-scalar ratio $r$')
ax.set_ylabel(r'field excursion $\Delta\phi/M_{\rm Pl}$')
ax.legend(frameon=False, loc='upper left')
save(fig, os.path.join(HERE, 'fig1_r_vs_dphi.pdf'))
