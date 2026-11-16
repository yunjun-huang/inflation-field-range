"""Fig. 4: joint (n_s, r) HPD regions of the public LCDM + r chains, with the
unit-excursion branches N = 50-70 overlaid. The regions are computed in
linear r (reflected KDE, src/joint_posterior.py) and only *displayed* on a
log r axis, so that the critical models (r ~ 1e-5) are visible.

Usage: python figures/fig4_joint_posterior.py ACT_CHAINDIR PLANCK_CHAINDIR"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src')); sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
import branch as br
import joint_posterior as jp
from style import setup, save, COL, COLORS, LABELS, GREY

act_dir, pl_dir = sys.argv[1], sys.argv[2]
grid = br.load_grid()
rh = json.load(open(os.path.join(HERE, '..', 'data', 'reheating.json')))
xs = np.linspace(0.950, 0.990, 241)
ys = np.geomspace(1e-6, 0.1, 90)
X, Y = np.meshgrid(xs, ys)

setup()
fig, ax = plt.subplots(figsize=(COL, 3.25))
for cd, col, fill in ((pl_dir, '0.45', False), (act_dir, '#3b6fb6', True)):
    hpd = jp.JointHPD(jp.load(cd))
    Z = hpd.density(X.ravel(), Y.ravel()).reshape(X.shape)
    l68, l95 = hpd.level(0.68), hpd.level(0.95)
    if fill:
        ax.contourf(X, Y, Z, levels=[l95, l68, Z.max() * 2], colors=['#c9d9f0', '#8fb0de'])
    if fill:
        ax.contour(X, Y, Z, levels=[l95, l68], colors=[col], linewidths=[0.7, 0.7])
    else:                                   # Planck: 95% boundary only
        ax.contour(X, Y, Z, levels=[l95], colors=[col], linewidths=[0.8], linestyles=['--'])
ax.plot([], [], color='#8fb0de', lw=5, label='P-ACT-LB + BK18 (68, 95%)')
ax.plot([], [], color='0.45', ls='--', lw=0.8, label='Planck-LB + BK18 (95%)')

Ns = np.linspace(50, 70, 200)
for m in br.MODELS:
    ns, r = br.spline(grid, m, 'ns1'), br.spline(grid, m, 'r1')
    ax.plot(ns(Ns), r(Ns), color=COLORS[m], lw=1.4, label=LABELS[m])
    for N, mk in ((50, 'o'), (60, 's'), (70, '^')):
        ax.plot(ns(N), r(N), mk, ms=3.2, color=COLORS[m], mfc='white', mew=0.8)
    Ni = rh[m]['N_inst']
    ax.plot(ns(Ni), r(Ni), '*', ms=6, color=COLORS[m], mec='k', mew=0.3)
ax.axhline(0.036, color=GREY, lw=0.6, ls=':')
ax.text(0.9505, 0.044, r'BK18 $r<0.036$', fontsize=6, color=GREY)
ax.set_yscale('log'); ax.set_xlim(0.950, 0.990); ax.set_ylim(1e-6, 0.1)
ax.set_xlabel(r'$n_s$'); ax.set_ylabel(r'$r$')
ax.legend(frameon=False, loc='lower center', fontsize=6, bbox_to_anchor=(0.5, 1.0), ncol=2,
          columnspacing=1.2, handlelength=1.8)
save(fig, os.path.join(HERE, 'fig4_joint_posterior.pdf'))
