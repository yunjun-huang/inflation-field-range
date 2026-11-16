"""Fig. 3: n_s of the Delta phi = M_Pl model against N for all four families
(first order solid, second order dashed), with the data edges and the
range of N that reheating can deliver.  Replaces the earlier E-model-only
fig2_ncrit.  Needs data/branch_grid.json, data/reheating.json,
data/joint_posterior.json."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src')); sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
import branch as br
from style import setup, save, COL, COLORS, LABELS, GREY

D = os.path.join(HERE, '..', 'data')
grid = br.load_grid()
rh = json.load(open(os.path.join(D, 'reheating.json')))
jp = json.load(open(os.path.join(D, 'joint_posterior.json')))
E_ACT = 0.9675
E_PL = round(jp['planck-bk-l-b_r_camb']['ns95_r0'][0], 4)
N_inst = np.mean([rh[m]['N_inst'] for m in br.MODELS])
N_kin = np.min([rh[m]['N_kin_bbn'] for m in br.MODELS])
print(f"N_inst (mean over models) = {N_inst:.2f}, N(w=1, T_reh=4.7 MeV) >= {N_kin:.2f}")

setup()
fig, ax = plt.subplots(figsize=(COL, 2.9))
Ns = np.linspace(44, 80, 400)
ax.axvspan(N_inst, N_kin, color='#fbe5c8', lw=0, zorder=0)
ax.axvspan(N_kin, 80, color='0.88', lw=0, zorder=0)
ax.axvline(N_inst, color=GREY, lw=0.7)
for m in br.MODELS:
    ax.plot(Ns, br.spline(grid, m, 'ns1')(Ns), color=COLORS[m], label=LABELS[m])
    ax.plot(Ns, br.spline(grid, m, 'ns2')(Ns), color=COLORS[m], ls='--', lw=0.8)
ax.axhline(E_ACT, color='k', lw=0.8)
ax.axhline(E_PL, color='k', lw=0.8, ls='-.')
ax.text(44.6, E_ACT + 0.0004, 'P-ACT-LB 95% (lower edge)', fontsize=6)
ax.text(79.6, E_PL - 0.0004, 'Planck-LB+BK18 95% (lower edge)', fontsize=6, ha='right', va='top')
ax.text(N_inst - 0.5, 0.9452, 'instant\nreheating', fontsize=6, ha='right', color=GREY)
ax.text(0.5 * (N_inst + N_kin), 0.9452, r'needs $w>1/3$', fontsize=6, ha='center', color='0.3')
ax.text(0.5 * (N_kin + 80), 0.9452, 'beyond\n$w=1$', fontsize=6, ha='center', color='0.3')
ax.plot([], [], color='k', ls='--', lw=0.8, label='second order')
ax.set_xlim(44, 80); ax.set_ylim(0.944, 0.978)
ax.set_xlabel(r'e-folds $N$ between pivot crossing and the end of inflation')
ax.set_ylabel(r'$n_s$ of the $\Delta\phi=M_{\rm Pl}$ model')
ax.legend(frameon=False, loc='lower right', fontsize=6.2, bbox_to_anchor=(1.0, 0.085))
save(fig, os.path.join(HERE, 'fig3_branch_reheating.pdf'))
