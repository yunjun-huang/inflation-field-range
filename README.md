# Field excursion in plateau and hilltop inflation under ACT DR6

Numerical study of the inflaton field excursion in single-field inflation, and
the critical tensor-to-scalar ratio at which it crosses the reduced Planck mass
(alpha-attractor E-model and hilltop potentials, P-ACT-LB spectral index).

**Author:** Yunjun Huang (ORCID 0009-0007-6289-749X)

Accompanies the manuscript "How small must r be for a sub-Planckian field
range? Plateau and hilltop inflation under ACT DR6".

## Two engines

| Module | Method | Used for |
|---|---|---|
| `src/exact_background.py` | **Exact background**: integrates phi'' + (3 - eps_H)(phi' + V_phi/V) = 0 in e-fold time from the slow-roll attractor; end of inflation at eps_H = 1; n_s = 1 - 2 eps_1 - eps_2 and r = 16 eps_1 (first order in the Hubble-flow parameters) | **All results of the manuscript** |
| `src/slowroll.py` | Original slow-roll engine: eps_V = 1 at the end, n_s = 1 - 6 eps_V + 2 eta_V, r = 16 eps_V; derivatives by central finite differences | Validation targets of Appendix A (small-mu limit, constant-tilt bound) and the slow-roll comparison (+43-58% in r_crit) |

Units: reduced Planck mass M_Pl = 1.

`data/critical_points.csv` is the output of the **slow-roll** engine
(`python src/slowroll.py`); the exact-background values quoted in the
manuscript are printed by `src/exact_background.py`.

## Reproducing the manuscript

Requirements: Python 3, numpy, scipy, matplotlib
(tested with numpy 2.4 / scipy 1.17 and numpy 2.5 / scipy 1.18).
Run all commands from the repository root.

| Command | What it reproduces |
|---|---|
| `python src/exact_background.py` | critical points (E-model, hilltop), constant-tilt ratios, N_crit for the adopted edge n_s = 0.9675, exact-vs-slow-roll comparison, validation target (v) |
| `python src/reproduce_tables.py` | E-model alpha scan, n_s(N) of the E-model critical model, swampland table, minimum-excursion table (hilltop), e-fold accumulation numbers. Prints `ALL MATCH` or the entries that differ from the manuscript |
| `python src/stability_check.py` | solver-tolerance / step-size check (Appendix B) |
| `python src/ncrit_edges.py` | N_crit for other n_s lower edges. Reads `data/ncrit_grid.json` (seconds); `--recompute` rebuilds the grid (slow) |
| `python src/branch.py` | unit-excursion branch for N = 44-80 (all four models), first- and second-order observables, V/V0 at pivot and end; writes `data/branch_grid.json` (~1 min on 2 cores) |
| `python src/joint_posterior.py ACT_DIR PLANCK_DIR` | joint (n_s, r) HPD regions from the public ACT DR6 LCDM+r chains (see below); writes `data/joint_posterior.json` (~2 min) |
| `python src/reheating.py` | reheating map: N for instantaneous reheating, required w and T_reh; writes `data/reheating.json` (seconds) |
| `python src/tables_v2.py` | prints the numbers of the second-order, N_crit and reheating tables; checks the earlier N_crit tables (`ALL MATCH`) |
| `python figures/fig1_r_vs_dphi.py` | Fig. 1 |
| `python figures/fig2_excursion_vs_efolds.py` | Fig. 2 (where the excursion is accumulated) and the accumulation fractions quoted in the text |
| `python figures/fig3_branch_reheating.py` | Fig. 3 (n_s on the critical branch, reheating ranges) |
| `python figures/fig4_joint_posterior.py ACT_DIR PLANCK_DIR` | Fig. 4 (joint posterior with the critical branches) |
| `python src/slowroll.py`, `pytest tests/` | slow-roll engine output, validation targets of Appendix A, regression tests (`tests/test_extensions.py` covers the v2.0 additions) |

Recommended order for a full rebuild: `branch.py` -> `joint_posterior.py` -> `reheating.py` -> `tables_v2.py` -> figures.

## Public chains (not included)

The joint-posterior analysis uses the ACT DR6.02 "R model" (LCDM + r) chains from LAMBDA,
<https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_r_get.html>:

- `p-actlite-bk-l-b_r_camb.tar.gz` - P-ACT + BK18 + CMB lensing + BAO (adopted)
- `planck-bk-l-b_r_camb.tar.gz` - Planck + BK18 + CMB lensing + BAO (comparison)

Unpack each and pass the directory (e.g. `.../p-actlite-bk-l-b_r_camb`) to the scripts.
The derived summary is committed in `data/joint_posterior.json`, so `reheating.py`,
`tables_v2.py` and Figs. 1-3 run without the chains.

## Second order and reheating (v2.0)

- Second-order observables use eps2 and eps3 obtained analytically from the
  equation of motion (needs V''), with n_s - 1 = -2e1 - e2 - 2e1^2 - (2C+3)e1e2 - Ce2e3,
  r = 16e1(1 + Ce2), C = gamma_E + ln 2 - 2.
- Reheating: N = 67 - ln(k*/a0H0) + 1/4 ln(V*^2/rho_end) + (1-3w)/(12(1+w)) ln(rho_th/rho_end)
  - 1/12 ln g_th, with V* fixed by A_s (first order) and rho_end = 3/2 V_end on the exact
  background; A_s and H0 are the posterior means of the adopted chain; T_reh >= 4.7 MeV.

## Numerical settings (exact background)

- `solve_ivp`, rtol = 1e-11, atol = 1e-13, max_step = 0.05 in N, dense output.
- End of inflation: terminal event eps_H - 1 = 0.
- eps_2 = d ln eps_1 / dN by a central difference (step 1e-5 in N) of phi'
  on the dense-output solution.
- E-model start: on the attractor, about 15 e-folds before the target N
  (moved further out in steps of 15% if fewer than N + 10 e-folds result).
- Hilltop start: the slow-roll e-fold integral from phi_end towards the top is
  used only to choose an initial field value (contracted in steps
  x -> 0.97 x); the exact equation is then always integrated forward.
- Stability: changing rtol/atol to 1e-13/1e-15 or max_step to 0.01 changes
  dphi, n_s, r by <= 2e-10 (Appendix B of the manuscript).
- Start-point independence: starting 15 or 25 e-folds before the target N
  gives the same alpha_crit and mu_crit (differences <= 1e-8).

## Layout

- `src/` - exact-background and slow-roll engines, stability, N_crit and table-reproduction scripts
- `tests/` - validation / regression tests
- `figures/` - scripts and generated figures
- `data/` - validation log, N_crit grid, slow-roll critical points, branch grid, joint-posterior and reheating summaries
- `crosscheck_emodel.py`, `crosscheck_hilltop.py` - independent slow-roll implementations (analytic derivatives) used as cross-checks
