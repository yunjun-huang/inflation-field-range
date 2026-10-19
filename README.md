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
| `python figures/fig1_r_vs_dphi.py`, `python figures/fig2_ncrit.py` | Figures 1 and 2 |
| `python src/slowroll.py`, `pytest tests/` | slow-roll engine output, validation targets of Appendix A, regression tests |

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
- `data/` - validation log, N_crit grid, slow-roll critical points
- `crosscheck_emodel.py`, `crosscheck_hilltop.py` - independent slow-roll implementations (analytic derivatives) used as cross-checks
