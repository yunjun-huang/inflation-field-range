# Field excursion in plateau and hilltop inflation under ACT DR6

Numerical study of the inflaton field excursion in single-field slow-roll
inflation, and the critical tensor-to-scalar ratio at which it crosses the
reduced Planck mass.

**Author:** Yunjun Huang (ORCID 0009-0007-6289-749X)

## Status

Work in progress.

## Method

A general slow-roll engine that accepts an arbitrary potential V(phi):

1. Solve eps_V = 1 for the end of inflation, phi_end
2. Integrate N(phi) and solve N(phi_star) = N for the pivot
3. Evaluate n_s = 1 - 6 eps_V + 2 eta_V and r = 16 eps_V at phi_star
4. Report Delta phi = phi_star - phi_end

Units: reduced Planck mass M_Pl = 1.
Derivatives are evaluated by central finite differences.

## Validation

| Test | Result |
|---|---|
| eps_V, eta_V vs m2phi2 closed form | rel. err <= 1.8e-5 |
| phi_end vs sqrt(2) | rel. err 8.9e-12 |
| phi_star vs sqrt(4N+2), N=60 | rel. err 1.5e-11 |
| Delta phi vs 10 sqrt(2), N=60 | 14.142136 |
| alpha-attractor E-model, alpha=1, N=60 | n_s=0.9678, r=2.96e-3, dphi=4.51 |

Full log: data/validation_log.md

## Layout

- src/ - slow-roll engine
- tests/ - validation tests
- figures/ - generated figures
- data/ - validation log and tabulated results

## Requirements

Python 3, numpy, scipy, matplotlib.
