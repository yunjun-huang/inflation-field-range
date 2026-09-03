"""
Regression tests for the exact background integrator.

Place next to exact_background.py and run:
    pytest test_exact_background.py -v

Test 12 is the external target: for V = m^2 phi^2 the slow-roll value of
phi_end is sqrt(2), while the exact background gives a value near 1.0.
Test 13 checks that the answer does not depend on where the integration is
started, which is what licenses starting from a slow-roll estimate.
Test 14 checks that the exact treatment tightens rather than relaxes the
bound, which is the direction claimed in the abstract.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from exact_background import *


# --- 12 --------------------------------------------------------------
def test_chaotic_exact_phi_end():
    """m^2 phi^2: exact phi_end near 1.0, well away from slow-roll sqrt(2)."""
    sol, Nend = trajectory(lambda p: (p * p, 2.0 * p), 18.0, ())
    phi_end = sol.sol(Nend)[0]
    assert 0.98 < phi_end < 1.04, phi_end
    assert abs(phi_end - np.sqrt(2)) > 0.3          # genuinely different


def test_chaotic_pivot_observables():
    """The two treatments agree at the pivot, where slow roll is good."""
    sol, Nend = trajectory(lambda p: (p * p, 2.0 * p), 18.0, ())
    _, ns, r = observables(sol, Nend - 60.0)
    assert abs(ns - (1 - 8 / 242)) < 5e-4           # slow roll: 0.96694
    assert abs(r - 32 / 242) / (32 / 242) < 0.03    # slow roll: 0.13223


# --- 13 --------------------------------------------------------------
def test_start_point_independence():
    """Moving the start 10 e-folds earlier must not move alpha_crit."""
    def alpha_crit_from(offset):
        def em(alpha, N):
            phi0 = np.sqrt(3 * alpha / 2) * np.log(4 * (N + offset) / (3 * alpha))
            for _ in range(60):
                _, Ne = trajectory(V_emodel, phi0, (alpha,))
                if Ne > N + 10.0:
                    break
                phi0 *= 1.15
            return analyse_exact(V_emodel, (alpha,), phi0, N)
        from scipy.optimize import brentq
        return brentq(lambda a: em(a, 60.0)["dphi"] - 1.0, 5e-3, 2.0, xtol=1e-10)

    assert abs(alpha_crit_from(15) - alpha_crit_from(25)) < 1e-8


# --- 14 --------------------------------------------------------------
@pytest.mark.parametrize("N,r_slowroll", [(50, 9.0252e-05),
                                          (60, 5.8325e-05),
                                          (70, 4.0411e-05)])
def test_exact_is_stricter_emodel(N, r_slowroll):
    """The exact critical r must sit below the slow-roll one, by 30-37%."""
    r_exact = emodel_exact(alpha_crit_exact(N), N)["r"]
    assert r_exact < r_slowroll
    assert 0.62 < r_exact / r_slowroll < 0.72


def test_published_critical_values():
    """Frozen values as they appear in Tables 3 and 4."""
    assert emodel_exact(alpha_crit_exact(60.0), 60.0)["r"] == pytest.approx(
        4.0760e-05, rel=1e-3)
    mu = mu_crit_exact(8, 60.0, 2.0119)
    assert hilltop_exact(mu, 8, 60.0)["r"] == pytest.approx(1.7513e-05, rel=1e-3)


def test_critical_excursion_is_unity():
    """By construction the critical models sit at exactly one Planck mass."""
    assert emodel_exact(alpha_crit_exact(60.0), 60.0)["dphi"] == pytest.approx(
        1.0, abs=1e-6)
