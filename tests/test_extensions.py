"""Regression tests for the revision (v2.0): second-order observables,
the critical branch, the reheating map and the joint posterior summary.

Run:  pytest tests/test_extensions.py      (about 10 s; uses data/*.json)
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import branch as br
import reheating as rh

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')


def test_branch_point_matches_first_order_tables():
    d = br.branch_point(("E", 60))
    assert abs(d["par"] - 0.0117044) < 1e-6
    assert abs(d["ns1"] - 0.9661082649) < 2e-9
    assert abs(d["r1"] / 4.0760e-5 - 1) < 1e-4
    h = br.branch_point(("H8", 60))
    assert abs(h["par"] - 1.6547187814) < 1e-7
    assert abs(h["ns1"] - 0.9604787278) < 2e-9


def test_analytic_eps2_equals_finite_difference():
    import exact_background as eb
    a = 0.0117044
    d = br.point("E", a, 60.0)
    _, ns_fd, _ = eb.observables(d["sol"], d["Nend"] - 60.0)
    assert abs(ns_fd - d["ns1"]) < 1e-9


def test_second_order_shift():
    d = br.branch_point(("E", 60))
    assert abs((d["ns2"] - d["ns1"]) - 4.17e-4) < 1e-5
    assert abs(d["r2"] / d["r1"] - 1 + 0.0247) < 5e-4


def test_ncrit_grid_reproduces_tables_5_and_6():
    g = br.load_grid()
    old = {0.9675: (62.54, 72.84, 68.69, 65.93), 0.9633: (55.46, 64.57, 60.90, 58.45),
           0.9613: (52.62, 61.26, 57.78, 55.46), 0.9566: (46.98, 54.68, 51.58, 49.50)}
    for e, vals in old.items():
        for m, v in zip(br.MODELS, vals):
            assert abs(br.n_crit(g, m, e, 1) - v) < 0.006
    for m in br.MODELS:          # uniform second-order shift
        assert abs(br.n_crit(g, m, 0.9675, 2) - br.n_crit(g, m, 0.9675, 1) + 0.74) < 0.02


def test_reheating_map():
    g = br.load_grid()
    M = rh.Model(g, "E")
    Ni = M.N_inst()
    assert 54.3 < Ni < 54.7
    # w = 1/3 is equivalent to instantaneous reheating up to the g_th term
    assert abs(M.N_pred(Ni, 1/3, 1e10) - M.N_pred(Ni)) < 0.05
    # the adopted ACT edge needs a stiff phase; p = 8 cannot reach it
    assert 0.6 < M.w_min(br.n_crit(g, "E", 0.9675)) < 0.7
    assert rh.Model(g, "H8").w_min(br.n_crit(g, "H8", 0.9675)) is None


def test_joint_posterior_summary():
    jp = json.load(open(os.path.join(DATA, "joint_posterior.json")))
    act = jp["p-actlite-bk-l-b_r_camb"]
    assert abs(act["ns_mean"] - 0.9741) < 3e-4 and abs(act["ns_std"] - 0.0033) < 2e-4
    assert abs(act["ns95_r0"][0] - 0.9671) < 3e-4
    pl = jp["planck-bk-l-b_r_camb"]
    assert 0.9595 < pl["ns95_r0"][0] < 0.9625
