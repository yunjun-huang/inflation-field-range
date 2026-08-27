"""三条验证的固化版本。

对应 data/validation_log.md 里的三个靶标：
  1. m^2 phi^2 的闭式解          —— 覆盖单调大场分支
  2. alpha-attractor 文献标准值  —— 覆盖无闭式解的 plateau 分支
  3. Lynker & Schimmrigk Eq.(10) —— 覆盖 hilltop 分支

运行方式（在仓库根目录）：
    pytest -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from slowroll import (V_quad, V_emodel, analyse, analyse_hilltop,
                      epsilon, eta, dphi_of_alpha, dphi_of_mu)
from scipy.optimize import brentq


# ================================================================
# 靶标 1：m^2 phi^2 闭式解
#   eps = eta = 2/phi^2,  phi_end = sqrt(2),  phi_* = sqrt(4N+2)
# ================================================================

def test_quadratic_slow_roll_parameters():
    """eps 与 eta 的数值值对闭式解"""
    for phi in [1.0, 2.0, 5.0, 10.0, 15.556]:
        exact = 2.0 / phi**2
        assert abs(epsilon(V_quad, phi) / exact - 1) < 1e-8
        assert abs(eta(V_quad, phi) / exact - 1) < 1e-4   # 二阶差分精度较低


def test_quadratic_trajectory():
    """phi_end、phi_* 与 Delta phi 对闭式解"""
    res = analyse(V_quad, 60.0, 0.1, 10.0)
    assert abs(res['phi_end'] / np.sqrt(2) - 1) < 1e-8
    assert abs(res['phi_star'] / np.sqrt(4 * 60 + 2) - 1) < 1e-8
    assert abs(res['dphi'] / (10 * np.sqrt(2)) - 1) < 1e-8


# ================================================================
# 靶标 2：alpha-attractor E-model 的文献标准值
#   alpha = 1, N = 60:  ns = 0.9678, r = 2.96e-3, dphi = 4.51
# ================================================================

def test_emodel_reference_values():
    """alpha=1, N=60 对文献值，四位有效数字"""
    V = V_emodel(1.0)
    res = analyse(V, 60.0, 1e-6, 15.0 * np.sqrt(1.5))
    assert abs(res['ns'] - 0.9678) < 5e-5
    assert abs(res['r'] / 2.964e-3 - 1) < 1e-3
    assert abs(res['dphi'] - 4.513) < 5e-3


# ================================================================
# 靶标 3：Lynker & Schimmrigk (arXiv:2507.15076) Eq. (10)
#   小场极限 ns -> 1 - 2(p-1)/[(p-2)N]
#   这是唯一覆盖 hilltop 分支的验证
# ================================================================

def test_hilltop_small_field_limit():
    """九个 (p, N) 点对解析公式，相对误差 < 1e-5"""
    for p in [8, 12, 20]:
        for N in [50.0, 60.0, 70.0]:
            res = analyse_hilltop(p, 0.1, N)
            formula = 1.0 - 2.0 * (p - 1) / ((p - 2) * N)
            assert abs(res['ns'] / formula - 1) < 1e-5, f"p={p}, N={N}"


# ================================================================
# 回归测试：论文里用到的三个临界点
#   数值来自 data/critical_points.csv，改动引擎后这几条会立刻报警
# ================================================================

def test_emodel_critical_point_N60():
    """E-model, N=60: alpha_crit = 0.017612, ns = 0.9668, r = 5.8325e-05"""
    log_a = brentq(lambda la: dphi_of_alpha(la, 60.0) - 1.0, -3.0, 1.0)
    alpha_c = 10.0 ** log_a
    res = analyse(V_emodel(alpha_c), 60.0, 1e-6, 15.0 * np.sqrt(1.5 * alpha_c))
    assert abs(alpha_c - 0.017612) < 1e-5
    assert abs(res['ns'] - 0.9668) < 5e-5
    assert abs(res['r'] / 5.8325e-05 - 1) < 1e-3
    assert abs(res['dphi'] - 1.0) < 1e-6          # 求根的自洽检验


def test_hilltop_critical_point_p8_N60():
    """hilltop p=8, N=60: mu_crit = 2.012, ns = 0.9614, r = 2.7563e-05"""
    lm = brentq(lambda l: dphi_of_mu(l, 8, 60.0) - 1.0, -1.0, 2.0)
    mu_c = 10.0 ** lm
    res = analyse_hilltop(8, mu_c, 60.0)
    assert abs(mu_c - 2.012) < 1e-3
    assert abs(res['ns'] - 0.9614) < 5e-5
    assert abs(res['r'] / 2.7563e-05 - 1) < 1e-3
    assert abs(res['dphi'] - 1.0) < 1e-6


def test_mu_crit_is_super_planckian():
    """mu_crit > 1 而 dphi = 1：论文正文点出的那个观察，锁住它"""
    for p in [8, 12, 20]:
        lm = brentq(lambda l: dphi_of_mu(l, p, 60.0) - 1.0, -1.0, 2.0)
        assert 10.0 ** lm > 1.0, f"p={p} 的 mu_crit 不再是超普朗克"