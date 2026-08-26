import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

# ================================================================
# 通用慢滚引擎
# ================================================================

def dV(V, phi, h=1e-5):
    """V 的一阶导数，中心差分"""
    return (V(phi + h) - V(phi - h)) / (2 * h)

def d2V(V, phi, h=1e-5):
    """V 的二阶导数，中心差分"""
    return (V(phi + h) - 2 * V(phi) + V(phi - h)) / (h ** 2)

def epsilon(V, phi):
    """epsilon_V = 0.5 * (V'/V)^2,  M_Pl = 1"""
    return 0.5 * (dV(V, phi) / V(phi)) ** 2

def eta(V, phi):
    """eta_V = V''/V,  M_Pl = 1"""
    return d2V(V, phi) / V(phi)

def phi_end(V, lo, hi):
    """解 epsilon_V(phi) = 1，返回暴胀结束时的 phi"""
    return brentq(lambda phi: epsilon(V, phi) - 1.0, lo, hi)

def N_efolds(V, phi, phi_e):
    """从 phi 滚到 phi_e 经历的 e-folds"""
    integrand = lambda p: 1.0 / np.sqrt(2.0 * epsilon(V, p))
    result, err = quad(integrand, phi_e, phi, limit=200)
    return result

def phi_star(V, phi_e, N_target, step=0.5, max_expand=200):
    """从 phi_e 出发按 step 扩张上界，直到 N 超过目标，再在该区间求根"""
    lo = phi_e * 1.001
    hi = lo + step
    for _ in range(max_expand):
        if N_efolds(V, hi, phi_e) > N_target:
            return brentq(lambda p: N_efolds(V, p, phi_e) - N_target, lo, hi)
        lo, hi = hi, hi + step
    raise RuntimeError(f"扩张 {max_expand} 次仍未框住 N={N_target}")

def analyse(V, N_target, end_lo, end_hi):
    """跑完整四步：phi_end -> phi_star -> 观测量"""
    pe = phi_end(V, end_lo, end_hi)
    ps = phi_star(V, pe, N_target)
    e = epsilon(V, ps)
    h = eta(V, ps)
    return dict(phi_end=pe, phi_star=ps,
                ns=1.0 - 6.0 * e + 2.0 * h, r=16.0 * e, dphi=ps - pe)


# ================================================================
# 验证 1：m^2 phi^2 有闭式解
# ================================================================

m = 1.0
V_quad = lambda phi: 0.5 * m**2 * phi**2

print(f"{'phi':>8} {'eps_num':>12} {'eps_exact':>12} {'rel.err':>10}")
for phi in [1.0, 2.0, 5.0, 10.0, 15.556]:
    e_num = epsilon(V_quad, phi)
    e_exact = 2.0 / phi**2
    print(f"{phi:8.3f} {e_num:12.6e} {e_exact:12.6e} {abs(e_num/e_exact - 1):10.2e}")

print()
print(f"{'phi':>8} {'eta_num':>12} {'eta_exact':>12} {'rel.err':>10}")
for phi in [1.0, 2.0, 5.0, 10.0, 15.556]:
    h_num = eta(V_quad, phi)
    h_exact = 2.0 / phi**2
    print(f"{phi:8.3f} {h_num:12.6e} {h_exact:12.6e} {abs(h_num/h_exact - 1):10.2e}")

print()
res_q = analyse(V_quad, 60.0, 0.1, 10.0)
print(f"phi_end  numeric = {res_q['phi_end']:.8f}   exact = {np.sqrt(2):.8f}")
print(f"phi_star numeric = {res_q['phi_star']:.6f}   exact = {np.sqrt(242):.6f}")
print(f"Delta phi        = {res_q['dphi']:.6f}   exact = {10*np.sqrt(2):.6f}")


# ================================================================
# 验证 2：alpha-attractor E-model（无闭式解）
# ================================================================

def V_emodel(alpha):
    """alpha-attractor E-model 势能"""
    b = np.sqrt(2.0 / (3.0 * alpha))
    return lambda phi: (1.0 - np.exp(-b * phi))**2

print()
print("alpha-attractor E-model, N = 60")
print(f"{'alpha':>8} {'phi_end':>9} {'phi_star':>9} {'ns':>8} {'r':>11} {'dphi':>8}")
for alpha in [0.01, 0.1, 1.0, 10.0]:
    V = V_emodel(alpha)
    end_hi = 15.0 * np.sqrt(1.5 * alpha)
    res = analyse(V, 60.0, 1e-6, end_hi)
    print(f"{alpha:8.3f} {res['phi_end']:9.4f} {res['phi_star']:9.4f} "
          f"{res['ns']:8.4f} {res['r']:11.3e} {res['dphi']:8.3f}")