import csv
import os

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

NS_ACT_LOW = 0.9743 - 2 * 0.0034      # P-ACT-LB 的 2sigma 下沿
NS_LO, NS_HI, R_MAX = 0.9675, 0.9811, 0.036

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
    result, err = quad(integrand, phi_e, phi, limit=400, epsabs=1e-8, epsrel=1e-6)
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

m = 1.0
V_quad = lambda phi: 0.5 * m**2 * phi**2

def V_emodel(alpha):
    """alpha-attractor E-model 势能"""
    b = np.sqrt(2.0 / (3.0 * alpha))
    return lambda phi: (1.0 - np.exp(-b * phi))**2

def V_hilltop(p, mu):
    """hilltop 势能"""
    return lambda phi: 1.0 - (phi / mu) ** p

def analyse_hilltop(p, mu, N_target, step_frac=0.02):
    """hilltop 的四步流程（场向外滚，phi_star < phi_end）"""
    V = V_hilltop(p, mu)

    # 1. phi_end: 解 eps=1，在 (0, mu) 内
    pe = brentq(lambda phi: epsilon(V, phi) - 1.0, 1e-12 * mu, mu * (1 - 1e-12))

    # 2. phi_star: 从 phi_end 向内（向山顶）收缩，直到 N 超过目标
    hi = pe * (1 - 1e-9)
    lo = hi
    for _ in range(4000):
        lo = lo * (1 - step_frac)
        if N_efolds(V, pe, lo) > N_target:
            break
    else:
        raise RuntimeError(f"未能框住 N={N_target} (p={p}, mu={mu})")
    ps = brentq(lambda phi: N_efolds(V, pe, phi) - N_target, lo, hi)

    e, h = epsilon(V, ps), eta(V, ps)
    return dict(phi_end=pe, phi_star=ps,
                ns=1.0 - 6.0 * e + 2.0 * h, r=16.0 * e,
                dphi=pe - ps, x_star=ps / mu, x_end=pe / mu,
                dphi_over_mu=(pe - ps) / mu)

def dphi_of_alpha(log_alpha, N_target):
    """给定 log10(alpha)，返回该模型的 Delta phi"""
    alpha = 10.0 ** log_alpha
    V = V_emodel(alpha)
    end_hi = 15.0 * np.sqrt(1.5 * alpha)
    return analyse(V, N_target, 1e-6, end_hi)['dphi']

def ns_at_critical_alpha(N_target):
    """给定 N，求 Delta phi = 1 的那个模型的 ns"""
    log_a = brentq(lambda la: dphi_of_alpha(la, N_target) - 1.0, -3.0, 1.0)
    alpha_c = 10.0 ** log_a
    V = V_emodel(alpha_c)
    return analyse(V, N_target, 1e-6, 15.0 * np.sqrt(1.5 * alpha_c))['ns']

def dphi_of_mu(log_mu, p, N_target):
    """给定 log10(mu)，返回 hilltop 的 Delta phi"""
    return analyse_hilltop(p, 10.0 ** log_mu, N_target)['dphi']

def ns_at_crit_mu(p, N_target):
    """给定 p 和 N，求 Delta phi = 1 的那个 hilltop 模型的 ns"""
    lm = brentq(lambda l: dphi_of_mu(l, p, N_target) - 1.0, -1.0, 2.0)
    return analyse_hilltop(p, 10.0 ** lm, N_target)['ns']

def swampland_c(res):
    """c = sqrt(2 eps) = sqrt(r/8)"""
    return np.sqrt(res['r'] / 8.0)

def lyth_bound_constant_tilt(ns, N_star):
    """常倾斜近似下的 Lyth bound 上限。

    Yang, Tao, Wang & Zhu (arXiv:2606.16711) Eq. (22)：
    在 r(N) = r_* exp(delta N) 的近似下，Delta chi <= M_Pl 要求
        r_* <= 2 delta^2 / (exp(delta N_*/2) - 1)^2,  delta = 1 - ns
    delta -> 0 时退化为 r_* <= 8/N_*^2。
    """
    delta = 1.0 - ns
    if delta == 0.0:
        return 8.0 / N_star**2
    return 2.0 * delta**2 / (np.exp(delta * N_star / 2.0) - 1.0)**2

def main():

    # --- 验证 1：m^2 phi^2 有闭式解 ---
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

    # --- 验证 2：alpha-attractor E-model（无闭式解）---
    print()
    print("alpha-attractor E-model, N = 60")
    print(f"{'alpha':>8} {'phi_end':>9} {'phi_star':>9} {'ns':>8} {'r':>11} {'dphi':>8}")
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        V = V_emodel(alpha)
        end_hi = 15.0 * np.sqrt(1.5 * alpha)
        res = analyse(V, 60.0, 1e-6, end_hi)
        print(f"{alpha:8.3f} {res['phi_end']:9.4f} {res['phi_star']:9.4f} "
              f"{res['ns']:8.4f} {res['r']:11.3e} {res['dphi']:8.3f}")

    # --- 临界点：Delta phi = 1 M_Pl ---
    print()
    print("E-model: Delta phi = 1 M_Pl 的临界点")
    print(f"{'N':>4} {'alpha_crit':>12} {'ns':>9} {'r':>12} {'dphi':>8}")
    for N_target in [50.0, 60.0, 70.0]:
        log_a = brentq(lambda la: dphi_of_alpha(la, N_target) - 1.0, -3.0, 1.0)
        alpha_c = 10.0 ** log_a
        V = V_emodel(alpha_c)
        res = analyse(V, N_target, 1e-6, 15.0 * np.sqrt(1.5 * alpha_c))
        print(f"{N_target:4.0f} {alpha_c:12.6f} {res['ns']:9.4f} {res['r']:12.4e} {res['dphi']:8.4f}")

    # --- 临界 e-folds：亚普朗克点何时进入 ACT 窗口 ---
    print()
    print(f"ACT 2sigma 下沿 ns = {NS_ACT_LOW:.4f}")
    print(f"{'N':>5} {'ns @ dphi=1':>12} {'判决':>8}")
    for N_target in [55.0, 58.0, 60.0, 62.0, 65.0, 70.0]:
        ns_c = ns_at_critical_alpha(N_target)
        verdict = "允许" if ns_c >= NS_ACT_LOW else "排除"
        print(f"{N_target:5.0f} {ns_c:12.4f} {verdict:>8}")

    N_crit = brentq(lambda N: ns_at_critical_alpha(N) - NS_ACT_LOW, 55.0, 70.0)
    print()
    print(f"临界 e-folds  N_crit = {N_crit:.2f}")

    # --- hilltop 小场极限验证 ---
    print()
    print("hilltop 小场极限验证：ns -> 1 - 2(p-1)/[(p-2)N]")
    print(f"{'p':>4} {'N':>4} {'mu':>6} {'ns_num':>9} {'ns_formula':>11} {'rel.err':>10}")
    for p in [8, 12, 20]:
        for N_t in [50.0, 60.0, 70.0]:
            res = analyse_hilltop(p, 0.1, N_t)
            formula = 1.0 - 2.0 * (p - 1) / ((p - 2) * N_t)
            print(f"{p:4} {N_t:4.0f} {0.1:6.2f} {res['ns']:9.5f} {formula:11.5f} "
                  f"{abs(res['ns']/formula - 1):10.2e}")

    # --- 旧 bug 诊断：ACT 窗口内最小场程点 ---
    print()
    print("旧 bug 诊断：ACT 窗口内最小场程点，看 dphi/mu")
    print(f"{'p':>4} {'mu':>8} {'x_star':>8} {'x_end':>8} {'dphi/mu':>9} {'dphi':>8} {'ns':>8} {'r':>10}")
    for p in [8, 12, 20]:
        best = None
        for mu in np.geomspace(1.0, 400.0, 200):
            try:
                s = analyse_hilltop(p, mu, 60.0)
            except Exception:
                continue
            if NS_LO <= s['ns'] <= NS_HI and s['r'] < R_MAX:
                if best is None or s['dphi'] < best['dphi']:
                    best = dict(s, mu=mu)
        if best:
            print(f"{p:4} {best['mu']:8.2f} {best['x_star']:8.4f} {best['x_end']:8.4f} "
                  f"{best['dphi_over_mu']:9.4f} {best['dphi']:8.2f} {best['ns']:8.4f} {best['r']:10.2e}")
        else:
            print(f"{p:4}    无解")

    # --- hilltop 临界点 ---
    print()
    print("hilltop: Delta phi = 1 M_Pl 的临界点")
    print(f"{'p':>4} {'N':>4} {'mu_crit':>9} {'ns':>9} {'r':>12} {'dphi':>8} {'verdict':>8}")
    for p in [8, 12, 20]:
        for N_t in [50.0, 60.0, 70.0]:
            try:
                lm = brentq(lambda l: dphi_of_mu(l, p, N_t) - 1.0, -1.0, 2.0)
            except Exception:
                print(f"{p:4} {N_t:4.0f}   root-finding failed")
                continue
            mu_c = 10.0 ** lm
            s = analyse_hilltop(p, mu_c, N_t)
            verdict = "allowed" if s['ns'] >= NS_ACT_LOW else "excluded"
            print(f"{p:4} {N_t:4.0f} {mu_c:9.3f} {s['ns']:9.4f} {s['r']:12.4e} "
                  f"{s['dphi']:8.4f} {verdict:>8}")

    # --- swampland 双支诊断 ---
    print()
    print("swampland 双支诊断（在 Delta phi = 1 的临界点上）")
    print(f"{'model':>18} {'N':>4} {'c=sqrt(r/8)':>12} {'|eta|':>9} {'ns':>8}")
    for N_t in [50.0, 60.0, 70.0]:
        la = brentq(lambda l: dphi_of_alpha(l, N_t) - 1.0, -3.0, 1.0)
        a_c = 10.0 ** la
        V = V_emodel(a_c)
        s = analyse(V, N_t, 1e-6, 15.0 * np.sqrt(1.5 * a_c))
        c = np.sqrt(s['r'] / 8.0)
        cp = abs(eta(V, s['phi_star']))
        print(f"{'E-model':>18} {N_t:4.0f} {c:12.5f} {cp:9.5f} {s['ns']:8.4f}")
    for N_t in [50.0, 60.0, 70.0]:
        lm = brentq(lambda l: dphi_of_mu(l, 8, N_t) - 1.0, -1.0, 2.0)
        mu_c = 10.0 ** lm
        s = analyse_hilltop(8, mu_c, N_t)
        Vh = V_hilltop(8, mu_c)
        c = np.sqrt(s['r'] / 8.0)
        cp = abs(eta(Vh, s['phi_star']))
        print(f"{'hilltop p=8':>18} {N_t:4.0f} {c:12.5f} {cp:9.5f} {s['ns']:8.4f}")

    # --- hilltop 的临界 e-folds ---
    print()
    print("hilltop 临界 e-folds")
    for p in [8, 12, 20]:
        try:
            Nc = brentq(lambda N: ns_at_crit_mu(p, N) - NS_ACT_LOW, 55.0, 95.0)
            print(f"  p={p:3}  N_crit = {Nc:.2f}")
        except Exception:
            print(f"  p={p:3}  在 [55, 95] 内无解")

    # --- 导出 CSV，供论文表格使用 ---
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    rows = []
    for N_t in [50.0, 60.0, 70.0]:
        la = brentq(lambda l: dphi_of_alpha(l, N_t) - 1.0, -3.0, 1.0)
        a_c = 10.0 ** la
        V = V_emodel(a_c)
        s = analyse(V, N_t, 1e-6, 15.0 * np.sqrt(1.5 * a_c))
        rows.append(['E-model', '', f"{a_c:.6f}", f"{N_t:.0f}", f"{s['ns']:.4f}",
                     f"{s['r']:.4e}", f"{np.sqrt(s['r']/8):.5f}",
                     f"{abs(eta(V, s['phi_star'])):.5f}",
                     'allowed' if s['ns'] >= NS_ACT_LOW else 'excluded'])
    for p in [8, 12, 20]:
        for N_t in [50.0, 60.0, 70.0]:
            lm = brentq(lambda l: dphi_of_mu(l, p, N_t) - 1.0, -1.0, 2.0)
            mu_c = 10.0 ** lm
            s = analyse_hilltop(p, mu_c, N_t)
            Vh = V_hilltop(p, mu_c)
            rows.append(['hilltop', f"{p}", f"{mu_c:.4f}", f"{N_t:.0f}",
                         f"{s['ns']:.4f}", f"{s['r']:.4e}",
                         f"{np.sqrt(s['r']/8):.5f}",
                         f"{abs(eta(Vh, s['phi_star'])):.5f}",
                         'allowed' if s['ns'] >= NS_ACT_LOW else 'excluded'])

    csv_path = os.path.join(out_dir, 'critical_points.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'p', 'param_crit', 'N', 'ns', 'r', 'c', 'c_prime', 'verdict'])
        w.writerows(rows)
    print()
    print(f"CSV 已导出: {csv_path}")


if __name__ == "__main__":
    main()