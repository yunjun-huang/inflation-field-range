"""Reheating map for the unit-excursion branch: which post-inflationary
histories deliver the N required for consistency with the data?

Run:  python src/reheating.py     (seconds; needs data/branch_grid.json from branch.py)

Matching equation (Liddle & Leach 2003; Planck 2018 X), reduced Planck units:
  N = 67 - ln(k_*/a0H0) + 1/4 ln(V_*^2/rho_end)
        + (1-3w)/(12(1+w)) ln(rho_th/rho_end) - 1/12 ln g_th
with k_* = 0.05 Mpc^-1, w the e-fold-averaged equation of state between the
end of inflation and thermalisation, rho_th = (pi^2/30) g_th T_reh^4.
The scale of inflation is fixed by the scalar amplitude, at first order
  A_s = H_*^2 / (8 pi^2 eps1),   V_* = H_*^2 (3 - eps1),
and on the exact background rho_end = 3 H_end^2 = (3/2) V_end.
A_s and H0 are the posterior means of the adopted P-ACT-LB + BK18 chain.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import brentq
import branch as br

MPL_GEV = 2.435e18
A_S = 2.1324e-9            # <A_s>, p-actlite-bk-l-b_r_camb (data/joint_posterior.json)
H0 = 68.19                 # <H0>,  same chain [km/s/Mpc]
K_STAR = 0.05              # Mpc^-1
C_KMS = 299792.458
LN_K = np.log(K_STAR / (H0 / C_KMS))
T_MIN_GEV = 4.7e-3         # lower bound on T_reh (de Salas et al. 2015)
G_HIGH = 106.75

# standard-model g_*(T), coarse table (T in GeV), log-interpolated
_GT = np.array([[1e-3, 10.75], [0.1, 10.75], [0.12, 14.25], [0.2, 17.25],
                [0.25, 61.75], [1.5, 75.75], [4.0, 86.25], [100., 96.25],
                [200., 106.75], [1e20, 106.75]])
def g_star(T_gev):
    return float(np.interp(np.log(T_gev), np.log(_GT[:, 0]), _GT[:, 1]))


class Model:
    def __init__(self, grid, model):
        self.name = model
        self.e1 = br.spline(grid, model, "eps1")
        self.vs = br.spline(grid, model, "v_star")
        self.ve = br.spline(grid, model, "v_end")
        self.ns1 = br.spline(grid, model, "ns1")
        self.ns2 = br.spline(grid, model, "ns2")
        self.r1 = br.spline(grid, model, "r1")

    def scales(self, N):
        e1 = self.e1(N)
        Vstar = 8 * np.pi ** 2 * e1 * A_S * (3 - e1)
        rho_end = 1.5 * Vstar * self.ve(N) / self.vs(N)
        return Vstar, rho_end

    def N_pred(self, N, w=None, T_gev=None):
        """Predicted pivot e-folds for the branch model labelled by N.
        w=None: instantaneous thermalisation (rho_th = rho_end)."""
        Vstar, rho_end = self.scales(N)
        base = 67 - LN_K + 0.25 * np.log(Vstar ** 2 / rho_end)
        if w is None:
            return base - np.log(G_HIGH) / 12
        g = g_star(T_gev)
        rho_th = np.pi ** 2 / 30 * g * (T_gev / MPL_GEV) ** 4
        return base + (1 - 3 * w) / (12 * (1 + w)) * np.log(rho_th / rho_end) - np.log(g) / 12

    def T_end_gev(self, N):
        _, rho_end = self.scales(N)
        return (rho_end / (np.pi ** 2 / 30 * G_HIGH)) ** 0.25 * MPL_GEV

    def N_inst(self):
        """Self-consistent N for instantaneous reheating."""
        return brentq(lambda N: self.N_pred(N) - N, 45, 79)

    def N_selfconsistent(self, w, T_gev):
        """Self-consistent N for (w, T_reh); None if it falls outside the grid."""
        f = lambda N: self.N_pred(N, w, T_gev) - N
        a, b = 44.01, 79.99
        if f(a) * f(b) > 0:
            return None
        return brentq(f, a, b)

    def T_required(self, N, w):
        """T_reh that delivers N on the branch for a given w > 1/3 (None if impossible)."""
        f = lambda lt: self.N_pred(N, w, np.exp(lt)) - N
        lo, hi = np.log(1e-4), np.log(0.999 * self.T_end_gev(N))
        if f(lo) * f(hi) > 0:
            return None
        return float(np.exp(brentq(f, lo, hi)))

    def w_min(self, N):
        """Smallest w for which N is reachable with T_reh >= T_MIN."""
        f = lambda w: self.N_pred(N, w, T_MIN_GEV) - N
        if f(1.0) < 0:
            return None                      # unreachable even with w = 1
        if f(1/3 + 1e-9) >= 0:
            return 1/3                       # no stiff phase needed
        return brentq(f, 1/3 + 1e-9, 1.0)


def run(edges):
    grid = br.load_grid()
    out = {}
    for m in br.MODELS:
        M = Model(grid, m)
        Ni = M.N_inst()
        Nw0 = M.N_selfconsistent(0.0, T_MIN_GEV)       # w = 0 down to the BBN bound
        Nk = M.N_selfconsistent(1.0, T_MIN_GEV)        # kination down to the BBN bound
        row = dict(N_inst=Ni, ns1_inst=float(M.ns1(Ni)), ns2_inst=float(M.ns2(Ni)),
                   r_inst=float(M.r1(Ni)), Vstar_quarter_gev=float(M.scales(Ni)[0] ** 0.25 * MPL_GEV),
                   T_end_gev=float(M.T_end_gev(Ni)),
                   N_w0_bbn=Nw0, N_w0_bbn_at_inst=float(M.N_pred(Ni, 0.0, T_MIN_GEV)),
                   N_kin_bbn=Nk, ns1_kin_bbn=None if Nk is None else float(M.ns1(Nk)), edges={})
        for label, (edge, order) in edges.items():
            Nc = br.n_crit(grid, m, edge, order)
            row["edges"][label] = dict(
                N_crit=Nc, dN_req=Nc - M.N_pred(Nc),
                w_min=M.w_min(Nc), T_req_w1=M.T_required(Nc, 1.0),
                T_req_w23=M.T_required(Nc, 2/3))
        out[m] = row
    return out


if __name__ == "__main__":
    jp = json.load(open(os.path.join(br.HERE, "..", "data", "joint_posterior.json")))
    e_act = round(jp["p-actlite-bk-l-b_r_camb"]["ns95_r0"][0], 4)   # as quoted in the paper
    e_pl = round(jp["planck-bk-l-b_r_camb"]["ns95_r0"][0], 4)
    edges = {"marginal 0.9675, 1st": (0.9675, 1), "marginal 0.9675, 2nd": (0.9675, 2),
             f"joint ACT {e_act:.4f}, 1st": (e_act, 1), f"joint ACT {e_act:.4f}, 2nd": (e_act, 2),
             f"joint Planck {e_pl:.4f}, 1st": (e_pl, 1), f"joint Planck {e_pl:.4f}, 2nd": (e_pl, 2)}
    res = run(edges)
    print(f"ln(k*/a0H0) = {LN_K:.3f}")
    for m, d in res.items():
        print(f"\n== {m}: N_inst = {d['N_inst']:.2f}  (ns1 {d['ns1_inst']:.4f}, ns2 {d['ns2_inst']:.4f},"
              f" r {d['r_inst']:.3e}, V*^1/4 = {d['Vstar_quarter_gev']:.3e} GeV, T_end = {d['T_end_gev']:.2e} GeV)")
        print(f"   w=0 down to T_reh=4.7 MeV: N ~ {d['N_w0_bbn_at_inst']:.2f} (evaluated on the N_inst model);"
              f"  w=1 down to 4.7 MeV: N = {d['N_kin_bbn']:.2f} (ns1 {d['ns1_kin_bbn']:.4f})")
        for lab, e in d["edges"].items():
            wm = "unreachable" if e["w_min"] is None else ("<=1/3 ok" if e["w_min"] <= 1/3 else f"{e['w_min']:.3f}")
            t1 = "-" if e["T_req_w1"] is None else f"{e['T_req_w1']:.2e}"
            t23 = "-" if e["T_req_w23"] is None else f"{e['T_req_w23']:.2e}"
            print(f"   {lab:28s} N_crit={e['N_crit']:.2f}  dN_req={e['dN_req']:+.2f}"
                  f"  w_min={wm}  T_req(w=1)={t1} GeV  T_req(w=2/3)={t23} GeV")
    with open(os.path.join(br.HERE, "..", "data", "reheating.json"), "w") as f:
        json.dump(res, f, indent=1)
