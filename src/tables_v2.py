"""Numbers for the tables added in the revised manuscript.

Run:  python src/tables_v2.py     (seconds; needs data/branch_grid.json,
                                  data/joint_posterior.json, data/reheating.json)

Prints: second-order corrections at the critical points; N_crit for all n_s
edges at first and second order (the first-order row for the adopted edge
and the dataset rows must reproduce the earlier Tables 5-6); the excursion
accumulation fractions; and the reheating table.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import branch as br

grid = br.load_grid()
D = os.path.join(br.HERE, "..", "data")
jp = json.load(open(os.path.join(D, "joint_posterior.json")))
rh = json.load(open(os.path.join(D, "reheating.json")))
NAMES = {"E": "E-model", "H8": "hilltop p=8", "H12": "hilltop p=12", "H20": "hilltop p=20"}
bad = []

print("=== Second-order corrections at the critical points ===")
print("  model          N   ns(1st)  ns(2nd)  dns       r(1st)      r(2nd)     dr/r")
for m in br.MODELS:
    for N in (50, 60, 70):
        d = next(x for x in grid if x["model"] == m and x["N"] == N)
        print(f"  {NAMES[m]:13s} {N}  {d['ns1']:.4f}  {d['ns2']:.4f}  {d['ns2']-d['ns1']:+.1e}"
              f"  {d['r1']:.4e}  {d['r2']:.4e}  {d['r2']/d['r1']-1:+.2%}")

e_act = jp["p-actlite-bk-l-b_r_camb"]["ns95_r0"][0]
e_pl = jp["planck-bk-l-b_r_camb"]["ns95_r0"][0]
EDGES = [("P-ACT-LB marginal (adopted)", 0.9675),
         ("P-ACT-LB+BK18 joint 95%", round(e_act, 4)),
         ("P-ACT marginal", 0.9633),
         ("SPT+Planck+ACT marginal", 0.9613),
         ("Planck-LB+BK18 joint 95%", round(e_pl, 4)),
         ("SPT+Planck marginal", 0.9566)]
OLD = {0.9675: (62.54, 72.84, 68.69, 65.93), 0.9633: (55.46, 64.57, 60.90, 58.45),
       0.9613: (52.62, 61.26, 57.78, 55.46), 0.9566: (46.98, 54.68, 51.58, 49.50)}
print("\n=== N_crit: first order / second order ===")
for lab, e in EDGES:
    row = []
    for i, m in enumerate(br.MODELS):
        n1, n2 = br.n_crit(grid, m, e, 1), br.n_crit(grid, m, e, 2)
        row.append(f"{n1:6.2f}/{n2:6.2f}")
        if e in OLD and abs(n1 - OLD[e][i]) > 0.0051:
            bad.append(f"N_crit {m} edge {e}: {n1:.3f} vs manuscript {OLD[e][i]}")
    print(f"  {lab:28s} {e:.4f}  " + "  ".join(row))

print("\n=== Reheating ===")
for m in br.MODELS:
    d = rh[m]
    print(f"  {NAMES[m]:13s} N_inst={d['N_inst']:.1f}  ns1={d['ns1_inst']:.4f}  ns2={d['ns2_inst']:.4f}"
          f"  r={d['r_inst']:.2e}  V*^1/4={d['Vstar_quarter_gev']:.2e} GeV"
          f"  N(w=1,T=4.7MeV)={d['N_kin_bbn']:.1f}")
    for lab, e in d["edges"].items():
        wm = e["w_min"]
        wm = "none" if wm is None else ("<=1/3" if wm <= 1/3 + 1e-12 else f"{wm:.2f}")
        t1 = "-" if e["T_req_w1"] is None else f"{e['T_req_w1']:.1e}"
        print(f"      {lab:26s} N_crit={e['N_crit']:.2f} dN={e['dN_req']:+.2f} w_min={wm} T(w=1)={t1}")

print("\nALL MATCH" if not bad else "\nDIFFERENCES:\n  " + "\n  ".join(bad))
