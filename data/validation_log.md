# 项目信息

作者：Yunjun Huang (June Huang)
ORCID：0009-0007-6289-749X
邮箱：fmddzlumos@gmail.com
项目：Field excursion in plateau and hilltop inflation under ACT DR6

---

# 验证记录

## 2026-08-26 数值导数 vs m²φ² 闭式解

epsilon_V:  相对误差 7.6e-11 ~ 1.4e-10  (phi = 1, 2, 5, 10, 15.556)
eta_V:      相对误差 8.3e-08 ~ 1.8e-05

eta 精度低于 epsilon：二阶中心差分需除以 h²=1e-10，放大浮点舍入。
量级符合预期，不影响后续计算。

## 2026-08-26 暴胀结束点 phi_end

m²φ²：解 epsilon_V = 1
  numeric = 1.41421356
  exact   = sqrt(2) = 1.41421356
  rel.err = 8.9e-12

brentq 求根区间 [0.1, 10.0]：两端 epsilon-1 异号（phi=0.1 时 eps=200，phi=10 时 eps=0.02）。

## 2026-08-26 e-folds 与场程

m²φ²，N=60：
  phi_star numeric = 15.556349
  phi_star exact   = sqrt(4N+2) = 15.556349
  rel.err = 1.5e-11
  Delta phi = 14.142136 = 10*sqrt(2)  ✓

phi_star 求根下界取 phi_end*1.001，避开 N=0 的平凡根。

## 2026-08-26 alpha-attractor E-model (N=60)

| alpha | phi_end | phi_star | ns     | r        | dphi  |
|-------|---------|----------|--------|----------|-------|
| 0.01  | 0.3098  | 1.1010   | 0.9667 | 3.318e-05| 0.791 |
| 0.10  | 0.5953  | 2.5937   | 0.9669 | 3.261e-04| 1.998 |
| 1.00  | 0.9402  | 5.4532   | 0.9678 | 2.964e-03| 4.513 |
| 10.0  | 1.2055  | 9.4578   | 0.9697 | 1.937e-02| 8.252 |

alpha=1 与文献标准值一致（ns=0.9678, r=2.96e-3, dphi=4.51）。
E-model 无闭式解，此为引擎在无参考答案情形下的独立验证。

### 数值问题与修正
E-model 平原区 V'->0，数值导数在 b*phi ~ 25-29 处因浮点相消失效
（b = sqrt(2/3alpha)）。原先写死的求根上界会踩进该区域，
产生 divide-by-zero 与 IntegrationWarning。
改为区间扩张法：从 phi_end 出发按 0.5 步长外推，直到 N 超过目标再求根。
修正后无警告，且 m2phi2 结果不变（回归通过）。

## 2026-08-27 临界点 Delta phi = 1 M_Pl（alpha-attractor E-model）

| N  | alpha_crit | ns     | r          | dphi   |
|----|------------|--------|------------|--------|
| 50 | 0.018956   | 0.9601 | 9.0252e-05 | 1.0000 |
| 60 | 0.017612   | 0.9668 | 5.8325e-05 | 1.0000 |
| 70 | 0.016591   | 0.9715 | 4.0411e-05 | 1.0000 |

求根方式：对 log10(alpha) 用 brentq，区间 [-3, 1]。
dphi 列全为 1.0000，是求根的自洽检验。

注意：N=60 的 ns=0.9668 低于 P-ACT-LB 的 2sigma 下沿 0.9675，
即该亚普朗克点被 ACT 排除；N=70 的 ns=0.9715 落在窗口内。
结论在 N=60 与 N=70 之间翻转，需进一步定出临界 N。

## 2026-08-27 临界 e-folds N_crit

判据：Delta phi = 1 的那个模型，其 ns 是否落在 P-ACT-LB 的 2sigma 窗口内。
窗口下沿 ns = 0.9743 - 2 x 0.0034 = 0.9675。

| N  | ns @ dphi=1 | 判决 |
|----|-------------|------|
| 55 | 0.9638      | 排除 |
| 58 | 0.9656      | 排除 |
| 60 | 0.9668      | 排除 |
| 62 | 0.9678      | 允许 |
| 65 | 0.9693      | 允许 |
| 70 | 0.9715      | 允许 |

**N_crit = 61.37**

求根方式：双层 brentq。内层对 log10(alpha) 解 dphi = 1；
外层对 N 解 ns(N) = 0.9675，区间 [55, 70]。

### 含义

亚普朗克场程在 N < 61.37 时被 ACT DR6 排除，在 N > 61.37 时允许。
常用的 e-folds 区间是 50-60，翻转点落在其外沿 1.37 个 e-fold 处。

因此结论对 N 的选取**不稳健**。此前 PPT 中"conclusion is robust to
reheating-driven e-fold uncertainty"的表述不成立，需在论文中明确更正，
并将 reheating 以边界条件的身份写入讨论。

### 待处理

quad 在 slowroll.py:32 仍有 IntegrationWarning（roundoff error）。
数值结果与独立计算一致（N_crit 独立值

### 已解决

quad 的 IntegrationWarning：将容差放宽至 epsabs=1e-8, epsrel=1e-6。
理由：N 的物理不确定性为 +-5 e-folds，原默认容差 1.49e-8 远超实际需求。
修改后警告消失，所有结果不变（N=55 处 ns 由 0.9638 变为 0.9637，
第四位小数级差异，判决不变）。

## 2026-08-27 Fig 1：场程 vs 张量-标量比

文件：figures/fig1_r_vs_dphi.py -> fig1_r_vs_dphi.pdf / .png

内容：alpha-attractor E-model，N=60，扫 alpha 从 1e-3 到 30（60 个点），
在 (r, dphi) 平面作图。标注：
- 灰色区域 dphi > 1 M_Pl（超普朗克）
- 黑点 r_crit = 5.8325e-05
- 竖虚线 CMB-S4 / LiteBIRD 目标灵敏度 sigma(r) ~ 1e-3

纵轴范围取 0.1 到 20，横轴 1e-6 到 1e-1。
横轴下限 1e-6 依教授建议：r_crit ~ 5.8e-5 落在图的中下部，
与 1e-3 竖线之间的间距清晰可读；若画到 1e-10 则全部挤在右上角。

图的结论：可探测的原初引力波（r > 1e-3）全部对应超普朗克场程。