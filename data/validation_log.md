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