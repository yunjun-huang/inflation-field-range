"""Joint (n_s, r) posterior from the public ACT DR6 MCMC chains.

Replaces the "marginal n_s interval + separate BK18 bound" filter by the
two-dimensional highest-posterior-density (HPD) region of the published
LCDM + r chains.

Chains (not stored in this repository, ~200 MB each once unpacked), from
LAMBDA, "ACT (DR6.02) MCMC Chains: R Model":
  https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_r_get.html
  p-actlite-bk-l-b_r_camb.tar.gz   P-ACT + BK18 + CMB lensing + DESI BAO  (adopted)
  planck-bk-l-b_r_camb.tar.gz      Planck + BK18 + CMB lensing + DESI BAO (comparison)

Usage:
  python src/joint_posterior.py CHAINDIR [CHAINDIR ...]
CHAINDIR is an unpacked chain directory, e.g. .../p-actlite-bk-l-b_r_camb

Method: weighted samples after discarding the first 30% of each chain;
Gaussian KDE in (n_s, r) with the samples reflected about the prior edge
r = 0 (so that the density is not biased low at r -> 0); the HPD level
enclosing a fraction p of the posterior weight is found from the KDE density
evaluated at the samples themselves. Because the critical models have
r ~ 1e-5, i.e. r = 0 to within the posterior width (~0.02), what matters is
the n_s interval cut out of the HPD region at r -> 0.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import brentq

BURN = 0.3
N_LEVEL_SAMPLES = 20000
RNG = np.random.default_rng(20261115)


def load(chaindir, burn=BURN):
    root = os.path.join(chaindir, os.path.basename(os.path.normpath(chaindir)))
    files = sorted(glob.glob(root + ".[0-9].txt"))
    if not files:
        raise FileNotFoundError(f"no chain files {root}.N.txt")
    cols = open(files[0]).readline().split()[1:]
    out = []
    for f in files:
        d = np.loadtxt(f)
        out.append(d[int(burn * len(d)):])
    d = np.vstack(out)
    get = lambda name: d[:, cols.index(name)]
    return dict(w=get("weight"), ns=get("ns"), r=get("r"), As=get("As"),
                H0=get("H0"), nchains=len(files))


class JointHPD:
    """Reflected 2D KDE of (n_s, r) and its HPD levels."""

    def __init__(self, ch):
        w, ns, r = ch["w"], ch["ns"], ch["r"]
        self.w = w / w.sum()
        # reflect about r = 0
        pts = np.vstack([np.concatenate([ns, ns]), np.concatenate([r, -r])])
        self.kde = gaussian_kde(pts, weights=np.concatenate([w, w]))
        # density at a weighted subsample of the posterior -> HPD levels
        idx = RNG.choice(len(w), size=min(N_LEVEL_SAMPLES, len(w)), p=self.w)
        self.sample_dens = self.density(ns[idx], r[idx])

    def density(self, ns, r):
        ns, r = np.atleast_1d(ns), np.atleast_1d(r)
        return 2.0 * self.kde(np.vstack([ns, r]))      # x2: reflected mass

    def level(self, p):
        """Density contour enclosing posterior fraction p."""
        return np.quantile(self.sample_dens, 1.0 - p)

    def ns_edges(self, p, r=0.0):
        """Lower/upper n_s edge of the p-HPD region at fixed r."""
        lev = self.level(p)
        f = lambda x: self.density(x, r)[0] - lev
        grid = np.linspace(0.95, 1.0, 501)
        vals = np.array([f(x) for x in grid])
        inside = np.where(vals > 0)[0]
        lo = brentq(f, grid[inside[0] - 1], grid[inside[0]], xtol=1e-6)
        hi = brentq(f, grid[inside[-1]], grid[inside[-1] + 1], xtol=1e-6)
        return lo, hi

    def inside(self, ns, r, p=0.95):
        return self.density(ns, r)[0] >= self.level(p)


def summary(chaindir, burn=BURN):
    ch = load(chaindir, burn)
    w = ch["w"] / ch["w"].sum()
    m = np.sum(w * ch["ns"]); s = np.sqrt(np.sum(w * (ch["ns"] - m) ** 2))
    o = np.argsort(ch["r"]); r95 = ch["r"][o][np.searchsorted(np.cumsum(w[o]), 0.95)]
    hpd = JointHPD(ch)
    e68 = hpd.ns_edges(0.68); e95 = hpd.ns_edges(0.95)
    return dict(name=os.path.basename(os.path.normpath(chaindir)), burn=burn,
                nchains=ch["nchains"], nsamples=len(w),
                ns_mean=m, ns_std=s, ns_2sig_lower=m - 2 * s, r95=r95,
                ns68_r0=e68, ns95_r0=e95,
                As_mean=float(np.sum(w * ch["As"])),
                H0_mean=float(np.sum(w * ch["H0"]))), hpd


if __name__ == "__main__":
    res = {}
    for cd in sys.argv[1:]:
        for burn in (0.3, 0.5):
            s, _ = summary(cd, burn)
            print(f"{s['name']}  burn={burn}: {s['nchains']} chains, {s['nsamples']} rows")
            print(f"   marginal  n_s = {s['ns_mean']:.4f} +- {s['ns_std']:.4f}"
                  f"  (mean-2sigma = {s['ns_2sig_lower']:.4f});  r < {s['r95']:.4f} (95%)")
            print(f"   joint HPD at r->0:  68%: {s['ns68_r0'][0]:.4f}-{s['ns68_r0'][1]:.4f}"
                  f"   95%: {s['ns95_r0'][0]:.4f}-{s['ns95_r0'][1]:.4f}")
            print(f"   <A_s> = {s['As_mean']:.4e},  <H0> = {s['H0_mean']:.2f}")
            if burn == BURN:
                res[s["name"]] = s
    out = os.path.join(os.path.dirname(__file__), "..", "data", "joint_posterior.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("written", os.path.normpath(out))
