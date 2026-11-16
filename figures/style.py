"""Shared figure style (PRD column width 3.4 in, full width 7.0 in)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = 3.4
FULL = 7.0
COLORS = {"E": "#1f4e9c", "H8": "#c0392b", "H12": "#e08a1e", "H20": "#7d3c98"}
LABELS = {"E": r"$\alpha$-attractor E-model", "H8": r"hilltop $p=8$",
          "H12": r"hilltop $p=12$", "H20": r"hilltop $p=20$"}
GREY = "0.35"


def setup():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5,
        "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 1.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, path):
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=220)
    print("saved", path)
