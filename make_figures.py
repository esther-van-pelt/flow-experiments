"""All figures for REPORT.md -> figures/*.png. Usage: python make_figures.py [e1|e2|e3|all]"""
import glob
import json
import math
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from targets import make_target
from train import load_flow, model_grid_logq

FIG = "figures"
os.makedirs(FIG, exist_ok=True)

# palette (dataviz reference instance)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
C_MLE, C_RKL, C_PAPER = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
DENS = LinearSegmentedColormap.from_list("dens", [SURFACE, "#b9d3f2", "#2a78d6", "#0b2f5e"])
DEPTH = LinearSegmentedColormap.from_list("depth", ["#b9d3f2", "#0b2f5e"])
SCORE = LinearSegmentedColormap.from_list("score", [SURFACE, "#7fb0ea", "#0b2f5e"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#d9d8d4", "#eb6834"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "font.size": 9, "axes.titlesize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
})


def load(exp):
    recs = {}
    for f in glob.glob(f"results/{exp}/*.pt"):
        r = torch.load(f, weights_only=False)
        c = r["cfg"]
        recs[(c["target"], c["L"], c["M"], c["param"], c["obj"])] = r
    return recs


def density_image(rec, n=201):
    t = make_target(rec["cfg"]["target"])
    pts, _ = t.grid(n)
    q = torch.exp(model_grid_logq(load_flow(rec), pts)).numpy().reshape(n, n)
    return q


def target_image(spec, n=201):
    t = make_target(spec)
    _, p = t.grid_density(n)
    return p.numpy().reshape(n, n)


def show(ax, img, title=None, vmax=None):
    ax.imshow(img, origin="lower", extent=[-5, 5, -5, 5], cmap=DENS,
              norm=PowerNorm(0.5, vmin=0, vmax=vmax or img.max()), interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8.5)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- E1
def fig_e1():
    R = load("e1_1d")
    Ks, Ls = [2, 3, 4, 6, 8], [1, 2, 4, 8, 16, 32]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
    for ax, (param, obj) in zip(axes, [("relaxed", "mle"), ("relaxed", "rkl"), ("paper", "mle"), ("paper", "rkl")]):
        M = np.full((len(Ks), len(Ls)), np.nan)
        for i, K in enumerate(Ks):
            for j, L in enumerate(Ls):
                r = R.get((f"mix1d:{K}", L, 1, param, obj))
                if r:
                    M[i, j] = r["metrics"]["modes_covered"]
        ax.imshow(M / np.array(Ks)[:, None], cmap=SCORE, vmin=0, vmax=1, aspect="auto")
        for i in range(len(Ks)):
            for j in range(len(Ls)):
                if not np.isnan(M[i, j]):
                    frac = M[i, j] / Ks[i]
                    ax.text(j, i, f"{int(M[i, j])}/{Ks[i]}", ha="center", va="center", fontsize=8,
                            color="white" if frac > 0.6 else INK)
        ax.set_xticks(range(len(Ls)), Ls); ax.set_yticks(range(len(Ks)), Ks)
        ax.set_xlabel("layers L")
        ax.set_title(f"{'MLE' if obj == 'mle' else 'reverse KL'} · {param} diag.")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("target modes K")
    fig.suptitle("1D: modes recovered (local max of q within σ of each target mode)", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "e1_modes_heatmap.png")

    # density overlays
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0))
    for ax, spec in zip(axes, ["mix1d:4", "mix1d:8", "sine1d"]):
        t = make_target(spec)
        pts, p = t.grid_density()
        x = pts[:, 0].numpy()
        ax.fill_between(x, p.numpy(), color="#d9d8d4", lw=0, label="target")
        Lshow = [2, 8, 32]
        for k, L in enumerate(Lshow):
            r = R.get((spec, L, 1, "relaxed", "mle"))
            if not r:
                continue
            q = torch.exp(model_grid_logq(load_flow(r), pts)).numpy()
            ax.plot(x, q, color=DEPTH(k / (len(Lshow) - 1)), lw=1.6, label=f"L = {L}")
        ax.set_title(t.name + "  (MLE, relaxed)")
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlim(-5, 5)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "e1_densities.png")


# ---------------------------------------------------------------- E2
PERIODS = [1, 2, 3, 4, 6]


def fig_e2():
    R = load("e2_sine")
    L = 64
    fig, axes = plt.subplots(len(PERIODS), 4, figsize=(12, 2.35 * len(PERIODS)),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]))
    n = 201
    g = np.linspace(-5, 5, n)
    for row, P in zip(axes, PERIODS):
        spec = f"sine2d:{P}"
        p = target_image(spec, n)
        show(row[0], p, f"target · {P} period{'s' if P > 1 else ''}")
        imgs = {}
        for ax, obj, lab in [(row[1], "mle", "MLE"), (row[2], "rkl", "reverse KL")]:
            r = R.get((spec, L, 2, "relaxed", obj))
            if r is None:
                ax.axis("off"); continue
            q = density_image(r, n)
            imgs[obj] = q
            m = r["metrics"]
            show(ax, q, f"{lab} · L={L}\nTV {m['tv']:.2f} · {m['cond_modes_found']}/{m['cond_modes_total']} cond. modes",
                 vmax=p.max())
        # conditional slice at y0 = 1
        ax = row[3]
        iy = int(np.argmin(np.abs(g - 1.0)))
        norm = lambda v: v / (v.sum() * (g[1] - g[0]) + 1e-300)
        ax.fill_between(g, norm(p[iy]), color="#d9d8d4", lw=0, label="target")
        if "mle" in imgs:
            ax.plot(g, norm(imgs["mle"][iy]), color=C_MLE, lw=1.6, label="MLE")
        if "rkl" in imgs:
            ax.plot(g, norm(imgs["rkl"][iy]), color=C_RKL, lw=1.6, label="reverse KL")
        ax.set_xlim(-5, 5); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_title("p(angle | outcome = 1)")
    axes[0, 3].legend(loc="upper right", fontsize=8, ncol=3)
    for a in axes[:, :3].ravel():
        a.set_aspect("equal")
    axes[-1, 3].set_xlabel("angle (x)")
    fig.tight_layout()
    save(fig, "e2_sine_gallery.png")

    # scaling plots
    Ls = [8, 16, 32, 64]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, obj, lab in [(axes[0], "mle", "MLE"), (axes[1], "rkl", "reverse KL")]:
        for k, L in enumerate(Ls):
            ys = []
            for P in PERIODS:
                r = R.get((f"sine2d:{P}", L, 2, "relaxed", obj))
                ys.append(r["metrics"]["cond_modes_found"] / r["metrics"]["cond_modes_total"] if r else np.nan)
            ax.plot(PERIODS, ys, "-o", color=DEPTH(k / (len(Ls) - 1)), ms=5, lw=1.8, label=f"L = {L}")
        ax.set_ylim(-0.03, 1.05); ax.set_xticks(PERIODS)
        ax.set_xlabel("periods in the target")
        ax.set_title(f"{lab}: share of conditional modes found")
        ax.grid(axis="y")
    axes[0].set_ylabel("fraction recovered")
    axes[1].legend(fontsize=8, loc="lower left")
    ax = axes[2]
    for obj, col, lab in [("mle", C_MLE, "MLE"), ("rkl", C_RKL, "reverse KL")]:
        for param, ls, mk in [("relaxed", "-", "o"), ("paper", "--", "s")]:
            ys = [R[(f"sine2d:{P}", 64, 2, param, obj)]["metrics"]["tv"] if (f"sine2d:{P}", 64, 2, param, obj) in R
                  else np.nan for P in PERIODS]
            ax.plot(PERIODS, ys, ls, marker=mk, color=col, ms=5, lw=1.8, label=f"{lab} · {param}")
    ax.set_xticks(PERIODS); ax.set_ylim(0, 1); ax.grid(axis="y")
    ax.set_xlabel("periods"); ax.set_ylabel("TV distance (lower is better)")
    ax.set_title("L = 64: relaxed vs paper diagonal")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "e2_scaling.png")

    # how the flow folds the Gaussian
    best = max((P for P in PERIODS if (f"sine2d:{P}", 64, 2, "relaxed", "mle") in R
                and R[(f"sine2d:{P}", 64, 2, "relaxed", "mle")]["metrics"]["cond_modes_found"]
                == R[(f"sine2d:{P}", 64, 2, "relaxed", "mle")]["metrics"]["cond_modes_total"]), default=None)
    if best is None:
        return
    rec = R[(f"sine2d:{best}", 64, 2, "relaxed", "mle")]
    flow = load_flow(rec)
    torch.manual_seed(1)
    z = torch.randn(6000, 2)
    with torch.no_grad():
        _, _, path = flow(z, return_path=True)
    steps = [0, 4, 8, 16, 32, 64]
    fig, axes = plt.subplots(1, len(steps), figsize=(13, 2.6))
    col = z[:, 0].numpy()
    for ax, s in zip(axes, steps):
        P_ = path[s].numpy()
        ax.scatter(P_[:, 0], P_[:, 1], c=col, cmap=DIVERGE, s=1.2, lw=0, vmin=-2.5, vmax=2.5, rasterized=True)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("base N(0, I)" if s == 0 else f"after {s} layers")
    fig.suptitle(f"How the flow deforms the Gaussian ({best}-period sine, MLE, L = 64). "
                 "Colour = first base coordinate z₁", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "e2_layers.png")


# ---------------------------------------------------------------- E3
GAL = ["two_gauss", "moons", "ring8", "rings", "grid16", "spiral", "checkerboard"]
GCOLS = [("mle", 16, 2, "MLE · L=16"), ("mle", 64, 2, "MLE · L=64"), ("rkl", 64, 2, "rev. KL · L=64"),
         ("mle", 64, 1, "MLE · L=64 · M=1")]


def fig_e3():
    R = load("e3_gallery")
    n = 181
    fig, axes = plt.subplots(len(GAL), 1 + len(GCOLS), figsize=(2.1 * (1 + len(GCOLS)), 2.15 * len(GAL)))
    for row, t in zip(axes, GAL):
        p = target_image(t, n)
        show(row[0], p, make_target(t).name)
        for ax, (obj, L, M, lab) in zip(row[1:], GCOLS):
            r = R.get((t, L, M, "relaxed", obj))
            if r is None:
                ax.axis("off"); continue
            m = r["metrics"]
            extra = f" · {m['modes_covered']}/{m['modes_total']} modes" if "modes_total" in m else ""
            show(ax, density_image(r, n), f"{lab}\nTV {m['tv']:.2f}{extra}", vmax=p.max())
        for a in row:
            a.set_aspect("equal")
    fig.tight_layout()
    save(fig, "e3_gallery.png")

    # dot plot of TV
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cols = [C_MLE, "#0b2f5e", C_RKL, C_PAPER]
    for k, ((obj, L, M, lab), c) in enumerate(zip(GCOLS, cols)):
        ys = np.arange(len(GAL)) + (k - 1.5) * 0.17
        xs = [R[(t, L, M, "relaxed", obj)]["metrics"]["tv"] if (t, L, M, "relaxed", obj) in R else np.nan for t in GAL]
        ax.scatter(xs, ys, color=c, s=34, label=lab, zorder=3, edgecolor=SURFACE, lw=1)
    ax.set_yticks(range(len(GAL)), [make_target(t).name for t in GAL])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis="x")
    ax.set_xlabel("TV distance to target (lower is better)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Gallery: fit quality per target", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "e3_tv.png")


def fig_targets():
    """Overview of every target distribution used, one row per experiment."""
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(3, 8, height_ratios=[0.75, 1, 1], width_ratios=[1.15] + [1] * 7,
                          hspace=0.45, wspace=0.15)

    def label(row, title, sub):
        ax = fig.add_subplot(gs[row, 0]); ax.axis("off")
        ax.text(0, 0.62, title, fontweight="bold", fontsize=10.5, va="center")
        ax.text(0, 0.38, sub, fontsize=8.5, color=INK2, va="center", linespacing=1.4)

    label(0, "E1 · 1D", "rows of bumps +\nthe 1D sine outcome")
    for j, spec in enumerate(["mix1d:2", "mix1d:3", "mix1d:4", "mix1d:6", "mix1d:8", "sine1d"]):
        ax = fig.add_subplot(gs[0, j + 1])
        t = make_target(spec)
        pts, p = t.grid_density()
        ax.fill_between(pts[:, 0].numpy(), p.numpy(), color=C_MLE, lw=0)
        ax.set_xlim(-5, 5); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xticks([-4, 0, 4])
        ax.set_title("outcome of\n2 sin(θ) + noise" if spec == "sine1d" else f"{spec.split(':')[1]} bumps", fontsize=8.5)
    label(1, "E2 · sine band", "horizontal = angle\nvertical = outcome")
    for j, P in enumerate(PERIODS):
        ax = fig.add_subplot(gs[1, j + 1])
        show(ax, target_image(f"sine2d:{P}", 161), f"{P} period{'s' if P > 1 else ''}")
        ax.set_aspect("equal")
    label(2, "E3 · gallery", "classic 2D shapes,\neasy → hard")
    for j, t in enumerate(GAL):
        ax = fig.add_subplot(gs[2, j + 1])
        show(ax, target_image(t, 161), make_target(t).name)
        ax.set_aspect("equal")
    save(fig, "targets.png")


def summary_tables():
    out = {}
    for exp in ["e1_1d", "e2_sine", "e3_gallery"]:
        rows = []
        for k, r in sorted(load(exp).items()):
            rows.append(dict(zip(["target", "L", "M", "param", "obj"], k)) | {
                kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r["metrics"].items()})
        out[exp] = rows
    with open("results/summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/summary.json")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, fn in [("targets", fig_targets), ("e1", fig_e1), ("e2", fig_e2), ("e3", fig_e3), ("tables", summary_tables)]:
        if which in (k, "all"):
            fn()
