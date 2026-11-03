"""Explanatory figures for MATH.md -> figures/m_*.png. Usage: python math_figs.py [name ...|all]"""
import math
import sys

import numpy as np
import torch

from make_figures import (DENS, DIVERGE, INK, INK2, MUTED, GRID, SCORE, SURFACE, plt, save, show,
                          target_image)
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from flows import OrthoSylvester
from targets import make_target
from train import load_flow

C_FLOW, C_DM, C_FM = "#2a78d6", "#eb6834", "#1baf7a"
FINAL = LinearSegmentedColormap.from_list("final", ["#7fb0ea", "#2a78d6", "#0b2f5e"])
torch.manual_seed(0)


def rec(path):
    return torch.load(path, weights_only=False)


def model(path):
    m = load_flow(rec(path))
    m.eval()
    return m


SINE_FLOW = "results/e2_sine/sine2d2_L64_M2_relaxed_mle_s0.pt"
SINE_DM = "results/c_sine/sine2d2_dm_h128x3_it45000_s0.pt"
SINE_FM = "results/c_sine/sine2d2_fm_h128x3_it45000_s0.pt"
GRID_FLOW = "results/e3_gallery/grid16_L64_M2_relaxed_mle_s0.pt"
GRID_DM = "results/c_gallery/grid16_dm_h128x3_it60000_s0.pt"
GRID_FM = "results/c_gallery/grid16_fm_h128x3_it60000_s0.pt"


def clean(ax, lim=5):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def hist_image(x, n=160, lim=5):
    H, _, _ = np.histogram2d(x[:, 1], x[:, 0], bins=n, range=[[-lim, lim], [-lim, lim]], density=True)
    return H


@torch.no_grad()
def ode_path(m, z, steps=60):
    """RK4 path from noise (t = 1) to data (t = t0); returns times and [steps+1, N, D] positions."""
    ts = torch.linspace(1.0, m.t0, steps + 1)
    xs, x = [z], z
    for a, b in zip(ts[:-1], ts[1:]):
        h = float(b - a)
        k1 = m._v(x, a); k2 = m._v(x + 0.5 * h * k1, a + 0.5 * h)
        k3 = m._v(x + 0.5 * h * k2, a + 0.5 * h); k4 = m._v(x + h * k3, b)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return ts.numpy(), torch.stack(xs).numpy()


def handmade_layer(M, rr, angle, b):
    """A 2D orthogonal Sylvester layer with chosen direction, diagonal products and offsets."""
    L = OrthoSylvester(2, M, "relaxed")
    th = angle
    W = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])[:, :M]
    with torch.no_grad():
        L.W.copy_(W)
        L.R_off.zero_(); L.Rt_off.zero_()
        rt = torch.full((M,), 3.0)
        L.c.copy_(torch.log(rt))
        target = torch.tensor(rr[:M]) + 1 - L.eps  # softplus(a) = rr + 1 - eps
        L.a.copy_(torch.log(torch.expm1(target)))
        L.b.copy_(torch.tensor(b[:M]))
    return L


# ------------------------------------------------------------------ 1. change of variables (1D)
def fig_cov():
    r, a, b = 2.5, 3.0, 0.0
    f = lambda z: z + r * np.tanh(a * z + b)
    fp = lambda z: 1 + r * a / np.cosh(a * z + b) ** 2
    pz = lambda z: np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    z = np.linspace(-4, 4, 2001)
    bands = [((-0.15, 0.15), C_FLOW), ((1.35, 1.65), C_DM)]

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]
    ax.plot(z, pz(z), color=INK)
    for (lo, hi), c in bands:
        zz = np.linspace(lo, hi, 50)
        ax.fill_between(zz, 0, pz(zz), color=c, alpha=0.75, lw=0)
    ax.set_title("(a) base density $p_Z(z) = \\mathcal{N}(z; 0, 1)$\ntwo strips of equal width 0.3")
    ax.set_xlabel("z"); ax.set_ylim(0, 0.45)

    ax = axs[1]
    ax.plot(z, f(z), color=INK)
    ax.plot(z, z, color=MUTED, lw=1, ls=":")
    for (lo, hi), c in bands:
        ax.axvspan(lo, hi, color=c, alpha=0.25, lw=0)
        ax.axhspan(f(lo), f(hi), color=c, alpha=0.25, lw=0)
    ax.text(-3.8, 4.5, "steep: strip is\nstretched ×%.1f" % fp(0.0), color=C_FLOW, fontsize=8, va="top")
    ax.text(1.0, -3.0, "flat: strip keeps\nits width (×%.1f)" % fp(1.5), color=C_DM, fontsize=8)
    ax.set_title("(b) the map $x = f(z) = z + R\\,\\tanh(\\tilde R z)$\n(a 1D Sylvester layer, $R = 2.5,\\ \\tilde R = 3$)")
    ax.set_xlabel("z"); ax.set_ylabel("x")

    ax = axs[2]
    ax.plot(f(z), pz(z) / fp(z), color=INK)
    for (lo, hi), c in bands:
        zz = np.linspace(lo, hi, 50)
        ax.fill_between(f(zz), 0, pz(zz) / fp(zz), color=c, alpha=0.75, lw=0)
    ax.set_title("(c) result $p_X(x) = p_Z(z)\\,/\\,|f'(z)|$\nsame probability per strip, one bump became two")
    ax.set_xlabel("x"); ax.set_ylim(0, 0.45)
    for ax in axs:
        ax.grid(True, axis="y")
    save(fig, "m_change_of_variables.png")


# ------------------------------------------------------------------ 2. one Sylvester layer (2D)
def grid_lines(lim=3.2, n=13, m=300):
    c = np.linspace(-lim, lim, n)
    s = np.linspace(-lim, lim, m)
    lines = [np.c_[np.full(m, ci), s] for ci in c] + [np.c_[s, np.full(m, ci)] for ci in c]
    return [torch.tensor(l, dtype=torch.float32) for l in lines]


@torch.no_grad()
def fig_layer():
    rows = [("M = 1, $R\\tilde R = 8$", handmade_layer(1, [8.0], 0.5, [0.0])),
            ("M = 2, $R_{ii}\\tilde R_{ii} = 4.5$", handmade_layer(2, [4.5, 4.5], 0.5, [0.0, 0.0]))]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.3), gridspec_kw=dict(wspace=0.08, hspace=0.25))
    n = 241
    g = torch.linspace(-5, 5, n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], -1)
    for r, (label, L) in enumerate(rows):
        Q = L.mats()[0].numpy()
        ax = axs[r, 0]
        base = np.exp(-0.5 * (pts ** 2).sum(-1).numpy()).reshape(n, n)
        ax.imshow(base, origin="lower", extent=[-5, 5, -5, 5], cmap=DENS, norm=PowerNorm(0.5))
        for l in grid_lines():
            ax.plot(l[:, 0], l[:, 1], color=INK2, lw=0.4, alpha=0.7)
        for i in range(Q.shape[1]):
            ax.annotate("", xy=2.6 * Q[:, i], xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=C_DM, lw=2))
            ax.text(*(3.0 * Q[:, i]), f"$u_{i + 1}$", color=C_DM, fontsize=11, ha="center", va="center")
        clean(ax)
        ax.set_title(f"{label}\ninput z ~ N(0, I) and a grid", fontsize=9)

        ax = axs[r, 1]
        for l in grid_lines():
            ax.plot(*L(l)[0].numpy().T, color=INK2, lw=0.4, alpha=0.7)
        clean(ax)
        ax.set_title("the same grid after the layer\n$z' = z + Q R\\,\\tanh(\\tilde R Q^\\top z + b)$", fontsize=9)

        ax = axs[r, 2]
        z = L.inverse(pts)
        _, ld = L(z)
        q = np.exp(-0.5 * (z ** 2).sum(-1).numpy() - ld.numpy()).reshape(n, n)
        show(ax, q)
        ax.set_title("density after the layer\n$p_Z(z)\\,/\\,|\\det J(z)|$", fontsize=9)

        ax = axs[r, 3]
        _, ld = L(pts)
        im = ax.imshow(ld.numpy().reshape(n, n), origin="lower", extent=[-5, 5, -5, 5], cmap=SCORE)
        clean(ax)
        ax.set_title("$\\log|\\det J| = \\sum_i \\log(1 + h_i'\\,R_{ii}\\tilde R_{ii})$\n(over the input plane)", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
    save(fig, "m_sylvester_layer.png")


# ------------------------------------------------------------------ 3. a deep flow, layer by layer
@torch.no_grad()
def fig_depth():
    f = model(SINE_FLOW)
    z = torch.randn(30000, 2)
    _, _, path = f(z, return_path=True)
    ks = [0, 2, 4, 8, 16, 32, 64]
    col = np.arctan2(z[:, 1], z[:, 0]).numpy()
    fig, axs = plt.subplots(1, len(ks), figsize=(14, 2.6), gridspec_kw=dict(wspace=0.05))
    for ax, k in zip(axs, ks):
        x = path[k].numpy()
        ax.scatter(x[:, 0], x[:, 1], c=col, cmap="twilight", s=0.3, lw=0, rasterized=True)
        clean(ax)
        ax.set_title("input z" if k == 0 else f"after {k} layers", fontsize=9)
    save(fig, "m_flow_depth.png")


# ------------------------------------------------------------------ 4. the exact inverse
def fig_inverse():
    from flows import _solve_monotone
    r, a, c, t = 2.0, 2.0, 0.4, 1.0  # R, R~, b, target w'
    g = lambda v: v + r * np.tanh(a * v + c)
    gp = lambda v: 1 + r * a / np.cosh(a * v + c) ** 2
    v = np.linspace(-3.5, 3.5, 800)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(wspace=0.28))
    ax = axs[0]
    ax.plot(v, g(v), color=INK)
    ax.axhline(t, color=C_DM, lw=1.2, ls="--")
    ax.text(1.0, t + 0.2, "target value $w'$", color=C_DM, fontsize=8)
    vk = t
    for k in range(5):
        vn = vk - (g(vk) - t) / gp(vk)
        ax.plot([vk, vk], [t, g(vk)], color=C_FLOW, lw=1, ls=":")
        ax.plot([vk, vn], [g(vk), t], color=C_FLOW, lw=1)
        ax.plot(vk, g(vk), "o", color=C_FLOW, ms=5)
        if k < 4:
            ax.text(vk - 0.32 if g(vk) > t else vk + 0.1, g(vk) + (0.25 if g(vk) > t else -0.45), f"$w_{k}$", color=C_FLOW, fontsize=9)
        vk = vn
    ax.set_xlabel("w"); ax.set_ylabel("$g(w) = w + R\\,\\tanh(\\tilde R w + b)$")
    ax.set_xlim(-1.6, 2.2); ax.set_ylim(-3.5, 4.5)
    ax.set_title("(a) undo one direction: solve $g(w) = w'$ ($R = \\tilde R = 2$, $b = 0.4$)\nstrictly increasing ⇒ exactly one solution. Newton: follow the tangent", fontsize=9)
    ax.grid(True)

    ax = axs[1]
    T = torch.linspace(-8, 8, 4000, dtype=torch.float64)
    its = np.arange(0, 31)
    for (rr, aa, col, lab) in [(1.0, 2.0, C_FLOW, "gentle layer, $R_{ii}\\tilde R_{ii} = 2$"),
                               (6.0, 3.0, C_DM, "sharp layer, $R_{ii}\\tilde R_{ii} = 18$")]:
        R, A, C = (torch.tensor(x, dtype=torch.float64) for x in (rr, aa, 0.4))
        res = []
        for n in its:
            vv = _solve_monotone(T, R, A, C, int(n))
            res.append((vv + R * torch.tanh(A * vv + C) - T).abs().numpy())
        res = np.maximum(np.array(res), 1e-16)
        ax.plot(its, np.median(res, 1), color=col, label=lab + " (median)")
        ax.plot(its, res.max(1), color=col, ls="--", lw=1.2, label="… worst of 4,000")
    ax.set_yscale("log"); ax.set_xlabel("iterations"); ax.set_ylabel("$|g(w) - w'|$")
    ax.set_title("(b) residual of the safeguarded solver used in flows.py", fontsize=9)
    ax.legend(fontsize=7.5); ax.grid(True)
    save(fig, "m_inverse.png")


# ------------------------------------------------------------------ 5. the noising paths
def fig_noising():
    from baselines import Diffusion
    tgt = make_target("sine2d:2")
    x0 = tgt.sample(300_000)
    eps = torch.randn_like(x0)
    ts = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    dm = Diffusion(2, 4)
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(2, len(ts) + 1, width_ratios=[1] * len(ts) + [1.5], wspace=0.06, hspace=0.45)
    for r, name in enumerate(["diffusion (VP)", "flow matching"]):
        for j, t in enumerate(ts):
            if r == 0:
                a, s = dm._alpha_sigma(torch.tensor(max(t, 1e-3)))
            else:
                a, s = torch.tensor(1 - t), torch.tensor(t)
            xt = a * x0 + s * eps
            ax = fig.add_subplot(gs[r, j])
            show(ax, hist_image(xt.numpy()))
            ax.set_title(f"t = {t:g}\n$\\alpha_t$ = {a:.2f}, $\\sigma_t$ = {s:.2f}", fontsize=8.5)
            if j == 0:
                ax.text(-6, 0, name, rotation=90, va="center", ha="right", fontsize=10, color=INK)
        ax = fig.add_subplot(gs[r, -1])
        tt = torch.linspace(1e-3, 1, 200)
        if r == 0:
            a, s = dm._alpha_sigma(tt)
            ax.set_title("VP schedule: $\\alpha_t^2 + \\sigma_t^2 = 1$", fontsize=9)
        else:
            a, s = 1 - tt, tt
            ax.set_title("straight path: $\\alpha_t + \\sigma_t = 1$", fontsize=9)
        ax.plot(tt, a, color=INK, label="$\\alpha_t$ (signal)")
        ax.plot(tt, s, color=C_DM if r == 0 else C_FM, label="$\\sigma_t$ (noise)")
        ax.set_xlabel("t"); ax.legend(fontsize=8, loc="center right"); ax.grid(True)
        ax.yaxis.tick_right()
    save(fig, "m_noising.png")


# ------------------------------------------------------------------ 6. the score
@torch.no_grad()
def fig_score():
    dm = model(SINE_DM)
    tgt = make_target("sine2d:2")
    x0 = tgt.sample(200_000)
    eps = torch.randn_like(x0)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.7), gridspec_kw=dict(wspace=0.06))

    # (a) one training example
    ax = axs[0]
    show(ax, target_image("sine2d:2"))
    t = 0.3
    a, s = dm._alpha_sigma(torch.tensor(t))
    p0 = torch.tensor([1.0, 2 * math.sin(math.pi / 2 * 1.0)])
    e = torch.tensor([-0.4, -1.7])
    pt = a * p0 + s * e
    ax.plot(*p0, "o", color=INK, ms=6); ax.text(p0[0] + 0.2, p0[1] + 0.3, "$x_0$", fontsize=11)
    ax.plot(*(a * p0), "o", color=MUTED, ms=5); ax.text(a * p0[0] - 1.3, a * p0[1] + 0.25, "$\\alpha_t x_0$", fontsize=10, color=INK2)
    ax.plot(*pt, "o", color=C_DM, ms=6); ax.text(pt[0] + 0.2, pt[1] - 0.4, "$x_t$", fontsize=11, color=C_DM)
    ax.annotate("", xy=pt, xytext=a * p0, arrowprops=dict(arrowstyle="-|>", color=C_DM, lw=1.8))
    ax.text(-4.6, -4.5, "network input: $(x_t, t)$\ntarget: $\\varepsilon$ (orange arrow / $\\sigma_t$)", fontsize=8.5)
    ax.set_title("(a) one training example, t = 0.3\n$x_t = \\alpha_t x_0 + \\sigma_t \\varepsilon$", fontsize=9)

    n = 19
    g = torch.linspace(-4.5, 4.5, n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], -1)
    for ax, t, lab in zip(axs[1:], [0.7, 0.3, 0.08], "bcd"):
        a, s = dm._alpha_sigma(torch.tensor(t))
        show(ax, hist_image((a * x0 + s * eps).numpy()))
        score = -dm.net(pts, torch.full((len(pts),), t)) / s
        mag = score.norm(dim=-1, keepdim=True)
        u = (score / mag).numpy()
        ax.quiver(pts[:, 0], pts[:, 1], u[:, 0], u[:, 1], mag.squeeze().numpy(), cmap=FINAL,
                  scale=26, width=0.005, headwidth=3.5)
        clean(ax)
        ax.set_title(f"({lab}) learned score $-\\hat\\varepsilon_\\theta(x, t)/\\sigma_t$, t = {t:g}\n"
                     f"(arrows: direction; darker = longer)", fontsize=9)
    save(fig, "m_score.png")


# ------------------------------------------------------------------ 7. paths of all three
@torch.no_grad()
def fig_paths():
    torch.manual_seed(3)
    z = torch.randn(400, 2)
    f = model(GRID_FLOW)
    _, _, path = f(z, return_path=True)
    flow_xs = torch.stack(path).numpy()
    rows = [("Sylvester flow", np.linspace(0, 1, len(flow_xs)), flow_xs, C_FLOW, "layer  (0 → 64)")]
    for name, p, c in [("diffusion", GRID_DM, C_DM), ("flow matching", GRID_FM, C_FM)]:
        ts, xs = ode_path(model(p), z, steps=80)
        rows.append((name, 1 - ts, xs, c, "time  (t = 1 → 0)"))
    fig, axs = plt.subplots(3, 2, figsize=(10.5, 13), gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.15, hspace=0.25))
    for r, (name, s, xs, c, xl) in enumerate(rows):
        end = xs[-1, :, 0]
        colors = FINAL((np.clip(end, -3.5, 3.5) + 3.5) / 7)
        ax = axs[r, 0]
        show(ax, target_image("grid16"))
        for i in range(0, xs.shape[1], 2 if r else 8):
            ax.plot(xs[:, i, 0], xs[:, i, 1], color=colors[i], lw=0.5 if r else 0.7, alpha=0.6 if r else 0.8)
        ax.scatter(xs[0, :, 0], xs[0, :, 1], s=3, color=MUTED, zorder=3)
        ax.scatter(xs[-1, :, 0], xs[-1, :, 1], s=4, color=INK, zorder=3)
        clean(ax)
        ax.set_title(f"{name}: paths in the plane\n(grey: noise start, black: sample" + (", 1 in 8 paths drawn)" if r == 0 else ")"), fontsize=9)
        ax = axs[r, 1]
        for i in range(xs.shape[1]):
            ax.plot(s, xs[:, i, 0], color=colors[i], lw=0.5, alpha=0.6)
        ax.set_xlabel(xl); ax.set_ylabel("first coordinate")
        ax.set_ylim(-4.5, 4.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["noise", "data"])
        ax.set_title(f"{name}: first coordinate along the path (colour = where it ends)", fontsize=9)
        for yv in (-3, -1, 1, 3):
            ax.axhline(yv, color=GRID, lw=0.8, zorder=0)
    save(fig, "m_paths.png")


# ------------------------------------------------------------------ 8. flow matching: crossing lines vs learned field
@torch.no_grad()
def fig_fm():
    fm = model(SINE_FM)
    tgt = make_target("sine2d:2")
    torch.manual_seed(5)
    x0 = tgt.sample(60)
    eps = torch.randn(60, 2)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw=dict(wspace=0.12))

    ax = axs[0]
    show(ax, target_image("sine2d:2"))
    for a, b in zip(x0, eps):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C_FM, lw=0.8, alpha=0.8)
    ax.scatter(eps[:, 0], eps[:, 1], s=8, color=MUTED, zorder=3)
    ax.scatter(x0[:, 0], x0[:, 1], s=8, color=INK, zorder=3)
    clean(ax)
    ax.set_title("(a) training pairs: straight lines from data $x_0$ (black)\nto noise $\\varepsilon$ (grey). Velocity $\\varepsilon - x_0$. They cross!", fontsize=9)

    ax = axs[1]
    t = 0.5
    x0b = tgt.sample(200_000)
    show(ax, hist_image(((1 - t) * x0b + t * torch.randn_like(x0b)).numpy()))
    n = 19
    g = torch.linspace(-4.5, 4.5, n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], -1)
    v = -fm.net(pts, torch.full((len(pts),), t))  # direction of generation (t decreasing)
    mag = v.norm(dim=-1, keepdim=True)
    u = (v / mag).numpy()
    ax.quiver(pts[:, 0], pts[:, 1], u[:, 0], u[:, 1], mag.squeeze().numpy(), cmap=FINAL,
              scale=26, width=0.005, headwidth=3.5)
    clean(ax)
    ax.set_title("(b) learned field at t = 0.5: average of the\nline velocities through each point (drawn as $-v_\\theta$)", fontsize=9)

    ax = axs[2]
    show(ax, target_image("sine2d:2"))
    ts, xs = ode_path(fm, eps, steps=60)
    for i in range(xs.shape[1]):
        ax.plot(xs[:, i, 0], xs[:, i, 1], color=C_FM, lw=0.8, alpha=0.8)
    ax.scatter(eps[:, 0], eps[:, 1], s=8, color=MUTED, zorder=3)
    ax.scatter(xs[-1, :, 0], xs[-1, :, 1], s=8, color=INK, zorder=3)
    clean(ax)
    ax.set_title("(c) solving $dx/dt = v_\\theta(x,t)$ from the\nsame noise points: curved paths", fontsize=9)
    save(fig, "m_flow_matching.png")


# ------------------------------------------------------------------ 9. the divergence and the density
@torch.no_grad()
def fig_divergence():
    fm = model(SINE_FM)
    n = 121
    g = torch.linspace(-5, 5, n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], -1)
    ts = [0.9, 0.5, 0.2, 0.05]
    tgt = make_target("sine2d:2")
    x0 = tgt.sample(200_000)
    eps = torch.randn_like(x0)
    fig, axs = plt.subplots(1, len(ts) + 1, figsize=(15, 3.6), gridspec_kw=dict(wspace=0.12))
    for ax, t in zip(axs, ts):
        _, div = fm._v_div(pts, t)
        D = div.numpy().reshape(n, n)
        lim = np.percentile(np.abs(D), 99)
        ax.imshow(D, origin="lower", extent=[-5, 5, -5, 5], cmap=DIVERGE,
                  norm=TwoSlopeNorm(0, -lim, lim), interpolation="bilinear")
        H = hist_image(((1 - t) * x0 + t * eps).numpy(), n=80)
        gg = np.linspace(-5, 5, 80)
        ax.contour(gg, gg, H, levels=[0.2 * H.max()], colors=INK, linewidths=0.7)
        clean(ax)
        ax.set_title(f"$\\nabla\\cdot v_\\theta(x, t)$,  t = {t:g}\ncolour range ±{lim:.0f}", fontsize=9)
    ax = axs[-1]
    from train import model_grid_logq
    q = torch.exp(model_grid_logq(fm, torch.stack([torch.meshgrid(torch.linspace(-5, 5, 161), torch.linspace(-5, 5, 161), indexing="xy")[k].flatten() for k in (0, 1)], -1).double())).numpy().reshape(161, 161)
    show(ax, q)
    ax.set_title("result: $q(x_0) = \\mathcal{N}(x_1)\\,\\exp\\int_0^1 \\nabla\\cdot v\\,dt$\n(flow matching, 35k params)", fontsize=9)
    save(fig, "m_divergence.png")


# ================================================================== small 1D building blocks
# A two-bump 1D data distribution with closed-form everything (used in the diffusion / FM parts).
MU, TAU, W = np.array([-2.0, 2.0]), 0.3, np.array([0.5, 0.5])
B0, B1 = 0.1, 20.0


def npdf(x, m, v):
    return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)


def path_coef(kind, t):
    """x_t = a x0 + s eps; returns a, s, da/dt, ds/dt."""
    t = np.asarray(t, dtype=float)
    if kind == "vp":
        beta = B0 + t * (B1 - B0)
        a = np.exp(-0.25 * t ** 2 * (B1 - B0) - 0.5 * t * B0)
        s = np.sqrt(1 - a ** 2)
        return a, s, -0.5 * beta * a, beta * a ** 2 / (2 * s)
    return 1 - t, t, -np.ones_like(t), np.ones_like(t)


def mix(x, t, kind):
    """Exact density, score and marginal velocity of the noised two-bump mixture."""
    a, s, da, ds = path_coef(kind, t)
    x = np.asarray(x, dtype=float)[..., None]
    var = a ** 2 * TAU ** 2 + s ** 2
    if np.ndim(var):
        var = var[..., None]; a_ = a[..., None]; s_ = s[..., None]; da_ = da[..., None]; ds_ = ds[..., None]
    else:
        a_, s_, da_, ds_ = a, s, da, ds
    comp = W * npdf(x, a_ * MU, var)
    p = comp.sum(-1)
    resp = comp / np.maximum(p[..., None], 1e-300)
    d = x - a_ * MU
    score = (resp * (-d / var)).sum(-1)
    ex0 = MU + a_ * TAU ** 2 / var * d
    eeps = s_ / var * d
    v = (resp * (da_ * ex0 + ds_ * eeps)).sum(-1)
    return p, score, v


def sample_data_1d(n, rng):
    k = rng.integers(0, 2, n)
    return MU[k] + TAU * rng.standard_normal(n)


def ode_1d(x1, kind, steps=200, t0=None):
    t0 = (1e-3 if kind == "vp" else 0.0) if t0 is None else t0
    ts = np.linspace(1.0, t0, steps + 1)
    xs, x = [x1], x1.copy()
    for a, b in zip(ts[:-1], ts[1:]):
        h = b - a
        f = lambda xx, tt: mix(xx, tt, kind)[2]
        k1 = f(x, a); k2 = f(x + 0.5 * h * k1, a + 0.5 * h); k3 = f(x + 0.5 * h * k2, a + 0.5 * h); k4 = f(x + h * k3, b)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return ts, np.array(xs)


def pt_image(kind, ts, xg):
    img = np.stack([mix(xg, t, kind)[0] for t in ts], 1)
    return img / img.max(0, keepdims=True)


def heat(ax, kind, t_left=1.0, t_right=0.0, xlim=4.5):
    ts = np.linspace(min(t_left, t_right), max(t_left, t_right), 300)
    ts = np.maximum(ts, 1e-3)
    xg = np.linspace(-xlim, xlim, 300)
    img = pt_image(kind, ts, xg)
    if t_left > t_right:
        img = img[:, ::-1]
    ax.imshow(img, origin="lower", aspect="auto", extent=[0, 1, -xlim, xlim], cmap=DENS,
              norm=PowerNorm(0.6), interpolation="bilinear")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([f"{t_left:g}", "0.5", f"{t_right:g}"])
    ax.set_xlabel(f"time t  ({'noise → data' if t_left > t_right else 'data → noise'})")
    ax.set_ylabel("x")


def tpos(t, t_left=1.0):
    """map time to the horizontal axis used by heat()"""
    return (t_left - np.asarray(t)) if t_left == 1.0 else np.asarray(t)


# ------------------------------------------------------------------ F1: linear map in 1D
def fig_linear():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200_000)
    x = 2 * z + 1
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.9), gridspec_kw=dict(wspace=0.25))
    g = np.linspace(-7, 9, 600)
    for ax, d, pdf, c, title in [(axs[0], z, npdf(g, 0, 1), C_FLOW, "samples of $z$ and $p_Z(z) = \\mathcal{N}(0, 1)$"),
                                 (axs[1], x, npdf(g, 1, 4) , C_DM, "samples of $x = 2z + 1$ and $p_X(x) = \\frac{1}{2}\\,p_Z\\!\\left(\\frac{x-1}{2}\\right)$")]:
        ax.hist(d, bins=40, range=(-7, 9), density=True, color=c, alpha=0.35, lw=0)
        ax.plot(g, pdf, color=INK, lw=1.6)
        ax.set_ylim(0, 0.45); ax.set_title(title, fontsize=9); ax.grid(True, axis="y")
    axs[1].annotate("twice as wide,\nhalf as high", xy=(3.2, 0.12), xytext=(5.0, 0.3), fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=MUTED))
    axs[0].set_xlabel("z"); axs[1].set_xlabel("x")
    save(fig, "m_linear_1d.png")


# ------------------------------------------------------------------ F2: area under 2D linear maps
def fig_area():
    maps = [("stretch", np.array([[2.0, 0], [0, 1.0]])), ("shear", np.array([[1.0, 1.0], [0, 1.0]])),
            ("rotate", np.array([[0.8, -0.6], [0.6, 0.8]])), ("squeeze", np.array([[0.5, 0], [0, 0.5]]))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.1), gridspec_kw=dict(wspace=0.1))
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)
    for ax, (name, A) in zip(axs, maps):
        ax.plot(*sq.T, color=MUTED, ls="--", lw=1)
        for k in np.linspace(0, 1, 5):
            for seg in (np.array([[k, 0], [k, 1]]), np.array([[0, k], [1, k]])):
                ax.plot(*(seg @ A.T).T, color=C_FLOW, lw=0.6, alpha=0.6)
        im = sq @ A.T
        ax.fill(*im.T, color=C_FLOW, alpha=0.18, lw=0)
        ax.plot(*im.T, color=C_FLOW, lw=1.6)
        ax.set_xlim(-0.8, 2.3); ax.set_ylim(-0.3, 1.7); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{name}:  A = [{A[0,0]:g}  {A[0,1]:g} ;  {A[1,0]:g}  {A[1,1]:g}]\n"
                     f"area ×{abs(np.linalg.det(A)):.2g}  →  density ×{1 / abs(np.linalg.det(A)):.2g}", fontsize=9)
    save(fig, "m_area_2d.png")


# ------------------------------------------------------------------ F3: stacking 1D layers adds valleys
def fig_compose():
    layers = [(2.0, 3.0, 0.0), (1.2, 4.0, -4.0 * 2.75), (1.2, 4.0, 4.0 * 4.0)]
    z = np.linspace(-5, 5, 20001)
    x, logp = z.copy(), np.log(npdf(z, 0, 1))
    stages = [(x.copy(), logp.copy())]
    for r, a, b in layers:
        logp = logp - np.log(1 + r * a / np.cosh(a * x + b) ** 2)
        x = x + r * np.tanh(a * x + b)
        stages.append((x.copy(), logp.copy()))
    fig, axs = plt.subplots(1, 4, figsize=(13, 2.8), gridspec_kw=dict(wspace=0.15), sharey=True)
    titles = ["input: $\\mathcal{N}(0,1)$", "after layer 1\n(step at 0)", "after layer 2\n(step at +2.75)",
              "after layer 3\n(step at −4)"]
    for k, (ax, (xx, lp)) in enumerate(zip(axs, stages)):
        ax.fill_between(xx, 0, np.exp(lp), color=C_FLOW, alpha=0.25, lw=0)
        ax.plot(xx, np.exp(lp), color=INK, lw=1.4)
        if k:
            r, a, b = layers[k - 1]
            ax.axvline(-b / a, color=C_DM, lw=1, ls="--")
        ax.set_xlim(-7, 7); ax.set_title(titles[k] + f"  →  {k + 1} bump{'s' if k else ''}", fontsize=9)
        ax.grid(True, axis="y"); ax.set_xlabel("x")
    save(fig, "m_compose_1d.png")


# ------------------------------------------------------------------ F4: the 1D step and the product R R~
def fig_profiles():
    v = np.linspace(-3, 3, 2001)
    at = 2.0
    prods = [(-1.5, "#b8b6ae"), (-0.5, "#7fb0ea"), (0.0, MUTED), (2.0, "#2a78d6"), (8.0, "#0b2f5e")]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw=dict(wspace=0.25))
    for pr, c in prods:
        r = pr / at
        y = v + r * np.tanh(at * v)
        lab = f"$R\\tilde R$ = {pr:g}" + ("  (folds back: NOT invertible)" if pr < -1 else "")
        axs[0].plot(v, y, color=c, lw=1.8 if pr >= -1 else 1.2, ls="-" if pr >= -1 else ":", label=lab)
        if pr >= -1:
            dens = npdf(v, 0, 1) / (1 + pr / np.cosh(at * v) ** 2)
            axs[1].plot(y, dens, color=c, lw=1.8, label=f"$R\\tilde R$ = {pr:g}")
    axs[0].set_xlabel("w (input coordinate along u)"); axs[0].set_ylabel("output coordinate")
    axs[0].set_title("the 1D map $w \\mapsto w + R\\,\\tanh(\\tilde R w)$, slope at 0 is $1 + R\\tilde R$", fontsize=9)
    axs[0].legend(fontsize=7.5); axs[0].grid(True)
    axs[1].set_xlabel("output coordinate"); axs[1].set_ylabel("density")
    axs[1].set_title("what it does to a Gaussian:\n$R\\tilde R > 0$ carves a valley, $-1 < R\\tilde R < 0$ squeezes into a peak", fontsize=9)
    axs[1].legend(fontsize=7.5); axs[1].grid(True); axs[1].set_xlim(-7, 7)
    save(fig, "m_layer_profiles.png")


# ------------------------------------------------------------------ F5: the layer in three sub-steps
@torch.no_grad()
def fig_layer_steps():
    th = 0.5
    q = np.array([math.cos(th), math.sin(th)])
    r, at, b = 1.0, 2.0, 0.0
    g = np.linspace(-2.5, 2.5, 9)
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    v = P @ q
    shift = r * np.tanh(at * v + b)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw=dict(wspace=0.22, width_ratios=[1, 1.15, 1]))
    ax = axs[0]
    lim = np.abs(v).max()
    ax.scatter(P[:, 0], P[:, 1], c=v, cmap=DIVERGE, vmin=-lim, vmax=lim, s=16)
    ax.annotate("", xy=2.4 * q, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.text(*(2.75 * q), "$u$", fontsize=12)
    nperp = np.array([-q[1], q[0]])
    ax.plot(*np.c_[-3.5 * nperp, 3.5 * nperp], color=INK2, lw=1, ls="--")
    clean(ax, 3.8)
    ax.set_title("step 1: project, $w = u^\\top z$\n(colour = w; dashed line: w = 0)", fontsize=9)
    ax = axs[1]
    vv = np.linspace(-4, 4, 400)
    ax.plot(vv, r * np.tanh(at * vv + b), color=C_DM)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("w"); ax.set_ylabel("how far to move along u")
    ax.set_title(f"step 2: a 1D soft step  $R\\,\\tanh(\\tilde R w + b)$\n($R$ = {r:g}: height, $\\tilde R$ = {at:g}: sharpness, $b$ = 0: centre at $w = -b/\\tilde R$)", fontsize=9)
    ax.grid(True)
    ax = axs[2]
    D = shift[:, None] * q[None]
    ax.quiver(P[:, 0], P[:, 1], D[:, 0], D[:, 1], angles="xy", scale_units="xy", scale=1, color=C_DM,
              width=0.004, headwidth=4)
    ax.scatter(P[:, 0], P[:, 1], s=5, color=INK2)
    ax.plot(*np.c_[-3.5 * nperp, 3.5 * nperp], color=INK2, lw=1, ls="--")
    clean(ax, 3.8)
    ax.set_title("step 3: move every point along u\n$z' = z + u\\,R\\,\\tanh(\\tilde R\\,u^\\top z + b)$", fontsize=9)
    save(fig, "m_layer_steps.png")


# ------------------------------------------------------------------ F6: Jacobian = local stretch (small circles -> ellipses)
@torch.no_grad()
def fig_jacobian():
    rows = [("M = 1", handmade_layer(1, [3.0], 0.5, [0.0])), ("M = 2", handmade_layer(2, [3.0, 3.0], 0.5, [0.0, 0.0]))]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw=dict(wspace=0.08))
    ang = torch.linspace(0, 2 * math.pi, 60)
    circ = 0.14 * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
    cs = torch.linspace(-3, 3, 9)
    for ax, (name, L) in zip(axs, rows):
        Q, R, Rt, rd, rtd = L.mats()
        for cx in cs:
            for cy in cs:
                c = torch.tensor([[cx, cy]])
                _, ld = L(c)
                out = c + (L(c + circ)[0] - L(c)[0])  # local shape J·circle, drawn at the input point
                col = SCORE(min(float(ld) / 3.0, 1.0) * 0.8 + 0.2)
                ax.fill(*out.numpy().T, color=col, lw=0)
                ax.plot(*out.numpy().T, color=INK2, lw=0.5)
        for i in range(Q.shape[1]):
            ax.annotate("", xy=4.2 * Q[:, i].numpy(), xytext=3.2 * Q[:, i].numpy(),
                        arrowprops=dict(arrowstyle="-|>", color=C_DM, lw=2))
            ax.text(*(4.6 * Q[:, i].numpy() + np.array([0.0, 0.35])), f"$u_{i + 1}$", color=C_DM, fontsize=11, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none"))
        clean(ax, 5)
        ax.set_title(f"{name}, $R_{{ii}}\\tilde R_{{ii}} = 3$: what a small circle becomes\n"
                     "(drawn at its starting point; darker = more area)", fontsize=9)
    save(fig, "m_jacobian.png")


# ------------------------------------------------------------------ F7: MLE training of a tiny 1D flow
def fig_mle_1d():
    from flows import Flow
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    data = torch.tensor(sample_data_1d(100_000, rng), dtype=torch.float32)[:, None]
    f = Flow(1, 6, 1, "relaxed")
    opt = torch.optim.Adam(f.parameters(), lr=2e-2)
    snaps, hist = {}, []
    xg = torch.linspace(-4, 4, 801)[:, None]
    keep = [0, 30, 150, 3000]
    for it in range(3001):
        if it in keep:
            with torch.no_grad():
                snaps[it] = f.log_prob(xg).exp().numpy()
        x = data[torch.randint(len(data), (512,))]
        loss = -f.log_prob(x).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
    fig, axs = plt.subplots(1, 5, figsize=(15, 2.8), gridspec_kw=dict(wspace=0.18, width_ratios=[1, 1, 1, 1, 1.2]))
    for ax, k in zip(axs, keep):
        ax.hist(data.numpy()[:20000, 0], bins=100, range=(-4, 4), density=True, color=MUTED, alpha=0.3, lw=0)
        ax.plot(xg[:, 0], snaps[k], color=C_FLOW)
        ax.set_title(f"step {k}", fontsize=9); ax.set_ylim(0, 0.75); ax.set_xlabel("x")
        ax.grid(True, axis="y")
    axs[0].set_ylabel("density")
    ax = axs[-1]
    ax.plot(np.convolve(hist, np.ones(20) / 20, mode="valid"), color=C_FLOW)
    ent = -np.mean(np.log(mix(sample_data_1d(200_000, rng), 0.0, "fm")[0]))
    ax.axhline(ent, color=INK2, ls="--", lw=1)
    ax.text(3000, ent + 0.04, "best possible\n(entropy of the data)", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("training step"); ax.set_title("loss $-\\frac{1}{n}\\sum\\log q(x_i)$", fontsize=9)
    ax.grid(True)
    save(fig, "m_mle_1d.png")


# ------------------------------------------------------------------ F8: noising in 1D
def fig_noise_1d():
    rng = np.random.default_rng(2)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(wspace=0.22))
    ax = axs[0]
    n = 400
    ts = np.linspace(0, 1, n + 1)
    for x0, c in [(-2.0, C_FLOW), (2.0, C_DM)]:
        a, s, _, _ = path_coef("vp", np.maximum(ts, 1e-4))
        ax.fill_between(ts, a * x0 - 2 * s, a * x0 + 2 * s, color=c, alpha=0.12, lw=0)
        for _ in range(4):
            x, xs = x0, [x0]
            for t0, t1 in zip(ts[:-1], ts[1:]):
                h = t1 - t0; beta = B0 + t0 * (B1 - B0)
                x = x - 0.5 * beta * x * h + math.sqrt(beta * h) * rng.standard_normal()
                xs.append(x)
            ax.plot(ts, xs, color=c, lw=0.8)
    ax.set_xlabel("time t  (data → noise)"); ax.set_ylabel("x"); ax.set_ylim(-4.5, 4.5); ax.grid(True)
    ax.set_title("two data points, noised 4 times each\nband: $\\alpha_t x_0 \\pm 2\\sigma_t$ (where $x_t$ lands 95% of the time)", fontsize=9)
    ax = axs[1]
    heat(ax, "vp", t_left=0.0, t_right=1.0)
    ax.set_title("the whole data distribution $p_t(x)$, two bumps at ±2\n(each column scaled to its maximum)", fontsize=9)
    save(fig, "m_noise_1d.png")


# ------------------------------------------------------------------ F9: score in 1D
def fig_score_1d():
    x = np.linspace(-4.5, 4.5, 900)
    ts = [0.02, 0.15, 0.3, 0.6]
    fig, axs = plt.subplots(2, len(ts), figsize=(13, 4.6), sharex=True, gridspec_kw=dict(hspace=0.12, wspace=0.18))
    for j, t in enumerate(ts):
        p, sc, _ = mix(x, t, "vp")
        axs[0, j].fill_between(x, 0, p, color=C_FLOW, alpha=0.25, lw=0); axs[0, j].plot(x, p, color=INK, lw=1.4)
        axs[0, j].set_title(f"t = {t:g}", fontsize=9.5); axs[0, j].grid(True, axis="y")
        sc_c = np.clip(sc, -12, 12)
        axs[1, j].plot(x, sc_c, color=C_DM)
        axs[1, j].axhline(0, color=MUTED, lw=0.8); axs[1, j].set_ylim(-12, 12); axs[1, j].grid(True)
        for xa in (-3.6, -1.0, 1.0, 3.6):
            s0 = float(np.interp(xa, x, sc_c))
            axs[1, j].annotate("", xy=(xa + 0.08 * s0, 0), xytext=(xa, 0),
                               arrowprops=dict(arrowstyle="-|>", color=C_DM, lw=1.3))
        axs[1, j].set_xlabel("x")
    axs[0, 0].set_ylabel("density $p_t(x)$"); axs[1, 0].set_ylabel("score $\\partial_x \\log p_t(x)$")
    axs[1, 0].text(-4.4, -11, "(cut off at ±12)", fontsize=7.5, color=INK2)
    save(fig, "m_score_1d.png")


# ------------------------------------------------------------------ F10: regression on noise = score
def fig_denoise():
    rng = np.random.default_rng(3)
    t = 0.25
    a, s, _, _ = path_coef("vp", t)
    x0 = sample_data_1d(400_000, rng)
    eps = rng.standard_normal(len(x0))
    xt = a * x0 + s * eps
    xg = np.linspace(-3.5, 3.5, 500)
    _, sc, _ = mix(xg, t, "vp")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw=dict(wspace=0.22))
    ax = axs[0]
    ax.scatter(xt[:3000], eps[:3000], s=2, color=MUTED, alpha=0.5, rasterized=True)
    ax.plot(xg, -s * sc, color=C_DM, lw=2.2, label="best guess  $\\mathbb{E}[\\varepsilon\\mid x_t] = -\\sigma_t\\,\\partial_x\\log p_t$")
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-4, 4)
    ax.set_xlabel("noisy input $x_t$"); ax.set_ylabel("noise $\\varepsilon$ that was added")
    ax.set_title(f"training pairs $(x_t, \\varepsilon)$ at t = {t:g}: the target is noisy,\nbut least squares learns its average", fontsize=9)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True)
    ax = axs[1]
    sel = np.abs(xt - 0.5) < 0.03
    ax.hist(eps[sel], bins=60, density=True, color=MUTED, alpha=0.5, lw=0)
    m = eps[sel].mean()
    ax.axvline(m, color=C_DM, lw=2.2)
    ax.text(m + 0.1, ax.get_ylim()[1] * 0.85, f"mean = {m:.2f}\n(the best single guess)", color=C_DM, fontsize=8.5)
    ax.text(-1.8, ax.get_ylim()[1] * 0.55, "came from\nthe +2 bump", ha="center", fontsize=8.5, color=INK2)
    ax.text(2.8, ax.get_ylim()[1] * 0.55, "came from\nthe −2 bump", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("$\\varepsilon$"); ax.set_yticks([])
    ax.set_title("all the noises that could have produced $x_t = 0.5$\n(the dashed slice on the left)", fontsize=9)
    save(fig, "m_denoise.png")


# ------------------------------------------------------------------ F11: reverse SDE and ODE in 1D
def fig_reverse_1d():
    rng = np.random.default_rng(4)
    n_paths, steps = 40, 500
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(wspace=0.18))
    x1 = rng.standard_normal(n_paths)
    ts = np.linspace(1.0, 1e-3, steps + 1)
    ax = axs[0]
    heat(ax, "vp")
    x, xs = x1.copy(), [x1.copy()]
    for t0, t1 in zip(ts[:-1], ts[1:]):
        h = t0 - t1; beta = B0 + t0 * (B1 - B0)
        _, sc, _ = mix(x, t0, "vp")
        x = x + (0.5 * beta * x + beta * sc) * h + np.sqrt(beta * h) * rng.standard_normal(n_paths)
        xs.append(x.copy())
    ax.plot(tpos(ts), np.array(xs), color=INK, lw=0.6, alpha=0.7)
    ax.set_title("reverse SDE: denoise with the score + fresh noise\n$dx = [-\\frac{1}{2}\\beta x - \\beta\\,\\partial_x\\log p_t]\\,dt + \\sqrt{\\beta}\\,d\\bar W$", fontsize=9)
    ax = axs[1]
    heat(ax, "vp")
    tt, xx = ode_1d(x1, "vp", steps=300)
    ax.plot(tpos(tt), xx, color=INK, lw=0.8, alpha=0.8)
    ax.set_title("probability-flow ODE: same distributions, no noise\n$dx/dt = -\\frac{1}{2}\\beta\\,(x + \\partial_x\\log p_t)$", fontsize=9)
    save(fig, "m_reverse_1d.png")


# ------------------------------------------------------------------ F12: density along paths in 1D
def fig_density_1d():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw=dict(wspace=0.22))
    from scipy.stats import norm
    x1 = norm.ppf((np.arange(41) + 0.5) / 41)  # probability 1/41 between neighbours
    ax = axs[0]
    heat(ax, "fm")
    tt, xx = ode_1d(x1, "fm", steps=300)
    ax.plot(tpos(tt), xx, color=INK, lw=0.7)
    ax.set_title("41 paths, probability 1/41 ≈ 2.4% between neighbours\nbunched up = high density, spread out = low density", fontsize=9)
    ax = axs[1]
    picks = [(40, C_FLOW), (20, MUTED), (31, C_DM)]
    for i, c in picks:
        xp = xx[:, i]
        exact = np.log(np.maximum(np.array([mix(xp[k], tt[k], "fm")[0] for k in range(len(tt))]), 1e-300))
        h = 1e-4
        div = np.array([(mix(xp[k] + h, tt[k], "fm")[2] - mix(xp[k] - h, tt[k], "fm")[2]) / (2 * h) for k in range(len(tt))])
        dt = tt[0] - tt[1]
        integ = np.log(npdf(xp[0], 0, 1)) + np.concatenate([[0], np.cumsum(0.5 * (div[1:] + div[:-1]) * dt)])
        ax.plot(tpos(tt), exact, color=c, lw=2.2, alpha=0.45)
        ax.plot(tpos(tt)[::15], integ[::15], "o", color=c, ms=3.5)
        ax.text(1.02, exact[-1] + {40: -0.9, 20: 0.0, 31: 0.9}[i], f"ends at x = {xp[-1]:+.2f}", color=c, fontsize=8, va="center")
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["1", "0.5", "0"]); ax.set_xlim(0, 1.3)
    ax.set_xlabel("time t  (noise → data)"); ax.set_ylabel("log density along the path")
    ax.set_title("lines: exact $\\log p_t(x(t))$.  dots: $\\log\\mathcal{N}(x_1) + \\int \\partial_x v\\,dt$\nthe divergence integral reproduces the density", fontsize=9)
    ax.grid(True)
    save(fig, "m_density_1d.png")


# ------------------------------------------------------------------ F13: flow matching in 1D
def fig_fm_1d():
    rng = np.random.default_rng(5)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.7), gridspec_kw=dict(wspace=0.2, width_ratios=[1, 1, 0.8]))
    n = 40
    x0 = sample_data_1d(n, rng); eps = rng.standard_normal(n)
    ax = axs[0]
    heat(ax, "fm")
    for a, e in zip(x0, eps):
        ax.plot([0, 1], [e, a], color=C_FM, lw=0.8)
    ax.set_title("training pairs: straight lines from noise $\\varepsilon$ to data $x_0$\nmany of them cross", fontsize=9)
    ax = axs[1]
    heat(ax, "fm")
    tt, xx = ode_1d(eps, "fm", steps=200)
    ax.plot(tpos(tt), xx, color=C_FM, lw=0.9)
    ax.plot(0.5, 0.5, "o", color=INK, ms=5)
    ax.set_title("ODE with the averaged velocity $v^*(x, t) = \\mathbb{E}[\\varepsilon - x_0 \\mid x_t = x]$\nfrom the same noise: paths never cross", fontsize=9)
    ax = axs[2]
    N = 2_000_000
    x0b = sample_data_1d(N, rng); eb = rng.standard_normal(N)
    t = 0.5
    xt = (1 - t) * x0b + t * eb
    sel = np.abs(xt - 0.5) < 0.01
    u = eb[sel] - x0b[sel]
    ax.hist(-u, bins=60, density=True, color=MUTED, alpha=0.5, lw=0)
    vstar = mix(0.5, t, "fm")[2]
    ax.axvline(-float(vstar), color=C_FM, lw=2.2)
    ax.text(-float(vstar) + 0.1, ax.get_ylim()[1] * 0.85, f"average = {-float(vstar):.2f}", color=C_FM, fontsize=8.5)
    ax.set_xlabel("velocity towards data, $x_0 - \\varepsilon$"); ax.set_yticks([])
    ax.set_title("all lines through the black dot (x = 0.5, t = 0.5):\ntheir velocities, and the average the network learns", fontsize=9)
    save(fig, "m_fm_1d.png")


# ------------------------------------------------------------------ F14: VP vs straight paths, and solver steps
def fig_dm_vs_fm_1d():
    rng = np.random.default_rng(6)
    x1 = rng.standard_normal(40)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.7), gridspec_kw=dict(wspace=0.22, width_ratios=[1, 1, 0.9]))
    for ax, kind, c, name in [(axs[0], "vp", C_DM, "diffusion (VP) path"), (axs[1], "fm", C_FM, "flow matching (straight) path")]:
        heat(ax, kind)
        tt, xx = ode_1d(x1, kind, steps=300)
        ax.plot(tpos(tt), xx, color=c, lw=0.9)
        ax.set_title(f"{name}: exact ODE paths", fontsize=9)
    ax = axs[2]
    N = 100_000
    z = rng.standard_normal(N)
    bins = np.linspace(-4, 4, 161)
    cen = 0.5 * (bins[1:] + bins[:-1])
    ptrue = mix(cen, 0.0, "fm")[0]
    ptrue = ptrue / ptrue.sum()
    for kind, c, name in [("vp", C_DM, "diffusion"), ("fm", C_FM, "flow matching")]:
        errs, Ns = [], [1, 2, 4, 8, 16, 32, 64, 128]
        for n in Ns:
            t0 = 1e-3 if kind == "vp" else 0.0
            ts = np.linspace(1.0, t0, n + 1)
            x = z.copy()
            for a, b in zip(ts[:-1], ts[1:]):
                x = x + (b - a) * mix(x, a, kind)[2]
            H, _ = np.histogram(x, bins=bins)
            errs.append(0.5 * np.abs(H / N - ptrue).sum())
        ax.plot(Ns, errs, "o-", color=c, label=name, ms=4)
    ax.set_xscale("log", base=2); ax.set_xlabel("Euler steps (network evaluations)"); ax.set_ylabel("TV error of the samples")
    ax.set_title("sampling error vs number of Euler steps\n(using the exact velocity, so only the solver errs)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True); ax.set_ylim(0, 1)
    save(fig, "m_dm_vs_fm_1d.png")


FIGS = dict(cov=fig_cov, layer=fig_layer, depth=fig_depth, inverse=fig_inverse, noising=fig_noising,
            score=fig_score, paths=fig_paths, fm=fig_fm, divergence=fig_divergence,
            linear=fig_linear, area=fig_area, compose=fig_compose, profiles=fig_profiles,
            steps=fig_layer_steps, jacobian=fig_jacobian, mle=fig_mle_1d, noise1d=fig_noise_1d,
            score1d=fig_score_1d, denoise=fig_denoise, reverse=fig_reverse_1d, density1d=fig_density_1d,
            fm1d=fig_fm_1d, dmfm=fig_dm_vs_fm_1d)

if __name__ == "__main__":
    torch.set_num_threads(8)
    which = sys.argv[1:] or ["all"]
    for k, fn in FIGS.items():
        if "all" in which or k in which:
            fn()
