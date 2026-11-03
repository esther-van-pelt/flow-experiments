"""Figures + tables for the comparison Sylvester flow vs diffusion (DM) vs flow matching (FM).
Usage: python compare_figs.py [sine|gallery|curves|nfe|oned|table|all]"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

import make_figures as mf  # reuse palette, rcParams, helpers
from make_figures import (C_MLE, C_RKL, C_PAPER, INK, INK2, MUTED, SURFACE, plt, save, show,
                          target_image)
from targets import make_target
from train import load_flow

torch.set_num_threads(int(os.environ.get("THREADS", "8")))

C_FLOW, C_DM, C_FM = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated all-pairs)
FAM_LABEL = {"sylvester": "Sylvester flow", "dm": "diffusion", "fm": "flow matching"}
FAM_COLOR = {"sylvester": C_FLOW, "dm": C_DM, "fm": C_FM}
PERIODS = [1, 2, 3, 4, 6]
GAL = mf.GAL


# ----------------------------------------------------------------------------- bookkeeping
def load_all():
    recs = []
    for f in glob.glob("results/*/*.pt"):
        if "/scratch/" in f:
            continue
        r = torch.load(f, weights_only=False)
        r["cfg"].setdefault("family", "sylvester")
        recs.append(r)
    return recs


def pick(recs, **kw):
    out = []
    for r in recs:
        c = r["cfg"]
        if all(c.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def one(recs, **kw):
    rs = pick(recs, **kw)
    return rs[0] if rs else None


# ----------------------------------------------------------------------------- FLOP model
# Rough per-sample FLOP counts (transcendentals counted as 10 FLOPs). Derivation in REPORT.md 7.3.
FLOW_FWD_LAYER = {2: 83, 1: 45}       # one forward layer, D = M = 2 (or 1)
FLOW_MLE_LAYER = {2: 3300, 1: 1700}   # inverse (2 x 30 Newton iterations) + re-forward + backward


def mlp_fwd_flops(D, h, n):
    macs = (D + 9) * h + (n - 1) * h * h + h * D
    return 2 * macs + 6 * h * n + 2 * D + 80  # + bias/SiLU + time features


def train_flops(cfg):
    D = make_target(cfg["target"]).D
    batch = cfg.get("batch", 512)
    if cfg["family"] == "sylvester":
        per = cfg["L"] * (FLOW_MLE_LAYER[D] if cfg["obj"] == "mle" else 3 * FLOW_FWD_LAYER[D])
    else:
        per = 3 * mlp_fwd_flops(D, cfg["hidden"], cfg["n_hidden"])
    return per * batch * cfg["iters"]


def sample_flops(cfg, nfe=None):
    D = make_target(cfg["target"]).D
    if cfg["family"] == "sylvester":
        return cfg["L"] * FLOW_FWD_LAYER[D]
    return nfe * mlp_fwd_flops(D, cfg["hidden"], cfg["n_hidden"])


REF_FLOPS = train_flops(dict(target="sine2d:3", family="sylvester", L=64, obj="mle", iters=3000))


# ----------------------------------------------------------------------------- density images
def dens(rec, n=161):
    t = make_target(rec["cfg"]["target"])
    pts, _ = t.grid(n)
    m = load_flow(rec)
    with torch.no_grad():
        if rec["cfg"]["family"] == "sylvester":
            lq = torch.cat([m.log_prob(pts[i:i + 20000].float()) for i in range(0, len(pts), 20000)])
        else:
            lq = torch.cat([m.log_prob(pts[i:i + 20000].float(), 25) for i in range(0, len(pts), 20000)])
    return torch.exp(lq.double()).numpy().reshape(n, n)


def fig_sine(recs):
    cols = [("Sylvester flow · 1.2k params\nMLE, L=64, 3k steps", dict(family="sylvester", L=64, M=2, param="relaxed", obj="mle", seed=0, exp="e2_sine")),
            ("diffusion · 1.1k params\n45k steps (≈ ½ flow FLOPs)", dict(family="dm", hidden=27, iters=45000, seed=0)),
            ("flow matching · 1.1k params\n45k steps", dict(family="fm", hidden=27, iters=45000, seed=0)),
            ("diffusion · 35k params\n45k steps", dict(family="dm", hidden=128, iters=45000, seed=0)),
            ("flow matching · 35k params\n45k steps", dict(family="fm", hidden=128, iters=45000, seed=0))]
    rows = [2, 3, 4, 6]
    fig, axes = plt.subplots(len(rows), 1 + len(cols), figsize=(2.15 * (1 + len(cols)), 2.0 * len(rows) + 0.6))
    for i, P in enumerate(rows):
        spec = f"sine2d:{P}"
        p = target_image(spec, 161)
        show(axes[i, 0], p, f"target · {P} periods")
        for j, (lab, kw) in enumerate(cols):
            r = one(recs, target=spec, **kw)
            ax = axes[i, j + 1]
            if r is None:
                ax.axis("off"); continue
            m = r["metrics"]
            show(ax, dens(r), (lab + "\n" if i == 0 else "") + f"TV {m['tv']:.2f} · {m['cond_modes_found']}/{m['cond_modes_total']} bumps", vmax=p.max())
        for a in axes[i]:
            a.set_aspect("equal")
    fig.tight_layout()
    save(fig, "c_sine_gallery.png")


def fig_gallery(recs):
    cols = [("Sylvester · MLE L=64\n1.2k params", dict(family="sylvester", L=64, M=2, obj="mle", exp="e3_gallery")),
            ("diffusion · 1.1k\n60k steps", dict(family="dm", hidden=27, iters=60000)),
            ("flow matching · 1.1k\n60k steps", dict(family="fm", hidden=27, iters=60000)),
            ("diffusion · 35k\n60k steps", dict(family="dm", hidden=128, iters=60000)),
            ("flow matching · 35k\n60k steps", dict(family="fm", hidden=128, iters=60000))]
    fig, axes = plt.subplots(len(GAL), 1 + len(cols), figsize=(2.05 * (1 + len(cols)), 2.05 * len(GAL)))
    for i, t in enumerate(GAL):
        p = target_image(t, 161)
        show(axes[i, 0], p, make_target(t).name)
        for j, (lab, kw) in enumerate(cols):
            r = one(recs, target=t, **kw)
            ax = axes[i, j + 1]
            if r is None:
                ax.axis("off"); continue
            m = r["metrics"]
            extra = f" · {m['modes_covered']}/{m['modes_total']}" if "modes_total" in m else ""
            show(ax, dens(r), (lab + "\n" if i == 0 else "") + f"TV {m['tv']:.2f}{extra}", vmax=p.max())
        for a in axes[i]:
            a.set_aspect("equal")
    fig.tight_layout()
    save(fig, "c_gallery.png")


# ----------------------------------------------------------------------------- learning curves
def fig_curves(recs):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, P in zip(axes, [2, 3, 4, 6]):
        spec = f"sine2d:{P}"
        for fam in ["dm", "fm"]:
            for h, ls, mk, lab in [(27, "-", "o", "1.1k params"), (128, "--", "s", "35k params")]:
                rs = sorted(pick(recs, target=spec, family=fam, hidden=h, seed=0), key=lambda r: r["cfg"]["iters"])
                if not rs:
                    continue
                xs = [train_flops(r["cfg"]) / REF_FLOPS for r in rs]
                ys = [r["metrics"]["tv"] for r in rs]
                ax.plot(xs, ys, ls, marker=mk, color=FAM_COLOR[fam], ms=5, lw=1.8,
                        mfc=FAM_COLOR[fam] if h == 27 else SURFACE, label=f"{FAM_LABEL[fam]} · {lab}")
            seeds = pick(recs, target=spec, family=fam, hidden=27, iters=45000)
            if len(seeds) > 1:
                tv = [r["metrics"]["tv"] for r in seeds]
                x = train_flops(seeds[0]["cfg"]) / REF_FLOPS
                ax.vlines(x, min(tv), max(tv), color=FAM_COLOR[fam], lw=4, alpha=0.35)
        fl = pick(recs, target=spec, family="sylvester", L=64, M=2, param="relaxed", obj="mle")
        f1 = [r["metrics"]["tv"] for r in fl if r["cfg"]["iters"] == 3000]
        if f1:
            ax.vlines(1, min(f1), max(f1), color=C_FLOW, lw=4, alpha=0.35)
            ax.plot([1], [np.median(f1)], "D", color=C_FLOW, ms=7, label="Sylvester flow · 1.2k params (MLE, L=64)")
        f4 = [r for r in fl if r["cfg"]["iters"] == 12000]
        if f4:
            ax.plot([1, 4], [np.median(f1), f4[0]["metrics"]["tv"]], ":", color=C_FLOW, lw=1.5)
            ax.plot([4], [f4[0]["metrics"]["tv"]], "D", color=C_FLOW, ms=7, mfc=SURFACE)
        ax.set_xscale("log"); ax.set_ylim(0, 1); ax.grid(axis="y")
        ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (2, 3)))
        ax.set_title(f"{P} periods")
        ax.set_xlabel("training FLOPs (1 = one flow MLE run)")
    axes[0].set_ylabel("TV distance (lower is better)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Sine band: fit quality versus training compute", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "c_sine_curves.png")


# ----------------------------------------------------------------------------- sampling cost
def bin_probs(target, bins=80, sub=5):
    n = bins * sub
    c = (torch.arange(n, dtype=torch.float64) + 0.5) / n * 10 - 5
    X, Y = torch.meshgrid(c, c, indexing="xy")
    lp = target.logp(torch.stack([X.flatten(), Y.flatten()], -1).float()).double()
    p = torch.exp(lp - lp.max()).reshape(n, n)
    p = p.reshape(bins, sub, bins, sub).sum((1, 3))
    return (p / p.sum()).numpy()


def hist_tv(x, P):
    bins = P.shape[0]
    x = x.numpy()
    H, _, _ = np.histogram2d(x[:, 1], x[:, 0], bins=bins, range=[[-5, 5], [-5, 5]])
    return 0.5 * np.abs(H / len(x) - P).sum() + 0.5 * (1 - H.sum() / len(x))


def nfe_data(recs, n=200_000):
    cache = "results/c_nfe.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {}
    setups = [("sine2d:3", dict(exp="e2_sine", seed=0), 45000), ("grid16", dict(exp="e3_gallery"), 60000)]
    for spec, fkw, it in setups:
        t = make_target(spec)
        P = bin_probs(t)
        torch.manual_seed(0)
        floor = hist_tv(t.sample(n), P)
        d = dict(floor=floor, models={})
        fr = one(recs, target=spec, family="sylvester", L=64, M=2, param="relaxed", obj="mle", **fkw)
        with torch.no_grad():
            t0 = time.time(); x = load_flow(fr).sample(n)[0]; dt = time.time() - t0
        d["models"]["sylvester"] = [dict(nfe=1, flops=sample_flops(fr["cfg"]), tv=hist_tv(x, P), sec=dt)]
        for fam in ["dm", "fm"]:
            for h in [27, 128]:
                r = one(recs, target=spec, family=fam, hidden=h, iters=it, seed=0)
                m = load_flow(r)
                rows = []
                for method, steps_list, k in [("euler", [1, 2, 4, 8, 16, 32, 64, 128], 1), ("rk4", [1, 2, 4, 8, 16, 32], 4)]:
                    for s in steps_list:
                        torch.manual_seed(0)
                        t0 = time.time(); x = m.sample(n, s, method)[0]; dt = time.time() - t0
                        rows.append(dict(method=method, nfe=s * k, flops=sample_flops(r["cfg"], s * k),
                                         tv=hist_tv(x, P), sec=dt))
                d["models"][f"{fam}_{h}"] = rows
                print(spec, fam, h, "done", flush=True)
        out[spec] = d
    json.dump(out, open(cache, "w"), indent=1)
    return out


def fig_nfe(recs):
    data = nfe_data(recs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (spec, d) in zip(axes, data.items()):
        for fam in ["dm", "fm"]:
            for h, mfc, lab in [(27, None, "1.1k"), (128, SURFACE, "35k")]:
                rows = d["models"].get(f"{fam}_{h}", [])
                for method, ls in [("euler", "-"), ("rk4", "--")]:
                    rr = [r for r in rows if r["method"] == method]
                    if rr:
                        ax.plot([r["flops"] for r in rr], [r["tv"] for r in rr], ls, marker="o", ms=4, lw=1.5,
                                color=FAM_COLOR[fam], mfc=mfc or FAM_COLOR[fam],
                                label=f"{FAM_LABEL[fam]} · {lab} · {method}")
        s = d["models"]["sylvester"][0]
        ax.plot([s["flops"]], [s["tv"]], "D", color=C_FLOW, ms=9, label="Sylvester flow (one pass)", zorder=5)
        ax.axhline(d["floor"], color=MUTED, lw=0.8, ls=":")
        ax.text(ax.get_xlim()[0] if False else 3e3, d["floor"] + 0.01, "sampling-noise floor", fontsize=7.5, color=INK2)
        ax.set_xscale("log"); ax.set_ylim(0, 1); ax.grid(axis="y")
        ax.set_xlabel("FLOPs to draw one sample (log scale)")
        ax.set_title(make_target(spec).name if spec != "sine2d:3" else "sine, 3 periods")
    axes[0].set_ylabel("TV from 200k samples")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Sampling cost: quality versus compute per sample", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "c_sampling_cost.png")


# ----------------------------------------------------------------------------- tables
def tables(recs):
    lines = []
    lines.append("SINE (TV, cond bumps) | flow MLE L64 (3 seeds) | DM 1.1k @45k (3 seeds) | FM 1.1k @45k (3 seeds) | DM 1.1k @150k | FM 1.1k @150k | DM 35k @45k | FM 35k @45k | flow MLE L64 12k")
    for P in PERIODS:
        spec = f"sine2d:{P}"
        def agg(rs):
            if not rs:
                return "-"
            tv = [r["metrics"]["tv"] for r in rs]
            cm = [r["metrics"]["cond_modes_found"] for r in rs]
            tot = rs[0]["metrics"]["cond_modes_total"]
            return f"{np.mean(tv):.2f}±{np.std(tv):.2f} ({np.mean(cm):.0f}/{tot})" if len(rs) > 1 else f"{tv[0]:.2f} ({cm[0]}/{tot})"
        row = [agg(pick(recs, target=spec, family="sylvester", L=64, M=2, param="relaxed", obj="mle", iters=3000)),
               agg(pick(recs, target=spec, family="dm", hidden=27, iters=45000)),
               agg(pick(recs, target=spec, family="fm", hidden=27, iters=45000)),
               agg(pick(recs, target=spec, family="dm", hidden=27, iters=150000)),
               agg(pick(recs, target=spec, family="fm", hidden=27, iters=150000)),
               agg(pick(recs, target=spec, family="dm", hidden=128, iters=45000)),
               agg(pick(recs, target=spec, family="fm", hidden=128, iters=45000)),
               agg(pick(recs, target=spec, family="sylvester", obj="mle", iters=12000))]
        lines.append(f"P={P}: " + " | ".join(row))
    lines.append("")
    lines.append("SINE rKL flow L64 (3 seeds): " + " | ".join(
        f"P={P}: {np.mean([r['metrics']['tv'] for r in pick(recs, target=f'sine2d:{P}', family='sylvester', L=64, M=2, param='relaxed', obj='rkl')]):.2f}" for P in PERIODS))
    lines.append("")
    lines.append("GALLERY TV (bumps) | flow MLE L64 | flow MLE L16 | DM 1.1k@4k | FM 1.1k@4k | DM 1.1k@60k | FM 1.1k@60k | DM 288@60k | FM 288@60k | DM 35k@60k | FM 35k@60k")
    for t in GAL:
        def g(**kw):
            r = one(recs, target=t, **kw)
            if r is None:
                return "-"
            m = r["metrics"]
            return f"{m['tv']:.2f}" + (f" ({m['modes_covered']}/{m['modes_total']})" if "modes_total" in m else "")
        lines.append(f"{t}: " + " | ".join([
            g(family="sylvester", L=64, M=2, obj="mle", exp="e3_gallery"), g(family="sylvester", L=16, M=2, obj="mle"),
            g(family="dm", hidden=27, iters=4000), g(family="fm", hidden=27, iters=4000),
            g(family="dm", hidden=27, iters=60000), g(family="fm", hidden=27, iters=60000),
            g(family="dm", hidden=11, iters=60000), g(family="fm", hidden=11, iters=60000),
            g(family="dm", hidden=128, iters=60000), g(family="fm", hidden=128, iters=60000)]))
    lines.append("")
    lines.append("1D TV (bumps) | flow MLE L32 | DM 199@2.5k | FM 199@2.5k | DM 199@37.5k | FM 199@37.5k | DM 35k@37.5k | FM 35k@37.5k")
    for t in ["mix1d:4", "mix1d:6", "mix1d:8", "sine1d"]:
        def g(**kw):
            r = one(recs, target=t, **kw)
            if r is None:
                return "-"
            m = r["metrics"]
            return f"{m['tv']:.2f}" + (f" ({m['modes_covered']}/{m['modes_total']})" if "modes_total" in m else "")
        lines.append(f"{t}: " + " | ".join([
            g(family="sylvester", L=32, obj="mle", param="relaxed"),
            g(family="dm", hidden=9, iters=2500), g(family="fm", hidden=9, iters=2500),
            g(family="dm", hidden=9, iters=37500), g(family="fm", hidden=9, iters=37500),
            g(family="dm", hidden=128, iters=37500), g(family="fm", hidden=128, iters=37500)]))
    lines.append("")
    lines.append("diverged_where_p (baselines, max over runs): " + str(max(
        (r["metrics"].get("diverged_where_p", 0) for r in recs if r["cfg"]["family"] != "sylvester"), default=0)))
    lines.append(f"REF training FLOPs (flow MLE L64, 3k steps): {REF_FLOPS:.3g}")
    for c in [dict(target="sine2d:3", family="dm", hidden=27, n_hidden=2, iters=it) for it in (3000, 15000, 45000, 150000)]:
        lines.append(f"  DM/FM 1.1k {c['iters']} steps = {train_flops(c) / REF_FLOPS:.3f} x REF")
    for c in [dict(target="sine2d:3", family="dm", hidden=128, n_hidden=3, iters=it) for it in (3000, 45000)]:
        lines.append(f"  DM/FM 35k {c['iters']} steps = {train_flops(c) / REF_FLOPS:.3f} x REF")
    lines.append(f"  flow rKL L64 3k = {train_flops(dict(target='sine2d:3', family='sylvester', L=64, obj='rkl', iters=3000)) / REF_FLOPS:.3f} x REF")
    lines.append(f"sample FLOPs: flow L64 {sample_flops(dict(target='sine2d:3', family='sylvester', L=64))}, MLP 1.1k per NFE {mlp_fwd_flops(2, 27, 2)}, MLP 35k per NFE {mlp_fwd_flops(2, 128, 3)}")
    txt = "\n".join(lines)
    open("results/compare_tables.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    recs = load_all()
    for k, fn in [("table", tables), ("curves", fig_curves), ("sine", fig_sine), ("gallery", fig_gallery), ("nfe", fig_nfe)]:
        if which in (k, "all"):
            fn(recs)
