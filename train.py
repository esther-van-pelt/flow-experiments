"""Training + evaluation of one configuration, and a multiprocessing sweep runner.

Objectives
  rkl : reverse KL  E_q[log q(x) - beta * log p~(x)], x = f(z), z ~ N(0, I). Only needs the
        (unnormalised) target density. beta is annealed 0.05 -> 1 over the first 40% of training
        (tempering, which counters the mode-seeking behaviour of reverse KL).
  mle : forward KL / maximum likelihood on samples from the target, using the exact inverse.

Families (cfg["family"], default "sylvester")
  sylvester : orthogonal Sylvester flow (flows.py), objective rkl or mle
  dm / fm   : diffusion / flow-matching baselines (baselines.py), trained with their own regression
              loss on the same 200k-sample pool, batch, optimiser and schedule as the MLE flows
"""
import json
import math
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import torch

from baselines import Diffusion, FlowMatching
from flows import Flow
from targets import make_target

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_name(cfg):
    fam = cfg.get("family", "sylvester")
    if fam == "sylvester":
        return f"{cfg['target'].replace(':', '')}_L{cfg['L']}_M{cfg['M']}_{cfg['param']}_{cfg['obj']}_s{cfg['seed']}"
    return f"{cfg['target'].replace(':', '')}_{fam}_h{cfg['hidden']}x{cfg['n_hidden']}_it{cfg['iters']}_s{cfg['seed']}"


def build_model(cfg, D):
    fam = cfg.get("family", "sylvester")
    if fam == "sylvester":
        return Flow(D, cfg["L"], cfg["M"], cfg["param"])
    cls = {"dm": Diffusion, "fm": FlowMatching}[fam]
    return cls(D, cfg["hidden"], cfg["n_hidden"])


def train(cfg):
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    target = make_target(cfg["target"])
    D = target.D
    fam = cfg.get("family", "sylvester")
    flow = build_model(cfg, D)
    iters, batch = cfg["iters"], cfg.get("batch", 512)
    opt = torch.optim.Adam(flow.parameters(), lr=cfg.get("lr", 3e-3))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 3e-3), total_steps=iters,
                                                pct_start=0.05)
    pool = target.sample(200_000) if fam != "sylvester" or cfg["obj"] == "mle" else None
    anneal = int(cfg.get("anneal", 0.4) * iters)
    hist, t0, skipped = [], time.time(), 0
    for it in range(iters):
        if fam != "sylvester":
            loss = flow.loss(pool[torch.randint(len(pool), (batch,))])
        elif cfg["obj"] == "rkl":
            beta = min(1.0, 0.05 + 0.95 * it / max(anneal, 1)) if anneal else 1.0
            x, logq = flow.sample(batch)
            loss = (logq - beta * target.logp(x)).mean()
        else:
            x = pool[torch.randint(len(pool), (batch,))]
            loss = -flow.log_prob(x).mean()
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            skipped += 1
        sched.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append((it, loss.item()))
    train_time = time.time() - t0
    metrics = evaluate(flow, target, cfg)
    metrics.update(train_time=train_time, skipped=skipped,
                   orth_err=flow.orth_error() if fam == "sylvester" else 0.0,
                   n_params=sum(p.numel() for p in flow.parameters()))
    return flow, target, metrics, hist


@torch.no_grad()
def model_grid_logq(flow, pts, chunk=20000):
    return torch.cat([flow.log_prob(pts[i:i + chunk].float()).double() for i in range(0, len(pts), chunk)])


@torch.no_grad()
def evaluate(flow, target, cfg):
    pts, p = target.grid_density()
    _, g = target.grid()
    cell = (g[1] - g[0]).item() ** target.D
    logq = model_grid_logq(flow, pts)
    q = torch.exp(logq)
    q_mass = (q.sum() * cell).item()
    tv = 0.5 * ((p - q).abs().sum() * cell).item() + 0.5 * max(0.0, 1 - q_mass)
    m = p > 1e-12
    kl_pq = (p[m] * (torch.log(p[m]) - logq[m]) * cell).sum().item()
    out = dict(tv=tv, kl_pq=kl_pq, q_mass_in_box=q_mass,
               diverged_where_p=int(((logq <= -1e4 + 1) & (p > 1e-6 * p.max())).sum()))
    if target.modes is not None:
        modes = torch.as_tensor(target.modes, dtype=torch.float64)
        sig = getattr(target, "sigma", 0.3)
        # a mode is "recovered" if q has a local maximum (height >= 5% of max q) within sigma of it
        peaks = local_maxima(q.numpy().reshape([len(g)] * target.D), g.numpy())
        covered = []
        for mu in modes.numpy():
            covered.append(bool(len(peaks) and np.min(np.linalg.norm(peaks - mu, axis=1)) <= max(sig, 0.15)))
        out["modes_total"] = len(covered)
        out["modes_covered"] = int(sum(covered))
    if target.D == 1:
        out["q_peaks"] = count_peaks(q.numpy())
        out["p_peaks"] = count_peaks(p.numpy())
    if cfg["target"].startswith("sine2d"):
        out.update(sine_conditional_metrics(target, pts, p, q, g))
    return out


def local_maxima(f, g, rel_height=0.05):
    """Coordinates of local maxima of a gridded density (1D or 2D, 2D stored as [y, x])."""
    from scipy.ndimage import maximum_filter
    mask = (f == maximum_filter(f, size=5, mode="constant")) & (f >= rel_height * f.max())
    idx = np.argwhere(mask)
    if f.ndim == 1:
        return g[idx[:, 0]][:, None]
    return np.c_[g[idx[:, 1]], g[idx[:, 0]]]


def count_peaks(f, rel_height=0.05, rel_prom=0.05):
    """Local maxima with height and prominence >= 5% of the global max."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(f, height=rel_height * f.max(), prominence=rel_prom * f.max())
    return int(len(pk))


def sine_conditional_metrics(target, pts, p, q, g):
    """For outcome levels y0 the conditional p(angle | y0) has 2*periods modes.
    Count how many of them appear as peaks (within 0.35 in angle) of q(angle, y0)."""
    from scipy.signal import find_peaks
    n = len(g)
    P, Q = p.reshape(n, n).numpy(), q.reshape(n, n).numpy()  # rows: y, cols: angle
    gx = g.numpy()
    found, total = 0, 0
    for y0 in (-1.0, 1.0):
        r = int(np.argmin(np.abs(gx - y0)))
        tp, _ = find_peaks(P[r], height=0.2 * P[r].max(), prominence=0.1 * P[r].max())
        qp, _ = find_peaks(Q[r], height=0.1 * max(Q[r].max(), 1e-30), prominence=0.05 * max(Q[r].max(), 1e-30))
        total += len(tp)
        for t in tp:
            if len(qp) and np.min(np.abs(gx[qp] - gx[t])) < 0.35:
                found += 1
    return dict(cond_modes_total=total, cond_modes_found=found)


def run_and_save(cfg):
    name = run_name(cfg)
    d = os.path.join(RESULTS, cfg["exp"])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".pt")
    if os.path.exists(path) and not cfg.get("force"):
        return name, "cached"
    flow, target, metrics, hist = train(cfg)
    torch.save(dict(cfg=cfg, state=flow.state_dict(), metrics=metrics, hist=hist), path)
    return name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}


def load_flow(rec):
    cfg = rec["cfg"]
    model = build_model(cfg, make_target(cfg["target"]).D)
    model.load_state_dict(rec["state"])
    return model


def sweep(cfgs, nproc=None):
    nproc = nproc or min(len(cfgs), max(1, os.cpu_count() - 2))
    # longest jobs first for better packing
    cfgs = sorted(cfgs, key=lambda c: -(c.get("L", 8) * c["iters"] * (6 if c.get("obj") == "mle" else 1)
                                         + c.get("hidden", 0) ** 2 * c.get("n_hidden", 0) * 2))
    t0 = time.time()
    with get_context("spawn").Pool(nproc) as pool:
        for i, (name, res) in enumerate(pool.imap_unordered(run_and_save, cfgs)):
            print(f"[{i + 1}/{len(cfgs)} {time.time() - t0:6.0f}s] {name}: {res}", flush=True)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    cfg.setdefault("exp", "scratch")
    cfg["force"] = True
    print(run_and_save(cfg))
