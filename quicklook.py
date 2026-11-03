"""Quick look: target vs learned density for every run in results/<exp>/ -> results/<exp>_quicklook.png"""
import glob, sys, math, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from train import load_flow, model_grid_logq
from targets import make_target
exp = sys.argv[1]
files = sorted(glob.glob(f"results/{exp}/*.pt"))
fig, axes = plt.subplots(len(files), 2, figsize=(7, 3.2 * len(files)), squeeze=False)
for row, f in zip(axes, files):
    rec = torch.load(f, weights_only=False); flow = load_flow(rec); t = make_target(rec["cfg"]["target"])
    pts, p = t.grid_density(161 if t.D == 2 else None); q = torch.exp(model_grid_logq(flow, pts))
    for ax, d, ttl in zip(row, (p, q), ("target", f.split('/')[-1][:-3] + f"\nTV={rec['metrics']['tv']:.2f}")):
        if t.D == 1: ax.plot(pts[:, 0], d)
        else: n = int(math.sqrt(len(d))); ax.imshow(d.reshape(n, n), origin="lower", extent=[-5, 5, -5, 5])
        ax.set_title(ttl, fontsize=8)
fig.tight_layout(); fig.savefig(f"results/{exp}_quicklook.png", dpi=70)
