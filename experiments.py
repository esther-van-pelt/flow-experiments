"""All sweeps. Usage: python experiments.py [e1|e2|e3|all]"""
import sys
from itertools import product
from train import sweep


def e1():  # 1D: how many modes per number of layers?
    cfgs = []
    for tgt, L, obj, param in product(["mix1d:2", "mix1d:3", "mix1d:4", "mix1d:6", "mix1d:8", "sine1d"],
                                      [1, 2, 4, 8, 16, 32], ["mle", "rkl"], ["relaxed", "paper"]):
        cfgs.append(dict(exp="e1_1d", target=tgt, L=L, M=1, param=param, obj=obj, seed=0, iters=2500))
    return cfgs


def e2():  # 2D angle/outcome sine with increasing number of periods
    cfgs = []
    for P, L, obj in product([1, 2, 3, 4, 6], [8, 16, 32, 64], ["mle", "rkl"]):
        cfgs.append(dict(exp="e2_sine", target=f"sine2d:{P}", L=L, M=2, param="relaxed", obj=obj, seed=0, iters=3000))
    for P, L, obj in product([1, 2, 3, 4, 6], [32, 64], ["mle", "rkl"]):
        cfgs.append(dict(exp="e2_sine", target=f"sine2d:{P}", L=L, M=2, param="paper", obj=obj, seed=0, iters=3000))
    return cfgs


def e3():  # 2D gallery of increasingly complex multimodal targets
    cfgs = []
    tg = ["two_gauss", "moons", "ring8", "rings", "grid16", "spiral", "checkerboard"]
    for t, L, obj in product(tg, [16, 64], ["mle", "rkl"]):
        cfgs.append(dict(exp="e3_gallery", target=t, L=L, M=2, param="relaxed", obj=obj, seed=0, iters=4000))
    for t, obj in product(tg, ["mle", "rkl"]):
        cfgs.append(dict(exp="e3_gallery", target=t, L=64, M=1, param="relaxed", obj=obj, seed=0, iters=4000))
    return cfgs


# ------------------------------------------------------------------ comparison with DM / FM
# Size tiers (trainable parameters):   flow L=64, M=2: 1152   |  MLP h=27x2: 1136   ("S", matched)
#                                       flow L=16, M=2:  288   |  MLP h=11x2:  288   ("XS", matched)
#                                       flow 1D, L=32:   192   |  MLP h=9x2:   199   ("S" in 1D)
#                                       -                      |  MLP h=128x3: ~35k  ("big", reference)
# Training budgets: 1x = same steps as the flows; 15x ~ same training FLOPs as an MLE flow step
# (inverse + backward); 50x = generous.
def _bl(exp, target, fam, h, n, iters, seed=0):
    return dict(exp=exp, target=target, family=fam, hidden=h, n_hidden=n, iters=iters, seed=seed)


def c_sine():
    cfgs = []
    for P, fam in product([1, 2, 3, 4, 6], ["dm", "fm"]):
        t = f"sine2d:{P}"
        for mult in [1, 5, 15, 50]:
            cfgs.append(_bl("c_sine", t, fam, 27, 2, 3000 * mult))
        for seed in [1, 2]:
            cfgs.append(_bl("c_sine", t, fam, 27, 2, 45000, seed))
        cfgs.append(_bl("c_sine", t, fam, 11, 2, 45000))
        for mult in [1, 15]:
            cfgs.append(_bl("c_sine", t, fam, 128, 3, 3000 * mult))
    # extra seeds for the flows (same settings as E2), and 4x longer flow training on the hard cases
    for P, obj, seed in product([1, 2, 3, 4, 6], ["mle", "rkl"], [1, 2]):
        cfgs.append(dict(exp="e2_sine", target=f"sine2d:{P}", L=64, M=2, param="relaxed", obj=obj, seed=seed, iters=3000))
    for P in [3, 4, 6]:
        cfgs.append(dict(exp="c_sine_flowlong", target=f"sine2d:{P}", L=64, M=2, param="relaxed", obj="mle", seed=0, iters=12000))
    return cfgs


def c_gallery():
    cfgs = []
    for t, fam in product(["two_gauss", "moons", "ring8", "rings", "grid16", "spiral", "checkerboard"], ["dm", "fm"]):
        cfgs += [_bl("c_gallery", t, fam, 27, 2, 4000), _bl("c_gallery", t, fam, 27, 2, 60000),
                 _bl("c_gallery", t, fam, 11, 2, 60000),
                 _bl("c_gallery", t, fam, 128, 3, 4000), _bl("c_gallery", t, fam, 128, 3, 60000)]
    return cfgs


def c_1d():
    cfgs = []
    for t, fam in product(["mix1d:4", "mix1d:6", "mix1d:8", "sine1d"], ["dm", "fm"]):
        cfgs += [_bl("c_1d", t, fam, 9, 2, 2500), _bl("c_1d", t, fam, 9, 2, 37500),
                 _bl("c_1d", t, fam, 128, 3, 37500)]
    return cfgs


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfgs = {"e1": e1, "e2": e2, "e3": e3, "c_sine": c_sine, "c_gallery": c_gallery, "c_1d": c_1d}
    groups = {"all": ["e1", "e2", "e3"], "compare": ["c_sine", "c_gallery", "c_1d"]}
    todo = [c for k, f in cfgs.items() if k in groups.get(which, [which]) for c in f()]
    print(len(todo), "runs", flush=True)
    sweep(todo, nproc=12)
