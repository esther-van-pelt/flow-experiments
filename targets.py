"""Toy target densities. Each target exposes an (unnormalised) log density, an exact-ish sampler
(rejection from a bounding box), a grid for evaluation and, for mixture targets, mode centres."""
import math

import numpy as np
import torch

BOX = 5.0  # evaluation / rejection box is [-BOX, BOX]^D


class Target:
    D = 2
    modes = None  # (K, D) array of mode centres for mode-coverage metrics, if meaningful

    def logp(self, x):
        raise NotImplementedError

    def grid(self, n=None):
        n = n or (4001 if self.D == 1 else 241)
        g = torch.linspace(-BOX, BOX, n, dtype=torch.float64)
        if self.D == 1:
            return g[:, None], g
        X, Y = torch.meshgrid(g, g, indexing="xy")
        return torch.stack([X.flatten(), Y.flatten()], -1), g

    def grid_density(self, n=None):
        pts, g = self.grid(n)
        with torch.no_grad():
            lp = self.logp(pts.float()).double()
        p = torch.exp(lp - lp.max())
        cell = (g[1] - g[0]) ** self.D
        p = p / (p.sum() * cell)
        return pts, p

    def sample(self, n, rng=None):
        if not hasattr(self, "_lmax"):
            pts, _ = self.grid()
            with torch.no_grad():
                lp = self.logp(pts.float())
            self._lmax = lp.max().item() + 0.05
        out, have = [], 0
        while have < n:
            x = (torch.rand(4 * n, self.D) * 2 - 1) * BOX
            keep = torch.log(torch.rand(4 * n)) < self.logp(x) - self._lmax
            out.append(x[keep]); have += int(keep.sum())
        return torch.cat(out)[:n]


def _lse_gauss(x, centres, sigma, weights=None):
    d2 = torch.cdist(x, centres) ** 2
    lw = 0.0 if weights is None else torch.log(weights)[None]
    return torch.logsumexp(-0.5 * d2 / sigma ** 2 + lw, -1)


class Mixture(Target):
    def __init__(self, centres, sigma, name, weights=None):
        self.centres = torch.as_tensor(np.asarray(centres), dtype=torch.float32)
        self.D = self.centres.shape[1]
        self.sigma, self.name = sigma, name
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        self.modes = self.centres.numpy()

    def logp(self, x):
        return _lse_gauss(x, self.centres.to(x.dtype), self.sigma,
                          None if self.weights is None else self.weights.to(x.dtype))


def gauss_mixture_1d(K):
    mu = np.linspace(-3.5, 3.5, K) if K > 1 else np.zeros(1)
    spacing = 7.0 / max(K - 1, 1)
    return Mixture(mu[:, None], min(0.5, spacing / 6), f"{K} Gaussians")


class NoisySine1D(Mixture):
    """Outcome y = 2 sin(theta) + eps, theta ~ U[0, 2pi), eps ~ N(0, 0.15^2): an arcsine density
    smoothed by noise -> bimodal with peaks at +-2. Built as a dense Gaussian mixture over theta."""

    def __init__(self):
        th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        super().__init__((2 * np.sin(th))[:, None], 0.15, "outcome 2 sin(θ) + noise")
        self.modes = np.array([[-2.0], [2.0]])


class Sine2D(Target):
    """Joint density of (angle, outcome): x ~ smooth box on [-4, 4] covering `periods` periods,
    y | x ~ N(2 sin(w x), 0.2^2). For |y| < 2 the conditional p(x | y) has 2*periods modes."""

    def __init__(self, periods, amp=2.0, sigma=0.2):
        self.periods, self.amp, self.sigma = periods, amp, sigma
        self.w = 2 * math.pi * periods / 8.0
        self.name = f"sine, {periods} period{'s' if periods != 1 else ''}"

    def logp(self, x):
        t, y = x[:, 0], x[:, 1]
        box = torch.nn.functional.logsigmoid(8 * (t + 4)) + torch.nn.functional.logsigmoid(8 * (4 - t))
        return box - 0.5 * ((y - self.amp * torch.sin(self.w * t)) / self.sigma) ** 2


class Curve(Mixture):
    """Density concentrated around a curve: dense Gaussian mixture along sampled curve points."""

    def __init__(self, pts, sigma, name, weights=None):
        super().__init__(pts, sigma, name, weights)
        self.modes = None


def ring8():
    a = np.arange(8) * 2 * np.pi / 8
    return Mixture(np.c_[3 * np.cos(a), 3 * np.sin(a)], 0.3, "8 Gaussians (ring)")


def two_gauss():
    return Mixture(np.array([[-2.0, 0.0], [2.0, 0.0]]), 0.5, "2 Gaussians")


def grid16():
    g = np.array([-3.0, -1.0, 1.0, 3.0])
    X, Y = np.meshgrid(g, g)
    return Mixture(np.c_[X.ravel(), Y.ravel()], 0.25, "4x4 grid")


def moons():
    t = np.linspace(0, np.pi, 200)
    upper = np.c_[np.cos(t), np.sin(t)]
    lower = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    pts = (np.r_[upper, lower] - [0.5, 0.25]) * 2.0
    return Curve(pts, 0.15, "two moons")


def rings():
    pts, w = [], []
    for r in (1.2, 2.4, 3.6):
        a = np.linspace(0, 2 * np.pi, int(120 * r), endpoint=False)
        pts.append(np.c_[r * np.cos(a), r * np.sin(a)])
    return Curve(np.concatenate(pts), 0.15, "3 rings")


def spiral():
    t = np.linspace(0.6, 3.6 * np.pi, 600)
    r = 0.37 * t
    return Curve(np.c_[r * np.cos(t), r * np.sin(t)], 0.18, "spiral")


class Checkerboard(Target):
    """4x4 checkerboard on [-4, 4]^2 with softened edges."""
    name = "checkerboard"

    def logp(self, x):
        ls = torch.nn.functional.logsigmoid
        k = 12.0
        terms = []
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    x0, y0 = -4 + 2 * i, -4 + 2 * j
                    terms.append(ls(k * (x[:, 0] - x0)) + ls(k * (x0 + 2 - x[:, 0]))
                                 + ls(k * (x[:, 1] - y0)) + ls(k * (y0 + 2 - x[:, 1])))
        return torch.logsumexp(torch.stack(terms, -1), -1)


GALLERY = {
    "two_gauss": two_gauss, "moons": moons, "ring8": ring8, "rings": rings,
    "grid16": grid16, "spiral": spiral, "checkerboard": Checkerboard,
}


def make_target(spec):
    kind, *rest = spec.split(":")
    if kind == "mix1d":
        return gauss_mixture_1d(int(rest[0]))
    if kind == "sine1d":
        return NoisySine1D()
    if kind == "sine2d":
        return Sine2D(int(rest[0]))
    return GALLERY[kind]()
