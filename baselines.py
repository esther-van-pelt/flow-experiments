"""Diffusion (VP-SDE, epsilon-prediction) and flow matching (rectified flow) baselines.

Both are written as a learned time-dependent velocity field v(x, t) that transports data (t = t0)
to N(0, I) (t = 1), so they share
  * the network (a small MLP on [x, t, Fourier features of t]),
  * the sampler (RK4 on dx/dt = v, integrated from t = 1 down to t0),
  * the exact density: log q(x) = log N(x(1); 0, I) + int_{t0}^{1} div v(x(t), t) dt,
    computed with RK4 and an exact divergence (D <= 2: one autograd pass per dimension).
This is the "probability-flow ODE" density. It is what the ODE sampler actually samples from, so
TV on a grid is computed exactly as for the Sylvester flows.
"""
import math

import torch
import torch.nn as nn

LOG2PI = math.log(2 * math.pi)


class MLP(nn.Module):
    def __init__(self, D, hidden, n_hidden=2, n_freq=4):
        super().__init__()
        self.register_buffer("freqs", 2 * math.pi * torch.arange(1, n_freq + 1, dtype=torch.float32))
        d_in = D + 1 + 2 * n_freq
        layers = []
        for i in range(n_hidden):
            layers += [nn.Linear(d_in if i == 0 else hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        tf = t[:, None] * self.freqs
        return self.net(torch.cat([x, t[:, None], torch.sin(tf), torch.cos(tf)], -1))


class ODEModel(nn.Module):
    t0 = 0.0

    def __init__(self, D, hidden, n_hidden=2):
        super().__init__()
        self.D = D
        self.net = MLP(D, hidden, n_hidden)

    def velocity(self, x, t):
        raise NotImplementedError

    def loss(self, x0):
        raise NotImplementedError

    def _v(self, x, t):
        return self.velocity(x, torch.full((x.shape[0],), float(t), dtype=x.dtype))

    @torch.no_grad()
    def sample(self, n, steps=50, method="rk4"):
        """ODE sampler from noise (t = 1) to data (t = t0). NFE = steps (euler) or 4 * steps (rk4)."""
        x = torch.randn(n, self.D)
        ts = torch.linspace(1.0, self.t0, steps + 1)
        for a, b in zip(ts[:-1], ts[1:]):
            h = float(b - a)
            if method == "euler":
                x = x + h * self._v(x, a)
                continue
            k1 = self._v(x, a)
            k2 = self._v(x + 0.5 * h * k1, a + 0.5 * h)
            k3 = self._v(x + 0.5 * h * k2, a + 0.5 * h)
            k4 = self._v(x + h * k3, b)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x, None

    def _v_div(self, x, t):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            v = self._v(x, t)
            div = torch.zeros(x.shape[0], dtype=x.dtype)
            for i in range(self.D):
                div = div + torch.autograd.grad(v[:, i].sum(), x, retain_graph=i < self.D - 1)[0][:, i]
        return v.detach(), div.detach()

    @torch.no_grad()
    def log_prob(self, x, steps=50):
        """Exact (up to RK4 error) log density of the probability-flow ODE model."""
        ts = torch.linspace(self.t0, 1.0, steps + 1)
        ld = torch.zeros(x.shape[0], dtype=x.dtype)
        for a, b in zip(ts[:-1], ts[1:]):
            h = float(b - a)
            k1, d1 = self._v_div(x, a)
            k2, d2 = self._v_div(x + 0.5 * h * k1, a + 0.5 * h)
            k3, d3 = self._v_div(x + 0.5 * h * k2, a + 0.5 * h)
            k4, d4 = self._v_div(x + h * k3, b)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            ld = ld + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        lq = -0.5 * (x ** 2).sum(-1) - 0.5 * self.D * LOG2PI + ld
        # Far outside the data the learned field extrapolates wildly and some trajectories diverge
        # (log q ~ -1e75 or overflow). The model density there is effectively 0: clamp, don't NaN.
        return torch.nan_to_num(lq, nan=-1e4, neginf=-1e4).clamp(min=-1e4)


class FlowMatching(ODEModel):
    """Rectified flow / conditional flow matching with the straight path x_t = (1 - t) x0 + t eps."""
    t0 = 0.0

    def velocity(self, x, t):
        return self.net(x, t)

    def loss(self, x0):
        eps = torch.randn_like(x0)
        t = torch.rand(x0.shape[0])
        xt = (1 - t[:, None]) * x0 + t[:, None] * eps
        return ((self.net(xt, t) - (eps - x0)) ** 2).sum(-1).mean()


class Diffusion(ODEModel):
    """VP-SDE (Song et al. 2021) with linear beta(t) in [0.1, 20], epsilon-prediction, trained with
    the standard DDPM loss. velocity() is the probability-flow ODE drift
        dx/dt = -1/2 beta(t) x - 1/2 beta(t) score,   score = -eps_hat / sigma_t."""
    t0 = 1e-3
    b0, b1 = 0.1, 20.0

    def _alpha_sigma(self, t):
        log_a = -0.25 * t ** 2 * (self.b1 - self.b0) - 0.5 * t * self.b0
        a = torch.exp(log_a)
        return a, torch.sqrt(1 - a ** 2)

    def velocity(self, x, t):
        beta = (self.b0 + t * (self.b1 - self.b0))[:, None]
        _, sig = self._alpha_sigma(t)
        return -0.5 * beta * x + 0.5 * beta * self.net(x, t) / sig[:, None]

    def loss(self, x0):
        eps = torch.randn_like(x0)
        t = self.t0 + (1 - self.t0) * torch.rand(x0.shape[0])
        a, s = self._alpha_sigma(t)
        xt = a[:, None] * x0 + s[:, None] * eps
        return ((self.net(xt, t) - eps) ** 2).sum(-1).mean()


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
