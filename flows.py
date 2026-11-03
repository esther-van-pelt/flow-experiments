"""Orthogonal Sylvester normalizing flows (van den Berg et al., 2018), non-amortized.

One layer:   z' = z + Q R h(R~ Q^T z + b),   h = tanh
  Q  : D x M with orthonormal columns (Bjorck iterative orthogonalisation, as in the paper)
  R, R~ : M x M upper triangular
  log|det J| = sum_i log(1 + h'(.)_i * R_ii * R~_ii)   (needs R_ii R~_ii > -1)

Two parametrisations of the diagonals:
  "paper"   : R_ii = tanh(a_i), R~_ii = tanh(c_i)            (as in the reference code; |R_ii|,|R~_ii| < 1)
  "relaxed" : R~_ii = exp(c_i) > 0, R_ii = (softplus(a_i) - 1 + eps) / R~_ii
              -> product in (-1+eps, inf); any sign pattern is reachable (h is odd, so the sign of
              R~_ii can be flipped together with row i of R~, b_i and column i of R).

Exact inverse: because R and R~ are upper triangular, coordinate v_M of v = Q^T z only depends on
itself, v_{M-1} on (v_{M-1}, v_M), ... so the inverse is M scalar, strictly monotone root-finds per
layer (back substitution). The component of z orthogonal to span(Q) is unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def orthogonalize(W, n_iter=25):
    """Bjorck iteration Q <- Q (I + 1/2 (I - Q^T Q)), started from W / ||W||_F."""
    M = W.shape[1]
    I = torch.eye(M, dtype=W.dtype)
    Q = W / W.norm()
    for _ in range(n_iter):
        Q = Q @ (1.5 * I - 0.5 * Q.T @ Q)
    return Q


class OrthoSylvester(nn.Module):
    def __init__(self, D, M, param="relaxed", eps=1e-3):
        super().__init__()
        assert 1 <= M <= D
        self.D, self.M, self.param, self.eps = D, M, param, eps
        self.W = nn.Parameter(torch.randn(D, M))
        self.R_off = nn.Parameter(0.1 * torch.randn(M, M))
        self.Rt_off = nn.Parameter(0.1 * torch.randn(M, M))
        if param == "paper":
            self.a = nn.Parameter(0.1 * torch.randn(M))
            self.c = nn.Parameter(0.1 * torch.randn(M))
        else:
            self.a = nn.Parameter(math.log(math.e - 1) + 0.1 * torch.randn(M))  # product ~ 0
            self.c = nn.Parameter(0.1 * torch.randn(M))
        self.b = nn.Parameter(torch.empty(M).uniform_(-2, 2))
        self.register_buffer("triu", torch.triu(torch.ones(M, M), 1))

    def mats(self):
        Q = orthogonalize(self.W)
        if self.param == "paper":
            rd, rtd = torch.tanh(self.a), torch.tanh(self.c)
        else:
            rtd = torch.exp(self.c)
            rd = (F.softplus(self.a) - 1 + self.eps) / rtd
        R = self.R_off * self.triu + torch.diag(rd)
        Rt = self.Rt_off * self.triu + torch.diag(rtd)
        return Q, R, Rt, rd, rtd

    def forward(self, z, mats=None):
        Q, R, Rt, rd, rtd = mats or self.mats()
        v = z @ Q
        h = torch.tanh(v @ Rt.T + self.b)
        z_new = z + (h @ R.T) @ Q.T
        logdet = torch.log1p((1 - h ** 2) * (rd * rtd)).sum(-1)
        return z_new, logdet

    @torch.no_grad()
    def inverse(self, x, n_iter=30, mats=None):
        Q, R, Rt, rd, rtd = mats or self.mats()
        vp = x @ Q  # v' = v + R h(R~ v + b)
        N, M = vp.shape
        v = torch.zeros_like(vp)
        h = torch.zeros_like(vp)
        for i in reversed(range(M)):
            t = vp[:, i] - (h[:, i + 1:] @ R[i, i + 1:]) if i < M - 1 else vp[:, i].clone()
            c = (v[:, i + 1:] @ Rt[i, i + 1:]) + self.b[i] if i < M - 1 else self.b[i].expand(N)
            v[:, i] = _solve_monotone(t, rd[i], rtd[i], c, n_iter)
            h[:, i] = torch.tanh(rtd[i] * v[:, i] + c)
        return x - (h @ R.T) @ Q.T


def _solve_monotone(t, r, a, c, n_iter):
    """Solve v + r tanh(a v + c) = t for v (strictly increasing in v). Safeguarded Newton:
    keep a bracket [lo, hi] around the root; fall back to bisection when a Newton step leaves the
    bracket, or when the last Newton step did not at least halve the residual (this stops Newton
    from cycling on very sharp layers, large r * a). The bracket test is strict so that a converged
    iterate sitting on the bracket edge is not thrown back to the midpoint."""
    lo, hi = t - r.abs() - 1e-6, t + r.abs() + 1e-6
    v = t.clone()
    g_last = torch.full_like(t, float("inf"))
    tol = 64 * torch.finfo(t.dtype).eps * (1 + t.abs() + r.abs())  # residual at rounding level
    for _ in range(n_iter):
        th = torch.tanh(a * v + c)
        g = v + r * th - t
        lo = torch.where(g < 0, v, lo)
        hi = torch.where(g > 0, v, hi)
        slow = (g.abs() > 0.5 * g_last) & (g.abs() > tol)
        if slow.any():
            v = torch.where(slow, 0.5 * (lo + hi), v)
            th = torch.tanh(a * v + c)
            g = v + r * th - t
            lo = torch.where(g < 0, v, lo)
            hi = torch.where(g > 0, v, hi)
        vn = v - g / (1 + r * a * (1 - th ** 2))
        bad = (vn < lo) | (vn > hi) | ~torch.isfinite(vn)
        v = torch.where(bad, 0.5 * (lo + hi), vn)
        g_last = g.abs()
    return v


class Flow(nn.Module):
    """Standard normal base N(0, I_D) pushed through L orthogonal Sylvester layers."""

    def __init__(self, D, L, M=None, param="relaxed"):
        super().__init__()
        self.D = D
        self.layers = nn.ModuleList([OrthoSylvester(D, M or D, param) for _ in range(L)])

    @staticmethod
    def base_logp(z):
        return -0.5 * (z ** 2).sum(-1) - 0.5 * z.shape[-1] * LOG2PI

    def forward(self, z, return_path=False):
        """Sampling direction. Returns x and log q(x) (given base samples z)."""
        logq = self.base_logp(z)
        path = [z]
        for layer in self.layers:
            z, ld = layer(z)
            logq = logq - ld
            if return_path:
                path.append(z)
        return (z, logq, path) if return_path else (z, logq)

    def sample(self, n):
        return self.forward(torch.randn(n, self.D))

    def log_prob(self, x):
        """Exact log q(x) via the inverse, with implicit-function-correct gradients.

        Per layer: the inverse value is found without grad, then one Newton step
        z <- z* - J^{-1}(f(z*) - x) (J detached) re-attaches the correct first-order dependence
        on parameters and on x; log det is then re-evaluated at that z.
        """
        need_grad = torch.is_grad_enabled()
        logdet = torch.zeros(x.shape[0])
        z = x
        for layer in reversed(self.layers):
            mats = layer.mats()
            z_star = layer.inverse(z.detach(), mats=mats)
            if need_grad:
                Q, R, Rt, _, _ = mats
                fz, _ = layer(z_star, mats)
                g = fz - z  # value ~0, carries gradients
                with torch.no_grad():  # Woodbury: J^-1 = I - Q A (I + A)^-1 Q^T, A = R diag(h') R~
                    hp = 1 - torch.tanh(z_star @ Q @ Rt.T + layer.b) ** 2
                    A = R.unsqueeze(0) * hp.unsqueeze(1) @ Rt  # (N, M, M)
                    I = torch.eye(layer.M).expand_as(A)
                    Qd = Q.detach()
                gq = g @ Qd  # (N, M)
                corr = torch.linalg.solve(I + A, gq.unsqueeze(-1))
                corr = (A @ corr).squeeze(-1)
                z = z_star - (g - corr @ Qd.T)
                _, ld = layer(z, mats)
            else:
                z = z_star
                _, ld = layer(z, mats)
            logdet = logdet + ld
        return self.base_logp(z) - logdet

    def orth_error(self):
        with torch.no_grad():
            errs = []
            for l in self.layers:
                Q = orthogonalize(l.W)
                errs.append((Q.T @ Q - torch.eye(l.M)).abs().max().item())
            return max(errs)
