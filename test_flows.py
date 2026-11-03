import torch
from flows import Flow
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
for D, M, param in [(1,1,'paper'),(1,1,'relaxed'),(2,1,'relaxed'),(2,2,'relaxed'),(2,2,'paper'),(3,2,'relaxed')]:
    f = Flow(D, 6, M, param)
    with torch.no_grad():  # make it nontrivial
        for l in f.layers:
            l.a.add_(torch.randn(M)); l.c.add_(torch.randn(M)); l.R_off.mul_(10); l.Rt_off.mul_(10)
    z = torch.randn(5, D)
    x, logq = f(z); x = x.detach(); logq = logq.detach()
    zr = x
    for l in reversed(f.layers): zr = l.inverse(zr)
    # logdet check
    J = torch.stack([torch.autograd.functional.jacobian(lambda u: f(u[None])[0][0], zi) for zi in z])
    ld_true = torch.logdet(J)
    ld_flow = f.base_logp(z) - logq
    lp = f.log_prob(x)
    # gradient check of log_prob wrt one parameter
    p = f.layers[2].b
    loss = f.log_prob(x).sum(); g = torch.autograd.grad(loss, p)[0][0]
    eps = 1e-6
    with torch.no_grad():
        p[0] += eps; up = f.log_prob(x).sum(); p[0] -= 2*eps; dn = f.log_prob(x).sum(); p[0] += eps
    fd = (up - dn) / (2*eps)
    print(f"D={D} M={M} {param:7s} inv_err={(zr-z).abs().max():.1e} logdet_err={(ld_true-ld_flow).abs().max():.1e} "
          f"logprob_err={(lp-logq).abs().max():.1e} grad={g.item():+.6f} fd={fd.item():+.6f} orth={f.orth_error():.1e}")
