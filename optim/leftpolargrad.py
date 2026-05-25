import torch
from torch.optim.optimizer import Optimizer, ParamsT

from .invsqrt import symmetric_matrix_invsqrt
from .utils import decoupled_weight_decay_, is_matrix_param


class LeftPolarGradM(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.95,
        alpha: float = 1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        backend: str = "polar_express",
        num_steps: int = 5,
        center_rows: bool = False,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            backend=backend,
            num_steps=num_steps,
            center_rows=center_rows,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            backend = group["backend"]
            num_steps = group["num_steps"]
            center_rows = group["center_rows"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not is_matrix_param(p):
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("LeftPolarGradM does not support sparse gradients.")

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)

                m = state["momentum"]
                m.mul_(beta).add_(g, alpha=1.0 - beta)

                if center_rows:
                    m_eff = m - m.mean(dim=0, keepdim=True)
                else:
                    m_eff = m

                decoupled_weight_decay_(p, lr, wd)

                C = m_eff @ m_eff.transpose(-1, -2)
                L = symmetric_matrix_invsqrt(C, eps=eps, backend=backend, num_steps=num_steps)

                # nu = tr(C L) = <m, mL>
                if alpha != 0:
                    nu = torch.trace(C @ L)
                    scale = nu.clamp_min(eps).pow(alpha)
                else:
                    scale = 1.0

                update = scale * (L @ m_eff)
                # For router quotient geometry, keep the actual update in the
                # centered expert subspace. This is redundant in exact
                # arithmetic for the left-spectral map, but protects against
                # numerical/inexact inverse-square-root errors.
                if center_rows:
                    update = update - update.mean(dim=0, keepdim=True)
                p.add_(update, alpha=-lr)

        return loss
