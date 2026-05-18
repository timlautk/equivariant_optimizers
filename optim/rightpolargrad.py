from __future__ import annotations

from itertools import repeat
from typing import Callable, List, Optional

import torch
from torch.optim.optimizer import Optimizer, ParamsT

try:
    from gram_newton_schulz import GramNewtonSchulz
except ImportError:
    GramNewtonSchulz = None

from .invsqrt import symmetric_matrix_invsqrt
from .utils import decoupled_weight_decay_, is_matrix_param


class RightPolarGradM(Optimizer):
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
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            backend=backend,
            num_steps=num_steps,
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

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not is_matrix_param(p):
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("RightPolarGradM does not support sparse gradients.")

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)

                m = state["momentum"]
                m.mul_(beta).add_(g, alpha=1.0 - beta)

                decoupled_weight_decay_(p, lr, wd)

                C = m.transpose(-1, -2) @ m
                R = symmetric_matrix_invsqrt(C, eps=eps, backend=backend, num_steps=num_steps)

                # nu = tr(C R) = <m, mR>
                if alpha != 0:
                    nu = torch.sum(C @ R)
                    scale = nu.clamp_min(eps).pow(alpha)
                else:
                    scale = 1.0

                update = scale * (m @ R)
                p.add_(update, alpha=-lr)

        return loss


_unmodified_polar_express_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
safety_factor = 1.05
# safety factor for numerical stability ( but exclude last polynomial )
coeffs_list = [(a / safety_factor, b / safety_factor**3, c / safety_factor**5) 
                for (a, b, c) in _unmodified_polar_express_coeffs_list[:-1]]
coeffs_list.append(_unmodified_polar_express_coeffs_list[-1])


class RightPolarGradM_GramNS(Optimizer):
    """
    RightPolarGradM using Gram Newton--Schulz / Polar Express-style orthogonalization.

    Update:
        m_k = beta * m_{k-1} + (1 - beta) * g_k
        u_k ≈ polar(m_k)  via GramNewtonSchulz
        nu_k = <m_k, u_k> = tr(m_k^T u_k)
        p <- p - lr * nu_k^alpha * u_k

    Notes
    -----
    - This is especially natural for tall-skinny matrices such as embeddings / LM heads.
    - The GramNewtonSchulz implementation already returns an orthogonalized matrix
      with the same shape as the input, so we do not explicitly form m^T m here.
    - For exact mathematical correctness on fused matrices, split them before calling
      the orthogonalizer.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.95,
        alpha: float = 1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients=None,
        use_gram_newton_schulz: bool = True,
        gram_newton_schulz_reset_iterations: Optional[List[int]] = None,
        num_steps: int = 5,
        compile_kwargs=None,
        filter_fn: Optional[Callable[[torch.nn.Parameter], bool]] = None,
    ):
        if GramNewtonSchulz is None:
            raise ImportError(
                "gram_newton_schulz is not installed. "
                "Install from https://github.com/Dao-AILab/gram-newton-schulz"
            )

        defaults = dict(
            lr=lr,
            beta=beta,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            num_steps=num_steps,
        )
        super().__init__(params, defaults)

        if ns_coefficients is None:
            ns_coefficients = coeffs_list[:num_steps] + list(repeat(coeffs_list[-1], num_steps - len(coeffs_list)))

        self.orthogonalizer = GramNewtonSchulz(
            ns_epsilon=ns_epsilon,
            ns_use_kernels=ns_use_kernels,
            ns_coefficients=ns_coefficients,
            use_gram_newton_schulz=use_gram_newton_schulz,
            gram_newton_schulz_reset_iterations=gram_newton_schulz_reset_iterations,
            compile_kwargs=compile_kwargs,
        )
        self.filter_fn = filter_fn

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
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not is_matrix_param(p):
                    continue
                if self.filter_fn is not None and not self.filter_fn(p):
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError(
                        "RightPolarGradM_GramNS does not support sparse gradients."
                    )

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)

                m = state["momentum"]
                m.mul_(beta).add_(g, alpha=1.0 - beta)

                decoupled_weight_decay_(p, lr, weight_decay)

                u = self.orthogonalizer(m)

                # Nuclear-norm surrogate scaling: <m, polar(m)>
                if alpha != 0:
                    nu = torch.sum(m * u)
                    scale = nu.clamp_min(eps).pow(alpha)
                else:
                    scale = 1.0

                update = scale * u
                p.add_(update, alpha=-lr)

        return loss
