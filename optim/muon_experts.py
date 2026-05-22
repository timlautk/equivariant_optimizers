from __future__ import annotations

import torch

from .invsqrt import symmetric_matrix_invsqrt
from .utils import decoupled_weight_decay_


# =========================
# Batched MoE expert Muon / polar optimizer
# =========================

_SUPPORTED_EXPERT_LAYOUTS = {
    "row",
    "col",
    "gpt_oss_gate_up_pair",
    "olmoe_gate_up_pair",
}


def _expert_orient(x: torch.Tensor, expert_layout: str) -> torch.Tensor:
    """Convert a 3D expert tensor into batched matrices [..., rows, cols]."""
    if expert_layout == "row":
        return x

    if expert_layout == "col":
        return x.transpose(-1, -2).contiguous()

    if expert_layout == "gpt_oss_gate_up_pair":
        # HF GPT-OSS: [E, d, 2*r] with interleaved gate/up columns.
        # Convert to [E, r, 2*d], so each intermediate neuron is a row.
        if x.ndim != 3 or x.shape[-1] % 2 != 0:
            raise ValueError(
                "expert_layout='gpt_oss_gate_up_pair' expects [E, d, 2*r], "
                f"got {tuple(x.shape)}"
            )
        E, d, two_r = x.shape
        r = two_r // 2
        return x.view(E, d, r, 2).permute(0, 2, 1, 3).contiguous().view(E, r, 2 * d)

    if expert_layout == "olmoe_gate_up_pair":
        # HF OLMoE: [E, 2*r, d] with gate and up concatenated along rows.
        # Convert to [E, r, 2*d], so each intermediate neuron is a row.
        if x.ndim != 3 or x.shape[1] % 2 != 0:
            raise ValueError(
                "expert_layout='olmoe_gate_up_pair' expects [E, 2*r, d], "
                f"got {tuple(x.shape)}"
            )
        E, two_r, d = x.shape
        r = two_r // 2
        gate = x[:, :r, :]
        up = x[:, r:, :]
        return torch.stack((gate, up), dim=-1).contiguous().view(E, r, 2 * d)

    raise ValueError(
        f"Unknown expert_layout={expert_layout}. "
        f"Expected one of {sorted(_SUPPORTED_EXPERT_LAYOUTS)}."
    )


def _expert_unorient(u: torch.Tensor, expert_layout: str, orig_shape: torch.Size) -> torch.Tensor:
    """Inverse of _expert_orient."""
    if expert_layout == "row":
        return u.reshape(orig_shape)

    if expert_layout == "col":
        return u.transpose(-1, -2).contiguous().reshape(orig_shape)

    if expert_layout == "gpt_oss_gate_up_pair":
        E, d, two_r = orig_shape
        r = two_r // 2
        return u.view(E, r, d, 2).permute(0, 2, 1, 3).contiguous().view(orig_shape)

    if expert_layout == "olmoe_gate_up_pair":
        E, two_r, d = orig_shape
        r = two_r // 2
        z = u.view(E, r, d, 2)
        gate = z[..., 0]
        up = z[..., 1]
        return torch.cat((gate, up), dim=1).contiguous().view(orig_shape)

    raise ValueError(
        f"Unknown expert_layout={expert_layout}. "
        f"Expected one of {sorted(_SUPPORTED_EXPERT_LAYOUTS)}."
    )


class BatchedExpertMuon(torch.optim.Optimizer):
    """Muon-style momentum polar optimizer for 3D MoE expert tensors.

    This is the correct `choice == "matrix"` optimizer for MoE expert tensors.
    It treats each expert as a separate matrix after applying `expert_layout`,
    computes a batched polar/Muon update, and maps the update back to the
    original tensor layout.

    Supported expert layouts:
      - "row":                 [E, rows, cols]
      - "col":                 [E, cols, rows], optimized through transpose
      - "gpt_oss_gate_up_pair": [E, d, 2*r] -> [E, r, 2*d]
      - "olmoe_gate_up_pair":  [E, 2*r, d] -> [E, r, 2*d]
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        alpha: float = 1.0,
        eps: float = 1e-8,
        backend: str = "polar_express",
        num_steps: int = 5,
        expert_layout: str = "row",
    ):
        if expert_layout not in _SUPPORTED_EXPERT_LAYOUTS:
            raise ValueError(
                f"Unknown expert_layout={expert_layout}. "
                f"Expected one of {sorted(_SUPPORTED_EXPERT_LAYOUTS)}."
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=alpha,
            eps=eps,
            backend=backend,
            num_steps=num_steps,
            expert_layout=expert_layout,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _batched_polar(
        x: torch.Tensor,
        *,
        eps: float,
        backend: str,
        num_steps: int,
    ) -> torch.Tensor:
        """Compute batched Muon/polar update on x with shape [E, m, n]."""
        if x.ndim != 3:
            raise ValueError(f"BatchedExpertMuonM expects 3D expert tensors after orientation, got {tuple(x.shape)}")

        m, n = x.shape[-2], x.shape[-1]
        xf = x.float()

        # Use the smaller Gram side per expert. Broadcasting handles [E, ...].
        if m <= n:
            gram = xf @ xf.transpose(-1, -2)
            L = symmetric_matrix_invsqrt(gram, eps=eps, backend=backend, num_steps=num_steps)
            u = L @ xf
        else:
            gram = xf.transpose(-1, -2) @ xf
            R = symmetric_matrix_invsqrt(gram, eps=eps, backend=backend, num_steps=num_steps)
            u = xf @ R

        return u.to(dtype=x.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]
            alpha = group["alpha"]
            eps = group["eps"]
            backend = group["backend"]
            num_steps = group["num_steps"]
            expert_layout = group["expert_layout"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("BatchedExpertMuonM does not support sparse gradients.")
                if p.ndim != 3:
                    raise RuntimeError(
                        f"BatchedExpertMuonM expects 3D MoE expert tensors, got {tuple(p.shape)}"
                    )

                g = p.grad
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)

                m_buf = state["momentum"]
                m_buf.mul_(beta).add_(g, alpha=1.0 - beta)

                decoupled_weight_decay_(p, lr, wd)

                work = _expert_orient(m_buf, expert_layout)
                u = self._batched_polar(work, eps=eps, backend=backend, num_steps=num_steps)

                if alpha != 0:
                    # Per-expert nuclear-style scaling.
                    nu = (work.float() * u.float()).sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
                    u = u * nu.pow(alpha).to(dtype=u.dtype)

                update = _expert_unorient(u, expert_layout, p.shape)
                p.add_(update, alpha=-lr)

        return loss
