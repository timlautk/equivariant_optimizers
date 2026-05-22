from __future__ import annotations

import torch

from .utils import decoupled_weight_decay_


# =========================
# MuonHeads-style batched MoE expert Muon / Polar Express optimizer
# =========================

_SUPPORTED_EXPERT_LAYOUTS = {
    "row",
    "col",
    "gpt_oss_gate_up_pair",
    "olmoe_gate_up_pair",
}


def _polar_express_stack(g: torch.Tensor) -> torch.Tensor:
    """Apply PolarExpressHeads independently to a stack of expert matrices."""
    from .polar_express import PolarExpressHeads

    if g.ndim != 3:
        raise ValueError(f"Expected a 3D expert stack [E, rows, cols], got {tuple(g.shape)}")
    return torch.stack(
        [PolarExpressHeads(g[e], compute_hermitian=False) for e in range(g.shape[0])],
        dim=0,
    )


def _muon_scale_for_expert_layout(g: torch.Tensor, expert_layout: str) -> float:
    """MuonHeadsPolarExpress-style scale max(1, rows / cols) ** 0.5."""
    if expert_layout == "row":
        rows, cols = g.shape[-2], g.shape[-1]
    elif expert_layout == "col":
        rows, cols = g.shape[-1], g.shape[-2]
    elif expert_layout == "gpt_oss_gate_up_pair":
        # HF GPT-OSS: [E, hidden_size, 2 * intermediate_size].
        # Split into gate/up matrices of shape [E, hidden_size, intermediate_size].
        rows, cols = g.shape[-2], g.shape[-1] // 2
    elif expert_layout == "olmoe_gate_up_pair":
        # HF OLMoE: [E, 2 * intermediate_size, hidden_size].
        # Split into gate/up matrices of shape [E, intermediate_size, hidden_size].
        rows, cols = g.shape[-2] // 2, g.shape[-1]
    else:
        raise ValueError(
            f"Unknown expert_layout={expert_layout}. "
            f"Expected one of {sorted(_SUPPORTED_EXPERT_LAYOUTS)}."
        )
    return max(1.0, float(rows) / float(cols)) ** 0.5


def _polarize_expert_update_like_muon_heads(g: torch.Tensor, expert_layout: str) -> torch.Tensor:
    """
    Polarize a 3D MoE expert tensor using the same convention as
    MuonHeadsPolarExpress: no nuclear-norm scaling and no alpha.

    For fused gate/up expert tensors, the gate and up branches are polarized
    separately, matching MuonHeadsPolarExpress.
    """
    if g.ndim != 3:
        raise ValueError(f"Expected a 3D MoE expert tensor, got {tuple(g.shape)}")

    if expert_layout == "row":
        return _polar_express_stack(g)

    if expert_layout == "col":
        return _polar_express_stack(g.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()

    if expert_layout == "gpt_oss_gate_up_pair":
        # GPT-OSS expert gate_up_proj has shape [E, H, 2I] and is interleaved:
        # gate = [..., ::2], up = [..., 1::2].
        if g.shape[-1] % 2 != 0:
            raise ValueError(
                "expert_layout='gpt_oss_gate_up_pair' expects last dimension 2*I, "
                f"got {tuple(g.shape)}"
            )
        gate = g[..., ::2].contiguous()
        up = g[..., 1::2].contiguous()
        gate_u = _polar_express_stack(gate)
        up_u = _polar_express_stack(up)
        out = torch.empty_like(g)
        out[..., ::2] = gate_u
        out[..., 1::2] = up_u
        return out

    if expert_layout == "olmoe_gate_up_pair":
        # OLMoE expert gate_up_proj has shape [E, 2I, H] and is concatenated:
        # gate = [:, :I, :], up = [:, I:, :].
        if g.shape[-2] % 2 != 0:
            raise ValueError(
                "expert_layout='olmoe_gate_up_pair' expects second-to-last dimension 2*I, "
                f"got {tuple(g.shape)}"
            )
        intermediate = g.shape[-2] // 2
        gate = g[:, :intermediate, :].contiguous()
        up = g[:, intermediate:, :].contiguous()
        return torch.cat([_polar_express_stack(gate), _polar_express_stack(up)], dim=1)

    raise ValueError(
        f"Unknown expert_layout={expert_layout}. "
        f"Expected one of {sorted(_SUPPORTED_EXPERT_LAYOUTS)}."
    )


class BatchedExpertMuonPolarExpress(torch.optim.Optimizer):
    """MuonHeadsPolarExpress-style optimizer for 3D MoE expert tensors.

    This is intended for the routing choice ``choice == 'matrix'`` on MoE expert
    tensors. It is a Muon-style optimizer, not a PolarGrad-style optimizer:

      1. It does not use nuclear-norm scaling and has no ``alpha`` argument.
      2. It uses momentum-first Polar Express orthogonalization.
      3. For fused gate/up expert tensors, it polarizes gate and up branches
         separately, matching ``MuonHeadsPolarExpress``.

    Supported expert layouts:
      - ``row``: [E, rows, cols]
      - ``col``: [E, cols, rows], optimized through transpose
      - ``gpt_oss_gate_up_pair``: [E, hidden_size, 2 * intermediate_size]
      - ``olmoe_gate_up_pair``: [E, 2 * intermediate_size, hidden_size]
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
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
            nesterov=nesterov,
            expert_layout=expert_layout,
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
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            expert_layout = group["expert_layout"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("BatchedExpertMuonPolarExpress does not support sparse gradients.")
                if p.ndim != 3:
                    raise RuntimeError(
                        f"BatchedExpertMuonPolarExpress expects 3D MoE expert tensors, got {tuple(p.shape)}"
                    )

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update_src = g.add(buf, alpha=momentum)
                else:
                    update_src = buf

                scale = _muon_scale_for_expert_layout(update_src, expert_layout)
                update = _polarize_expert_update_like_muon_heads(update_src, expert_layout)
                update = update.mul(scale)

                decoupled_weight_decay_(p, lr, weight_decay)
                p.add_(update.to(dtype=p.dtype), alpha=-lr)

        return loss


# Short aliases for convenience.
BatchedExpertMuonPE = BatchedExpertMuonPolarExpress
BatchExpertMuonPolarExpress = BatchedExpertMuonPolarExpress
