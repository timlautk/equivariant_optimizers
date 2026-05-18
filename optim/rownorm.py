from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from .row_ops import apply_row_scaling
from .utils import decoupled_weight_decay_


_SUPPORTED_ORIENTATIONS = {
    "row",
    "col",
    "gpt_oss_gate_up_pair",
    "olmoe_gate_up_pair",
}


def _is_supported_param(p: torch.Tensor) -> bool:
    # 2D: ordinary matrices. 3D: batched MoE expert tensors [E, m, n].
    return p.ndim in (2, 3)


def _center_rows(x: torch.Tensor) -> torch.Tensor:
    """Center along the row/neuron axis for both 2D and batched 3D matrices."""
    return x - x.mean(dim=-2, keepdim=True)


def _apply_row_scaling_batched(
    x: torch.Tensor,
    *,
    mode: str,
    eps: float,
) -> torch.Tensor:
    """Apply row scaling to the last axis, supporting batched matrices.

    If your project-level apply_row_scaling already supports tensors of shape
    [..., rows, cols], this simply delegates. If it only supports 2D matrices,
    the fallback flattens all leading dimensions into one row batch.
    """
    try:
        return apply_row_scaling(x, mode=mode, eps=eps)
    except Exception:
        if x.ndim <= 2:
            raise
        orig_shape = x.shape
        y = x.reshape(-1, orig_shape[-1])
        y = apply_row_scaling(y, mode=mode, eps=eps)
        return y.reshape(orig_shape)


def _orient(x: torch.Tensor, orientation: str) -> torch.Tensor:
    """Convert x to a work matrix/tensor whose rows are the geometry units.

    orientation="row":
        [m, n] or [E, m, n] unchanged.

    orientation="col":
        [m, n] -> [n, m], [E, m, n] -> [E, n, m].

    orientation="gpt_oss_gate_up_pair":
        gpt-oss gate_up_proj is [E, d, 2*r] with interleaved gate/up columns.
        Convert to [E, r, 2*d], one row per intermediate neuron, containing
        both the gate and up vectors for that neuron.

    orientation="olmoe_gate_up_pair":
        OLMoE gate_up_proj is [E, 2*r, d] with non-interleaved gate/up rows.
        Convert to [E, r, 2*d], one row per intermediate neuron, containing
        both the gate and up rows for that neuron.
    """
    if orientation == "row":
        return x

    if orientation == "col":
        return x.transpose(-1, -2).contiguous()

    if orientation == "gpt_oss_gate_up_pair":
        if x.ndim != 3 or x.shape[-1] % 2 != 0:
            raise ValueError(
                "orientation='gpt_oss_gate_up_pair' expects [E, d, 2*r], "
                f"got {tuple(x.shape)}"
            )
        E, d, two_r = x.shape
        r = two_r // 2
        return x.view(E, d, r, 2).permute(0, 2, 1, 3).contiguous().view(E, r, 2 * d)

    if orientation == "olmoe_gate_up_pair":
        if x.ndim != 3 or x.shape[1] % 2 != 0:
            raise ValueError(
                "orientation='olmoe_gate_up_pair' expects [E, 2*r, d], "
                f"got {tuple(x.shape)}"
            )
        E, two_r, d = x.shape
        r = two_r // 2
        gate = x[:, :r, :]
        up = x[:, r:, :]
        return torch.stack((gate, up), dim=-1).contiguous().view(E, r, 2 * d)

    raise ValueError(f"Unknown orientation={orientation}")


def _unorient(u: torch.Tensor, orientation: str, orig_shape: torch.Size) -> torch.Tensor:
    """Inverse of _orient."""
    if orientation == "row":
        return u.reshape(orig_shape)

    if orientation == "col":
        return u.transpose(-1, -2).contiguous().reshape(orig_shape)

    if orientation == "gpt_oss_gate_up_pair":
        E, d, two_r = orig_shape
        r = two_r // 2
        return u.view(E, r, d, 2).permute(0, 2, 1, 3).contiguous().view(orig_shape)

    if orientation == "olmoe_gate_up_pair":
        E, two_r, d = orig_shape
        r = two_r // 2
        z = u.view(E, r, d, 2)
        gate = z[..., 0]
        up = z[..., 1]
        return torch.cat((gate, up), dim=1).contiguous().view(orig_shape)

    raise ValueError(f"Unknown orientation={orientation}")


class RowNormM(Optimizer):
    """
    Momentum row-norm optimizer with orientation support.

    Supported orientations:
        row:
            Apply row normalization directly to the stored matrix/tensor.
            Use for embeddings [v, d], LM heads [v, d], routers [e, d],
            dense SwiGLU gate/up [d_ff, d], OLMoE gate_up [E, 2*r, d],
            and gpt-oss down_proj [E, r, d].

        col:
            Apply row normalization to the transpose and transpose back.
            Use for dense down_proj [d, d_ff] and OLMoE down_proj [E, d, r].

        gpt_oss_gate_up_pair:
            For gpt-oss gate_up_proj [E, d, 2*r] with interleaved gate/up
            column pairs. Each intermediate neuron row becomes
            [gate_j, up_j] in [E, r, 2*d].

        olmoe_gate_up_pair:
            Optional pair-aware OLMoE gate_up_proj [E, 2*r, d]. The default
            cheaper OLMoE choice can remain orientation='row'.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        row_mode: str = "inverse_eps",
        eps: float = 1e-8,
        center_rows: bool = False,
        orientation: str = "row",
        expert_layout: str | None = None,
    ):
        if expert_layout is not None:
            orientation = expert_layout
        if orientation not in _SUPPORTED_ORIENTATIONS:
            raise ValueError(
                f"Unknown orientation={orientation}. Expected one of "
                f"{sorted(_SUPPORTED_ORIENTATIONS)}."
            )

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            row_mode=row_mode,
            eps=eps,
            center_rows=center_rows,
            orientation=orientation,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _oriented_row_update(
        m: torch.Tensor,
        *,
        row_mode: str,
        eps: float,
        center_rows: bool,
        orientation: str,
    ) -> torch.Tensor:
        orig_shape = m.shape
        work = _orient(m, orientation)
        if center_rows:
            work = _center_rows(work)
        update_work = _apply_row_scaling_batched(work, mode=row_mode, eps=eps)
        return _unorient(update_work, orientation, orig_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            row_mode = group["row_mode"]
            eps = group["eps"]
            center_rows = group["center_rows"]
            orientation = group["orientation"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not _is_supported_param(p):
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("RowNormM does not support sparse gradients.")

                g = p.grad
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)

                m = state["momentum"]
                m.mul_(beta).add_(g, alpha=1.0 - beta)

                decoupled_weight_decay_(p, lr, wd)

                update = self._oriented_row_update(
                    m,
                    row_mode=row_mode,
                    eps=eps,
                    center_rows=center_rows,
                    orientation=orientation,
                )
                p.add_(update, alpha=-lr)

        return loss


class BatchedExpertRowNormM(RowNormM):
    """Backward-compatible alias for 3D MoE expert tensors.

    Use expert_layout="row", "col", "gpt_oss_gate_up_pair", or
    "olmoe_gate_up_pair". Internally this is just RowNormM with orientation.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        row_mode: str = "inverse_eps",
        eps: float = 1e-8,
        expert_layout: str = "row",
    ):
        super().__init__(
            params,
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            row_mode=row_mode,
            eps=eps,
            center_rows=False,
            orientation=expert_layout,
        )
