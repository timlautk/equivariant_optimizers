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

    If project-level apply_row_scaling supports [..., rows, cols], this simply
    delegates. If it only supports 2D matrices, the fallback flattens all
    leading dimensions into one row batch.
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
    """Convert x to a work matrix/tensor whose rows are the geometry units."""
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


class HybridPolarGradM(Optimizer):
    """
    Hybrid spectral / row-norm optimizer with orientation support.

    Supported orders:
      - order="polar_then_row": polarize first, then row-normalize.
      - order="row_then_polar": row-normalize first, then polarize.

    Supported orientations:
      - row: apply row geometry directly to [m, n] or [E, m, n].
      - col: apply row geometry to the transpose and transpose back.
      - gpt_oss_gate_up_pair: [E, d, 2*r] -> [E, r, 2*d].
      - olmoe_gate_up_pair: [E, 2*r, d] -> [E, r, 2*d].
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.95,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        row_mode: str = "inverse_eps",
        eps: float = 1e-8,
        backend: str = "polar_express",
        num_steps: int = 5,
        left: bool = False,
        center_rows: bool = False,
        order: str = "polar_then_row",   # {"polar_then_row", "row_then_polar"}
        orientation: str = "row",        # {"row", "col", "gpt_oss_gate_up_pair", "olmoe_gate_up_pair"}
        # Backward-compatible alias used by earlier routing code.
        expert_layout: str | None = None,
    ):
        if expert_layout is not None:
            orientation = expert_layout
        if order not in ("polar_then_row", "row_then_polar"):
            raise ValueError(f"Unsupported order={order}")
        if orientation not in _SUPPORTED_ORIENTATIONS:
            raise ValueError(
                f"Unsupported orientation={orientation}. Expected one of "
                f"{sorted(_SUPPORTED_ORIENTATIONS)}."
            )

        defaults = dict(
            lr=lr,
            beta=beta,
            alpha=alpha,
            weight_decay=weight_decay,
            row_mode=row_mode,
            eps=eps,
            backend=backend,
            num_steps=num_steps,
            left=left,
            center_rows=center_rows,
            order=order,
            orientation=orientation,
        )
        super().__init__(params, defaults)

    def _polarize(
        self,
        X: torch.Tensor,
        *,
        left: bool,
        eps: float,
        backend: str,
        num_steps: int,
    ) -> torch.Tensor:
        # Supports both 2D and batched 3D X through torch.matmul broadcasting,
        # assuming symmetric_matrix_invsqrt supports batched Grams. If not, use
        # HybridPolarGradM_GramNS or row-only experts for very large runs.
        if left:
            C = X @ X.transpose(-1, -2)
            L = symmetric_matrix_invsqrt(C, eps=eps, backend=backend, num_steps=num_steps)
            return L @ X
        else:
            C = X.transpose(-1, -2) @ X
            R = symmetric_matrix_invsqrt(C, eps=eps, backend=backend, num_steps=num_steps)
            return X @ R

    @torch.no_grad()
    def _compute_update(
        self,
        m: torch.Tensor,
        *,
        alpha: float,
        row_mode: str,
        eps: float,
        backend: str,
        num_steps: int,
        left: bool,
        center_rows: bool,
        order: str,
        orientation: str,
    ) -> torch.Tensor:
        orig_shape = m.shape
        work = _orient(m, orientation)
        if center_rows:
            work = _center_rows(work)

        if order == "polar_then_row":
            u = self._polarize(work, left=left, eps=eps, backend=backend, num_steps=num_steps)
            if alpha != 0:
                # Sum over all entries. For batched experts this couples the
                # scalar scale across experts in the same parameter tensor.
                nu = torch.sum(work.float() * u.float())
                scale = nu.clamp_min(eps).pow(alpha).to(dtype=u.dtype)
            else:
                scale = 1.0
            update_work = _apply_row_scaling_batched(scale * u, mode=row_mode, eps=eps)

        elif order == "row_then_polar":
            work_row = _apply_row_scaling_batched(work, mode=row_mode, eps=eps)
            u = self._polarize(work_row, left=left, eps=eps, backend=backend, num_steps=num_steps)
            if alpha != 0:
                nu = torch.sum(work_row.float() * u.float())
                scale = nu.clamp_min(eps).pow(alpha).to(dtype=u.dtype)
            else:
                scale = 1.0
            update_work = scale * u

        else:
            raise ValueError(f"Unsupported order={order}")

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
            alpha = group["alpha"]
            wd = group["weight_decay"]
            row_mode = group["row_mode"]
            eps = group["eps"]
            backend = group["backend"]
            num_steps = group["num_steps"]
            left = group["left"]
            center_rows = group["center_rows"]
            order = group["order"]
            orientation = group["orientation"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not _is_supported_param(p):
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("HybridPolarGradM does not support sparse gradients.")

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)

                m = state["momentum"]
                m.mul_(beta).add_(p.grad, alpha=1.0 - beta)

                decoupled_weight_decay_(p, lr, wd)

                update = self._compute_update(
                    m,
                    alpha=alpha,
                    row_mode=row_mode,
                    eps=eps,
                    backend=backend,
                    num_steps=num_steps,
                    left=left,
                    center_rows=center_rows,
                    order=order,
                    orientation=orientation,
                )
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
    (1.875, -1.25, 0.375),
]

safety_factor = 1.05
coeffs_list = [
    (a / safety_factor, b / safety_factor**3, c / safety_factor**5)
    for (a, b, c) in _unmodified_polar_express_coeffs_list[:-1]
]
coeffs_list.append(_unmodified_polar_express_coeffs_list[-1])


class HybridPolarGradM_GramNS(Optimizer):
    """
    HybridPolarGradM using Gram Newton--Schulz / Polar Express-style
    orthogonalization, with the same orientation support as HybridPolarGradM.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.95,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        row_mode: str = "inverse_eps",
        eps: float = 1e-8,
        backend: str = "polar_express",
        num_steps: int = 5,
        left: bool = False,
        center_rows: bool = False,
        order: str = "polar_then_row",
        orientation: str = "row",
        # Backward-compatible alias used by earlier routing code.
        expert_layout: str | None = None,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients=None,
        use_gram_newton_schulz: bool = True,
        gram_newton_schulz_reset_iterations: Optional[List[int]] = None,
        compile_kwargs=None,
        filter_fn: Optional[Callable[[torch.nn.Parameter], bool]] = None,
    ):
        if GramNewtonSchulz is None:
            raise ImportError(
                "gram_newton_schulz is not installed. "
                "Install from https://github.com/Dao-AILab/gram-newton-schulz"
            )
        if expert_layout is not None:
            orientation = expert_layout
        if order not in ("polar_then_row", "row_then_polar"):
            raise ValueError(f"Unsupported order={order}")
        if orientation not in _SUPPORTED_ORIENTATIONS:
            raise ValueError(
                f"Unsupported orientation={orientation}. Expected one of "
                f"{sorted(_SUPPORTED_ORIENTATIONS)}."
            )

        defaults = dict(
            lr=lr,
            beta=beta,
            alpha=alpha,
            weight_decay=weight_decay,
            row_mode=row_mode,
            eps=eps,
            backend=backend,
            num_steps=num_steps,
            left=left,
            center_rows=center_rows,
            order=order,
            orientation=orientation,
        )
        super().__init__(params, defaults)

        if ns_coefficients is None:
            ns_coefficients = coeffs_list[:num_steps] + list(
                repeat(coeffs_list[-1], max(0, num_steps - len(coeffs_list)))
            )

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
    def _compute_update(
        self,
        m: torch.Tensor,
        *,
        alpha: float,
        row_mode: str,
        eps: float,
        center_rows: bool,
        order: str,
        orientation: str,
    ) -> torch.Tensor:
        orig_shape = m.shape
        work = _orient(m, orientation)
        if center_rows:
            work = _center_rows(work)

        if order == "polar_then_row":
            u = self.orthogonalizer(work)
            if alpha != 0:
                nu = torch.sum(work.float() * u.float())
                scale = nu.clamp_min(eps).pow(alpha).to(dtype=u.dtype)
            else:
                scale = 1.0
            update_work = _apply_row_scaling_batched(scale * u, mode=row_mode, eps=eps)

        elif order == "row_then_polar":
            work_row = _apply_row_scaling_batched(work, mode=row_mode, eps=eps)
            u = self.orthogonalizer(work_row)
            if alpha != 0:
                nu = torch.sum(work_row.float() * u.float())
                scale = nu.clamp_min(eps).pow(alpha).to(dtype=u.dtype)
            else:
                scale = 1.0
            update_work = scale * u

        else:
            raise ValueError(f"Unsupported order={order}")

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
            alpha = group["alpha"]
            wd = group["weight_decay"]
            row_mode = group["row_mode"]
            eps = group["eps"]
            center_rows = group["center_rows"]
            order = group["order"]
            orientation = group["orientation"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not _is_supported_param(p):
                    continue
                if self.filter_fn is not None and not self.filter_fn(p):
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("HybridPolarGradM_GramNS does not support sparse gradients.")

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)

                m = state["momentum"]
                m.mul_(beta).add_(p.grad, alpha=1.0 - beta)

                decoupled_weight_decay_(p, lr, wd)

                update = self._compute_update(
                    m,
                    alpha=alpha,
                    row_mode=row_mode,
                    eps=eps,
                    center_rows=center_rows,
                    order=order,
                    orientation=orientation,
                )
                p.add_(update, alpha=-lr)

        return loss


class BatchedExpertHybridPolarGradM(HybridPolarGradM):
    """Backward-compatible exact/backbone alias for 3D MoE expert tensors."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.95,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        row_mode: str = "inverse_eps",
        eps: float = 1e-8,
        order: str = "row_then_polar",
        left: bool = False,
        expert_layout: str = "row",
        backend: str = "polar_express",
        num_steps: int = 5,
    ):
        super().__init__(
            params,
            lr=lr,
            beta=beta,
            alpha=alpha,
            weight_decay=weight_decay,
            row_mode=row_mode,
            eps=eps,
            backend=backend,
            num_steps=num_steps,
            left=left,
            center_rows=False,
            order=order,
            orientation=expert_layout,
        )


class BatchedExpertHybridPolarGradM_GramNS(HybridPolarGradM_GramNS):
    """
    Backward-compatible GramNS alias for 3D MoE expert tensors.

    This wrapper intentionally accepts the same extra keyword arguments that
    the generic routing code passes to HybridPolarGradM_GramNS, including
    backend, num_steps, center_rows, orientation, and expert_layout.  The
    important one is num_steps: it controls the number of Polar-Express /
    Gram-Newton-Schulz coefficients used by the orthogonalizer.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.95,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        row_mode: str = "inverse_eps",
        eps: float = 1e-8,
        backend: str = "polar_express",
        num_steps: int = 5,
        order: str = "row_then_polar",
        left: bool = False,
        center_rows: bool = False,
        expert_layout: str = "row",
        orientation: str | None = None,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients=None,
        use_gram_newton_schulz: bool = True,
        gram_newton_schulz_reset_iterations: Optional[List[int]] = None,
        compile_kwargs=None,
        filter_fn: Optional[Callable[[torch.nn.Parameter], bool]] = None,
        **unused_kwargs,
    ):
        if orientation is None:
            orientation = expert_layout

        super().__init__(
            params,
            lr=lr,
            beta=beta,
            alpha=alpha,
            weight_decay=weight_decay,
            row_mode=row_mode,
            eps=eps,
            backend=backend,
            num_steps=num_steps,
            left=left,
            center_rows=center_rows,
            order=order,
            orientation=orientation,
            ns_epsilon=ns_epsilon,
            ns_use_kernels=ns_use_kernels,
            ns_coefficients=ns_coefficients,
            use_gram_newton_schulz=use_gram_newton_schulz,
            gram_newton_schulz_reset_iterations=gram_newton_schulz_reset_iterations,
            compile_kwargs=compile_kwargs,
            filter_fn=filter_fn,
        )
