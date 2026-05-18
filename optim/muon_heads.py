import os
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.distributed as dist

from .polar_express import PolarExpressHeads
from .utils import decoupled_weight_decay_


def _polarize_expert_stack(g: torch.Tensor) -> torch.Tensor:
    """
    Apply PolarExpressHeads independently to a stack of expert matrices.

    g shape: (num_experts, rows, cols)
    returns same shape.
    """
    outs = []
    for e in range(g.shape[0]):
        outs.append(PolarExpressHeads(g[e], compute_hermitian=False))
    return torch.stack(outs, dim=0)


def _polarize_gate_up_split(g: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal polar factors of gate_proj and up_proj separately in 
    MoE expert's gate_up_proj.

    Original g shape:
        (num_experts, hidden_size, 2 * intermediate_size)

    MoE expert uses:
        gate = gate_up[..., ::2]
        up   = gate_up[..., 1::2]

    We polarize gate and up separately per expert, then interleave back.
    """
    if g.ndim != 3:
        raise ValueError(f"gate_up split expects a 3D tensor, got shape {tuple(g.shape)}")

    if g.shape[-1] % 2 != 0:
        raise ValueError(
            f"gate_up last dimension must be even, got shape {tuple(g.shape)}"
        )

    gate = g[..., ::2].contiguous()   # (E, H, I)
    up = g[..., 1::2].contiguous()    # (E, H, I)

    gate_polar = _polarize_expert_stack(gate)
    up_polar = _polarize_expert_stack(up)

    out = torch.empty_like(g)
    out[..., ::2] = gate_polar
    out[..., 1::2] = up_polar

    return out


def _polarize_olmoe_gate_up_split(g: torch.Tensor) -> torch.Tensor:
    """
    Variant B for OLMoE gate_up_proj.

    OLMoE gate_up_proj shape:
        (num_experts, 2 * intermediate_size, hidden_size)

    F.linear(x, weight) uses weight shape:
        (out_features, in_features)

    Therefore:
        gate branch = g[:, :I, :]   # (E, I, H)
        up branch   = g[:, I:, :]   # (E, I, H)

    We polarize gate and up separately per expert, then concatenate back.
    """
    if g.ndim != 3:
        raise ValueError(f"OLMoE gate_up split expects 3D tensor, got {tuple(g.shape)}")

    if g.shape[-2] % 2 != 0:
        raise ValueError(
            f"OLMoE gate_up second-to-last dim must be even, got {tuple(g.shape)}"
        )

    intermediate_size = g.shape[-2] // 2

    gate = g[:, :intermediate_size, :].contiguous()
    up = g[:, intermediate_size:, :].contiguous()

    gate_polar = _polarize_expert_stack(gate)
    up_polar = _polarize_expert_stack(up)

    return torch.cat([gate_polar, up_polar], dim=1)


class MuonHeadsPolarExpress(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by head-wise Polar Express.

    This optimizer applies SGD momentum first, then replaces the update for designated
    attention projection weights by a per-head Polar Express transform.

    Supported head-aware layouts:
        q_proj / k_proj / v_proj: layout='qkv'
            weight shape = (num_heads * head_dim, hidden_size)
        o_proj: layout='o'
            weight shape = (hidden_size, num_heads * head_dim)

    To activate per-head behavior, provide `head_configs`, a mapping whose keys can be the
    parameter objects themselves, `id(parameter)`, or parameter names when using
    `named_muon_params`. Each value should be a dict like:

        {'num_heads': ..., 'head_dim': ..., 'layout': 'qkv'}
        {'num_heads': ..., 'head_dim': ..., 'layout': 'o'}

    Parameters without a matching head config fall back to whole-matrix PolarExpressHeads,
    which in turn falls back to whole-matrix PolarExpress.
    """

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        adamw_params: Optional[Iterable[torch.nn.Parameter]] = None,
        adamw_lr: float = 0.001,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0,
        named_muon_params: Optional[Iterable[tuple[str, torch.nn.Parameter]]] = None,
        head_configs: Optional[Mapping[Any, Dict[str, Any]]] = None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        muon_params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = list(muon_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)

        self._param_names: Dict[int, str] = {}
        if named_muon_params is not None:
            for name, p in named_muon_params:
                self._param_names[id(p)] = name

        self._head_configs: Dict[int, Dict[str, Any]] = {}
        if head_configs is not None:
            for key, cfg in head_configs.items():
                normalized = dict(cfg)
                if "layout" not in normalized:
                    normalized["layout"] = "qkv"
                if isinstance(key, torch.nn.Parameter):
                    self._head_configs[id(key)] = normalized
                elif isinstance(key, int):
                    self._head_configs[key] = normalized
                elif isinstance(key, str):
                    for pid, name in self._param_names.items():
                        if name == key:
                            self._head_configs[pid] = normalized
                else:
                    raise TypeError(
                        "head_configs keys must be parameter objects, parameter ids, or names"
                    )

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            if p.ndim >= 2:
                self.state[p]['use_muon'] = True
            else:
                self.state[p]['use_muon'] = False
        for p in adamw_params:
            self.state[p]['use_muon'] = False

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0
    
    def _polarize_update(self, p: torch.nn.Parameter, g: torch.Tensor) -> torch.Tensor:
        cfg = self._head_configs.get(id(p))
        if cfg is None:
            return PolarExpressHeads(g, compute_hermitian=False)
        layout = cfg.get("layout", "qkv")
        if layout in {"qkv", "o"}:
            return PolarExpressHeads(
                g,
                num_heads=cfg["num_heads"],
                head_dim=cfg["head_dim"],
                layout=layout,
                compute_hermitian=False,
            )
        if layout == "gate_up_split":
            # gpt-oss style: (E, H, 2I), split columns.
            return _polarize_gate_up_split(g)
        if layout == "olmoe_gate_up_split":
            # OLMoE style: (E, 2I, H), split rows.
            return _polarize_olmoe_gate_up_split(g)
        if layout in {"expert", "down", "olmoe_down"}:
            return _polarize_expert_stack(g)
        raise ValueError(f"Unknown layout: {layout}")
    
    def _muon_scale(self, p: torch.nn.Parameter, g: torch.Tensor) -> float:
        cfg = self._head_configs.get(id(p))
        if cfg is None:
            rows, cols = g.size(-2), g.size(-1)
        else:
            layout = cfg.get("layout", "qkv")
            if layout == "qkv":
                rows, cols = int(cfg["head_dim"]), g.size(-1)
            elif layout == "o":
                rows, cols = g.size(-2), int(cfg["head_dim"])
            elif layout == "gate_up_split":
                # gpt-oss: (E, H, 2I), split into two (H, I) matrices.
                rows, cols = g.size(-2), g.size(-1) // 2
            elif layout == "olmoe_gate_up_split":
                # OLMoE: (E, 2I, H), split into two (I, H) matrices.
                rows, cols = g.size(-2) // 2, g.size(-1)
            elif layout in {"expert", "down", "olmoe_down"}:
                rows, cols = g.size(-2), g.size(-1)
            else:
                raise ValueError(f"Unknown layout: {layout}")
        return max(1.0, rows / cols) ** 0.5

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group['params'] if self.state[p]['use_muon']]
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            total_params = sum(p.numel() for p in params)
            ref_param = params[0] if params else None
            updates_device = ref_param.device if ref_param is not None else torch.device('cpu')
            updates_flat = torch.zeros(total_params, device=updates_device, dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue
                    state = self.state[p]
                    # Only flatten generic higher-rank tensors.
                    # Do NOT flatten MoE expert tensors.
                    cfg = self._head_configs.get(id(p))
                    layout = cfg.get("layout") if cfg is not None else None
                    if g.ndim > 2 and layout not in {
                        "expert",
                        "gate_up",
                        "gate_up_split",
                        "down",
                        "olmoe_gate_up_split",
                        "olmoe_down",
                    }:
                        g = g.view(g.size(0), -1)
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    scale = self._muon_scale(p, g)
                    g = self._polarize_update(p, g)
                    g *= scale
                    updates_flat[curr_idx:curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            if self.world_size > 1 and total_params > 0:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in params:
                g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                if p.grad is not None:
                    decoupled_weight_decay_(p.data, lr, weight_decay)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group['params'] if not self.state[p]['use_muon']]
            lr = group['adamw_lr_ratio'] * group['lr']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1
                step = state['step']
                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
