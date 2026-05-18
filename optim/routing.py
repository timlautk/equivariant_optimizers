from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# The 2D oriented and 3D MoE-expert-aware row/hybrid optimizers live in
# rownorm.py and hybrid.py.  Routing should not duplicate optimizer logic.
from .rownorm import BatchedExpertRowNormM
from .hybrid import BatchedExpertHybridPolarGradM, BatchedExpertHybridPolarGradM_GramNS


# =========================
# Mixed optimizer wrapper
# =========================

@dataclass
class MixedOptimizerConfig:
    role_to_optimizer: Dict[str, tuple]


class MixedOptimizer:
    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            for opt in self.optimizers.values():
                opt.step()
            return loss

        for opt in self.optimizers.values():
            opt.step()
        return loss

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        for k, sd in state_dict.items():
            if k in self.optimizers:
                self.optimizers[k].load_state_dict(sd)


def build_mixed_optimizer(
    param_groups: List[Dict[str, Any]],
    mixed_cfg: MixedOptimizerConfig,
) -> MixedOptimizer:
    role_to_group = {g["role"]: g for g in param_groups}
    optimizers: Dict[str, torch.optim.Optimizer] = {}

    for role, group in role_to_group.items():
        if role not in mixed_cfg.role_to_optimizer:
            raise ValueError(f"No optimizer configured for role '{role}'")

        opt_cls, base_kwargs = mixed_cfg.role_to_optimizer[role]
        kwargs = dict(base_kwargs)

        for k, v in group.items():
            if k not in {"params", "role", "param_names"}:
                kwargs[k] = v

        optimizers[role] = opt_cls(group["params"], **kwargs)

    return MixedOptimizer(optimizers)


# =========================
# Transformer routing config
# =========================

@dataclass
class TransformerRouteConfig:
    skip_frozen: bool = True
    role_hparams: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    embedding_keywords: Tuple[str, ...] = (
        "embed_tokens", "tok_embeddings", "wte", "word_embeddings",
        "token_embedding", "embedding",
    )
    lm_head_keywords: Tuple[str, ...] = ("lm_head", "output_projection")
    norm_keywords: Tuple[str, ...] = ("norm", "ln_", "layernorm", "rmsnorm")
    bias_keywords: Tuple[str, ...] = ("bias",)
    attention_keywords: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj",
        "attn", "wq", "wk", "wv", "wo",
    )
    mlp_keywords: Tuple[str, ...] = (
        "up_proj", "down_proj", "gate_proj", "gate_up_proj", "up_projs",
        "down_projs", "experts", "fc1", "fc2", "mlp", "feed_forward",
        "ffn", "w1", "w2", "w3",
    )


# =========================
# Low-level helpers
# =========================

def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    text = text.lower()
    return any(k.lower() in text for k in keywords)


def _module_lookup(model: nn.Module) -> Dict[str, nn.Module]:
    out = {"": model}
    for name, mod in model.named_modules():
        out[name] = mod
    return out


def _module_name(param_name: str) -> str:
    parts = param_name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def _is_matrix(p: nn.Parameter) -> bool:
    return p.ndim == 2


def _ancestor_modules(module_name: str, modules: Dict[str, nn.Module]) -> List[nn.Module]:
    out: List[nn.Module] = []
    parts = module_name.split(".")
    for i in range(len(parts) - 1, -1, -1):
        name = ".".join(parts[:i])
        if name in modules:
            out.append(modules[name])
    return out


def _is_olmoe_sparse_moe_module(mod: Optional[nn.Module]) -> bool:
    if mod is None:
        return False
    cls = mod.__class__.__name__.lower()
    return ("olmoe" in cls and ("moe" in cls or "router" in cls or "expert" in cls)) or (
        hasattr(mod, "gate") and hasattr(mod, "experts")
    )


def _is_gpt_oss_moe_module(mod: Optional[nn.Module]) -> bool:
    if mod is None:
        return False
    cls = mod.__class__.__name__.lower()
    return ("gptoss" in cls or "gpt_oss" in cls) and (
        "moe" in cls or "router" in cls or "expert" in cls or hasattr(mod, "experts")
    )


def _is_router_name(name: str, module_name: str) -> bool:
    lname = name.lower()
    m = module_name.lower()
    return (
        m == "gate" or m.endswith(".gate") or
        m == "router" or m.endswith(".router") or
        lname.endswith(".router.weight") or lname.endswith(".router.bias")
    )


def _infer_transformer_role(
    name: str,
    p: nn.Parameter,
    modules: Dict[str, nn.Module],
    cfg: TransformerRouteConfig,
    tied_lm_head_weight: Optional[nn.Parameter] = None,
) -> str:
    lname = name.lower()
    module_name = _module_name(name)
    module_name_l = module_name.lower()
    mod = modules.get(module_name, None)
    ancestors = _ancestor_modules(module_name, modules)
    mod_cls = mod.__class__.__name__.lower() if mod is not None else ""
    ancestor_classes = " ".join(a.__class__.__name__.lower() for a in ancestors)

    if tied_lm_head_weight is not None and p is tied_lm_head_weight:
        return "lm_head"

    if isinstance(mod, nn.Embedding) or _contains_any(lname, cfg.embedding_keywords):
        return "embedding"

    if _contains_any(lname, cfg.lm_head_keywords):
        return "lm_head"

    # Router weights:
    # OLMoE: .gate.weight [E, d]
    # gpt-oss: .router.weight [E, d], .router.bias [E]
    if _is_router_name(name, module_name_l):
        if p.ndim == 1:
            return "other"
        if _is_matrix(p):
            return "moe_router"

    # 3D MoE expert tensors:
    # OLMoE:   experts.gate_up_proj [E, 2*r, d], experts.down_proj [E, d, r]
    # gpt-oss: experts.gate_up_proj [E, d, 2*r], experts.down_proj [E, r, d]
    if p.ndim == 3 and "experts" in lname:
        is_gpt_oss = "gptoss" in mod_cls or "gpt_oss" in mod_cls or "gptoss" in ancestor_classes or "gpt_oss" in ancestor_classes
        if "gate_up_proj" in lname:
            return "gpt_oss_expert_gate_up" if is_gpt_oss else "moe_expert_gate_up"
        if "down_proj" in lname:
            return "gpt_oss_expert_down" if is_gpt_oss else "moe_expert_down"

    if _contains_any(lname, cfg.bias_keywords) or p.ndim == 1:
        return "other"

    if _contains_any(lname, cfg.norm_keywords):
        return "other"

    # Dense SwiGLU MLP neuron geometry:
    # gate_proj/up_proj [d_ff, d], down_proj [d, d_ff].
    if _is_matrix(p):
        if "gate_proj" in lname or "up_proj" in lname or "gate_up_proj" in lname:
            return "mlp_gate_up"
        if "down_proj" in lname:
            return "mlp_down"

        if _contains_any(lname, cfg.attention_keywords):
            return "matrix_attention"
        if _contains_any(lname, cfg.mlp_keywords):
            return "matrix_mlp"
        return "matrix_other"

    return "other"


# =========================
# Group builder
# =========================

def build_transformer_param_groups(
    model: nn.Module,
    cfg: Optional[TransformerRouteConfig] = None,
    tied_lm_head_weight: Optional[nn.Parameter] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    if cfg is None:
        cfg = TransformerRouteConfig()

    modules = _module_lookup(model)

    grouped_params: Dict[str, List[nn.Parameter]] = {
        "embedding": [],
        "lm_head": [],
        "moe_router": [],
        "mlp_gate_up": [],
        "mlp_down": [],
        "moe_expert_gate_up": [],
        "moe_expert_down": [],
        "gpt_oss_expert_gate_up": [],
        "gpt_oss_expert_down": [],
        "matrix_attention": [],
        "matrix_mlp": [],
        "matrix_other": [],
        "other": [],
    }
    grouped_names: Dict[str, List[str]] = {k: [] for k in grouped_params}

    seen = set()
    for name, p in model.named_parameters():
        if cfg.skip_frozen and not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))

        role = _infer_transformer_role(
            name=name,
            p=p,
            modules=modules,
            cfg=cfg,
            tied_lm_head_weight=tied_lm_head_weight,
        )
        grouped_params[role].append(p)
        grouped_names[role].append(name)

    groups: List[Dict[str, Any]] = []
    for role, params in grouped_params.items():
        if not params:
            continue
        g: Dict[str, Any] = {
            "params": params,
            "role": role,
            "param_names": grouped_names[role],
        }
        if role in cfg.role_hparams:
            g.update(cfg.role_hparams[role])
        groups.append(g)

    if verbose:
        print("Transformer parameter routing summary:")
        for g in groups:
            role = g["role"]
            n_tensors = len(g["params"])
            n_elems = sum(p.numel() for p in g["params"])
            print(f"  - {role:22s}: {n_tensors:4d} tensors, {n_elems:14d} params")
            for n in g["param_names"][:5]:
                print(f"      {n}")
            if len(g["param_names"]) > 5:
                print(f"      ... ({len(g['param_names']) - 5} more)")

    return groups


# =========================
# Public mixed-optimizer builder
# =========================

def build_transformer_mixed_optimizer(
    model: nn.Module,
    *,
    RightPolarGradM: torch.optim.Optimizer,
    LeftPolarGradM: torch.optim.Optimizer,
    RowNormM: torch.optim.Optimizer,
    HybridPolarGradM: torch.optim.Optimizer,
    MatrixOptimizerCls: torch.optim.Optimizer,
    OtherOptimizerCls: torch.optim.Optimizer,
    tied_lm_head_weight: Optional[nn.Parameter] = None,
    verbose: bool = True,
    # role-specific learning rates
    lr_other: float = 2e-4,
    lr_matrix: float = 8e-4,
    lr_embed: float = 1e-3,
    lr_lm_head: float = 1e-3,
    lr_mlp_gate_up: Optional[float] = None,
    lr_mlp_down: Optional[float] = None,
    lr_router: float = 5e-4,
    lr_moe_expert_gate_up: Optional[float] = None,
    lr_moe_expert_down: Optional[float] = None,
    # role-specific weight decay
    wd_other: float = 0.01,
    wd_matrix: float = 0.0,
    wd_embed: float = 0.0,
    wd_lm_head: float = 0.0,
    wd_mlp_gate_up: Optional[float] = None,
    wd_mlp_down: Optional[float] = None,
    wd_router: float = 0.0,
    wd_moe_expert_gate_up: Optional[float] = None,
    wd_moe_expert_down: Optional[float] = None,
    # role-specific momentum
    beta_matrix: float = 0.95,
    beta_embed: float = 0.95,
    beta_lm_head: float = 0.95,
    beta_mlp_gate_up: Optional[float] = None,
    beta_mlp_down: Optional[float] = None,
    beta_router: float = 0.95,
    beta_moe_expert_gate_up: Optional[float] = None,
    beta_moe_expert_down: Optional[float] = None,
    # geometry-aware optimizer params
    alpha: float = 1.0,
    eps: float = 1e-8,
    num_steps: int = 5,
    # matrix optimizer extras
    matrix_named_params=None,
    attention_head_configs=None,
    backend: str = "polar_express",
    right_optimizer_kwargs: Optional[Dict[str, Any]] = None,
    hybrid_optimizer_kwargs: Optional[Dict[str, Any]] = None,
    # practical choices
    lm_head_optimizer: str = "hybrid",      # {"right", "row", "hybrid", "adamw"}
    embed_optimizer: str = "right",         # {"right", "row", "hybrid", "adamw"}
    router_optimizer: str = "left",         # {"left", "row", "hybrid", "adamw"}
    mlp_up_gate_optimizer: str = "matrix",  # {"matrix", "row", "hybrid", "adamw"}
    mlp_down_optimizer: str = "matrix",     # {"matrix", "row", "hybrid", "adamw"}
    moe_expert_gate_up_optimizer: str = "row",  # {"row", "hybrid", "adamw", "matrix"}
    moe_expert_down_optimizer: str = "row",     # {"row", "hybrid", "adamw", "matrix"}
    row_mode: str = "inverse_eps",
    embed_hybrid_order: str = "row_then_polar",
    lm_head_hybrid_order: str = "row_then_polar",
    router_hybrid_order: str = "row_then_polar",
    mlp_hybrid_order: str = "row_then_polar",
    moe_expert_hybrid_order: str = "row_then_polar",
):
    if matrix_named_params is None:
        matrix_named_params = []
    if attention_head_configs is None:
        attention_head_configs = {}
    if right_optimizer_kwargs is None:
        right_optimizer_kwargs = {}
    if hybrid_optimizer_kwargs is None:
        hybrid_optimizer_kwargs = {}

    # Defaults
    lr_mlp_gate_up = lr_matrix if lr_mlp_gate_up is None else lr_mlp_gate_up
    lr_mlp_down = lr_matrix if lr_mlp_down is None else lr_mlp_down
    wd_mlp_gate_up = wd_matrix if wd_mlp_gate_up is None else wd_mlp_gate_up
    wd_mlp_down = wd_matrix if wd_mlp_down is None else wd_mlp_down
    beta_mlp_gate_up = beta_matrix if beta_mlp_gate_up is None else beta_mlp_gate_up
    beta_mlp_down = beta_matrix if beta_mlp_down is None else beta_mlp_down

    lr_moe_expert_gate_up = lr_matrix if lr_moe_expert_gate_up is None else lr_moe_expert_gate_up
    lr_moe_expert_down = lr_matrix if lr_moe_expert_down is None else lr_moe_expert_down
    wd_moe_expert_gate_up = wd_matrix if wd_moe_expert_gate_up is None else wd_moe_expert_gate_up
    wd_moe_expert_down = wd_matrix if wd_moe_expert_down is None else wd_moe_expert_down
    beta_moe_expert_gate_up = beta_matrix if beta_moe_expert_gate_up is None else beta_moe_expert_gate_up
    beta_moe_expert_down = beta_matrix if beta_moe_expert_down is None else beta_moe_expert_down

    valid_layer_opts = {"matrix", "row", "hybrid", "adamw"}
    for label, value in {
        "mlp_up_gate_optimizer": mlp_up_gate_optimizer,
        "mlp_down_optimizer": mlp_down_optimizer,
        "moe_expert_gate_up_optimizer": moe_expert_gate_up_optimizer,
        "moe_expert_down_optimizer": moe_expert_down_optimizer,
    }.items():
        if value not in valid_layer_opts:
            raise ValueError(f"Unknown {label}={value}")

    valid_hybrid_orders = {"polar_then_row", "row_then_polar"}
    for label, value in {
        "embed_hybrid_order": embed_hybrid_order,
        "lm_head_hybrid_order": lm_head_hybrid_order,
        "router_hybrid_order": router_hybrid_order,
        "mlp_hybrid_order": mlp_hybrid_order,
        "moe_expert_hybrid_order": moe_expert_hybrid_order,
    }.items():
        if value not in valid_hybrid_orders:
            raise ValueError(f"Unknown {label}={value}")

    route_cfg = TransformerRouteConfig(
        role_hparams={
            "embedding": {"lr": lr_embed},
            "lm_head": {"lr": lr_lm_head},
            "moe_router": {"lr": lr_router},
            "mlp_gate_up": {"lr": lr_mlp_gate_up},
            "mlp_down": {"lr": lr_mlp_down},
            "moe_expert_gate_up": {"lr": lr_moe_expert_gate_up},
            "moe_expert_down": {"lr": lr_moe_expert_down},
            "gpt_oss_expert_gate_up": {"lr": lr_moe_expert_gate_up},
            "gpt_oss_expert_down": {"lr": lr_moe_expert_down},
            "matrix_attention": {"lr": lr_matrix},
            "matrix_mlp": {"lr": lr_matrix},
            "matrix_other": {"lr": lr_matrix},
            "other": {"lr": lr_other},
        }
    )

    param_groups = build_transformer_param_groups(
        model,
        cfg=route_cfg,
        tied_lm_head_weight=tied_lm_head_weight,
        verbose=verbose,
    )

    def _filter_kwargs(optimizer_cls: torch.optim.Optimizer, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted = inspect.signature(optimizer_cls.__init__).parameters
        return {k: v for k, v in kwargs.items() if k in accepted}

    def _right_opt_kwargs(lr: float, beta: float, weight_decay: float) -> Dict[str, Any]:
        kwargs = {
            "lr": lr,
            "beta": beta,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "backend": backend,
            "num_steps": num_steps,
        }
        kwargs.update(right_optimizer_kwargs)
        return _filter_kwargs(RightPolarGradM, kwargs)

    def _hybrid_opt_kwargs(
        lr: float,
        beta: float,
        weight_decay: float,
        *,
        left: bool,
        center_rows: bool,
        order: str,
        orientation: str = "row",
    ) -> Dict[str, Any]:
        kwargs = {
            "lr": lr,
            "beta": beta,
            "alpha": alpha,
            "weight_decay": weight_decay,
            "row_mode": row_mode,
            "eps": eps,
            "backend": backend,
            "num_steps": num_steps,
            "left": left,
            "center_rows": center_rows,
            "order": order,
            "orientation": orientation,
        }
        kwargs.update(hybrid_optimizer_kwargs)
        kwargs.update({"left": left, "center_rows": center_rows, "order": order, "orientation": orientation})
        return _filter_kwargs(HybridPolarGradM, kwargs)

    def _adamw_opt(lr: float, weight_decay: float):
        return (
            OtherOptimizerCls,
            {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.95), "fused": True},
        )

    def _embed_opt():
        if embed_optimizer == "right":
            return (RightPolarGradM, _right_opt_kwargs(lr_embed, beta_embed, wd_embed))
        if embed_optimizer == "row":
            return (
                RowNormM,
                {"lr": lr_embed, "beta": beta_embed, "weight_decay": wd_embed,
                 "row_mode": row_mode, "eps": eps, "center_rows": False, "orientation": "row"},
            )
        if embed_optimizer == "hybrid":
            return (
                HybridPolarGradM,
                _hybrid_opt_kwargs(lr_embed, beta_embed, wd_embed, left=False, center_rows=False, order=embed_hybrid_order),
            )
        if embed_optimizer == "adamw":
            return _adamw_opt(lr_embed, wd_embed)
        raise ValueError(f"Unknown embed_optimizer={embed_optimizer}")

    def _lm_head_opt():
        if lm_head_optimizer == "right":
            return (RightPolarGradM, _right_opt_kwargs(lr_lm_head, beta_lm_head, wd_lm_head))
        if lm_head_optimizer == "row":
            return (
                RowNormM,
                {"lr": lr_lm_head, "beta": beta_lm_head, "weight_decay": wd_lm_head,
                 "row_mode": row_mode, "eps": eps, "center_rows": False, "orientation": "row"},
            )
        if lm_head_optimizer == "hybrid":
            return (
                HybridPolarGradM,
                _hybrid_opt_kwargs(lr_lm_head, beta_lm_head, wd_lm_head, left=False, center_rows=False, order=lm_head_hybrid_order),
            )
        if lm_head_optimizer == "adamw":
            return _adamw_opt(lr_lm_head, wd_lm_head)
        raise ValueError(f"Unknown lm_head_optimizer={lm_head_optimizer}")

    def _router_opt():
        if router_optimizer == "left":
            return (
                LeftPolarGradM,
                {
                    "lr": lr_router, "beta": beta_router, "alpha": alpha, "eps": eps,
                    "weight_decay": wd_router, "backend": backend, "num_steps": num_steps,
                    "center_rows": True,
                },
            )
        if router_optimizer == "row":
            return (
                RowNormM,
                {"lr": lr_router, "beta": beta_router, "weight_decay": wd_router,
                 "row_mode": row_mode, "eps": eps, "center_rows": True, "orientation": "row"},
            )
        if router_optimizer == "hybrid":
            return (
                HybridPolarGradM,
                _hybrid_opt_kwargs(lr_router, beta_router, wd_router, left=True, center_rows=True, order=router_hybrid_order),
            )
        if router_optimizer == "adamw":
            return _adamw_opt(lr_router, wd_router)
        raise ValueError(f"Unknown router_optimizer={router_optimizer}")

    def _oriented_mlp_opt(choice: str, *, orientation: str, lr: float, weight_decay: float, beta: float):
        if choice == "matrix":
            return (
                MatrixOptimizerCls,
                {"lr": lr, "weight_decay": weight_decay, "momentum": beta,
                 "named_muon_params": matrix_named_params, "head_configs": attention_head_configs},
            )
        if choice == "row":
            return (
                RowNormM,
                {"lr": lr, "beta": beta, "weight_decay": weight_decay,
                 "row_mode": row_mode, "eps": eps, "center_rows": False, "orientation": orientation},
            )
        if choice == "hybrid":
            return (
                HybridPolarGradM,
                _hybrid_opt_kwargs(lr, beta, weight_decay, left=False, center_rows=False, order=mlp_hybrid_order, orientation=orientation),
            )
        if choice == "adamw":
            return _adamw_opt(lr, weight_decay)
        raise ValueError(f"Unknown MLP optimizer choice={choice}")

    def _expert_opt(choice: str, *, layout: str, lr: float, weight_decay: float, beta: float):
        if choice == "row":
            return (
                BatchedExpertRowNormM,
                {"lr": lr, "beta": beta, "weight_decay": weight_decay,
                 "row_mode": row_mode, "eps": eps, "expert_layout": layout},
            )
        if choice == "hybrid":
            # Match the expert hybrid implementation to the main hybrid class
            # selected by the training script.  If HybridPolarGradM is the
            # GramNS implementation, use the expert GramNS wrapper and pass
            # through the same GramNS kwargs.
            use_expert_gramns = "gramns" in getattr(HybridPolarGradM, "__name__", "").lower()
            expert_hybrid_cls = (
                BatchedExpertHybridPolarGradM_GramNS
                if use_expert_gramns
                else BatchedExpertHybridPolarGradM
            )
            kwargs = {
                "lr": lr,
                "beta": beta,
                "alpha": alpha,
                "weight_decay": weight_decay,
                "row_mode": row_mode,
                "eps": eps,
                "backend": backend,
                "num_steps": num_steps,
                "order": moe_expert_hybrid_order,
                "left": False,
                "center_rows": False,
                "expert_layout": layout,
            }
            if use_expert_gramns:
                kwargs.update(hybrid_optimizer_kwargs)
            return (expert_hybrid_cls, _filter_kwargs(expert_hybrid_cls, kwargs))
        if choice == "adamw":
            return _adamw_opt(lr, weight_decay)
        if choice == "matrix":
            # The generic matrix optimizer usually expects 2D tensors. Use AdamW
            # as a safe fallback unless you have a 3D-aware MatrixOptimizerCls.
            return _adamw_opt(lr, weight_decay)
        raise ValueError(f"Unknown MoE expert optimizer choice={choice}")

    mixed_cfg = MixedOptimizerConfig(
        role_to_optimizer={
            "embedding": _embed_opt(),
            "lm_head": _lm_head_opt(),
            "moe_router": _router_opt(),
            "mlp_gate_up": _oriented_mlp_opt(
                mlp_up_gate_optimizer, orientation="row",
                lr=lr_mlp_gate_up, weight_decay=wd_mlp_gate_up, beta=beta_mlp_gate_up,
            ),
            "mlp_down": _oriented_mlp_opt(
                mlp_down_optimizer, orientation="col",
                lr=lr_mlp_down, weight_decay=wd_mlp_down, beta=beta_mlp_down,
            ),
            # OLMoE expert tensor layouts
            "moe_expert_gate_up": _expert_opt(
                moe_expert_gate_up_optimizer, layout="row",
                lr=lr_moe_expert_gate_up, weight_decay=wd_moe_expert_gate_up, beta=beta_moe_expert_gate_up,
            ),
            "moe_expert_down": _expert_opt(
                moe_expert_down_optimizer, layout="col",
                lr=lr_moe_expert_down, weight_decay=wd_moe_expert_down, beta=beta_moe_expert_down,
            ),
            # gpt-oss expert tensor layouts
            "gpt_oss_expert_gate_up": _expert_opt(
                moe_expert_gate_up_optimizer, layout="gpt_oss_gate_up_pair",
                lr=lr_moe_expert_gate_up, weight_decay=wd_moe_expert_gate_up, beta=beta_moe_expert_gate_up,
            ),
            "gpt_oss_expert_down": _expert_opt(
                moe_expert_down_optimizer, layout="row",
                lr=lr_moe_expert_down, weight_decay=wd_moe_expert_down, beta=beta_moe_expert_down,
            ),
            "matrix_attention": (
                MatrixOptimizerCls,
                {"lr": lr_matrix, "weight_decay": wd_matrix, "momentum": beta_matrix,
                 "named_muon_params": matrix_named_params, "head_configs": attention_head_configs},
            ),
            "matrix_mlp": (
                MatrixOptimizerCls,
                {"lr": lr_matrix, "weight_decay": wd_matrix, "momentum": beta_matrix,
                 "named_muon_params": matrix_named_params, "head_configs": attention_head_configs},
            ),
            "matrix_other": (
                MatrixOptimizerCls,
                {"lr": lr_matrix, "weight_decay": wd_matrix, "momentum": beta_matrix,
                 "named_muon_params": matrix_named_params, "head_configs": attention_head_configs},
            ),
            "other": _adamw_opt(lr_other, wd_other),
        }
    )

    return build_mixed_optimizer(param_groups, mixed_cfg)


# Backward-compatible no-op expert config builders.
# The new routing handles MoE expert tensors directly.
def build_olmoe_expert_configs(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    return {}


def build_gpt_oss_expert_configs(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    return {}


# =========================
# Vision builder placeholder/import compatibility
# =========================
# Keep your existing build_vision_mixed_optimizer below this point if needed.
