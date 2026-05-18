import torch
import torch.nn as nn


def is_matrix_param(p: torch.Tensor) -> bool:
    return p.ndim == 2


def decoupled_weight_decay_(p: torch.Tensor, lr: float, wd: float) -> None:
    if wd != 0.0:
        p.mul_(1.0 - lr * wd)


def centered_router_momentum(M: torch.Tensor) -> torch.Tensor:
    """
    Center rows for MoE routers: Pi_perp M
    """
    return M - M.mean(dim=0, keepdim=True)


def _pick_int(*values):
    for value in values:
        if value is not None:
            return int(value)
    return None


def _resolve_attention_head_spec(module: nn.Module):
    config = getattr(module, "config", None)
    q_proj = module.q_proj
    k_proj = module.k_proj
    v_proj = module.v_proj
    o_proj = module.o_proj

    num_attention_heads = _pick_int(
        getattr(module, "num_heads", None),
        getattr(module, "num_attention_heads", None),
        getattr(config, "num_attention_heads", None),
    )
    num_key_value_heads = _pick_int(
        getattr(module, "num_key_value_heads", None),
        getattr(module, "n_kv_heads", None),
        getattr(config, "num_key_value_heads", None),
    )
    head_dim = _pick_int(
        getattr(module, "head_dim", None),
        getattr(config, "head_dim", None),
    )

    q_out = q_proj.weight.shape[0]
    k_out = k_proj.weight.shape[0]
    v_out = v_proj.weight.shape[0]
    o_in = o_proj.weight.shape[1]

    if head_dim is None and num_attention_heads is not None and q_out % num_attention_heads == 0:
        head_dim = q_out // num_attention_heads
    if head_dim is None and num_key_value_heads is not None and k_out % num_key_value_heads == 0:
        head_dim = k_out // num_key_value_heads

    if num_attention_heads is None and head_dim is not None and q_out % head_dim == 0:
        num_attention_heads = q_out // head_dim
    if num_key_value_heads is None and head_dim is not None and k_out % head_dim == 0:
        num_key_value_heads = k_out // head_dim

    if num_attention_heads is None or num_key_value_heads is None or head_dim is None:
        return None

    expected_q_out = num_attention_heads * head_dim
    expected_kv_out = num_key_value_heads * head_dim
    if q_out != expected_q_out or k_out != expected_kv_out or v_out != expected_kv_out or o_in != expected_q_out:
        raise ValueError(
            f"Could not infer a consistent head layout for {module.__class__.__name__}: "
            f"q_proj={tuple(q_proj.weight.shape)}, "
            f"k_proj={tuple(k_proj.weight.shape)}, "
            f"v_proj={tuple(v_proj.weight.shape)}, "
            f"o_proj={tuple(o_proj.weight.shape)}, "
            f"num_attention_heads={num_attention_heads}, "
            f"num_key_value_heads={num_key_value_heads}, "
            f"head_dim={head_dim}"
        )

    return num_attention_heads, num_key_value_heads, head_dim


def build_attention_head_configs(model: nn.Module):
    """
    Build per-parameter head configs for HF-style attention modules exposing
    `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

    This handles grouped-query attention layouts where `q_proj`/`o_proj` use
    `num_attention_heads`, while `k_proj`/`v_proj` use `num_key_value_heads`.
    Head counts may live on either the module itself or `module.config`, so we
    resolve from both and validate the inferred spec against the projection
    weight shapes before wiring `PolarExpressHeads`.
    """
    head_configs = {}

    for module_name, module in model.named_modules():
        has_all_projs = all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))
        if not has_all_projs:
            continue

        head_spec = _resolve_attention_head_spec(module)
        if head_spec is None:
            raise ValueError(
                f"Could not resolve attention head config for module {module_name or module.__class__.__name__}"
            )
        num_attention_heads, num_key_value_heads, head_dim = head_spec

        q_name = f"{module_name}.q_proj.weight" if module_name else "q_proj.weight"
        k_name = f"{module_name}.k_proj.weight" if module_name else "k_proj.weight"
        v_name = f"{module_name}.v_proj.weight" if module_name else "v_proj.weight"
        o_name = f"{module_name}.o_proj.weight" if module_name else "o_proj.weight"

        head_configs[q_name] = {
            "num_heads": int(num_attention_heads),
            "head_dim": int(head_dim),
            "layout": "qkv",
        }
        head_configs[k_name] = {
            "num_heads": int(num_key_value_heads),
            "head_dim": int(head_dim),
            "layout": "qkv",
        }
        head_configs[v_name] = {
            "num_heads": int(num_key_value_heads),
            "head_dim": int(head_dim),
            "layout": "qkv",
        }
        head_configs[o_name] = {
            "num_heads": int(num_attention_heads),
            "head_dim": int(head_dim),
            "layout": "o",
        }

    for name, _ in model.named_parameters():
        if "experts.gate_up_proj" in name:
            head_configs[name] = {
                "layout": "gate_up_split",
            }
        if "experts.down_proj" in name:
            head_configs[name] = {
                "layout": "down",
            }

    return head_configs


def build_olmoe_expert_configs(model: nn.Module):
    """
    Build per-parameter configs for HF OlmoeExperts.

    OlmoeExperts stores:
        gate_up_proj: (num_experts, 2 * intermediate_size, hidden_size)
        down_proj:    (num_experts, hidden_size, intermediate_size)

    These are raw nn.Parameter tensors, not nn.Linear modules, so parameter
    names do not end with ".weight".
    """
    expert_configs = {}
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name != "OlmoeExperts":
            continue
        if hasattr(module, "gate_up_proj"):
            p = module.gate_up_proj
            if p.ndim != 3:
                raise ValueError(
                    f"{module_name}.gate_up_proj should be 3D, got {tuple(p.shape)}"
                )
            if p.shape[-2] % 2 != 0:
                raise ValueError(
                    f"{module_name}.gate_up_proj second-to-last dim should be even, got {tuple(p.shape)}"
                )
            name = f"{module_name}.gate_up_proj" if module_name else "gate_up_proj"
            expert_configs[name] = {
                "layout": "olmoe_gate_up_split",
            }
        if hasattr(module, "down_proj"):
            p = module.down_proj
            if p.ndim != 3:
                raise ValueError(
                    f"{module_name}.down_proj should be 3D, got {tuple(p.shape)}"
                )
            name = f"{module_name}.down_proj" if module_name else "down_proj"
            expert_configs[name] = {
                "layout": "olmoe_down",
            }
    return expert_configs


def build_gpt_oss_expert_configs(model: nn.Module):
    """
    Build per-parameter configs for gpt-oss GptOssExperts parameters.

    gpt-oss expert tensors are not attention-head tensors.

    gate_up_proj:
        shape = (num_experts, hidden_size, 2 * intermediate_size)
        layout = "gate_up_split"

    down_proj:
        shape = (num_experts, intermediate_size, hidden_size)
        layout = "down"
    """
    expert_configs = {}
    for module_name, module in model.named_modules():
        if hasattr(module, "gate_up_proj"):
            p = module.gate_up_proj
            if p.ndim != 3:
                raise ValueError(
                    f"{module_name}.gate_up_proj should be 3D, got {tuple(p.shape)}"
                )
            if p.shape[-1] % 2 != 0:
                raise ValueError(
                    f"{module_name}.gate_up_proj last dim should be even, got {tuple(p.shape)}"
                )
            name = f"{module_name}.gate_up_proj" if module_name else "gate_up_proj"
            expert_configs[name] = {
                "layout": "gate_up_split",
            }
        if hasattr(module, "down_proj"):
            p = module.down_proj
            if p.ndim != 3:
                raise ValueError(
                    f"{module_name}.down_proj should be 3D, got {tuple(p.shape)}"
                )
            name = f"{module_name}.down_proj" if module_name else "down_proj"
            expert_configs[name] = {
                "layout": "down",
            }
    return expert_configs