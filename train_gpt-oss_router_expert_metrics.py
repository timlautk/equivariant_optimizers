import os
import time
import random
import glob
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    GptOssConfig,
    GptOssForCausalLM,
    get_cosine_schedule_with_warmup,
)

from torch.utils.tensorboard import SummaryWriter

from optim import (
    RightPolarGradM,
    RightPolarGradM_GramNS,
    LeftPolarGradM,
    RowNormM,
    HybridPolarGradM,
    HybridPolarGradM_GramNS,
    MuonHeadsPolarExpress,
    build_transformer_mixed_optimizer,
    build_attention_head_configs,
    build_gpt_oss_expert_configs,
)


# --------------------------
# DDP utils
# --------------------------

def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0

def ddp_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1

def ddp_is_main() -> bool:
    return ddp_rank() == 0

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------
# Simple multi-scheduler wrapper
# --------------------------

class MixedScheduler:
    def __init__(self, schedulers: Dict[str, object]):
        self.schedulers = schedulers

    def step(self):
        for sched in self.schedulers.values():
            sched.step()

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.schedulers.items()}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.schedulers:
                self.schedulers[k].load_state_dict(v)


# --------------------------
# Binary shard loader
# --------------------------

MAGIC = 20260317
HEADER_SIZE_INT32 = 256
VERSION = 2


def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(HEADER_SIZE_INT32 * 4), dtype=np.int32)

    if len(header) != HEADER_SIZE_INT32:
        raise ValueError(f"{filename}: incomplete header")
    if int(header[0]) != MAGIC:
        raise ValueError(f"{filename}: magic mismatch, got {int(header[0])}, expected {MAGIC}")
    if int(header[1]) != VERSION:
        raise ValueError(f"{filename}: version mismatch, got {int(header[1])}, expected {VERSION}")

    ntok = int(header[2])
    bytes_per_token = int(header[3])
    if bytes_per_token not in (2, 4):
        raise ValueError(f"{filename}: unsupported bytes_per_token={bytes_per_token}")

    return {
        "ntok": ntok,
        "bytes_per_token": bytes_per_token,
        "delimiter_id": int(header[4]),
        "n_prepend": int(header[5]),
        "n_append": int(header[6]),
    }


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(HEADER_SIZE_INT32 * 4), dtype=np.int32)

        if int(header[0]) != MAGIC:
            raise ValueError(f"{filename}: magic mismatch, got {int(header[0])}, expected {MAGIC}")
        if int(header[1]) != VERSION:
            raise ValueError(f"{filename}: version mismatch, got {int(header[1])}, expected {VERSION}")

        ntok = int(header[2])
        bytes_per_token = int(header[3])

        if bytes_per_token == 2:
            dtype = np.uint16
        elif bytes_per_token == 4:
            dtype = np.uint32
        else:
            raise ValueError(f"{filename}: unsupported bytes_per_token={bytes_per_token}")

        tokens = np.frombuffer(f.read(), dtype=dtype)

    if len(tokens) != ntok:
        raise ValueError(f"{filename}: read {len(tokens)} tokens, header says {ntok}")

    return tokens


class DistributedTokenLoader:
    def __init__(self, filename_pattern, B, T, dp_rank, dp_world_size, device=None):
        self.dp_rank = int(dp_rank)
        self.dp_world_size = int(dp_world_size)
        self.B = int(B)
        self.T = int(T)
        self.device = device

        self.files = sorted(glob.glob(filename_pattern))
        if len(self.files) < 1:
            raise FileNotFoundError(f"Could not find any files matching: {filename_pattern}")

        self.shard_meta = [_peek_data_shard(f) for f in self.files]

        min_needed = self.dp_world_size * self.B * self.T + 1
        bad = [f for f, meta in zip(self.files, self.shard_meta) if meta["ntok"] < min_needed]
        if bad:
            raise ValueError(
                f"Some shards are too small. Need at least world_size*B*T+1={min_needed} tokens. "
                f"Examples: {bad[:3]}"
            )

        self.reset()

    def _load_tokens_for_current_shard(self):
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def reset(self):
        self.current_shard = 0
        self.current_position = self.dp_rank * self.B * self.T
        self._load_tokens_for_current_shard()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.dp_rank * self.B * self.T
        self._load_tokens_for_current_shard()

    def next_batch(self):
        B, T = self.B, self.T
        need = B * T + 1

        buf = self.tokens[self.current_position : self.current_position + need]
        if len(buf) < need:
            self.advance()
            buf = self.tokens[self.current_position : self.current_position + need]
            if len(buf) < need:
                raise RuntimeError(
                    f"Shard {self.files[self.current_shard]} too small at position "
                    f"{self.current_position}: need {need}, got {len(buf)}"
                )

        buf = torch.tensor(buf.astype(np.int64), dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.dp_world_size

        if self.current_position + (B * T * self.dp_world_size + 1) > len(self.tokens):
            self.advance()

        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

        return x, y

    def state_dict(self):
        return {
            f"current_shard_rank_{self.dp_rank}": self.current_shard,
            f"current_position_rank_{self.dp_rank}": self.current_position,
            "dataloader_world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict):
        world_size_ckpt = state_dict.get("dataloader_world_size")
        if world_size_ckpt != self.dp_world_size:
            raise NotImplementedError(
                f"Cannot restore loader with different world size: ckpt={world_size_ckpt}, "
                f"current={self.dp_world_size}"
            )
        self.current_shard = state_dict[f"current_shard_rank_{self.dp_rank}"]
        self.current_position = state_dict[f"current_position_rank_{self.dp_rank}"]
        self._load_tokens_for_current_shard()


def resolve_bin_pattern(data_dir: str, pattern: str) -> str:
    if os.path.isabs(pattern):
        return pattern
    return os.path.join(data_dir, pattern)


def maybe_reinit_for_lm(module: nn.Module, std: float = 0.02):
    """
    Optional scratch-style reinit if you instantiate from config instead of
    loading pretrained weights.
    """
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



# --------------------------
# Router auxiliary losses and diagnostics
# --------------------------

def _router_logits_list(router_logits):
    if router_logits is None:
        return []
    if isinstance(router_logits, torch.Tensor):
        return [router_logits]
    return [x for x in router_logits if x is not None]


def get_router_logits_from_outputs(outputs):
    if hasattr(outputs, "router_logits"):
        return outputs.router_logits
    if isinstance(outputs, dict):
        return outputs.get("router_logits", None)
    return None


def router_load_balancing_loss(router_logits, num_experts: int, top_k: int):
    """
    Differentiable-through-probabilities load-balancing loss, following the
    Switch/MoE style statistic N * sum_i f_i P_i. The hard assignment fraction
    f_i is computed from top-k router choices and detached; P_i is the mean
    router probability mass and remains differentiable.
    """
    logits_list = _router_logits_list(router_logits)
    if not logits_list:
        return None

    losses = []
    k = min(int(top_k), int(num_experts))
    for logits in logits_list:
        flat = logits.float().reshape(-1, logits.shape[-1])
        if flat.numel() == 0:
            continue
        probs = torch.softmax(flat, dim=-1)
        topk_idx = torch.topk(probs, k=k, dim=-1).indices
        assignment = F.one_hot(topk_idx, num_classes=num_experts).sum(dim=1).to(probs.dtype)
        # f_i sums to 1 over experts; P_i also sums to 1.
        f_i = assignment.mean(dim=0) / float(k)
        p_i = probs.mean(dim=0)
        losses.append(num_experts * torch.sum(f_i.detach() * p_i))

    if not losses:
        return None
    return torch.stack(losses).mean()


def router_z_loss(router_logits):
    """
    Router z-loss: mean square log-sum-exp of router logits, averaged over MoE layers.
    """
    logits_list = _router_logits_list(router_logits)
    if not logits_list:
        return None
    vals = []
    for logits in logits_list:
        flat = logits.float().reshape(-1, logits.shape[-1])
        if flat.numel() == 0:
            continue
        vals.append(torch.logsumexp(flat, dim=-1).square().mean())
    if not vals:
        return None
    return torch.stack(vals).mean()


@torch.no_grad()
def compute_router_metrics(router_logits, num_experts: int, top_k: int, dead_expert_threshold_scale: float = 0.1):
    """
    Computes global router diagnostics and per-layer expert-assignment
    distributions. In DDP, token/expert counts and scalar sums are reduced
    across ranks before metrics are formed.

    Returns:
        metrics: dict[str, Tensor]
            Metrics averaged across MoE layers.
        per_layer: list[dict[str, Tensor]]
            Per-layer summary metrics.
        load_fractions: Tensor | None
            Tensor of shape [num_router_layers, num_experts] whose (l, i)
            entry is the fraction of top-k assignments in layer l sent to
            expert i. Rows sum to 1.
        prob_masses: Tensor | None
            Tensor of shape [num_router_layers, num_experts] whose (l, i)
            entry is the average softmax probability mass assigned to expert i.
            Rows sum to 1.
    """
    logits_list = _router_logits_list(router_logits)
    if not logits_list:
        return {}, [], None, None

    k = min(int(top_k), int(num_experts))
    eps = 1e-12
    per_layer = []
    load_fractions = []
    prob_masses = []
    accum = {
        "router/load_balancing_loss": [],
        "router/load_cv": [],
        "router/load_entropy": [],
        "router/dead_expert_fraction": [],
        "router/top_expert_load": [],
        "router/prob_cv": [],
        "router/prob_entropy": [],
        "router/token_entropy": [],
        "router/z_loss": [],
        "router/logit_rms": [],
        "router/logit_max_abs": [],
        "router/logsumexp_mean": [],
        "router/logsumexp_max": [],
    }

    for layer_idx, logits in enumerate(logits_list):
        flat = logits.float().reshape(-1, logits.shape[-1])
        if flat.numel() == 0:
            continue
        probs = torch.softmax(flat, dim=-1)
        topk_idx = torch.topk(probs, k=k, dim=-1).indices
        assignment = F.one_hot(topk_idx, num_classes=num_experts).sum(dim=1).to(probs.dtype)

        counts = assignment.sum(dim=0)                           # [E], sums to N*k
        prob_sums = probs.sum(dim=0)                             # [E], sums to N
        token_count = torch.tensor(float(flat.shape[0]), device=flat.device)
        logsumexp = torch.logsumexp(flat, dim=-1)
        z_sum = logsumexp.square().sum()
        logsumexp_sum = logsumexp.sum()
        logsumexp_max = logsumexp.max()
        entropy_sum = (-(probs * (probs.clamp_min(eps).log())).sum(dim=-1)).sum()
        logit_sq_sum = flat.square().sum()
        logit_count = torch.tensor(float(flat.numel()), device=flat.device)
        logit_max_abs = flat.abs().max()

        if ddp_is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(prob_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(z_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(logsumexp_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(entropy_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(logit_sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(logit_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(logit_max_abs, op=dist.ReduceOp.MAX)
            dist.all_reduce(logsumexp_max, op=dist.ReduceOp.MAX)

        f_i = counts / (token_count * float(k)).clamp_min(eps)
        p_i = prob_sums / token_count.clamp_min(eps)
        uniform = 1.0 / float(num_experts)
        dead_threshold = float(dead_expert_threshold_scale) * uniform

        load_mean = f_i.mean().clamp_min(eps)
        prob_mean = p_i.mean().clamp_min(eps)
        load_entropy = -(f_i * f_i.clamp_min(eps).log()).sum() / np.log(num_experts)
        prob_entropy = -(p_i * p_i.clamp_min(eps).log()).sum() / np.log(num_experts)
        layer_metrics = {
            "router/load_balancing_loss": num_experts * torch.sum(f_i * p_i),
            "router/load_cv": f_i.std(unbiased=False) / load_mean,
            "router/load_entropy": load_entropy,
            "router/dead_expert_fraction": (f_i < dead_threshold).float().mean(),
            "router/top_expert_load": f_i.max(),
            "router/prob_cv": p_i.std(unbiased=False) / prob_mean,
            "router/prob_entropy": prob_entropy,
            "router/token_entropy": (entropy_sum / token_count.clamp_min(eps)) / np.log(num_experts),
            "router/z_loss": z_sum / token_count.clamp_min(eps),
            "router/logit_rms": torch.sqrt(logit_sq_sum / logit_count.clamp_min(eps)),
            "router/logit_max_abs": logit_max_abs,
            "router/logsumexp_mean": logsumexp_sum / token_count.clamp_min(eps),
            "router/logsumexp_max": logsumexp_max,
        }
        per_layer.append({k_: v_.detach() for k_, v_ in layer_metrics.items()})
        load_fractions.append(f_i.detach())
        prob_masses.append(p_i.detach())
        for k_, v_ in layer_metrics.items():
            accum[k_].append(v_.detach())

    metrics = {}
    for k_, vals in accum.items():
        if vals:
            metrics[k_] = torch.stack(vals).mean()

    load_fractions_tensor = torch.stack(load_fractions) if load_fractions else None
    prob_masses_tensor = torch.stack(prob_masses) if prob_masses else None
    return metrics, per_layer, load_fractions_tensor, prob_masses_tensor


def tensor_metrics_to_float(metrics):
    return {k: float(v.detach().cpu()) for k, v in metrics.items()}


def maybe_limit_layers(x, max_layers: int):
    if x is None:
        return None
    if max_layers is not None and int(max_layers) > 0:
        return x[: int(max_layers)]
    return x


def log_router_assignment_tensors(
    writer,
    prefix: str,
    step: int,
    load_fractions,
    prob_masses,
    *,
    log_expert_scalars: bool = False,
    log_heatmaps: bool = True,
    log_histograms: bool = True,
    max_layers: int = 0,
    heatmap_clip: float = 3.0,
):
    """
    Logs per-layer expert assignment diagnostics.

    load_fractions and prob_masses are [num_layers, num_experts] tensors.
    The heatmaps are normalized by the uniform value 1 / num_experts, so a
    value of 1 means perfectly uniform load/probability mass. Values are
    clipped at heatmap_clip for visualization.
    """
    if writer is None or load_fractions is None:
        return

    load_fractions = maybe_limit_layers(load_fractions.detach().float().cpu(), max_layers)
    if prob_masses is not None:
        prob_masses = maybe_limit_layers(prob_masses.detach().float().cpu(), max_layers)

    num_layers, num_experts = load_fractions.shape
    uniform = 1.0 / max(float(num_experts), 1.0)
    clip = max(float(heatmap_clip), 1e-6)

    if log_heatmaps:
        # TensorBoard image shape is [C, H, W]. H=layer index, W=expert index.
        rel_load = (load_fractions / uniform).clamp(0.0, clip) / clip
        writer.add_image(f"{prefix}/expert_load_fraction_heatmap", rel_load.unsqueeze(0), step)
        if prob_masses is not None:
            rel_prob = (prob_masses / uniform).clamp(0.0, clip) / clip
            writer.add_image(f"{prefix}/expert_probability_mass_heatmap", rel_prob.unsqueeze(0), step)

    if log_histograms:
        writer.add_histogram(f"{prefix}/expert_load_fraction_hist", load_fractions.flatten(), step)
        if prob_masses is not None:
            writer.add_histogram(f"{prefix}/expert_probability_mass_hist", prob_masses.flatten(), step)

    if log_expert_scalars:
        # This can create many TensorBoard time series: num_layers * num_experts.
        for li in range(num_layers):
            for ei in range(num_experts):
                writer.add_scalar(f"{prefix}/layer_{li}/expert_{ei}/load_fraction", float(load_fractions[li, ei]), step)
                if prob_masses is not None:
                    writer.add_scalar(f"{prefix}/layer_{li}/expert_{ei}/prob_mass", float(prob_masses[li, ei]), step)


def save_router_assignment_npz(out_dir: str, prefix: str, step: int, load_fractions, prob_masses):
    """Save raw per-layer expert assignment arrays for later heatmap plots."""
    if out_dir is None or out_dir == "" or load_fractions is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    safe_prefix = prefix.replace("/", "_")
    path = os.path.join(out_dir, f"{safe_prefix}_step_{int(step):08d}.npz")
    arrays = {
        "step": np.array(int(step), dtype=np.int64),
        "load_fractions": load_fractions.detach().float().cpu().numpy(),
    }
    if prob_masses is not None:
        arrays["prob_masses"] = prob_masses.detach().float().cpu().numpy()
    np.savez_compressed(path, **arrays)

def get_model_and_loaders(
    model_name,
    hidden_size,
    device_batch_size,
    num_experts,
    num_hidden_layers,
    seq_len,
    data_dir,
    train_bin_pattern,
    val_bin_pattern,
    device,
    output_router_logits=False,
):
    resolved_train_pattern = resolve_bin_pattern(data_dir, train_bin_pattern)
    resolved_val_pattern = resolve_bin_pattern(data_dir, val_bin_pattern)

    train_loader = DistributedTokenLoader(
        filename_pattern=resolved_train_pattern,
        B=device_batch_size,
        T=seq_len,
        dp_rank=ddp_rank(),
        dp_world_size=ddp_world_size(),
        device=device,
    )

    val_loader = DistributedTokenLoader(
        filename_pattern=resolved_val_pattern,
        B=device_batch_size,
        T=seq_len,
        dp_rank=ddp_rank(),
        dp_world_size=ddp_world_size(),
        device=device,
    )

    if model_name == "gpt-oss":
        config = GptOssConfig(
            attention_bias=True,
            attention_dropout=0.0,
            eos_token_id=200002,
            experts_per_token=4,
            head_dim=64,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=hidden_size,
            max_position_embeddings=131072,
            model_type="gpt_oss",
            num_attention_heads=64,
            num_experts_per_tok=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
            num_local_experts=num_experts,
            output_router_logits=bool(output_router_logits),
            pad_token_id=199999,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=150000,
            router_aux_loss_coef=0.0,  # computed manually below for ablations
            sliding_window=128,
            swiglu_limit=7.0,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
            vocab_size=201088,
        )
        model = GptOssForCausalLM(config)
        init_std = getattr(config, "initializer_range", 0.02)
        model.apply(lambda m: maybe_reinit_for_lm(m, std=init_std))
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader, val_loader


# --------------------------
# Wrapper so routing can use MuonHeadsPolarExpress role-wise
# --------------------------

class MuonHeadsPolarExpressWrapper(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0.0,
        momentum=0.95,
        named_muon_params=None,
        head_configs=None,
    ):
        if named_muon_params is None:
            named_muon_params = []
        if head_configs is None:
            head_configs = {}

        self._inner = MuonHeadsPolarExpress(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
            named_muon_params=named_muon_params,
            head_configs=head_configs,
        )
        self.param_groups = self._inner.param_groups
        self.defaults = getattr(self._inner, "defaults", {})
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        return self._inner.step(closure=closure)

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)


class LeftPolarGradMUncentered(LeftPolarGradM):
    """LeftPolarGradM variant for router ablations without row-centering.

    The normal router-compatible LeftPolarGradM uses center_rows=True to remove
    the shared-logit-shift direction. This class forces center_rows=False, so it
    applies a Muon/left-polar update in the full expert space. It is intended as
    a deliberately misspecified ablation, not as the symmetry-compatible router
    update.
    """

    def __init__(self, params, *args, **kwargs):
        kwargs["center_rows"] = False
        super().__init__(params, *args, **kwargs)


if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-oss")
    parser.add_argument("--lm_head_optimizer", type=str, default="row", choices=["right", "row", "hybrid", "adamw"])
    parser.add_argument("--embed_optimizer", type=str, default="row", choices=["right", "row", "hybrid", "adamw"])
    parser.add_argument("--router_optimizer", type=str, default="row", choices=["left", "left_uncentered", "muon_uncentered", "row", "hybrid", "adamw"])
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.0)
    parser.add_argument("--router_z_loss_coef", type=float, default=0.0)
    parser.add_argument("--output_router_logits", type=bool, default=False)
    parser.add_argument("--log_router_metrics", type=bool, default=True)
    parser.add_argument("--router_metrics_every", type=int, default=500)
    parser.add_argument("--router_metrics_per_layer", type=bool, default=True)
    parser.add_argument("--router_metrics_log_expert_scalars", type=bool, default=True)
    parser.add_argument("--router_metrics_log_heatmaps", type=bool, default=True)
    parser.add_argument("--router_metrics_log_histograms", type=bool, default=True)
    parser.add_argument("--router_metrics_save_npz", type=bool, default=True)
    parser.add_argument("--router_metrics_npz_dir", type=str, default="router_metrics_npz")
    parser.add_argument("--router_metrics_max_layers", type=int, default=0)
    parser.add_argument("--router_heatmap_clip", type=float, default=3.0)
    parser.add_argument("--dead_expert_threshold_scale", type=float, default=0.1)
    parser.add_argument("--right_optimizer_impl", type=str, default="gramns", choices=["standard", "gramns"])
    parser.add_argument("--hybrid_optimizer_impl", type=str, default="gramns", choices=["standard", "gramns"])
    parser.add_argument("--embed_hybrid_order", type=str, default="row_then_polar", choices=["polar_then_row", "row_then_polar"])
    parser.add_argument("--lm_head_hybrid_order", type=str, default="row_then_polar", choices=["polar_then_row", "row_then_polar"])
    parser.add_argument("--router_hybrid_order", type=str, default="row_then_polar", choices=["polar_then_row", "row_then_polar"])
    parser.add_argument("--moe_expert_gate_up_optimizer", type=str, default="matrix", choices=["row", "hybrid", "adamw", "matrix"])
    parser.add_argument("--moe_expert_down_optimizer", type=str, default="matrix", choices=["row", "hybrid", "adamw", "matrix"])
    parser.add_argument("--moe_expert_hybrid_order", type=str, default="row_then_polar", choices=["polar_then_row", "row_then_polar"])
    parser.add_argument("--ns_epsilon", type=float, default=1e-7)
    parser.add_argument("--ns_use_kernels", type=bool, default=True)
    parser.add_argument("--use_gram_newton_schulz", type=bool, default=True)
    parser.add_argument("--gram_newton_schulz_reset_iterations", type=list[int], default=None)
    parser.add_argument("--row_mode", type=str, default="inverse_eps")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr_muon", type=float, default=1e-3)
    parser.add_argument("--lr_embed", type=float, default=1e-1)
    parser.add_argument("--lr_lm_head", type=float, default=1e-3)
    parser.add_argument("--lr_router", type=float, default=7.5e-4)
    parser.add_argument("--lr_moe_expert_gate_up", type=float, default=None)
    parser.add_argument("--lr_moe_expert_down", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--wd_muon", type=float, default=1e-3)
    parser.add_argument("--wd_embed", type=float, default=0.0)
    parser.add_argument("--wd_lm_head", type=float, default=0.0)
    parser.add_argument("--wd_router", type=float, default=0.0)
    parser.add_argument("--wd_moe_expert_gate_up", type=float, default=None)
    parser.add_argument("--wd_moe_expert_down", type=float, default=None)
    parser.add_argument("--beta_matrix", type=float, default=0.95)
    parser.add_argument("--beta_embed", type=float, default=0.95)
    parser.add_argument("--beta_lm_head", type=float, default=0.95)
    parser.add_argument("--beta_router", type=float, default=0.95)
    parser.add_argument("--beta_moe_expert_gate_up", type=float, default=None)
    parser.add_argument("--beta_moe_expert_down", type=float, default=None)
    parser.add_argument("--backend", type=str, default="polar_express", choices=["polar_express", "newton_schulz"])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device_batch_size", type=int, default=8)
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=2880)
    parser.add_argument("--num_hidden_layers", type=int, default=24)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", type=bool, default=True)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--tensorboard", type=bool, default=False)
    parser.add_argument("--tb_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--train_bin_pattern", type=str, default="*_train_*.bin")
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--val_tokens", type=int, default=10_485_760)
    parser.add_argument("--val_loss_every", type=int, default=500)
    parser.add_argument("--val_bin_pattern", type=str, default="*_val_*.bin")
    args = parser.parse_args()

    # Defaults for 3D MoE expert tensors.
    # OLMoE:
    #   experts.gate_up_proj [E, 2*r, d] -> row layout
    #   experts.down_proj    [E, d, r]   -> column layout
    # gpt-oss:
    #   experts.gate_up_proj [E, d, 2*r] -> interleaved gate/up column-pair layout
    #   experts.down_proj    [E, r, d]   -> row layout
    if args.lr_moe_expert_gate_up is None:
        args.lr_moe_expert_gate_up = args.lr_muon
    if args.lr_moe_expert_down is None:
        args.lr_moe_expert_down = args.lr_muon

    if args.wd_moe_expert_gate_up is None:
        args.wd_moe_expert_gate_up = args.wd_muon
    if args.wd_moe_expert_down is None:
        args.wd_moe_expert_down = args.wd_muon

    if args.beta_moe_expert_gate_up is None:
        args.beta_moe_expert_gate_up = args.beta_matrix
    if args.beta_moe_expert_down is None:
        args.beta_moe_expert_down = args.beta_matrix

    need_router_logits = (
        bool(args.output_router_logits)
        or bool(args.log_router_metrics)
        or args.router_aux_loss_coef != 0.0
        or args.router_z_loss_coef != 0.0
    )


    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_all(args.seed + ddp_rank())
    torch.set_float32_matmul_precision("high")

    writer = None
    if args.tensorboard and ddp_is_main():
        os.makedirs(args.tb_dir, exist_ok=True)
        if args.run_name:
            run_name = args.run_name
        else:
            run_name = (
                f"{args.model}_lr{args.lr}_ws{ddp_world_size()}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        logdir = os.path.join(args.tb_dir, run_name)
        writer = SummaryWriter(log_dir=logdir)

        # Log CLI/config arguments as TensorBoard hparams
        hparams = {}
        for k, v in vars(args).items():
            if isinstance(v, (int, float, str, bool)):
                hparams[k] = v
            elif v is None:
                hparams[k] = "None"
            elif isinstance(v, (list, tuple)):
                hparams[k] = str(list(v))
            else:
                hparams[k] = str(v)

        writer.add_hparams(hparams, {"hparam/init": 0.0})
        writer.add_text("hparams", "\n".join(f"{k}: {v}" for k, v in vars(args).items()), global_step=0)

    tokens_in_global_batch = args.device_batch_size * args.seq_len * ddp_world_size()
    assert args.val_tokens % tokens_in_global_batch == 0, (
        f"Invalid val_tokens={args.val_tokens}. "
        f"It must be divisible by device_batch_size * seq_len * world_size = "
        f"{tokens_in_global_batch}."
    )
    val_steps = args.val_tokens // tokens_in_global_batch

    if ddp_is_main():
        print(f"Validation tokens: {args.val_tokens}", flush=True)
        print(f"Validation steps: {val_steps}", flush=True)

    model, train_loader, val_loader = get_model_and_loaders(
        model_name=args.model,
        hidden_size=args.hidden_size,
        device_batch_size=args.device_batch_size,
        num_experts=args.num_experts,
        num_hidden_layers=args.num_hidden_layers,
        seq_len=args.seq_len,
        data_dir=args.data_dir,
        train_bin_pattern=args.train_bin_pattern,
        val_bin_pattern=args.val_bin_pattern,
        device=device,
        output_router_logits=need_router_logits,
    )
    model.to(device)
    model.train()

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False, dynamic=False)

    if ddp_is_main():
        print(
            f"number of trainable parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}",
            flush=True,
        )

    if ddp_is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if ddp_is_initialized() else model
    router_num_experts = int(getattr(raw_model.config, "num_local_experts", getattr(raw_model.config, "num_experts", args.num_experts)))
    router_top_k = int(getattr(raw_model.config, "num_experts_per_tok", getattr(raw_model.config, "experts_per_token", 1)))

    # ---- shape assertions for role-specific optimizers ----
    if hasattr(raw_model, "model") and hasattr(raw_model.model, "embed_tokens"):
        embed_w = raw_model.model.embed_tokens.weight
        assert embed_w.ndim == 2, f"embed_tokens.weight must be 2D, got {embed_w.shape}"
        assert embed_w.shape == (raw_model.config.vocab_size, raw_model.config.hidden_size), (
            f"embed_tokens.weight expected {(raw_model.config.vocab_size, raw_model.config.hidden_size)}, "
            f"got {tuple(embed_w.shape)}"
        )
        assert embed_w.shape[0] >= embed_w.shape[1], (
            f"embed_tokens.weight expected v x d with v>=d, got {tuple(embed_w.shape)}"
        )

    if hasattr(raw_model, "lm_head") and hasattr(raw_model.lm_head, "weight"):
        lm_head_w = raw_model.lm_head.weight
        assert lm_head_w.ndim == 2, f"lm_head.weight must be 2D, got {lm_head_w.shape}"
        assert lm_head_w.shape == (raw_model.config.vocab_size, raw_model.config.hidden_size), (
            f"lm_head.weight expected {(raw_model.config.vocab_size, raw_model.config.hidden_size)}, "
            f"got {tuple(lm_head_w.shape)}"
        )
        assert lm_head_w.shape[0] >= lm_head_w.shape[1], (
            f"lm_head.weight expected v x d with v>=d, got {tuple(lm_head_w.shape)}"
        )
    
    for module_name, module in raw_model.named_modules():
        if hasattr(module, "gate") and hasattr(module, "experts"):
            gate = getattr(module, "gate", None)
            if gate is None:
                continue

            for pname, p in gate.named_parameters():
                if p.ndim != 2:
                    continue
                rows, cols = p.shape
                assert rows <= cols, (
                    f"Router parameter {module_name}.gate.{pname} expected e x d with e<=d, "
                    f"got {tuple(p.shape)}"
                )
                if hasattr(module, "num_experts"):
                    assert rows == int(module.num_experts), (
                        f"Router parameter {module_name}.gate.{pname} expected rows=num_experts="
                        f"{int(module.num_experts)}, got {tuple(p.shape)}"
                    )


    # ---- gpt-oss expert tensor shape assertions ----
    # HF GptOssExperts stores:
    #   gate_up_proj [num_experts, hidden_size, 2 * intermediate_size]
    #   down_proj    [num_experts, intermediate_size, hidden_size]
    # gate_up_proj uses interleaved gate/up columns.
    for module_name, module in raw_model.named_modules():
        if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
            gate_up = getattr(module, "gate_up_proj")
            down = getattr(module, "down_proj")
            if isinstance(gate_up, torch.nn.Parameter) and isinstance(down, torch.nn.Parameter):
                assert gate_up.ndim == 3, f"{module_name}.gate_up_proj must be 3D, got {tuple(gate_up.shape)}"
                assert down.ndim == 3, f"{module_name}.down_proj must be 3D, got {tuple(down.shape)}"
                assert gate_up.shape[0] == args.num_experts, (
                    f"{module_name}.gate_up_proj expected first dim=num_experts={args.num_experts}, "
                    f"got {tuple(gate_up.shape)}"
                )
                assert down.shape[0] == args.num_experts, (
                    f"{module_name}.down_proj expected first dim=num_experts={args.num_experts}, "
                    f"got {tuple(down.shape)}"
                )
                assert gate_up.shape[1] == raw_model.config.hidden_size, (
                    f"{module_name}.gate_up_proj expected second dim=hidden_size={raw_model.config.hidden_size}, "
                    f"got {tuple(gate_up.shape)}"
                )
                assert gate_up.shape[2] % 2 == 0, (
                    f"{module_name}.gate_up_proj expected interleaved 2*intermediate last dim, "
                    f"got {tuple(gate_up.shape)}"
                )
                assert down.shape[2] == raw_model.config.hidden_size, (
                    f"{module_name}.down_proj expected shape [E, intermediate_size, hidden_size], "
                    f"got {tuple(down.shape)}"
                )

    named_params = list(raw_model.named_parameters())
    attention_head_configs = {}
    attention_head_configs.update(build_attention_head_configs(raw_model))
    attention_head_configs.update(build_gpt_oss_expert_configs(raw_model))
    right_optimizer_cls = RightPolarGradM if args.right_optimizer_impl == "standard" else RightPolarGradM_GramNS
    hybrid_optimizer_cls = HybridPolarGradM if args.hybrid_optimizer_impl == "standard" else HybridPolarGradM_GramNS
    gramns_optimizer_kwargs = {
        "ns_epsilon": args.ns_epsilon,
        "ns_use_kernels": args.ns_use_kernels,
        "use_gram_newton_schulz": args.use_gram_newton_schulz,
    }
    if args.gram_newton_schulz_reset_iterations is not None:
        gramns_optimizer_kwargs["gram_newton_schulz_reset_iterations"] = (
            args.gram_newton_schulz_reset_iterations
        )

    hidden_matrix_named_params = [
        (name, p) for name, p in named_params
        if (
            p.ndim >= 2
            and "embed_tokens" not in name
            and "lm_head" not in name
            and "experts.gate_up_proj" not in name
            and "experts.down_proj" not in name
        )
    ]

    tied_weight = None
    if hasattr(raw_model, "lm_head") and hasattr(raw_model.lm_head, "weight"):
        tied_weight = raw_model.lm_head.weight

    # Router centering ablation.
    # The mixed-optimizer builder only understands router_optimizer="left".
    # For the uncentered ablation, we keep the builder route as "left" but
    # swap the LeftPolarGradM class for a wrapper that forces center_rows=False.
    if args.router_optimizer in ("left_uncentered", "muon_uncentered"):
        left_router_optimizer_cls = LeftPolarGradMUncentered
        router_optimizer_for_builder = "left"
    else:
        left_router_optimizer_cls = LeftPolarGradM
        router_optimizer_for_builder = args.router_optimizer

    optimizer = build_transformer_mixed_optimizer(
        raw_model,
        RightPolarGradM=right_optimizer_cls,
        LeftPolarGradM=left_router_optimizer_cls,
        RowNormM=RowNormM,
        HybridPolarGradM=hybrid_optimizer_cls,
        MatrixOptimizerCls=MuonHeadsPolarExpressWrapper,
        OtherOptimizerCls=torch.optim.AdamW,
        tied_lm_head_weight=tied_weight,
        verbose=ddp_is_main(),
        lr_other=args.lr,
        lr_matrix=args.lr_muon,
        lr_embed=args.lr_embed,
        lr_lm_head=args.lr_lm_head,
        lr_router=args.lr_router,
        lr_moe_expert_gate_up=args.lr_moe_expert_gate_up,
        lr_moe_expert_down=args.lr_moe_expert_down,
        wd_other=args.weight_decay,
        wd_matrix=args.wd_muon,
        wd_embed=args.wd_embed,
        wd_lm_head=args.wd_lm_head,
        wd_router=args.wd_router,
        wd_moe_expert_gate_up=args.wd_moe_expert_gate_up,
        wd_moe_expert_down=args.wd_moe_expert_down,
        beta_matrix=args.beta_matrix,
        beta_embed=args.beta_embed,
        beta_lm_head=args.beta_lm_head,
        beta_router=args.beta_router,
        beta_moe_expert_gate_up=args.beta_moe_expert_gate_up,
        beta_moe_expert_down=args.beta_moe_expert_down,
        alpha=args.alpha,
        eps=args.eps,
        num_steps=args.inner_steps,
        matrix_named_params=hidden_matrix_named_params,
        attention_head_configs=attention_head_configs,
        backend=args.backend,
        right_optimizer_kwargs=gramns_optimizer_kwargs,
        hybrid_optimizer_kwargs=gramns_optimizer_kwargs,
        lm_head_optimizer=args.lm_head_optimizer,
        embed_optimizer=args.embed_optimizer,
        router_optimizer=router_optimizer_for_builder,
        moe_expert_gate_up_optimizer=args.moe_expert_gate_up_optimizer,
        moe_expert_down_optimizer=args.moe_expert_down_optimizer,
        row_mode=args.row_mode,
        embed_hybrid_order=args.embed_hybrid_order,
        lm_head_hybrid_order=args.lm_head_hybrid_order,
        router_hybrid_order=args.router_hybrid_order,
        moe_expert_hybrid_order=args.moe_expert_hybrid_order,
    )

    num_iterations = args.train_steps
    cooldown_frac = 0.4

    def get_lr_scale(it):
        t = 1 - it / num_iterations
        t = max(t, 1e-12)
        if t >= cooldown_frac:
            return 1.0
        return t / cooldown_frac

    role_schedulers = {}
    for role, opt in optimizer.optimizers.items():
        if role == "other":
            role_schedulers[role] = get_cosine_schedule_with_warmup(
                optimizer=opt,
                num_warmup_steps=100,
                num_training_steps=num_iterations,
                num_cycles=0.5,
            )
        else:
            role_schedulers[role] = torch.optim.lr_scheduler.LambdaLR(opt, get_lr_scale)

    scheduler = MixedScheduler(role_schedulers)

    torch.cuda.synchronize()
    training_time_ms = 0.0
    t0 = time.perf_counter()

    for step in range(args.train_steps + 1):
        if step == 10:
            training_time_ms = 0.0
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        timed_steps = (step - 10) if step > 10 else float("nan")
        last_step = step == args.train_steps

        # Validation
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)

            model.eval()
            val_loader.reset()
            val_loss = torch.tensor(0.0, device=device)
            val_ce_loss = torch.tensor(0.0, device=device)
            val_router_metrics_sum = {}
            val_router_load_fraction_sum = None
            val_router_prob_mass_sum = None
            val_router_metric_steps = 0

            for _ in range(val_steps):
                with torch.no_grad():
                    x_val, y_val = val_loader.next_batch()
                    outputs = model(input_ids=x_val, labels=y_val, output_router_logits=need_router_logits)
                    router_logits = get_router_logits_from_outputs(outputs)
                    ce_loss = outputs.loss
                    total_loss = ce_loss
                    aux_loss = router_load_balancing_loss(
                        router_logits,
                        num_experts=router_num_experts,
                        top_k=router_top_k,
                    ) if need_router_logits else None
                    z_loss = router_z_loss(router_logits) if need_router_logits else None
                    if aux_loss is not None and args.router_aux_loss_coef != 0.0:
                        total_loss = total_loss + args.router_aux_loss_coef * aux_loss
                    if z_loss is not None and args.router_z_loss_coef != 0.0:
                        total_loss = total_loss + args.router_z_loss_coef * z_loss

                    val_loss += total_loss.detach()
                    val_ce_loss += ce_loss.detach()

                    if args.log_router_metrics and need_router_logits:
                        metrics, _, load_fractions, prob_masses = compute_router_metrics(
                            router_logits,
                            num_experts=router_num_experts,
                            top_k=router_top_k,
                            dead_expert_threshold_scale=args.dead_expert_threshold_scale,
                        )
                        if metrics:
                            for mk, mv in metrics.items():
                                val_router_metrics_sum[mk] = val_router_metrics_sum.get(mk, torch.tensor(0.0, device=device)) + mv.detach()
                            if load_fractions is not None:
                                val_router_load_fraction_sum = (
                                    load_fractions.detach()
                                    if val_router_load_fraction_sum is None
                                    else val_router_load_fraction_sum + load_fractions.detach()
                                )
                            if prob_masses is not None:
                                val_router_prob_mass_sum = (
                                    prob_masses.detach()
                                    if val_router_prob_mass_sum is None
                                    else val_router_prob_mass_sum + prob_masses.detach()
                                )
                            val_router_metric_steps += 1

            if ddp_is_initialized():
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_ce_loss, op=dist.ReduceOp.AVG)

            val_loss = val_loss.item() / val_steps
            val_ce_loss = val_ce_loss.item() / val_steps
            val_router_metrics = {
                k: (v / max(val_router_metric_steps, 1)).detach()
                for k, v in val_router_metrics_sum.items()
            }
            val_router_load_fractions = (
                val_router_load_fraction_sum / max(val_router_metric_steps, 1)
                if val_router_load_fraction_sum is not None else None
            )
            val_router_prob_masses = (
                val_router_prob_mass_sum / max(val_router_metric_steps, 1)
                if val_router_prob_mass_sum is not None else None
            )

            if ddp_is_main():
                if step > 10:
                    step_avg = training_time_ms / timed_steps
                    print(
                        f"step:{step}/{args.train_steps} | "
                        f"val_loss:{val_loss:.6f} | "
                        f"val_ce:{val_ce_loss:.6f} | "
                        f"train_time:{training_time_ms:.0f}ms | "
                        f"step_avg:{step_avg:.2f}ms",
                        flush=True,
                    )
                else:
                    print(
                        f"step:{step}/{args.train_steps} | "
                        f"val_loss:{val_loss:.6f} | "
                        f"val_ce:{val_ce_loss:.6f} | "
                        f"train_time:{training_time_ms:.0f}ms",
                        flush=True,
                    )

            if args.tensorboard and ddp_is_main():
                writer.add_scalar("val_loss", val_loss, step)
                writer.add_scalar("val_ce_loss", val_ce_loss, step)
                for mk, mv in tensor_metrics_to_float(val_router_metrics).items():
                    writer.add_scalar(f"val/{mk}", mv, step)
                log_router_assignment_tensors(
                    writer,
                    "val/router_assignments",
                    step,
                    val_router_load_fractions,
                    val_router_prob_masses,
                    log_expert_scalars=args.router_metrics_log_expert_scalars,
                    log_heatmaps=args.router_metrics_log_heatmaps,
                    log_histograms=args.router_metrics_log_histograms,
                    max_layers=args.router_metrics_max_layers,
                    heatmap_clip=args.router_heatmap_clip,
                )
                if args.router_metrics_save_npz and ddp_is_main():
                    save_router_assignment_npz(
                        args.router_metrics_npz_dir,
                        "val_router_assignments",
                        step,
                        val_router_load_fractions,
                        val_router_prob_masses,
                    )

            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # Training step
        step_t0 = time.perf_counter()

        input_ids, labels = train_loader.next_batch()
        outputs = model(input_ids=input_ids, labels=labels, output_router_logits=need_router_logits)
        router_logits = get_router_logits_from_outputs(outputs)
        ce_loss = outputs.loss
        loss = ce_loss
        aux_loss = router_load_balancing_loss(
            router_logits,
            num_experts=router_num_experts,
            top_k=router_top_k,
        ) if need_router_logits else None
        z_loss = router_z_loss(router_logits) if need_router_logits else None
        if aux_loss is not None and args.router_aux_loss_coef != 0.0:
            loss = loss + args.router_aux_loss_coef * aux_loss
        if z_loss is not None and args.router_z_loss_coef != 0.0:
            loss = loss + args.router_z_loss_coef * z_loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            loss_detached = loss.detach()
            ce_loss_detached = ce_loss.detach()
            aux_loss_detached = aux_loss.detach() if aux_loss is not None else torch.tensor(0.0, device=device)
            z_loss_detached = z_loss.detach() if z_loss is not None else torch.tensor(0.0, device=device)
            if ddp_is_initialized():
                dist.all_reduce(loss_detached, op=dist.ReduceOp.AVG)
                dist.all_reduce(ce_loss_detached, op=dist.ReduceOp.AVG)
                dist.all_reduce(aux_loss_detached, op=dist.ReduceOp.AVG)
                dist.all_reduce(z_loss_detached, op=dist.ReduceOp.AVG)

            train_router_metrics = {}
            train_router_layer_metrics = []
            train_router_load_fractions = None
            train_router_prob_masses = None
            should_log_router_metrics = (
                args.log_router_metrics
                and need_router_logits
                and args.router_metrics_every > 0
                and step % args.router_metrics_every == 0
            )
            if should_log_router_metrics:
                (
                    train_router_metrics,
                    train_router_layer_metrics,
                    train_router_load_fractions,
                    train_router_prob_masses,
                ) = compute_router_metrics(
                    router_logits,
                    num_experts=router_num_experts,
                    top_k=router_top_k,
                    dead_expert_threshold_scale=args.dead_expert_threshold_scale,
                )

        torch.cuda.synchronize()
        now = time.perf_counter()
        step_time_ms = 1000 * (now - step_t0)
        approx_time = training_time_ms + 1000 * (now - t0)

        main_lr_role = "matrix_attention" if "matrix_attention" in optimizer.optimizers else next(iter(optimizer.optimizers))
        main_lr = optimizer.optimizers[main_lr_role].param_groups[0]["lr"]

        if ddp_is_main():
            print(
                f"step:{step}/{args.train_steps} | "
                f"train_loss:{loss_detached.item():.6f} | "
                f"ce:{ce_loss_detached.item():.6f} | "
                f"aux:{aux_loss_detached.item():.6f} | "
                f"z:{z_loss_detached.item():.6f} | "
                f"lr:{main_lr:.8f} | "
                f"step_time:{step_time_ms:.2f}ms | "
                f"train_time:{approx_time:.0f}ms",
                flush=True,
            )

        if args.tensorboard and ddp_is_main():
            writer.add_scalar("train_loss", loss_detached.item(), step)
            writer.add_scalar("train_ce_loss", ce_loss_detached.item(), step)
            writer.add_scalar("train_router_aux_loss", aux_loss_detached.item(), step)
            writer.add_scalar("train_router_z_loss", z_loss_detached.item(), step)
            for mk, mv in tensor_metrics_to_float(train_router_metrics).items():
                writer.add_scalar(f"train/{mk}", mv, step)
            if args.router_metrics_per_layer:
                for li, lm in enumerate(train_router_layer_metrics):
                    for mk, mv in tensor_metrics_to_float(lm).items():
                        short_name = mk.replace("router/", "")
                        writer.add_scalar(f"train/router_layer_{li}/{short_name}", mv, step)
            log_router_assignment_tensors(
                writer,
                "train/router_assignments",
                step,
                train_router_load_fractions,
                train_router_prob_masses,
                log_expert_scalars=args.router_metrics_log_expert_scalars,
                log_heatmaps=args.router_metrics_log_heatmaps,
                log_histograms=args.router_metrics_log_histograms,
                max_layers=args.router_metrics_max_layers,
                heatmap_clip=args.router_heatmap_clip,
            )
            if args.router_metrics_save_npz and ddp_is_main():
                save_router_assignment_npz(
                    args.router_metrics_npz_dir,
                    "train_router_assignments",
                    step,
                    train_router_load_fractions,
                    train_router_prob_masses,
                )
            writer.add_scalar("lr", main_lr, step)

    if args.tensorboard and ddp_is_main():
        writer.flush()
        writer.close()

    if ddp_is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
