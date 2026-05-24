import os
import time
import random
import glob
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    OlmoeConfig,
    OlmoeForCausalLM,
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
    build_olmoe_expert_configs,
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

    if model_name == "olmoe-1b-7b":
        config = OlmoeConfig(
            attention_bias=False,
            attention_dropout=0.0,
            clip_qkv=None,
            eos_token_id=50279,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=1024,
            max_position_embeddings=4096,
            model_type="olmoe",
            norm_topk_prob=False,
            num_attention_heads=16,
            num_experts=num_experts,
            num_experts_per_tok=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=16,
            output_router_logits=False,
            pad_token_id=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=10000.0,
            router_aux_loss_coef=0.0,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
            vocab_size=50304,
        )
        model = OlmoeForCausalLM(config)
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


if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="olmoe-1b-7b")
    parser.add_argument("--lm_head_optimizer", type=str, default="row", choices=["right", "row", "hybrid", "adamw"])
    parser.add_argument("--embed_optimizer", type=str, default="row", choices=["right", "row", "hybrid", "adamw"])
    parser.add_argument("--router_optimizer", type=str, default="row", choices=["left", "row", "hybrid", "adamw"])
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
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_muon", type=float, default=5e-1)
    parser.add_argument("--lr_embed", type=float, default=5e-3)
    parser.add_argument("--lr_lm_head", type=float, default=5e-3)
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
    parser.add_argument("--device_batch_size", type=int, default=20)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
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


    # ---- OLMoE expert tensor shape assertions ----
    # HF OlmoeExperts stores:
    #   gate_up_proj [num_experts, 2 * intermediate_size, hidden_size]
    #   down_proj    [num_experts, hidden_size, intermediate_size]
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
                assert gate_up.shape[2] == raw_model.config.hidden_size, (
                    f"{module_name}.gate_up_proj expected last dim=hidden_size={raw_model.config.hidden_size}, "
                    f"got {tuple(gate_up.shape)}"
                )
                assert down.shape[1] == raw_model.config.hidden_size, (
                    f"{module_name}.down_proj expected shape [E, hidden_size, intermediate_size], "
                    f"got {tuple(down.shape)}"
                )

    named_params = list(raw_model.named_parameters())
    attention_head_configs = {}
    attention_head_configs.update(build_attention_head_configs(raw_model))
    attention_head_configs.update(build_olmoe_expert_configs(raw_model))
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

    optimizer = build_transformer_mixed_optimizer(
        raw_model,
        RightPolarGradM=right_optimizer_cls,
        LeftPolarGradM=LeftPolarGradM,
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
        router_optimizer=args.router_optimizer,
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

            for _ in range(val_steps):
                with torch.no_grad():
                    x_val, y_val = val_loader.next_batch()
                    outputs = model(input_ids=x_val, labels=y_val)
                    val_loss += outputs.loss.detach()

            if ddp_is_initialized():
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            val_loss = val_loss.item() / val_steps

            if ddp_is_main():
                if step > 10:
                    step_avg = training_time_ms / timed_steps
                    print(
                        f"step:{step}/{args.train_steps} | "
                        f"val_loss:{val_loss:.6f} | "
                        f"train_time:{training_time_ms:.0f}ms | "
                        f"step_avg:{step_avg:.2f}ms",
                        flush=True,
                    )
                else:
                    print(
                        f"step:{step}/{args.train_steps} | "
                        f"val_loss:{val_loss:.6f} | "
                        f"train_time:{training_time_ms:.0f}ms",
                        flush=True,
                    )

            if args.tensorboard and ddp_is_main():
                writer.add_scalar("val_loss", val_loss, step)

            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # Training step
        step_t0 = time.perf_counter()

        input_ids, labels = train_loader.next_batch()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            loss_detached = loss.detach()
            if ddp_is_initialized():
                dist.all_reduce(loss_detached, op=dist.ReduceOp.AVG)

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
                f"lr:{main_lr:.8f} | "
                f"step_time:{step_time_ms:.2f}ms | "
                f"train_time:{approx_time:.0f}ms",
                flush=True,
            )

        if args.tensorboard and ddp_is_main():
            writer.add_scalar("train_loss", loss_detached.item(), step)
            writer.add_scalar("lr", main_lr, step)

    if args.tensorboard and ddp_is_main():
        writer.flush()
        writer.close()

    if ddp_is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
