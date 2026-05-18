"""
Symmetry-compatible optimizers for matrix-valued parameters.
"""

from .rightpolargrad import RightPolarGradM, RightPolarGradM_GramNS
from .leftpolargrad import LeftPolarGradM

from .rownorm import RowNormM, BatchedExpertRowNormM

from .hybrid import (
    HybridPolarGradM,
    HybridPolarGradM_GramNS,
    BatchedExpertHybridPolarGradM,
    BatchedExpertHybridPolarGradM_GramNS,
)
from .polar_express import PolarExpress, PolarExpressHeads

from .invsqrt import (
    symmetric_matrix_invsqrt,
    symmetric_matrix_invsqrt_newton_schulz,
)

from .row_ops import (
    row_norms,
    row_scale,
    apply_row_scaling,
)

from .utils import build_attention_head_configs

from .routing import (
    MixedOptimizer,
    MixedOptimizerConfig,
    TransformerRouteConfig,
    build_mixed_optimizer,
    build_transformer_param_groups,
    build_transformer_mixed_optimizer,
    build_olmoe_expert_configs,
    build_gpt_oss_expert_configs,
)

from .muon_heads import MuonHeadsPolarExpress


__all__ = [
    "RightPolarGradM",
    "RightPolarGradM_GramNS",
    "LeftPolarGradM",

    "RowNormM",
    "BatchedExpertRowNormM",

    "HybridPolarGradM",
    "HybridPolarGradM_GramNS",
    "BatchedExpertHybridPolarGradM",
    "BatchedExpertHybridPolarGradM_GramNS",

    "PolarExpress",
    "PolarExpressHeads",

    "symmetric_matrix_invsqrt",
    "symmetric_matrix_invsqrt_newton_schulz",

    "row_norms",
    "row_scale",
    "apply_row_scaling",

    "MixedOptimizer",
    "MixedOptimizerConfig",
    "TransformerRouteConfig",
    "build_mixed_optimizer",
    "build_transformer_param_groups",
    "build_transformer_mixed_optimizer",
    "build_attention_head_configs",
    "build_olmoe_expert_configs",
    "build_gpt_oss_expert_configs",

    "MuonHeadsPolarExpress",
]