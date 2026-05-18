# Adopted from The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm
# by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower at https://arxiv.org/abs/2505.16932

import torch


# Computed for num_iters=5, safety_factor=2e-2, cushion=2
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]


def _polar_express_impl(
    G: torch.Tensor,
    compute_hermitian: bool = False,
):
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT  # this reduces FLOPs

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    for a, b, c in coeffs_list:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5

    if compute_hermitian:
        H = G.type_as(X).mT @ X.mT
        H = (H + H.mT) / 2

    if transposed:
        X = X.mT
        if compute_hermitian:
            H = H.mT

    if compute_hermitian:
        return X, H
    return X


@torch.compile(dynamic=False, fullgraph=True)
def PolarExpress(
    G: torch.Tensor,
    compute_hermitian: bool = False,
) -> torch.Tensor:
    return _polar_express_impl(G, compute_hermitian=compute_hermitian)


def _reshape_heads(
    G: torch.Tensor,
    num_heads: int,
    head_dim: int,
    layout: str,
):
    if G.ndim != 2:
        raise ValueError(f"PolarExpressHeads expects a 2D tensor, got shape {tuple(G.shape)}")

    if layout == "qkv":
        out_dim, hidden_size = G.shape
        expected = num_heads * head_dim
        if out_dim != expected:
            raise ValueError(
                f"For layout='qkv', expected shape ({expected}, hidden_size), got {tuple(G.shape)}"
            )
        return G.view(num_heads, head_dim, hidden_size), G.shape

    if layout == "o":
        hidden_size, in_dim = G.shape
        expected = num_heads * head_dim
        if in_dim != expected:
            raise ValueError(
                f"For layout='o', expected shape (hidden_size, {expected}), got {tuple(G.shape)}"
            )
        return G.view(hidden_size, num_heads, head_dim).permute(1, 0, 2).contiguous(), G.shape

    raise ValueError(f"Unknown layout: {layout}. Expected 'qkv' or 'o'.")


def _unreshape_heads(
    X_heads: torch.Tensor,
    original_shape,
    layout: str,
):
    if layout == "qkv":
        return X_heads.reshape(original_shape)
    if layout == "o":
        num_heads, hidden_size, head_dim = X_heads.shape
        return X_heads.permute(1, 0, 2).contiguous().view(hidden_size, num_heads * head_dim)
    raise ValueError(f"Unknown layout: {layout}. Expected 'qkv' or 'o'.")


@torch.compile(dynamic=False, fullgraph=True)
def _polar_express_heads_compiled(
    G_heads: torch.Tensor,
    compute_hermitian: bool = False,
):
    return _polar_express_impl(G_heads, compute_hermitian=compute_hermitian)


def PolarExpressHeads(
    G: torch.Tensor,
    num_heads: int | None = None,
    head_dim: int | None = None,
    layout: str = "qkv",
    compute_hermitian: bool = False,
) -> torch.Tensor:
    """
    Head-aware Polar Express.

    Args:
        G:
            q/k/v layout: (num_heads * head_dim, hidden_size)
            o layout:     (hidden_size, num_heads * head_dim)
        num_heads: number of heads in this projection block.
        head_dim: dimension per head.
        layout: 'qkv' for q_proj/k_proj/v_proj, 'o' for o_proj.
        compute_hermitian: whether to also return the per-head Hermitian factor.

    Returns:
        Tensor with the same shape as G, obtained by applying Polar Express independently
        to each head block. If num_heads/head_dim are not provided, this falls back to the
        whole-matrix PolarExpress behavior.
    """
    if num_heads is None or head_dim is None:
        return PolarExpress(G, compute_hermitian=compute_hermitian)

    G_heads, original_shape = _reshape_heads(G, num_heads=num_heads, head_dim=head_dim, layout=layout)
    out = _polar_express_heads_compiled(G_heads, compute_hermitian=compute_hermitian)
    if compute_hermitian:
        X_heads, H_heads = out
        return _unreshape_heads(X_heads, original_shape, layout), H_heads
    return _unreshape_heads(out, original_shape, layout)
