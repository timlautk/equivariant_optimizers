import torch


def row_norms(X: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    return X.norm(dim=-1, keepdim=True).clamp_min(eps)


def row_scale(
    X: torch.Tensor,
    mode: str = "inverse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns rowwise scaling factors with shape (n_rows, 1).
    mode:
      - 'inverse'         : 1 / ||row||
      - 'inverse_eps'     : 1 / (||row|| + eps)
      - 'unit'            : same as inverse with clamp
    """
    norms = X.norm(dim=-1, keepdim=True)
    if mode == "inverse":
        return 1.0 / norms.clamp_min(eps)
    if mode == "inverse_eps":
        return 1.0 / (norms + eps)
    if mode == "unit":
        return 1.0 / norms.clamp_min(eps)
    raise ValueError(f"Unsupported row scaling mode: {mode}")


def apply_row_scaling(
    X: torch.Tensor,
    mode: str = "inverse_eps",
    eps: float = 1e-8,
) -> torch.Tensor:
    return row_scale(X, mode=mode, eps=eps) * X