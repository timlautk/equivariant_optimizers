from itertools import repeat
import torch


@torch.no_grad()
def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


@torch.compile(dynamic=False, fullgraph=True)
def symmetric_matrix_invsqrt_newton_schulz(
    A: torch.Tensor,
    eps: float = 1e-8,
    num_steps: int = 5,
    return_sqrt: bool = False,
) -> torch.Tensor:
    """
    Newton-Schulz inverse square root for a symmetric PSD matrix.
    Assumes A is small enough that explicit Gram formation is acceptable.
    """
    assert A.ndim >= 2
    assert A.size(-1) == A.size(-2), "A must be square"

    dtype_in = A.dtype
    device = A.device
    n = A.size(-1)

    # Work in fp32 for stability, cast output back at end.
    A = A.float()

    I = torch.eye(n, device=device, dtype=A.dtype)
    while I.ndim < A.ndim:
        I = I.unsqueeze(0)

    # Symmetrize + damp
    A = symmetrize(A) + eps * I

    # Normalize for stability
    normA = A.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    Y = A / normA
    Z = I.expand_as(A).clone()

    for _ in range(num_steps):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    if return_sqrt:
        Y = Y * normA.sqrt()
        return (Z / normA.sqrt()).to(dtype_in), Y.to(dtype_in)
    
    return (Z / normA.sqrt()).to(dtype_in)


_unmodified_polar_express_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
safety_factor = 1.05
# safety factor for numerical stability ( but exclude last polynomial )
coeffs_list = [(a / safety_factor, b / safety_factor**3, c / safety_factor**5) 
                for (a, b, c) in _unmodified_polar_express_coeffs_list[:-1]]
coeffs_list.append(_unmodified_polar_express_coeffs_list[-1])


@torch.compile(dynamic=False, fullgraph=True)
def symmetric_matrix_invsqrt_polar_express(
    A: torch.Tensor,
    eps: float = 1e-8,
    num_steps: int = 5,
    return_sqrt: bool = False,
) -> torch.Tensor:
    """
    Compute A^{-1/2} for SPD / Hermitian PSD matrices using the
    Polar Express iteration.

    Args:
        A: (..., n, n) SPD matrix
        eps: diagonal damping and scaling safeguard
        num_steps: number of steps for the Polar Express method
        return_sqrt: if True, also return A^{1/2}

    Returns:
        Z: (..., n, n) approximate A^{-1/2}
        optionally Y: (..., n, n) approximate A^{1/2}
    """
    assert A.ndim >= 2
    assert A.size(-1) == A.size(-2), "A must be square"

    dtype_in = A.dtype
    device = A.device
    n = A.size(-1)

    # Work in fp32 for stability, cast output back at end.
    A = A.float()

    I = torch.eye(n, device=device, dtype=A.dtype)
    while I.ndim < A.ndim:
        I = I.unsqueeze(0)

    # Symmetrize + damp
    A = symmetrize(A) + eps * I

    # Scale so spectrum is in a safe range.
    # Since inverse square root rescales as (alpha A)^(-1/2)=alpha^(-1/2)A^(-1/2),
    # we track the scaling outside the iteration.
    alpha = A.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + eps
    Y = A / alpha
    Z = I.expand_as(A).clone()

    hs = coeffs_list[:num_steps] + list(repeat(coeffs_list[-1], num_steps - len(coeffs_list)))
    for a, b, c in hs:
        # p(x) = a x + b x^3 + c x^5 = x * phi(1 - x^2)
        # phi(r) = (a+b+c) + (-b-2c) r + c r^2
        alpha0 = a + b + c
        beta0 = -b - 2 * c
        gamma0 = c

        R = I - Z @ Y
        R = symmetrize(R)  # optional stabilization
        R2 = R @ R
        M = alpha0 * I + beta0 * R + gamma0 * R2

        Y = Y @ M
        Z = M @ Z

    # Undo scaling: if Y0 = A / alpha, then
    # Y_inf = (A / alpha)^{1/2} = alpha^{-1/2} A^{1/2}
    # Z_inf = (A / alpha)^{-1/2} = alpha^{1/2} A^{-1/2}
    #
    # Hence A^{-1/2} = Z_inf / sqrt(alpha)
    sqrt_alpha = alpha.sqrt()
    Z = Z / sqrt_alpha

    if return_sqrt:
        Y = Y * sqrt_alpha
        return Z.to(dtype_in), Y.to(dtype_in)

    return Z.to(dtype_in)


def symmetric_matrix_invsqrt(
    A: torch.Tensor,
    eps: float = 1e-8,
    backend: str = "newton_schulz",
    num_steps: int = 5,
) -> torch.Tensor:
    if backend == "newton_schulz":
        return symmetric_matrix_invsqrt_newton_schulz(A, eps=eps, num_steps=num_steps)
    if backend == "polar_express":
        return symmetric_matrix_invsqrt_polar_express(A, eps=eps, num_steps=num_steps)

    raise ValueError(f"Unsupported backend: {backend}")
