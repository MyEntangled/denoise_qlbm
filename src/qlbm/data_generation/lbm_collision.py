from src.qlbm.lbm_lattices import get_lattice
import numpy as np


# def get_equilibrium(rho: np.ndarray, u: np.ndarray, lattice: str) -> np.ndarray:
#     """
#     Maxwell–Boltzmann 2nd-order equilibrium:
#         f_i^eq = w_i * rho * [ 1 + (c_i·u)/cs^2 + (c_i·u)^2/(2 cs^4) - (u·u)/(2 cs^2) ]
#     Args:
#         rho: (B,)
#         u:   (B,d)
#         lattice: one of {"D1Q3","D2Q9","D3Q15","D3Q19","D3Q27"}
#     Returns:
#         F_eq: (B,Q)
#     """
#     c, w = get_lattice(lattice, as_array=True)   # c: (Q,d), w: (Q,)
#     cs2 = 1./3
#
#     B = rho.shape[0]
#     d = c.shape[1]
#
#     u = np.asarray(u, dtype=float).reshape(B, d)
#     rho = np.asarray(rho, dtype=float).reshape(B)
#
#     # cu = u · c_i  -> (B,Q)
#     cu = np.einsum('bd,qd->bq', u, c, optimize=True)
#     uu = np.einsum('bd,bd->b', u, u, optimize=True)[:, None]  # (B,1)
#
#     w_row = w[None, :]  # (1,Q)
#     F_eq = w_row * rho[:, None] * (
#         1.0 + (cu / cs2) + 0.5 * (cu**2) / (cs2**2) - 0.5 * (uu / cs2)
#     )
#     return F_eq  # (B,Q)

def get_equilibrium(rho: np.ndarray, u: np.ndarray, lattice: str, eq_dist_deg: int) -> np.ndarray:
    """
    Maxwell–Boltzmann equilibrium:
        1st-order: f_i^eq = w_i * rho * [ 1 + (c_i·u)/cs^2 ]
        2nd-order: f_i^eq = w_i * rho * [ 1 + (c_i·u)/cs^2 + (c_i·u)^2/(2 cs^4) - (u·u)/(2 cs^2) ]

    Args:
        rho: (...,)          arbitrary batch shape
        u:   (..., d)        same batch shape as rho, with last dim = d
        lattice: one of {"D1Q3","D2Q9","D3Q15","D3Q19","D3Q27"}

    Returns:
        F_eq: (..., Q)
    """
    assert eq_dist_deg in [1,2], "eq_dist_deg must be 1 or 2."

    c, w = get_lattice(lattice, as_array=True)   # c: (Q,d), w: (Q,)

    cs2_inv = 3.
    cs4_inv = 9.

    # Check shapes
    if u.ndim < 1:
        raise ValueError("u must have at least one dimension (last is velocity).")
    if rho.shape != u.shape[:-1]:
        raise ValueError(
            f"rho.shape {rho.shape} must match u.shape[:-1] {u.shape[:-1]}"
        )

    d = c.shape[1]
    if u.shape[-1] != d:
        raise ValueError(
            f"Last dim of u ({u.shape[-1]}) must equal lattice dimension d={d}"
        )

    # cu = u · c_i  -> shape (..., Q)
    cu = np.einsum('...d,qd->...q', u, c)

    # uu = |u|^2 -> shape (..., 1)
    uu = np.einsum('...d,...d->...', u, u)[..., None]

    # Broadcast weights over batch axes: shape (1,1,...,1,Q)
    w_shape = (1,) * rho.ndim + (w.shape[0],)
    w_b = w.reshape(w_shape)

    # rho[..., None] has shape (..., 1), broadcast with w_b (..., Q) ⇒ (..., Q)
    if eq_dist_deg == 1:
        F_eq = w_b * rho[..., None] * (1.0 + cu * cs2_inv)
    else:   # eq_dist_deg == 2
        F_eq = w_b * rho[..., None] * (
            1.0 + cu * cs2_inv + 0.5 * (cu**2) * cs4_inv - 0.5 * uu * cs2_inv
        )
    return F_eq



def collide(F: np.ndarray, lattice: str, omega=1.0) -> np.ndarray:
    """
    Single-relaxation-time BGK collision:
        F_post = (1 - omega) * F + omega * F_eq(rho, u)
    Args:
        F:    (B,Q)
        lattice: same choices as above
        omega: scalar or array-like of shape (B,) for per-batch omega
    Returns:
        F_post: (B,Q)
    """
    F = np.asarray(F, dtype=float)
    B, Q = F.shape
    c, w = get_lattice(lattice, as_array=True)
    d = c.shape[1]

    rho = np.sum(F, axis=1)                  # (B,)
    # momentum = sum_i F_i c_i  -> (B,d)
    mom = F @ c
    # avoid division by zero (if any rho==0)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(rho[:, None] != 0, mom / rho[:, None], 0.0)

    F_eq = get_equilibrium(rho, u, lattice)  # (B,Q)

    omega_arr = np.asarray(omega)
    if omega_arr.ndim == 0:
        # Scalar omega
        return (1.0 - float(omega_arr)) * F + float(omega_arr) * F_eq
    else:
        omega_arr = omega_arr.reshape(B, 1)  # broadcast across Q
        return (1.0 - omega_arr) * F + omega_arr * F_eq