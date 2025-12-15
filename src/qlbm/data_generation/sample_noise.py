from src.lattices.lbm_lattices import get_lattice
from scipy.linalg import null_space
import numpy as np

# --- Build moment matrix and weighted null basis for 0th/1st moments ---
def build_zero_first_moment_basis(lattice: str):
    """
    Returns:
        M  : (d+1, Q) moment matrix for density and momentum (rows: [1, cx, cy, (cz)])
        D  : (Q, Q) diag(1/sqrt(w))
        Dinv: (Q, Q) diag(sqrt(w))
        Nw : (Q, Q - (d+1)) orthonormal columns spanning weighted null space of M
    """
    c, w = get_lattice(lattice, as_array=True)      # c: (Q,d), w: (Q,)
    Q, d = c.shape

    # M rows: [1, c_x, c_y, (c_z)]
    M = np.vstack([np.ones(Q, dtype=float), c.T])  # shape (d+1, Q)

    Dinv = np.diag(np.sqrt(w))
    D    = np.diag(1.0/np.sqrt(w))

    # Null space of M (right null space in column form)
    N = null_space(M)                  # shape (Q, Q - (d+1)), columns basis
    # Weight-orthonormalize via QR on D @ N, then map back
    Qw, _ = np.linalg.qr(D @ N)        # Qw has orthonormal cols in standard metric
    Nw = Dinv @ Qw                     # columns form an orthonormal basis in weighted metric

    return M, N, Nw


# --- Sample noise with zero density/momentum ---
def sample_zero_moments_noise(n_samples: int,
                              noise_strength,
                              lattice: str,
                              weighted: bool = True,
                              rng=None):
    """
    Sample fluctuations in the null-space of 0th and 1st moments
    (density and momentum = 0).

    Args:
        n_samples: int
        noise_strength: float or (n_samples,) array
        lattice: name (D1Q3, D2Q9, D3Q15, D3Q19, D3Q27)
        weighted: if True use weight-aware null basis Nw (Var(eps_i) ~ w_i), else use N
        rng: optional np.random.Generator
    Returns:
        eps: (n_samples, Q)
    """
    if rng is None:
        rng = np.random.default_rng()

    M, N, Nw = build_zero_first_moment_basis(lattice)
    basis = Nw if weighted else N

    Q = basis.shape[0]
    k = basis.shape[1]
    coeffs = rng.standard_normal(size=(n_samples, k))
    eps = coeffs @ basis.T                     # (n_samples, Q)

    # scale by noise_strength
    ns = np.asarray(noise_strength)
    if ns.ndim == 0:
        eps *= float(ns)
    else:
        eps *= ns.reshape(-1, 1)

    # verify zero moments (numerical)
    resid = M @ eps.T
    if not np.allclose(resid, 0, atol=1e-12):
        raise AssertionError(f"Nonzero moment residual: max|M eps|={np.max(np.abs(resid)):.2e}")

    return eps


if __name__ == "__main__":
    # Example usage
    eps_w  = sample_zero_moments_noise(100000, 1, "D2Q9", weighted=True)
    eps_un = sample_zero_moments_noise(100000, 1, "D2Q9", weighted=False)
    print(eps_w.shape)   # (5, 9)
    print((eps_w**2).mean(axis=0))  # variance per component
    print((eps_un**2).mean(axis=0))