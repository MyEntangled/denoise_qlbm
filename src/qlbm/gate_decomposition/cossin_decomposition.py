import numpy as np
from numpy.linalg import norm
from scipy.linalg import block_diag, cossin

def csd_equal_blocks(U: np.ndarray):
    """
    Cosine–Sine Decomposition for a 2n x 2n unitary/orthogonal matrix U
    partitioned into 4 equal n x n blocks.

    Returns:
        U1, U2, V1, V2, theta
    such that:
        U = blkdiag(U1, U2) @ [[C, -S],
                               [S,  C]] @ blkdiag(V1, V2)
    where C = diag(cos(theta)), S = diag(sin(theta)).
    """
    # --- shape checks
    m, k = U.shape
    if m != k or m % 2 != 0:
        raise ValueError("U must be square with even dimension (2n x 2n).")

    # --- basic unitary check (tolerant)
    print("Unitary precheck for CSD:", norm(U.conj().T @ U - np.eye(m)))
    if norm(U.conj().T @ U - np.eye(m)) > 1e-6:
        raise ValueError("U is not unitary within tolerance.")

    n = m // 2

    # use SciPy's CSD (separated form)
    # SciPy API (>=1.10) supports separate=True returning (U1, U2, V1H, V2H, theta).
    # We pass p=n (top block rows), q=n (left block cols) for equal partition.
    (U1, U2), theta, (V1H, V2H) = cossin(U, p=n, q=n, separate=True)

    # --- optional: verify reconstruction
    C = np.diag(np.cos(theta))
    S = np.diag(np.sin(theta))
    middle = np.block([[C, -S],
                       [S,  C]])
    L = block_diag(U1, U2)
    R = block_diag(V1H, V2H)
    U_rec = L @ middle @ R

    err = norm(U - U_rec)
    if err > 1e-8:
        # Not fatal—just warn in a way that doesn't hide problems.
        print(f"[CSD] Warning: reconstruction residual ‖U − L·CS·R‖ = {err:.3e}")

    return (U1, U2), theta, (V1H, V2H)


# ------------------ Example usage & test ------------------
if __name__ == "__main__":
    rng = np.random.default_rng(18)

    def random_unitary(n, real_orthog=False):
        if real_orthog:
            # QR on real Gaussian -> Haar orthogonal
            X = rng.standard_normal((n, n))
            Q, _ = np.linalg.qr(X)
            return Q
        else:
            # QR on complex Gaussian -> Haar-ish unitary
            X = (rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))) / np.sqrt(2)
            Q, _ = np.linalg.qr(X)
            return Q

    n = 18
    U = random_unitary(2*n, True)

    (U1, U2), theta, (V1H, V2H) = csd_equal_blocks(U)

    # Inspect angles and reconstruction error
    C = np.diag(np.cos(theta))
    S = np.diag(np.sin(theta))
    L = block_diag(U1, U2)
    R = block_diag(V1H, V2H)
    middle = np.block([[C, -S],
                       [S,  C]])
    rec_err = norm(U - L @ middle @ R)
    print("theta (radians):", theta)
    print("reconstruction error:", rec_err)