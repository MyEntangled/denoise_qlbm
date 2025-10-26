from src.qlbm.operations.collision.denoiser import DenoisingCollision
from src.qlbm.gate_decomposition.clements_decomposition import unitary_to_givens_ops
from src.qlbm.gate_decomposition.givens_rotation_constructor import ops_to_rots

import numpy as np
import scipy

def nagy_block_encoding(A, rescale=True):
    """
    Construct the unitary block-encoding:
        U = [[A, sqrt(I - AA†)],
             [sqrt(I - A†A), -A†]]

    Returns
        U : ((m+n), (m+n)) complex ndarray
            Unitary matrix containing A in its top-left block.
        alpha : float
            Scaling factor used (1 if rescale=False and ||A||<=1).
    """
    m, n = A.shape

    # Operator (spectral) norm via largest singular value
    smax = np.linalg.norm(A, 2)
    alpha = max(smax, 1.0) if rescale else 1.0
    B = A / alpha

    # Hermitian square roots
    I_m = np.eye(m, dtype=B.dtype)
    I_n = np.eye(n, dtype=B.dtype)
    X = scipy.linalg.sqrtm(I_n - B @ B.conj().T)
    Y = scipy.linalg.sqrtm(I_m - B.conj().T @ B)

    # Fix possible small non-hermitian noise from sqrtm, since X and Y should be Hermitian.
    X = (X + X.conj().T) / 2
    Y = (Y + Y.conj().T) / 2

    # Assemble full unitary
    U = np.block([
        [B, X],
        [Y, -B.conj().T]
    ])

    # Realify U if possible
    if np.allclose(U.imag, 0):
        U = U.real

    # Sanity check
    err = np.linalg.norm(U.conj().T @ U - np.eye(m+n))
    if err > 1e-8:
        raise ValueError(f"U may not be exactly unitary (‖U* @ U − I‖={err:.2e})")


    return U, alpha


def schlimgen_block_encoding(A: np.ndarray, rescale=True):
    """
    Block-encoding of Schlimgen et al. [https://arxiv.org/pdf/2205.02826, Figure 2].
    Builds the (2d)x(2d) unitary U acting on 1 ancilla + d-dim system s.t.
        (⟨0|⊗I) U (|0⟩⊗·)  =  (A/α)  (·)
    with α = max(1, ||A||_2) if rescale=True, else α=1 (user must ensure ||A||≤1).

    Returns:
        U_total : complex ndarray, shape (2d, 2d) — the dilation unitary 𝓤
        alpha   : float — the scaling actually used (≥1 if rescaled)
        U, Vh, Sigma : (U, V†, singular values) from SVD of A
    """
    d_in, d_out = A.shape
    if d_in != d_out:
        raise ValueError("This implementation assumes square A (d×d). "
                         "General rectangular A is possible but requires slight wiring changes.")

    d = d_in
    # SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    smax = float(np.max(s)) if s.size else 0.0
    alpha = max(1.0, smax) if rescale else 1.0
    s_tilde = s / (alpha if alpha > 0 else 1.0)  # singular values normalized to [0,1]

    # Build the (k+1)-qubit diagonal unitary U_Σ = Σ^+ ⊕ Σ^-  (Eq. (1)-(2))
    # For singular values σ ∈ [0,1], Eq. (2) simplifies to:
    #   Σ_i^± = σ_i  ±  i sqrt(1 - σ_i^2)     # Handle σ=0 edge case robustly.

    sqrt_terms = np.sqrt(np.maximum(0.0, 1.0 - s_tilde**2))
    Sigma_plus  = s_tilde + 1j * sqrt_terms
    Sigma_minus = s_tilde - 1j * sqrt_terms

    # Assemble U_Σ as diagonal over the doubled space: diag([Σ^+_1,...,Σ^+_d, Σ^-_1,...,Σ^-_d])
    USigma = np.diag(np.concatenate([Sigma_plus, Sigma_minus])).astype(np.complex128)

    # Build (I ⊗ V†) and (I ⊗ U)
    I_a = np.eye(2, dtype=np.complex128)
    Id  = np.eye(d, dtype=np.complex128)
    I_kron_Vh = np.kron(I_a, Vh)            # (2d x 2d)
    I_kron_U  = np.kron(I_a, U)             # (2d x 2d)

    # Hadamard on ancilla
    H = (1/np.sqrt(2)) * np.array([[1, 1],
                                   [1,-1]])
    H_kron_I = np.kron(H, Id)

    # Full unitary: (H⊗I) (I⊗U) U_Σ (I⊗V†) (H⊗I)
    U_total = H_kron_I @ I_kron_U @ USigma @ I_kron_Vh @ H_kron_I

    # Optional sanity check
    resid = np.linalg.norm(U_total.conj().T @ U_total - np.eye(2*d))
    if resid > 1e-8:
        print(f"Warning: non-unitary residual ‖U†U−I‖ ≈ {resid:.2e}")

    return U_total, alpha, U, Vh, s


if __name__ == "__main__":
    lattice = "D2Q9"
    take_sqrt = False

    denoiser = DenoisingCollision(lattice=lattice)
    encoding_type = 'sqrt' if take_sqrt else 'full'
    manifold_aware = True

    D = denoiser.build_denoising_op(encoding_type, np.zeros(2), manifold_aware)
    print(D)

    # U, alpha = nagy_block_encoding(D, rescale=True)
    # print(U.round(4))
    #
    # givens_ops, rel_phases = unitary_to_givens_ops(U)
    # givens_rots = ops_to_rots(givens_ops)


    U_sch, alpha, U_svd, Vh_svd, s_svd = schlimgen_block_encoding(D, rescale=True)
    print("Scaling factor:", alpha)
    print("Scaled singular values:", s_svd)
    print("Unitary U:")
    print(U_sch[:9, :9].round(3))


    U_nagy, alpha = nagy_block_encoding(D, rescale=True)
    print("Sz.-Nagy block unitary:")
    print(U_nagy[:9,:9].round(3))

    print("Difference between block-encodings:")
    print(np.linalg.norm(U_sch - U_nagy))