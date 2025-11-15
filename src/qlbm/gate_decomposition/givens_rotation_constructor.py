import numpy as np

def get_givens_angle(G: np.ndarray, atol: float = 1e-8):
    """
    Given a 2x2 REAL Givens block G = [[c, -s],
                                       [s,  c]]
    with c = cos(theta), s = sin(theta),
    return theta = atan2(s, c).

    Raises:
        ValueError: if shape is not (2,2)
        NotImplementedError: if M is complex (we only handle real case now)
        ValueError: if M is not approximately orthogonal with det ≈ 1
    """
    G = np.asarray(G)
    if G.shape != (2, 2):
        raise ValueError("M must be 2x2.")

    # Complex not supported (for now)
    if np.iscomplexobj(G):
        raise NotImplementedError("Complex inputs are not handled yet.")

    # Basic orthogonality & det checks (cheap sanity)
    if not np.allclose(G.T @ G, np.eye(2), atol=atol):
        raise ValueError("M is not orthogonal within tolerance.")
    det = np.linalg.det(G)
    if not np.isclose(det, 1.0, atol=atol):
        raise ValueError(f"det(M) ≈ {det:.6g} ≠ 1; not a proper rotation.")

    # Extract c, s from the canonical positions
    c,s  = G[0, 0], G[1, 0]

    # Angle
    theta = float(np.arctan2(s, c))
    return theta


def givens_rot_to_mat(N: int, m: int, n: int, theta: float):
    """Convert Givens rotations to matrices."""

    c, s = np.cos(theta), np.sin(theta)
    G = np.array([[c, -s], [s,  c]])

    G_embed = np.eye(N, dtype=float)
    G_embed[np.ix_([m, n], [m, n])] = G

    return G_embed


def check_decomposition(givens_rots, rel_phases, U: np.ndarray, atol: float = 1e-8):

    if np.iscomplexobj(U):
        raise NotImplementedError("Complex matrices are not handled yet.")

    N = len(rel_phases)

    if U.shape[0] != U.shape[1]:
        raise ValueError("U must be square.")
    if U.shape[0] != N:
        raise ValueError(f"U must have size {N} x {N}.")

    givens_mats = [givens_rot_to_mat(N, m, n, theta) for (m, n, theta) in givens_rots]

    U_rec = np.eye(N, dtype=float)
    for M in givens_mats:
        U_rec = M @ U_rec

    # Multiply relative phase
    U_rec = np.diag(rel_phases) @ U_rec

    if not np.allclose(U_rec, U, atol=atol):
        raise ValueError("Decomposition of U is not correct within tolerance.")

    return U_rec


if __name__ == "__main__":
    from src.qlbm.gate_decomposition.clements_decomposition import unitary_to_givens_ops

    # 1. Create a random orthogonal matrix
    N = 18
    #U_original = np.eye(N)
    U_original, _ = np.linalg.qr(np.random.randn(N, N))

    print("Original orthog. matrix:")
    print(np.round(U_original, 3))
    print("-" * 30)

    # 2. Decompose the matrix
    givens_ops, rel_phases = unitary_to_givens_ops(U_original)

    # 3. Convert to rotation form
    givens_rots = [(m,n,get_givens_angle(op)) for m,n,op in givens_ops]
    print(givens_rots)

    # 4. Verify the decomposition
    U_rec = check_decomposition(givens_rots, rel_phases, U_original)
    #print("Reconstructed ortho. matrix")
    #print(np.round(U_rec, 3))


