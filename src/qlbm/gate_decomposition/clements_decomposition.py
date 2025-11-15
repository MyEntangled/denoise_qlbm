### Implementation of Clements decomposition (Supplementary Material)
### https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460

import numpy as np

def _right_givens_for_row(a, b):
    """
    Construct a 2×2 unitary/orthogonal matrix G such that [a, b] @ G = [0, r]
    - Nullify the FIRST entry, r = sqrt(|a|^2 + |b|^2)
    - If both a,b are real → G is real orthogonal (float dtype)
    - Otherwise → G is complex unitary (complex dtype)
    Right multiplication → mixes columns (nulls element in a row).
    """
    is_real = np.isrealobj(a) and np.isrealobj(b)
    dtype = float if is_real else complex

    r = np.sqrt(abs(a)**2 + abs(b)**2)
    if r < 1e-9:
        return np.eye(2, dtype=dtype), 0.0

    # Edge case: a=0 (already null)
    if abs(a) < 1e-9:
        if is_real:
            if b >= 0:
                G = np.eye(2)
            else:
                G = np.array([[1, 0], [0, -1]])
        else:
            phase = np.exp(-1j * np.angle(b))
            G = np.array([[1, 0], [0, phase]])
        return G.astype(dtype), float(r)

    # Normalization
    a_n = a / r
    b_n = b / r

    if is_real:
        # Real orthogonal version: [a, b] G = [0, r]
        G = np.array([[b_n,  a_n],
                      [-a_n, b_n]])
    else:
        # Complex unitary version
        G = np.array([[b_n,   np.conj(a_n)],
                      [-a_n,  np.conj(b_n)]])

    return G.astype(dtype), float(r)


def _left_givens_for_col(a, b):
    """
    Construct a 2×2 unitary/orthogonal matrix G such that G @ [a, b]^T = [r, 0]^T
    - Nullify the SECOND entry, r = sqrt(|a|^2 + |b|^2)
    - If both a,b are real → G is real orthogonal (float dtype)
    - Otherwise → G is complex unitary (complex dtype)
    Left multiplication → mixes rows (nulls element in a column).
    """
    is_real = np.isrealobj(a) and np.isrealobj(b)
    dtype = float if is_real else complex

    r = np.sqrt(abs(a)**2 + abs(b)**2)
    if r < 1e-9:
        return np.eye(2, dtype=dtype), 0.0

    if abs(b) < 1e-9:
        if is_real:
            if a >= 0:
                G = np.eye(2)
            else:
                G = np.array([[-1, 0], [0, 1]])
        else:
            phase = np.exp(-1j * np.angle(a))
            G = np.array([[phase, 0], [0, 1]])

        return G.astype(dtype), float(r)

    a_n = a / r
    b_n = b / r

    if is_real:
        # Real orthogonal Givens
        G = np.array([[a_n, b_n],
                      [-b_n, a_n]])
    else:
        # Complex unitary version
        G = np.array([[np.conj(a_n), np.conj(b_n)],
                      [-b_n,          a_n]])

    return G.astype(dtype), float(r)


def clements_algorithm(U: np.ndarray, atol: float = 1e-8):
    """
    Algorithm S1 (Clements et al. 2016), dtype-aware:
         (Left) * U * (Right) = D
    - If U is real orthogonal → keep float ops, D has ±1.
    - Else treat as complex unitary → complex ops, D has unit-modulus phases.
    """
    U = np.asarray(U)
    N = U.shape[0]
    if U.shape != (N, N):
        raise ValueError("U must be square.")

    # Detect real-orthogonal vs complex-unitary
    is_real = np.isrealobj(U)
    dtype = float if is_real else complex

    # Sanity: unitary check
    if not np.allclose(U.conj().T @ U, np.eye(N), atol=atol):
        raise ValueError("U is not unitary within tolerance.")


    Uhat = U.astype(dtype, copy=True)

    left_ops, right_ops = [], []

    # Main sweep: i = 1..N-1
    for i in range(1, N):
        if i % 2 == 1:
            # Right-multiply: null elements in row via column mixing (T^{-1})
            for j in range(0, i):
                r_idx = N - j - 1
                c_idx = i - j - 1
                a = Uhat[r_idx, c_idx]
                b = Uhat[r_idx, c_idx + 1]
                if abs(a) > atol:
                    # nullfy a = Uhat[r_idx, c_idx] (left element)
                    k, l = c_idx, c_idx + 1
                    G, _ = _right_givens_for_row(a, b)  # dtype-adaptive version
                    Uhat[:, [k, l]] = Uhat[:, [k, l]] @ G
                    right_ops.append((k, l, G))

                    # print(f'Mixing cols {k} and {l} by', G)
                    # print(Uhat.round(3))
        else:
            # Left-multiply: null elements in column via row mixing (T)
            for j in range(1, i + 1):
                r_idx = N + j - i - 1
                c_idx = j - 1

                a = Uhat[r_idx - 1, c_idx]
                b = Uhat[r_idx, c_idx]
                if abs(b) > atol:
                    # nullfy b = Uhat[r_idx, c_idx] (bottom element)
                    p, q = r_idx - 1, r_idx
                    G, _ = _left_givens_for_col(a, b)   # dtype-adaptive version
                    Uhat[[p, q], :] = G @ Uhat[[p, q], :]
                    left_ops.append((p, q, G))

                    # print(f'Mixing rows {p} and {q} by', G)
                    # print(Uhat.round(3))


    # Diagonal D
    diag = np.diagonal(Uhat)
    D = np.diag(diag)

    # Must be diagonal within tolerance

    if not np.allclose(Uhat, D, atol=atol):
        raise AssertionError("Decomposition failed: Uhat is not diagonal within tolerance.")

    if not np.allclose(np.abs(np.diagonal(Uhat)), 1.0, atol=atol):
        raise AssertionError("Decomposition failed: diagonal elements are not unit-modulus.")


    return (left_ops,     # (p, q, 2x2 block)
            right_ops,   # (k, l, 2x2 block)
            diag,           # diagonal phases (±1 for real case)
            Uhat,)             # should be diagonal ≈ D


# ---- Helper: verify clements's algorithm ----

def _verify_clements_algorithm(U, left_ops, right_ops):
    """Verify Clements's algorithm:
    Re-apply recorded operations to confirm (Left) * U * (Right) == D."""
    Uhat = U.copy()
    for (k, l, G) in right_ops:
        Uhat[:, [k, l]] = Uhat[:, [k, l]] @ G
    for (p, q, G) in left_ops:
        Uhat[[p, q], :] = G @ Uhat[[p, q], :]
    return Uhat


def unitary_to_givens_ops(U: np.ndarray):

    # Clements decomposition: (Left) * U * (Right) = D
    left_ops, right_ops, diag, _ = clements_algorithm(U)

    # -> U = (Left_l ... Left_0)^* @ D @ (Right_0 ... Right_r)^*
    #      = Left_0^* ... Left_l^* @ D @ Right_r^* ... Right_0^*
    #      = D @ (D^* Left_0^* D) ... (D^* @ Left_l^* @ D) @ Right_r^* ... Right_0^*
    #      := D @ L_0 ... L_l @ R_r ... R_0

    # We store operators by order of application, i.e. (R_0, ..., R_r, L_l, ... L_0)

    ops = []
    for (k,l,G) in right_ops:
        assert k < l
        G_dag = G.conj().T
        ops.append((k, l, G_dag))

    for (p,q,G) in reversed(left_ops):
        assert p < q
        G_dag = G.conj().T
        # Conjugate by D
        D_eff = np.diag([diag[p], diag[q]])
        G_eff = D_eff.conj() @ G_dag @ D_eff
        ops.append((p, q, G_eff))

    return ops, diag


if __name__ == '__main__':
    # 1. Create a random unitary matrix
    N = 4
    #U_original = np.eye(N)
    U_original, _ = np.linalg.qr(np.random.randn(N, N))
    #U_original = unitary_group.rvs(N)

    print("Original Unitary Matrix U:")
    print(np.round(U_original, 3))
    print("-" * 30)

    # 2. Decompose the matrix
    left_ops, right_ops, diag, Uhat_algo = clements_algorithm(U_original)
    Uhat_test = _verify_clements_algorithm(U_original, left_ops, right_ops)

    assert np.allclose(Uhat_algo, Uhat_test), "Re-applied ops do not match Uhat from algorithm."
    assert np.allclose(Uhat_algo, np.diag(diag)), "Uhat from algorithm does not match diagonal D."
    print("Diagonal", diag)

    ops, rel_phases = unitary_to_givens_ops(U_original)

    print(ops)
    print(rel_phases)

