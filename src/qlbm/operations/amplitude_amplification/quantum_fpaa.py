import numpy as np
from src.qlbm.operations.amplitude_amplification.yoder_fixed_point import compute_L, compute_phases

def setup_fpaa(lambda_min, delta):
    """Setup parameters for Fixed-Point Amplitude Amplification (FPAA).

    Args:
        lambda_min (float): Minimum success probability threshold.
        delta (float): Error tolerance parameter in (0, 1].

    Returns:
        L (int): The chosen odd integer for the number of iterations.
        l (int): (L - 1) // 2
        alphas (np.ndarray): Phase angles alpha_j for j=1..l, where L=2l+1.
        betas (np.ndarray): Phase angles beta_j for j=1..l, where beta_{l-j+1} = -alpha_j.
    """
    # Input validation
    if not (0.0 < lambda_min <= 1.0):
        raise ValueError("lambda_min must be in (0, 1].")
    if not (0.0 < delta <= 1.0):
        raise ValueError("delta must be in (0, 1].")

    # Get L from Yoder's computation
    L = compute_L(lambda_min, delta)
    l = (L - 1) // 2

    if l == 0:
        return L, l, np.array([]), np.array([])

    # Get the phase angles from Yoder's computation
    alphas, betas = compute_phases(L, delta)

    return L, l, alphas, betas

def _conjugate_subsystem_operators_(op_list):
    """Yield the conjugate (dagger) of each operator in the input list.
    """
    conj_list = []
    for op in op_list[::-1]:
        if len(op) == 2:
            Ua, Ud = op
            conj_list.append((Ua.conj().T, Ud.conj().T))
        elif len(op) == 1:
            U = op[0]
            conj_list.append((U.conj().T,))
        else:
            raise ValueError("Each element of op_list must be either a single unitary or a pair of unitaries.")
    return conj_list

def fpaa_full_unitaries(alphas, betas, A):
    assert len(alphas) == len(betas), "Length of alphas and betas must be equal."
    assert isinstance(A, np.ndarray) and A.shape[0] == A.shape[1], "U_init must be a square numpy array."

    l = len(alphas)
    Q = A.shape[0] // 2

    # Define single-qubit Rz
    RZ = lambda theta: np.diag([np.cos(theta/2) - 1j * np.sin(theta/2), np.cos(theta/2) + 1j * np.sin(theta/2)])

    # Define S_t and S_s (generalized Grover operator G = - S_s S_t)
    S_t = lambda beta: np.kron(RZ(-beta), np.eye(Q))  # phase exp(1j * beta/2)
    S_s = lambda alpha: A @ np.kron(RZ(alpha), np.eye(Q)) @ A.conj().T  # phase exp(-1j * alpha/2)

    unitaries = []

    for j in range(l):
        unitaries.append(S_t(betas[j]))
        unitaries.append(S_s(alphas[j]))

    return unitaries

def fpaa_subsystem_unitaries(alphas, betas, A_seq):

    assert len(alphas) == len(betas), "Length of alphas and betas must be equal."
    assert isinstance(A_seq, list)

    l = len(alphas)

    # Define single-qubit rotation operators
    RZ = lambda theta: np.diag([np.cos(theta/2) - 1j * np.sin(theta/2), np.cos(theta/2) + 1j * np.sin(theta/2)])

    for A in A_seq:
        if not (isinstance(A, tuple) and (len(A) == 1 or len(A) == 2)):
            raise ValueError("Each element of A_seq must be either a single unitary or a pair of unitaries.")

    A = A_seq[0]
    if len(A) == 1: # A = (U,)
        Q = A[0].shape[0] // 2
    else:   # A = (Ua, Ud)
        Q = A[1].shape[0]

    A_dagger = _conjugate_subsystem_operators_(A_seq)

    unitaries = []

    for j in range(l):
        unitaries.append((RZ(-betas[j]), np.eye(Q)))  # S_t(beta) = Rz(-beta) ⊗ I

        # S_a(alpha) = A @ (Rz(alpha) ⊗ I)  @ A^dagger
        unitaries += A_dagger
        unitaries.append((RZ(alphas[j]), np.eye(Q)))
        unitaries += A

    return unitaries

def hnufpaa_full_unitaries(alphas, betas, A):
    """Heralded non-unitary fixed-point amplitude amplification."""
    assert len(alphas) == len(betas), "Length of alphas and betas must be equal."
    assert isinstance(A, np.ndarray) and A.shape[0] == A.shape[1], "U_init must be a square numpy array."

    l = len(alphas)
    Q = A.shape[0] // 2

    # Define single-qubit Rz
    RZ = lambda theta: np.diag([np.cos(theta/2) - 1j * np.sin(theta/2), np.cos(theta/2) + 1j * np.sin(theta/2)])

    # Define S_t and S_s (generalized Grover operator G = - S_s S_t)
    S_t = lambda beta: np.kron(RZ(-beta), np.eye(Q))  # phase exp(1j * beta/2)
    S_s = lambda alpha: A @ np.kron(RZ(alpha), np.eye(Q)) @ A.conj().T  # phase exp(-1j * alpha/2)

    unitaries = []

    for j in range(l):
        unitaries.append(S_t(betas[j]))
        unitaries.append(S_s(alphas[j]))

    return unitaries


if __name__ == '__main__':
    from src.qlbm.operations.collision.denoiser import DenoisingCollision
    from src.qlbm.gate_decomposition.block_encoding import schlimgen_block_encoding

    # Example usage
    lmbda_min = 0.4
    delta = 0.00001 ** 0.5
    L, l, alphas, betas = setup_fpaa(lmbda_min, delta)
    print(f"L: {L}, l: {l}")
    print(f"Alphas: {alphas}")
    print(f"Betas: {betas}")


    lattice = "D2Q9"
    Q = 9
    take_sqrt = True

    denoiser = DenoisingCollision(lattice=lattice)
    encoding_type = 'sqrt' if take_sqrt else 'full'
    manifold_aware = True

    #D = denoiser.build_denoising_op(encoding_type, np.zeros(2), manifold_aware)

    D, _ = np.linalg.qr(np.random.randn(Q, Q))
    #D = np.random.randn(Q, Q)

    D /= np.linalg.norm(D, 2) * 1.5 # ensure ||D|| < 1

    # Schlimgen block encoding of D
    # Full unitary: U_col = (H ⊗ I) (I ⊗ U) U_Σ (I ⊗ V†) (H ⊗ I)
    U_col, alpha, U_svd, USigma, Vh_svd = schlimgen_block_encoding(D, rescale=True)

    Ia, Id = np.eye(2), np.eye(Q)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    U_col_seq = [(H, Id), (Ia, Vh_svd), (USigma,), (Ia, U_svd), (H, Id)]


    state = np.random.rand(Q)
    state /= np.linalg.norm(state)
    state = np.kron(np.array([1, 0]), state)  # add ancilla
    state = U_col @ state
    print(f"Initial success probability: {np.linalg.norm(state[:Q])**2}")
    print(f"Initial failure probability: {np.linalg.norm(state[Q:])**2}")

    op_list = list(fpaa_full_unitaries(alphas, betas, A=U_col))
    print(f"unitary operators: {len(op_list)}")
    for op in op_list:
        #print(len(op), [len(each) for each in op])
        state = op @ state

    prob = np.linalg.norm(state[:Q])
    print(f"Final success probability: {np.linalg.norm(state[:Q]) ** 2}")
    print(f"Final failure probability: {np.linalg.norm(state[Q:]) ** 2}")