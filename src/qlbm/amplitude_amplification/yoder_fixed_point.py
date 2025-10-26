# Fixed-point amplitude amplification: https://arxiv.org/pdf/1409.3305
import numpy as np


def chebyshev_T(n, z):
    """Chebyshev polynomial of the first kind, T_n(z), for real z (vectorized)."""
    z = np.asarray(z, dtype=float)
    z_clip = np.clip(z, -1.0, 1.0)
    # piecewise for numerical stability
    inside = np.abs(z) <= 1.0 + 1e-15
    out = np.empty_like(z, dtype=float)
    out[inside] = np.cos(n * np.arccos(z_clip[inside]))
    # for |z|>1: T_n(z) = cosh(n arccosh(|z|)) * sgn(z)^n
    if np.any(~inside):
        zabs = np.abs(z[~inside])
        out[~inside] = np.cosh(n * np.arccosh(zabs)) * (np.sign(z[~inside]) ** n)
    return out


def choose_L(lambda_min, delta):
    """Return the smallest odd L such that width <= lambda_min, where
       width w = 1 - 1 / T_{1/L}(1/delta)^2 (success probability).
    """
    if not (0 < delta <= 1):
        raise ValueError("delta must be in (0,1].")

    # # Brute-force search (slower):
    # L = 1  # must be odd; we'll increment by 2
    # while True:
    #     L += 2
    #     Gamma = chebyshev_T(1.0 / L, 1.0 / delta)  # T_{1/L}(1/delta); here n can be fractional
    #     w = 1.0 - 1.0 / (Gamma ** 2)
    #     if w <= lambda_min:
    #         return int(L)

    num = np.arccosh(1.0 / delta)
    den = np.arccosh(1.0/np.sqrt(1.0 - lambda_min))
    L_float = num / den
    L_odd = int(np.ceil(L_float))
    # Ensure L is odd
    if L_odd % 2 == 0:
        L_odd += 1
    return L_odd


# ---------- Phase schedule ----------
def fixed_point_phases(L, delta):
    """Compute alpha_j and beta_j for j=1..l, where L=2l+1, and beta_{l-j+1} = -alpha_j."""
    if L % 2 != 1:
        raise ValueError("L must be odd.")
    l = (L - 1) // 2

    gamma_inv = chebyshev_T(1.0 / L, 1.0 / delta)  # = 1/gamma
    gamma = 1.0 / gamma_inv
    s = np.sqrt(1.0 - gamma**2)    # sqrt(1 - gamma^2)

    alphas = np.empty(l, dtype=float)

    for j in range(1, l + 1):
        y = np.tan(2.0 * np.pi * j / L) * s

        # α_j = 2 * cot^{-1}(y) in (0, π]; implement as 2*atan(1/y), being careful around y=0
        alpha_j = 2.0 * np.arctan2(1.0, y)

        # map to (0, 2π) consistently if desired; probabilities are phase-invariant so not needed
        alphas[j - 1] = alpha_j

    betas = -alphas[::-1]  # β_{l-j+1} = -α_j
    return alphas, betas


# ---------- Build the 2D operators on span{|bar_t>, |t>} ----------
def S_s(alpha, lmbda):
    """Selective phase on |s>, in the basis {|bar_t>, |t>} where |s> = [sqrt(1-λ), sqrt(λ)]."""
    if not (0 < lmbda <= 1):
        raise ValueError("lambda must be in (0,1].")

    a = np.sqrt(1.0 - lmbda)
    b = np.sqrt(lmbda)

    svec = np.array([[a], [b]], dtype=complex)
    P = svec @ svec.conj().T
    return np.eye(2, dtype=complex) - (1.0 - np.exp(-1j * alpha)) * P

def S_t(beta):
    """Selective phase on |t>, i.e., phase on the |t> basis vector (index 1)."""
    # projector |t><t| = [[0,0],[0,1]]
    P = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    return np.eye(2, dtype=complex) - (1.0 - np.exp(1j * beta)) * P

def G_iter(alpha, beta, lmbda):
    """Generalized Grover iterate: G = - S_s(alpha) S_t(beta) (global phase irrelevant for prob)."""
    if not (0 < lmbda <= 1):
        raise ValueError("lambda must be in (0,1].")
    return - S_s(alpha, lmbda) @ S_t(beta)


# ---------- Theoretical probabilities ----------
def P_prefix_theory(h, lmbda, L, delta):
    if not (0 < lmbda <= 1):
        raise ValueError("lambda must be in (0,1].")

    gamma_inv = chebyshev_T(1.0 / L, 1.0 / delta)
    x = np.sqrt(1.0 - lmbda)

    num = chebyshev_T(h, gamma_inv * x)
    den = chebyshev_T(h, gamma_inv)
    return 1.0 - (num / den) ** 2

def P_final_theory(lmbda, L, delta):
    if not (0 < lmbda <= 1):
        raise ValueError("lambda must be in (0,1].")

    gamma_inv = chebyshev_T(1.0 / L, 1.0 / delta)
    x = np.sqrt(1.0 - lmbda)
    return 1.0 - (delta ** 2) * (chebyshev_T(L, gamma_inv * x)) ** 2


# ---------- Simulation on the 2D invariant subspace ----------
def simulate_prefix_probabilities(lmbda, L, delta):
    """Return list P_h (h=0..l) from explicit matrix products of G(α_j,β_j)."""
    if not (0 < lmbda <= 1):
        raise ValueError("lambda must be in (0,1].")

    l = (L - 1) // 2
    alphas, betas = fixed_point_phases(L, delta)

    # initial |s> = [sqrt(1-λ), sqrt(λ)]
    state = np.array([np.sqrt(max(0.0, 1.0 - lmbda)), np.sqrt(max(0.0, lmbda))], dtype=complex)
    probs = [abs(state[1])**2]  # h=0

    for h in range(1, l + 1):
        G = G_iter(alphas[h-1], betas[h-1], lmbda)
        state = G @ state
        probs.append(abs(state[1])**2)
    return probs


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # ---------- Demo parameters ----------
    lambda_min = 0.03         # assumed lower bound on λ
    delta = 0.1 ** 0.5        # choose delta so that δ^2 = 0.1 (for visibility)
    L = choose_L(lambda_min, delta)
    l = (L - 1) // 2

    print(f"Chosen parameters: lambda_min={lambda_min}, delta={delta:.6f}, L={L} (queries = {L-1})")

    # ---------- Verify designed endpoint equals theory, and bound >= 1 - δ^2 for λ ≥ λ_min ----------
    lmbdas = np.linspace(lambda_min, 0.9, 100)
    P_sim = []
    P_th = []
    violations = 0

    for lam in lmbdas:
        # simulate full prefix sequence to endpoint (h = l)
        P_prefix = simulate_prefix_probabilities(lam, L, delta)
        P_end_sim = P_prefix[-1]
        P_end_th = P_final_theory(lam, L, delta)
        P_sim.append(P_end_sim)
        P_th.append(P_end_th)
        if P_end_sim + 1e-10 < (1.0 - delta**2):  # numerical tolerance
            violations += 1

    P_sim = np.array(P_sim)
    P_th = np.array(P_th)
    max_abs_err = np.max(np.abs(P_sim - P_th))
    idx_worst = int(np.argmax(np.abs(P_sim - P_th)))
    print(f"Max |simulation - theory| over λ∈[{lambda_min},0.9] is {max_abs_err:.3e} at λ={lmbdas[idx_worst]:.4f}")
    print(f"All λ ≥ λ_min satisfy P_final ≥ 1-δ^2 = {1-delta**2:.6f} ?  {'YES' if violations==0 else 'NO'} (violations={violations})")

    # ---------- Plot: simulated vs theory ----------
    plt.figure()
    plt.plot(lmbdas, P_sim, label="Simulated endpoint")
    plt.plot(lmbdas, P_th, linestyle="--", label="Theory endpoint")
    plt.axhline(1.0 - delta**2, linestyle=":", label="Guarantee 1 - δ²")
    plt.xlabel("λ")
    plt.ylabel("Success probability at designed endpoint")
    plt.title(f"Fixed-point Grover (L={L}, l={l}, queries={L-1})")
    plt.legend()
    plt.show()

    # ---------- Also verify prefix-by-prefix for a single λ ----------
    lam_test = 0.08  # ≥ lambda_min
    P_prefix_sim = simulate_prefix_probabilities(lam_test, L, delta)
    P_prefix_th = [P_prefix_theory(h, lam_test, L, delta) for h in range(0, l + 1)]
    diffs = np.array(P_prefix_sim) - np.array(P_prefix_th)
    print(f"Prefix check at λ={lam_test:.3f}: max |sim-theory| over h=0..{l} is {np.max(np.abs(diffs)):.3e}")
    print("Prefix probabilities (h, sim, theory):")
    for h in range(0, l + 1):
        print(f"h={h:2d}:  {P_prefix_sim[h]:.9f}   {P_prefix_th[h]:.9f}")
