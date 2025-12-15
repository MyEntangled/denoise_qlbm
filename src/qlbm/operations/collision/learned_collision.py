from src.lattices.lbm_lattices import get_lattice

import numpy as np
import jax.numpy as jnp
import jax
import scipy.linalg
from functools import partial

# --- JITted functions: perform optimization on the orthogonal group using landing algorithm [arXiv:2102.07432]---
@partial(jax.jit, static_argnames=("Q"))
def loss_function(U: jnp.ndarray,
                  input_tensorstates: jnp.ndarray,
                  target_states: jnp.ndarray,
                  Q: int) -> jnp.ndarray:
    """
    L(U) = 1 - mean_i  |⟨0^r,y_i| U |a^r,x_i⟩| / sqrt(λ_i),
    where λ_i = || (⟨0^r|⊗I) U |a^r,x_i⟩ ||^2  (postselect ancilla = |0^r⟩).
    """
    #Y = input_tensorstates @ U.T                 # apply collision: (U @ X^T)^T
    Y = input_tensorstates @ U
    Y_post = Y[..., :Q]                            # ancilla postselection: keep first Q comps

    lam_sqrt = jnp.linalg.norm(Y_post, axis=-1, keepdims=True)   # sqrt(λ_i)
    lam_sqrt = jnp.where(lam_sqrt > 0, lam_sqrt, 1e-12)  # numeric guard
    Y_post_nml = Y_post / lam_sqrt

    overlap = jnp.abs(jnp.sum(target_states * Y_post_nml, axis=1))
    overlap_error = 1.0 - jnp.mean(overlap)

    return overlap_error

@jax.jit
def relative_grad(X: jnp.ndarray, euclid_grad_X: jnp.ndarray) -> jnp.ndarray:
    """ψ(X) = Skew(∇f(X) Xᵀ)  (Riemannian / relative gradient on O(n))."""
    GXT = euclid_grad_X @ X.T
    return 0.5 * (GXT - GXT.T)

@jax.jit
def landing_field(X: jnp.ndarray, relative_grad_X: jnp.ndarray, mu: float) -> jnp.ndarray:
    """
    Λ(X) = ψ(X) X + μ ∇N(X),  with  N(X) = 1/4||XXᵀ−I||²  ⇒  ∇N(X)=(XXᵀ−I)X.
    """
    XXT = X @ X.T
    grad_N = (XXT - jnp.eye(X.shape[0], dtype=X.dtype)) @ X
    return relative_grad_X @ X + mu * grad_N

@jax.jit
def safe_stepsize(X: jnp.ndarray, relative_grad_X: jnp.ndarray, mu: float, eps: float) -> jnp.ndarray:
    """
    Stepsize from landing algorithm analysis.
    d = N(X) = ¼||XXᵀ−I||²,  a = ||ψ(X)||.
    η_safe = ( √(α² + 4β(ε−d)) + α ) / (2β),
    α = 2μd − 2ad − 2μd²,   β = a² + μ²d³ + 2μad² + a²d.
    """
    I = jnp.eye(X.shape[0], dtype=X.dtype)
    d = 0.25 * jnp.sum((X @ X.T - I) ** 2)
    a = jnp.linalg.norm(relative_grad_X)

    alpha = 2 * mu * d - 2 * a * d - 2 * mu * (d ** 2)
    beta  = a**2 + (mu**2) * (d**3) + 2 * mu * a * (d**2) + (a**2) * d

    disc = alpha**2 + 4 * beta * (eps - d)
    disc = jnp.maximum(disc, 0.0)  # numeric guard
    return (jnp.sqrt(disc) + alpha) / (2 * beta + 1e-18)

def get_tensorstate(states: jnp.ndarray, r: int, ancilla: str) -> jnp.ndarray:
    """
    Construct composite tensor states |0^r>⊗|ψ> or |+^r>⊗|ψ>.

    Args:
        states: (B, D) array of base statevectors |ψ>.
        r: number of qubits in the first register.
        ancilla: either 'zero' or 'plus'.

    Returns:
        new_states: (B, 2^r * D) array of tensor product states.
    """
    if ancilla not in ("zero", "plus"):
        raise ValueError("ancilla must be 'zero' or 'plus'.")

    B, D = states.shape
    factor = 2 ** r

    if ancilla == "zero":
        # |0^r> ⊗ |ψ> → pad with zeros
        pad_shape = (B, (factor - 1) * D)
        zeros = jnp.zeros(pad_shape, dtype=states.dtype)
        new_states = jnp.concatenate([states, zeros], axis=1)
    else:  # first_reg == "plus"
        # |+^r> ⊗ |ψ> = (1/√factor) Σ_k |k>⊗|ψ>
        tiled = jnp.tile(states, reps=(1, factor))
        new_states = tiled / jnp.sqrt(factor)

    return new_states

# ------------------------------------------


class LearnedCollision:
    def __init__(self, lattice: str):
        c, self.weights = get_lattice(lattice, as_array=True)
        self.vels = c.T
        self.Q, self.D = c.shape

        self.cs = 1./np.sqrt(3)  # speed of sound

    def train_collision_unitary(self,
                                X,
                                Y,
                                r: int,
                                ancilla: str = "zero",
                                eta: float = 0.5,
                                mu: float = 1.0,
                                eps: float = 0.5,
                                max_steps: int = 10000,
                                tol: float = 1e-9,
                                rng: np.random.Generator | None = None,
                                verbose: bool = True):
        """
        Train an orthogonal (unitary) collision operator U for the quantum LBM model.

        The algorithm embeds input and target state vectors into a higher-dimensional
        tensor space via ancillary qubits, then optimizes a collision matrix U to
        minimize a fidelity-based loss function. The update follows a Riemannian
        gradient descent on the orthogonal manifold using the relative gradient,
        safe step size, and landing field projection.

        Args:
            X: (N, Q) array of input state vectors.
            Y: (N, Q) array of target state vectors.
            r: Number of ancilla qubits used in the tensor embedding.
            ancilla: Type of ancilla initialization, either:
                     - "zero" → ancilla prepared in |0^r⟩
                     - "plus" → ancilla prepared in |+^r⟩
            eta: Base learning rate (step size) for gradient descent.
            mu: Hyperparameter for the landing-field projection (stabilizes updates).
            eps: Numerical safeguard in step-size computation (avoids divergence).
            max_steps: Maximum number of gradient steps to perform.
            tol: Stopping criterion for change in loss between iterations.
            rng: Optional NumPy random number generator for reproducible initialization.
            verbose: If True, prints initial and final overlaps and intermediate info.

        Returns:
            U: (Q·2^r, Q·2^r) JAX array representing the trained orthogonal collision operator.
            history: dict containing the evolution of the training loss:
                     - 'error': list of loss values at each iteration.

        Notes:
            - The optimization is performed directly on the orthogonal manifold:
                  U_{k+1} = U_k - η * landing_field(U_k, relative_grad(U_k, ∇L))
            - Initialization is done via QR decomposition of a Gaussian random matrix.
            - The loss function measures the mean anti-overlap between U|ψ⟩ and |ϕ⟩.
            - Convergence is detected when loss change < tol or fidelity saturates.
        """
        if rng is None:
            rng = np.random.default_rng()

        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        Q = X.shape[1]

        # Tensor product states with ancillas |0^r> or |+^r>
        Xt = get_tensorstate(X, r, ancilla)  # (B, 2^r * Q)
        #Yt = get_tensorstate(Y, r, ancilla)
        new_dim = Q * (2 ** r)

        # Random orthogonal init (NumPy QR → JAX array)
        U0, _ = np.linalg.qr(rng.standard_normal((new_dim, new_dim)))
        U = jnp.asarray(U0)

        err0 = float(jnp.mean(loss_function(U, Xt, Y, Q)))
        history = {"error": [err0]}
        if verbose:
            print("Initial overlap:", 1.0 - err0)

        for step in range(max_steps):
            egrad = jax.grad(loss_function)(U, Xt, Y, Q)
            rgrad = relative_grad(U, egrad)
            eta_safe = safe_stepsize(U, rgrad, mu, eps)
            U = U - min(eta, eta_safe) * landing_field(U, rgrad, mu)

            antifid = float(loss_function(U, Xt, Y, Q))
            history["error"].append(antifid)
            if antifid <= 1.0 and abs(history["error"][-1] - history["error"][-2]) < tol:
                break

            if step > 0 and step % 10 == 0 and verbose:
                print(f"Step {step}: overlap = {1.0 - antifid:.6f}")

        U = np.asarray(U, dtype=float)
        U, _ = scipy.linalg.polar(U) # re-orthogonalize to fix numeric drift

        if verbose:
            print("Final overlap (with re-orthog.):", 1.0 - float(loss_function(U, Xt, Y, Q)))

        return np.asarray(U), history


    def test_collision_unitary(self,
                               U,
                               X_test,
                               Y_test,
                               r: int,
                               ancilla: str = "zero") -> dict[str, float]:
        """
        Evaluate a trained collision operator U on test data.

        Args:
            U: (Q·2^r, Q·2^r) trained unitary / orthogonal collision operator
            X_test: (N, Q) input state vectors
            Y_test: (N, Q) target state vectors
            r: number of ancilla qubits used in training
            ancilla: 'zero' or 'plus' — ancilla register initialization

        Returns:
            dict with:
              - 'loss': mean loss_function(U, X_test, Y_test, Q)
              - 'overlap': 1 - loss (mean fidelity)
        """
        U = jnp.asarray(U)
        X_test = jnp.asarray(X_test)
        Y_test = jnp.asarray(Y_test)

        Q = X_test.shape[1]

        Xt = get_tensorstate(X_test, r, ancilla)
        Yt = get_tensorstate(Y_test, r, ancilla)

        # Compute loss (anti-overlap)
        test_loss = float(jnp.mean(loss_function(U, Xt, Y_test, Q)))
        test_overlap = 1.0 - test_loss

        print("Learned Collision Test Results:")
        print(f"Test overlap: {test_overlap:.6f}  |  Test loss: {test_loss:.6e}")
        return {"loss": test_loss, "overlap": test_overlap}


if __name__ == '__main__':
    from src.qlbm.data_generation.sample_distribution import sample_low_mach_data
    from src.qlbm.data_generation.create_states import distributions_to_statevectors
    from src.qlbm.data_generation.helper import split_train_test
    from src.lattices.lbm_symmetries import get_symmetry

    # --- choose lattice + data spec ---
    lattice = "D2Q9"
    n_samples = 25000
    r = 1  # ancilla qubits
    ancilla = "zero"  # or "plus"

    rng = np.random.default_rng(42)

    # 1) Sample distributions (F) and their post-collision targets (F_target)
    rho = 1.0  # base density
    mean_norm_u = 0.2/np.sqrt(3)  # mean |u| (in lattice units, small => low Mach)
    std_norm_u = 0.2/np.sqrt(3)  # std deviation of |u|
    rel_noise_strength = 0.1  # relative magnitude of thermal/noise fluctuations

    F, F_target = sample_low_mach_data(
        n_samples=n_samples,
        lattice=lattice,
        rho=rho,
        mean_norm_u=mean_norm_u,
        std_norm_u=std_norm_u,
        rel_noise_strength=rel_noise_strength,
        omega = 1.,
        rng=rng,
        weighted_noise=True
    )

    # 2) Convert distributions → state vectors (optionally sqrt + norm) and augment by symmetries
    sym = get_symmetry(lattice)  # permutation dict
    X = distributions_to_statevectors(F, take_sqrt=True, normalize=True, symmetries=sym)
    Y = distributions_to_statevectors(F_target, take_sqrt=True, normalize=True, symmetries=sym)

    # 3) Train/test split
    Xtr, Ytr, Xte, Yte = split_train_test(X, Y, train_ratio=0.8, shuffle=True, rng=rng)
    print("Train/test sizes:", Xtr.shape[0], Xte.shape[0])

    # 4) Train collision operator U by landing algorithm
    collision_model = LearnedCollision(lattice=lattice)

    _, w = get_lattice(lattice, as_array=True)

    U, history = collision_model.train_collision_unitary(
        Xtr, Ytr, r=r,
        ancilla=ancilla,
        eta=0.5, mu=1.0, eps=0.5,
        max_steps=5000, tol=1e-9,
        rng=rng, verbose=True
    )


    # 5) Unitary check and accuracy check
    unitarity_error = np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))
    print(f"Trained collision operator unitarity error: {unitarity_error:.2e}")
    print(U[:9, :9].round(3))

    # 6) Evaluate on test set
    metrics = collision_model.test_collision_unitary(U, Xte, Yte, r=r, ancilla=ancilla)
    print(metrics)