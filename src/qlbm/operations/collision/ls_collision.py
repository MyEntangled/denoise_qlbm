from src.qlbm.lbm_lattices import get_lattice
from src.qlbm.gate_decomposition.block_encoding import nagy_block_encoding, schlimgen_block_encoding

import numpy as np

def proj_opnorm_ball(A: np.ndarray, radius: float = 1.0):
    """
    Projection of A onto {B : ||B||_2 <= radius} by clipping its singular values.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    s_clipped = np.minimum(s, radius)
    return U @ np.diag(s_clipped) @ Vt


class LSQCollision:
    def __init__(self, lattice: str):
        c, self.weights = get_lattice(lattice, as_array=True)
        self.vels = c.T
        self.Q, self.D = c.shape

        self.cs = 1./np.sqrt(3)  # speed of sound


    def train_ls_collision(self, X, Y, radius: float = 1.0):
        """
        Solve  min_B ||B X^T - Y^T||_F^2   s.t.  ||B||_2 <= radius

        Parameters
        ----------
        X: (N, Q) array of input state vectors.
        Y: (N, Q) array of target state vectors.
        radius: Spectral-norm bound for B (default 1.0).

        Returns
        -------
        B : (Q, Q) array, the spectral-norm–constrained estimator.
        loss : Objective value ||B X^T - Y^T||_F^2.
        """
        if X.shape != Y.shape:
            raise ValueError(f"X and Y must have the same shape; got {X.shape} vs {Y.shape}")
        N, Q = X.shape

        # Least-squares solution: B_ls = Y^T X (X^T X)^{-1}
        H = X.T @ X  # (Q, Q)
        C = Y.T @ X  # (Q, Q)
        H_pinv = np.linalg.pinv(H)
        B_ls = C @ H_pinv  # (Q, Q)

        print("B_ls matrix", B_ls)
        print(B_ls.round(4))
        print("svals:", np.linalg.svdvals(B_ls))

        # Project onto spectral-norm ball
        B = proj_opnorm_ball(B_ls, radius=radius)

        # Compute loss
        residual = B @ X.T - Y.T    # (Q, N)
        error = np.sum(residual * residual, axis=0)  # ||·||_F^2
        avg_loss = np.mean(error)

        print("B matrix")
        print(B.round(4))
        print("clipped svals:", np.linalg.svdvals(B))

        print("Avg least-square loss", avg_loss)

        return B, avg_loss


    def test_ls_collision(self, B_col, input_states, target_states):
        output_states = (B_col @ input_states.T).T

        output_norms = np.linalg.norm(output_states, axis=1, keepdims=True)
        output_states /= output_norms

        test_overlap = float(np.mean(np.abs(np.sum(output_states * target_states, axis=1))))
        test_loss = 1 - test_overlap

        print("--- LSQ-Collision Test Results ---")
        print(f"Test overlap: {test_overlap:.6f}  |  Test loss: {test_loss:.6e}")
        return {"loss": test_loss, "overlap": test_overlap}


if __name__ == '__main__':
    from src.qlbm.data_generation.sample_distribution import sample_low_mach_data
    from src.qlbm.data_generation.create_states import distributions_to_statevectors
    from src.qlbm.lbm_symmetries import get_symmetry

    rng = np.random.default_rng(0)

    # Parameters
    n_samples = 50000
    lattice = "D2Q9"

    rho = 1
    cs = 1./np.sqrt(3)
    mean_norm_u = 0.2 * cs
    std_norm_u = 0.2 * cs
    rel_noise_strength = 0.1  # relative magnitude of thermal/noise fluctuations

    take_sqrt = True
    normalize = True
    symmetries = get_symmetry(lattice)

    # Generate samples
    F, F_target = sample_low_mach_data(
        n_samples=n_samples,
        lattice=lattice,
        rho=rho,
        mean_norm_u=mean_norm_u,
        std_norm_u=std_norm_u,
        rel_noise_strength=rel_noise_strength,
        omega=1,
        rng=rng,
        weighted_noise=True
    )

    input_states = distributions_to_statevectors(F, take_sqrt=take_sqrt, normalize=normalize, symmetries=symmetries)
    target_states = distributions_to_statevectors(F_target, take_sqrt=take_sqrt, normalize=normalize, symmetries=symmetries)


    collision = LSQCollision(lattice=lattice)
    B_col, _ = collision.train_ls_collision(X=input_states, Y=target_states)
    metrics = collision.test_ls_collision(B_col, input_states, target_states)

    print(metrics)