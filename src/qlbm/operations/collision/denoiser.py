from src.qlbm.lbm_lattices import get_lattice

import numpy as np
from itertools import combinations


class DenoisingCollision:
    def __init__(self, lattice: str):
        c, self.weights = get_lattice(lattice, as_array=True)
        self.vels = c.T
        self.Q, self.D = c.shape

        self.cs = 1./np.sqrt(3)  # speed of sound

    def build_discrete_hermite_basis(self):
        """
        Construct the discrete Hermite basis
        :return: V = [v_0, v_alpha \forall \alpha, v_{\alpha\beta} \forall commutative pairs (\alpha,\beta)]
        """
        sqrt_w = np.sqrt(self.weights)
        cs2 = self.cs * self.cs

        V = sqrt_w.reshape(-1,1)

        for i in range(self.D):
            V = np.concatenate([V, (self.vels[i] * sqrt_w / self.cs).reshape(-1,1)], axis=1)

        for i in range(self.D):
            V = np.concatenate([V, ((self.vels[i]*self.vels[i] - cs2)*sqrt_w / (cs2 * np.sqrt(2))).reshape(-1,1)], axis=1)

        for vel_a, vel_b in combinations(self.vels, 2):
            V = np.concatenate([V, ((vel_a * vel_b)*sqrt_w / cs2).reshape(-1,1)], axis=1)

        return V


    def build_local_chart(self, u0, encoding_type):
        """
        Construct the matrix B whose image is the tangent plane of equilibrium distributions at u0.

        :param u0: Reference point
        :param encoding_type: 'full' or 'sqrt'
        :return: B = [b(u0), b_\alpha (u0) \forall \alpha], shape (Q, D+1)
        """
        assert encoding_type in ['full', 'sqrt'], "encoding_type must be full (f) or sqrt (f^{0.5})."
        assert len(u0) == self.D

        cs2 = self.cs * self.cs
        components = []

        if encoding_type == 'full':
            ## Position b(u0)
            components.append([
                1,
                *[x / self.cs for x in u0],
                *[x*x / (cs2 * np.sqrt(2)) for x in u0],
                *[x*y / cs2 for x,y in combinations(u0,2)]
            ])

            ## Derivative b_i(u0) for all directions i
            for i in range(self.D):
                components.append([
                    0,
                    *[1/self.cs if p == i else 0 for p in range(self.D)],
                    *[2*u0[p] / (cs2 * np.sqrt(2)) if p == i else 0 for p in range(self.D)],
                    *[(u0[q]/cs2 if p == i else u0[p]/cs2 if q == i else 0) for p, q in combinations(range(len(u0)), 2)]
                ])

        else:
            ## Position b(u0)
            components.append([
                1 - np.sum(u0 * u0) / (8*cs2),
                *[x / (2*self.cs) for x in u0],
                *[x*x / (4*cs2 * np.sqrt(2)) for x in u0],
                *[x*y / (4*cs2) for x,y in combinations(u0,2)]
            ])

            ## Derivative b_i(u0) for all directions i
            for i in range(self.D):
                components.append([
                    -u0[i] / (4*cs2),
                    *[1/(2*self.cs) if p == i else 0 for p in range(self.D)],
                    *[u0[p] / (2*cs2 * np.sqrt(2)) if p == i else 0 for p in range(self.D)],
                    *[(u0[q]/(4*cs2) if p == i else u0[p]/(4*cs2) if q == i else 0) for p, q in combinations(range(len(u0)), 2)]
                ])

        B = np.array(components).T
        return B


    @staticmethod
    def build_manifold_retraction(local_basis):
        """
        Construct the retraction on the equilibrium manifold,
        which is the orthogonal projector onto the tangent plane at reference point.

        :param local_basis: output from build_local_chart
        :return: retraction operator, shape (Q,Q)
        """

        G = np.linalg.inv(local_basis.T @ local_basis) @ local_basis.T  ## pseudo-inv of B = local_basis
        L = local_basis @ G
        return L


    def build_denoising_op(self, encoding_type, u0, manifold_aware):
        assert encoding_type in ['full', 'sqrt'], "encoding_type must be full (f) or sqrt (f^{0.5})."
        assert len(u0) == self.D

        sqrt_w = np.sqrt(self.weights)
        W_sqrt = np.diag(sqrt_w)
        W_sqrt_inv = np.diag(1/sqrt_w)

        V = self.build_discrete_hermite_basis()
        B = self.build_local_chart(u0, encoding_type)
        L = self.build_manifold_retraction(B)

        if encoding_type == 'full':
            if manifold_aware:
                D = W_sqrt @ V @ L @ V.T @ W_sqrt_inv
            else:
                D = W_sqrt @ V @ V.T @ W_sqrt_inv
        else:
            if manifold_aware:
                D = V @ L @ V.T
            else:
                D = V @ V.T

        # spec = np.linalg.eigvals(D.T @ D)
        # print("Spectrum", np.min(spec), np.max(spec))
        return D


    def apply_denoiser(self, input_states, encoding_type, u0, manifold_aware):
        D = self.build_denoising_op(encoding_type, u0, manifold_aware)
        output_states = D @ input_states.T  # shape (Q, n_samples)
        return output_states.T


    def test_denoiser(self, input_states, target_states, encoding_type, u0, manifold_aware):
        output_states = self.apply_denoiser(input_states, encoding_type, u0, manifold_aware)

        output_norms = np.linalg.norm(output_states, axis=1, keepdims=True)
        output_states /= output_norms

        test_overlap = float(np.mean(np.abs(np.sum(output_states * target_states, axis=1))))
        test_loss = 1 - test_overlap

        print("--- Denoiser Test Results ---")
        print(f"Test overlap: {test_overlap:.6f}  |  Test loss: {test_loss:.6e}")
        return {"loss": test_loss, "overlap": test_overlap}


if __name__ == '__main__':
    from src.qlbm.data_generation.sample_distribution import sample_low_mach_data
    from src.qlbm.data_generation.create_states import distributions_to_statevectors
    from src.qlbm.lbm_symmetries import get_symmetry

    rng = np.random.default_rng(123)

    # Parameters
    n_samples = 50000
    lattice = "D2Q9"

    rho = 1.0
    cs = 1/np.sqrt(3)
    mean_norm_u = 0.1 * cs
    std_norm_u = 0.2 * cs
    rel_noise_strength = 0.1  # relative magnitude of thermal/noise fluctuations

    take_sqrt = False
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
        rng=rng,
        weighted_noise=True
    )

    input_states = distributions_to_statevectors(F, take_sqrt=take_sqrt, normalize=normalize, symmetries=symmetries)
    target_states = distributions_to_statevectors(F_target, take_sqrt=take_sqrt, normalize=normalize, symmetries=symmetries)


    denoiser = DenoisingCollision(lattice=lattice)
    encoding_type = 'sqrt' if take_sqrt else 'full'
    manifold_aware = True


    metrics = denoiser.test_denoiser(input_states, target_states, encoding_type, np.zeros(2), manifold_aware)
    print(metrics)