from src.qlbm.lbm_lattices import get_lattice
from src.qlbm.lbm_symmetries import get_symmetry
from src.qlbm.operations.boundary_conditions.bounce_back import bounce_back_obstacles

from src.qlbm.operations.streaming.streaming_by_assignment import streaming_periodic
from src.qlbm.operations.collision.denoiser import DenoisingCollision
from src.qlbm.operations.collision.learned_collision import LearnedCollision

from src.qlbm.data_generation.sample_distribution import sample_low_mach_data
from src.qlbm.data_generation.create_states import distributions_to_statevectors

from src.qlbm.gate_decomposition.block_encoding import schlimgen_block_encoding
from src.qlbm.gate_decomposition.cossin_decomposition import csd_equal_blocks


import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

### Simulator class for Quantum Lattice Boltzmann Method
### The computation takes place on W \otimes C^2

class QLBMSimulator:
    def __init__(self, lattice: str, grid_size: tuple, collision_model_type: str):
        self.lattice = lattice
        self.c, self.w = get_lattice(lattice, True)
        self.Q, self.d = self.c.shape

        self.grid_size = grid_size # (nZ, nY, nX) for 3D, (nY, nX) for 2D

        assert collision_model_type in ['denoising', 'learned']
        self.collision_model_type = collision_model_type
        if collision_model_type == 'denoising':
            self.collision_model = DenoisingCollision(lattice)
        elif collision_model_type == 'learned':
            self.collision_model = LearnedCollision(lattice)

        self.U_col = None  # full collision unitary
        self.U_col_seq = None # to-be-realized sequence of high-level operators on two registers, ancilla & system


    def _init_learned_collision_(self, encoding_type: str, omega: float, seed: int = 0):
        ancilla = 'zero'
        r = 1  # # of ancilla qubits
        n_samples = 25000

        rng = np.random.default_rng(seed)

        # Sample distributions (F) and their post-collision targets (F_target)
        rho = 1.0  # base density
        mean_norm_u = 0.2 / np.sqrt(3)  # mean |u| (in lattice units, small => low Mach)
        std_norm_u = 0.3 / np.sqrt(3)  # std deviation of |u|
        rel_noise_strength = 0.01  # relative magnitude of thermal/noise fluctuations

        F, F_target = sample_low_mach_data(
            n_samples=n_samples,
            lattice=self.lattice,
            rho=rho,
            mean_norm_u=mean_norm_u,
            std_norm_u=std_norm_u,
            rel_noise_strength=rel_noise_strength,
            omega=omega,
            rng=rng,
            weighted_noise=True
        )

        # Convert distributions → state vectors (optionally sqrt + norm) and augment by symmetries
        sym = get_symmetry(self.lattice)  # permutation dict
        if encoding_type == 'sqrt':
            X = distributions_to_statevectors(F, take_sqrt=True, normalize=True, symmetries=sym)
            Y = distributions_to_statevectors(F_target, take_sqrt=True, normalize=True, symmetries=sym)
        elif encoding_type == 'full':
            X = distributions_to_statevectors(F, take_sqrt=True, normalize=True, symmetries=sym)
            Y = distributions_to_statevectors(F_target, take_sqrt=True, normalize=True, symmetries=sym)

        # Train collision operator (don't need to test here, if want to test use learned_collision.py)
        # U_col, _ = self.collision_model.train_collision_op(
        #     X, Y, r=r,
        #     comp_weights=self.w,
        #     ancilla=ancilla,
        #     eta=0.5, mu=1.0, eps=0.5,
        #     max_steps=5000, tol=1e-9,
        #     rng=rng, verbose=True
        # )

        U_col, loss = self.collision_model.get_ls_collision_op(X, Y)

        # Cosine-Sine decomposition.
        (U1, U2), theta, (V1H, V2H) = csd_equal_blocks(U_col)
        C = np.diag(np.cos(theta))
        S = np.diag(np.sin(theta))
        middle = np.block([[C, -S], [S, C]])
        seq = [(V1H, V2H), middle, (U1, U2)]

        return U_col, seq



    def _init_denoising_collision_(self, encoding_type: str):
        # For denoising collision, we directly build the operator
        #u0 = np.zeros(self.d, dtype=float)  # reference point
        u0 = np.array([0.1,0])
        manifold_aware = True

        D = self.collision_model.build_denoising_op(
            encoding_type=encoding_type,
            u0=u0,
            manifold_aware=manifold_aware
        )

        # Schlimgen block encoding of D
        # Full unitary: U_col = (H ⊗ I) (I ⊗ U) U_Σ (I ⊗ V†) (H ⊗ I)
        U_col, alpha, U_svd, USigma, Vh_svd = schlimgen_block_encoding(D, rescale=True)

        I_a = np.eye(2)
        H = np.array([[1, 1], [1,-1]]) / np.sqrt(2)
        Id = np.eye(self.d)

        seq = [(H, Id), (I_a, Vh_svd), (USigma,), (I_a, U_svd), (H, Id)]

        return U_col, seq



    def init_collision_operator(self, encoding_type: str, seed: int = 0):
        if encoding_type not in ['sqrt', 'full']:
            raise ValueError("encoding_type must be 'sqrt' or 'full'.")

        ## Train or build collision operator (many parameters are specified instead of passed for simplicity)
        if self.collision_model_type == 'learned':
            U_col, seq = self._init_learned_collision_(encoding_type, seed)

        elif self.collision_model_type == 'denoising':
            U_col, seq = self._init_denoising_collision_(encoding_type)
        else:
            raise ValueError("Unknown collision model type.")

        self.U_col, self.U_col_seq = U_col, seq

        return


    def collide(self, states, apply_collision_as: str, embed_fn: Callable):
        # Implement collision step
        norm = np.linalg.norm(states) # compute norm before embedding

        # Embed states into larger space
        system_states = embed_fn(states)

        if apply_collision_as == 'full_unitary':
            # Apply self.U_col
            outputs = system_states @ self.U_col
            postselects = outputs[..., :self.Q]  # take the system part
            postselect_norm = np.linalg.norm(postselects)  # single scalar, norm of superpositioned state
            return postselects * norm / (postselect_norm + 1e-12)
        else:
            NotImplementedError("Only 'full_unitary' collision application is implemented.")


    def stream(self, states):
        # Implement streaming step
        return streaming_periodic(states, lattice=self.lattice, dims=self.grid_size)

    def apply_boundary_conditions(self, states, obstacles):
        # Implement boundary conditions
        return bounce_back_obstacles(states, obstacles, lattice=self.lattice, dims=self.grid_size)


    def step(self, states: np.ndarray, obstacles: np.ndarray, embed_fn: Callable) -> np.ndarray:
        # Streaming
        states = self.stream(states)
        #print("After streaming:", np.linalg.norm(states[obstacles]))

        # Boundary conditions: bounce-back
        states = self.apply_boundary_conditions(states, obstacles)
        #print("After bouncing back:", np.linalg.norm(states[obstacles]))

        # Embed into larger space then collide
        states = self.collide(states, 'full_unitary', embed_fn)
        #print("After collision:",np.linalg.norm(states[obstacles]))

        return states.real

    def simulate(self, init_states: np.ndarray, obstacles: np.ndarray, num_steps: int, show_every: int = 10):
        multiply_ket0 = lambda psi: np.concatenate([psi, np.zeros_like(psi)], axis=-1)  # function |psi> --> |0>|psi>

        states = init_states.copy()
        #norm = np.linalg.norm(states)
        #states = states / norm
        #print("Initial norm:", norm)

        for i in range(num_steps):
            states = self.step(states, obstacles=obstacles, embed_fn=multiply_ket0)
            print("Iteration", i)
            print(states[10,3,:].round(4))
            if i % show_every == 0:
                self.plot_field(states.real, obstacles)
        return states #* norm


    def plot_field(self, states, obstacles):

        selected_sign  = np.sign(np.sum(states, axis=-1, keepdims=True))
        states = states * selected_sign


        F = np.clip(states, a_min=0, a_max=None)**2
        F[obstacles] = 0

        rho = np.sum(F, axis=-1, keepdims=True)
        u = np.matmul(F, self.c) / (rho + 1e-12)
        print(u[10, 3, :])

        norm_u_sq = np.linalg.norm(u, axis=-1)
        plt.imshow(norm_u_sq)

        # ux, uy = u[..., 0], u[...,1]
        # dfydx = ux[2:, 1:-1] - ux[:-2, 1:-1]
        # dfxdy = uy[1:-1, 2:] - uy[1:-1, :-2]
        # curl = dfxdy - dfydx
        # plt.imshow(curl, cmap='bwr')

        plt.pause(.01)
        plt.cla()


if __name__ == "__main__":
    from src.qlbm.domain_settings import axial_flow

    # Example usage of QLBMSimulator
    lattice = "D2Q9"
    grid_size = (400, 100)  # 2D grid

    denoise_qlbm = QLBMSimulator(lattice, grid_size, collision_model_type ='denoising')
    denoise_qlbm.init_collision_operator(encoding_type='sqrt', seed=0)
    A_denoise = denoise_qlbm.U_col[:denoise_qlbm.Q, :denoise_qlbm.Q].real
    s_denoise = np.linalg.svd(A_denoise, compute_uv=False)
    print("Denoising collision unitary:")
    print(A_denoise.round(4))
    print("Singular values:", s_denoise.round(4))
    print()

    # learned_qlbm = QLBMSimulator(lattice, grid_size, collision_model_type ='learned')
    # learned_qlbm.init_collision_operator(encoding_type='sqrt', seed=None)
    # A_learned = learned_qlbm.U_col[:learned_qlbm.Q, :learned_qlbm.Q].real
    # s_learned = np.linalg.svd(A_learned, compute_uv=False)
    # print("Learned collision unitary:")
    # print(A_learned.round(4))
    # print("Singular values:", s_learned)


    # Initialize states
    #input_states = np.ones((grid_size[1], grid_size[0], learned_qlbm.Q)) / np.sqrt(learned_qlbm.Q)

    Nx, Ny = grid_size
    obstacles = [
        ('round', (Ny // 2, Nx / 4), 13),
        #('box', (Ny * 0.7, Nx * 0.6), (14, 16)),
    ]


    F, solid, c, w = axial_flow.setup_domain((Ny, Nx), 'D2Q9', obstacles, flow_axis=0, flow_boost=2.3)
    input_states = np.sqrt(F)
    #input_states = F

    print("Input states:")
    print(input_states[10,3,:].round(4))



    # denoise_output_states = denoise_qlbm.simulate(input_states, obstacles=solid, num_steps=1000)
    # denoise_output_states = denoise_output_states.reshape(*grid_size, denoise_qlbm.Q)


    # learned_output_states = learned_qlbm.simulate(input_states, obstacles=solid, num_steps=1000)
    # learned_output_states = learned_output_states.reshape(*grid_size, learned_qlbm.Q)






