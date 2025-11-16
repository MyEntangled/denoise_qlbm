from src.qlbm.lbm_lattices import get_lattice
from src.qlbm.lbm_symmetries import get_symmetry
from src.qlbm.operations.boundary_conditions.bounce_back import bounce_back_obstacles

from src.qlbm.operations.streaming.streaming_by_assignment import streaming_periodic
from src.qlbm.operations.collision.denoiser import DenoisingCollision
from src.qlbm.operations.collision.ls_collision import LSCollision
from src.qlbm.operations.collision.learned_collision import LearnedCollision

from src.qlbm.data_generation.sample_distribution import sample_low_mach_data
from src.qlbm.data_generation.create_states import distributions_to_statevectors

from src.qlbm.gate_decomposition.block_encoding import schlimgen_block_encoding
from src.qlbm.gate_decomposition.cossin_decomposition import csd_equal_blocks

from src.qlbm.gate_decomposition.clements_decomposition import unitary_to_givens_ops
from src.qlbm.gate_decomposition.givens_rotation_constructor import get_givens_angle, givens_rot_to_mat

import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

### Simulator class for Quantum Lattice Boltzmann Method
### The computation takes place on W \otimes C^2

class QuantumLBMSimulator:
    def __init__(self,
                 lattice: str,
                 grid_size: tuple,
                 encoding_type: str,
                 collision_model_type: str,
                 eq_dist_degree: int,
                 apply_operators_as: str = 'full_unitary'):

        self.lattice = lattice
        self.c, self.w = get_lattice(lattice, True)
        self.Q, self.d = self.c.shape

        assert len(grid_size) == self.d, f"Grid size dimension {len(grid_size)} does not match lattice dimension {self.d}."
        self.grid_size = grid_size # (nZ, nY, nX) for 3D, (nY, nX) for 2D

        assert encoding_type in ['sqrt', 'full'], "encoding_type must be 'sqrt' or 'full'."
        self.encoding_type = encoding_type

        if collision_model_type not in ['denoising', 'least-square']:
            raise ValueError("collision_model_type must be 'denoising' or 'least-square'.")
        self.collision_model_type = collision_model_type

        assert eq_dist_degree in [1, 2], "eq_dist_degree must be 1 or 2."
        self.eq_dist_degree = eq_dist_degree

        if collision_model_type == 'denoising':
            self.collision_model = DenoisingCollision(lattice, eq_dist_degree)
        elif collision_model_type == 'least-square':
            self.collision_model = LSCollision(lattice)



        self.U_col = None  # full collision unitary
        self.U_col_op_seq = None # to-be-realized sequence of high-level operators on two registers, ancilla & system
        self.U_col_gate_seq = None # to-be-realized sequence of quantum gates

        if apply_operators_as not in ['full_unitary', 'subsystem_unitary', 'quantum_gates']:
            raise ValueError("apply_operators_as must be 'full_unitary', 'subsystem_unitary', or 'quantum_gates'.")
        self.apply_operators_as = apply_operators_as


    def _sample_distributions_(self, num_samples: int, omega: float, rng):
        # Sample distributions (F) and their post-collision targets (F_target)
        rho = 1.0  # base density
        mean_norm_u = 0.2 / np.sqrt(3)  # mean |u| (in lattice units, small => low Mach)
        std_norm_u = 0.2 / np.sqrt(3)  # std deviation of |u|
        rel_noise_strength = 0.05  # relative magnitude of thermal/noise fluctuations

        F, F_target = sample_low_mach_data(
            n_samples=num_samples,
            lattice=self.lattice,
            rho=rho,
            mean_norm_u=mean_norm_u,
            std_norm_u=std_norm_u,
            rel_noise_strength=rel_noise_strength,
            omega=omega,
            rng=rng,
            weighted_noise=True
        )

        return F, F_target

    def _prepare_statevectors_(self, F, F_target, include_symmetries: bool = True):
        if include_symmetries:
            symmetries = get_symmetry(self.lattice)  # permutation dict
        else:
            symmetries = None

        if self.encoding_type == 'sqrt':
            X = distributions_to_statevectors(F, take_sqrt=True, normalize=True, symmetries=symmetries)
            Y = distributions_to_statevectors(F_target, take_sqrt=True, normalize=True, symmetries=symmetries)
        elif self.encoding_type == 'full':
            X = distributions_to_statevectors(F, take_sqrt=True, normalize=True, symmetries=symmetries)
            Y = distributions_to_statevectors(F_target, take_sqrt=True, normalize=True, symmetries=symmetries)
        else:
            raise ValueError("'encoding_type' must be either 'sqrt', 'full'.")
        return X, Y

    def _convert_operator_to_gates_(self, U: np.ndarray):
        """Convert a unitary operator U into a sequence of quantum gates using Givens rotations"""

        # Givens operators have the form [[c, -s], [s,  c]] submatrix in the rows/cols (m,n) of a larger identity
        givens_ops, rel_phases = unitary_to_givens_ops(U)

        # list of (m,n,theta) for each Givens rotation
        givens_rots = [(m,n,get_givens_angle(op)) for m,n,op in givens_ops]

        # Reconstruct the matrix of the gate sequence for simulation.
        # For running quantum circuit, use actual givens rotations as gates.
        givens_gates = [givens_rot_to_mat(self.Q, m, n, theta) for (m, n, theta) in givens_rots]

        # Define gate sequence
        gate_seq = []
        gate_seq.extend([(np.eye(2), G) for G in givens_gates])
        gate_seq.append((np.eye(2), np.diag(rel_phases)))

        return gate_seq

    def _init_denoising_collision_(self, u0: np.ndarray):
        # For denoising collision, we directly build the operator
        assert u0.ndim == 1 and len(u0) == self.d

        manifold_aware = True

        D = self.collision_model.build_denoising_op(
            encoding_type=self.encoding_type,
            u0=u0,
            manifold_aware=manifold_aware
        )

        # Schlimgen block encoding of D
        # Full unitary: U_col = (H ⊗ I) (I ⊗ U) U_Σ (I ⊗ V†) (H ⊗ I), where U_Σ is diagonal
        U_col, alpha, U_svd, USigma, Vh_svd = schlimgen_block_encoding(D, rescale=True)
        self.U_col = U_col

        Ia, Id = np.eye(2), np.eye(self.Q)
        H = np.array([[1, 1], [1,-1]]) / np.sqrt(2)
        op_seq = [(H, Id), (Ia, Vh_svd), (USigma,), (Ia, U_svd), (H, Id)]
        self.U_col_op_seq = op_seq


        # Define gate sequence
        gate_seq = [(H, Id)]
        gate_seq.extend(self._convert_operator_to_gates_(Vh_svd))
        gate_seq.append((USigma,))
        gate_seq.extend(self._convert_operator_to_gates_(U_svd))
        gate_seq.append((H, Id))

        self.U_col_gate_seq = gate_seq

        return U_col, op_seq, gate_seq

    def _init_ls_collision_(self, omega, seed: int = None):
        rng = np.random.default_rng(seed)

        # Sample data & convert to statevectors
        F, F_target = self._sample_distributions_(num_samples=25000, omega=omega, rng=rng)
        X, Y = self._prepare_statevectors_(F, F_target, True)

        B, _ = self.collision_model.train_ls_collision(X, Y)

        # Schlimgen block encoding of B
        # Full unitary: U_col = (H ⊗ I) (I ⊗ U) U_Σ (I ⊗ V†) (H ⊗ I)
        U_col, alpha, U_svd, USigma, Vh_svd = schlimgen_block_encoding(B, rescale=True)
        self.U_col = U_col

        Ia, Id = np.eye(2), np.eye(self.Q)
        H = np.array([[1, 1], [1,-1]]) / np.sqrt(2)

        op_seq = [(H, Id), (Ia, Vh_svd), (USigma,), (Ia, U_svd), (H, Id)]
        self.U_col_op_seq = op_seq

        # Define gate sequence
        gate_seq = [(H, Id)]
        gate_seq.extend(self._convert_operator_to_gates_(Vh_svd))
        gate_seq.append((USigma,))
        gate_seq.extend(self._convert_operator_to_gates_(U_svd))
        gate_seq.append((H, Id))
        self.U_col_gate_seq = gate_seq

        return U_col, op_seq, gate_seq

    # def _init_learned_collision_(self, omega: float, seed: int = None):
    #     ancilla = 'zero'
    #     r = 1  # # of ancilla qubits
    #     rng = np.random.default_rng(seed)
    #
    #     # Sample data & convert to statevectors
    #     F, F_target = self._sample_distributions_(num_samples=25000, omega=omega, rng=rng)
    #     X, Y = self._prepare_statevectors_(F, F_target, True)
    #
    #     # Train collision operator (don't need to test here, if want to test use learned_collision.py)
    #     U_col, _ = self.collision_model.train_collision_unitary(
    #         X, Y, r=r,
    #         ancilla=ancilla,
    #         eta=0.5, mu=1.0, eps=0.5,
    #         max_steps=5000, tol=1e-9,
    #         rng=rng, verbose=True
    #     )
    #
    #     # Cosine-Sine decomposition.
    #     (U1, U2), theta, (V1H, V2H) = csd_equal_blocks(U_col)
    #     C = np.diag(np.cos(theta))
    #     S = np.diag(np.sin(theta))
    #     middle = np.block([[C, -S], [S, C]])
    #     seq = [(V1H, V2H), middle, (U1, U2)]
    #     return U_col, seq

    def init_collision_operator(self, u0 = None, seed: int = None):

        ## Train or build collision operator (many parameters are specified instead of passed for simplicity)
        if self.collision_model_type == 'denoising':
            if u0 is None:
                raise ValueError("'u0' must be provided for denoising collision.")
            U_col, op_seq, gate_seq = self._init_denoising_collision_(np.array(u0))

        elif self.collision_model_type == 'least-square':
            U_col, op_seq, gate_seq = self._init_ls_collision_(omega=1, seed=seed)

        else:
            raise ValueError("Unknown collision model type.")

        self.U_col, self.U_col_op_seq, self.U_col_gate_seq = U_col, op_seq, gate_seq
        return


    def apply_system_unitary(self, U, system_states: np.ndarray):
        if self.apply_operators_as == 'full_unitary':
            # Apply a full unitary U to the states
            return np.einsum('ij,...j->...i', U, system_states)

        elif self.apply_operators_as == 'subsystem_unitary':
            # Apply unitaries in the ancilla and the data subsystems as provided by U
            if len(U) == 1:  # one big unitary for whole system (ancilla + data)
                U_sys = U[0]
            elif len(U) == 2:  # two unitaries for ancilla and data subsystems
                U_anc, U_vel = U
                U_sys = np.kron(U_anc, U_vel)
            else:
                raise ValueError("There are only 2 subsystems for ancilla and velocities.")
            return np.einsum('ij,...j->...i', U_sys, system_states)

        elif self.apply_operators_as == 'quantum_gates':
            # Apply quantum gates in the ancilla and the data subsystems as provided by U
            if len(U) == 1:  # one big unitary for whole system (ancilla + data)
                U_sys = U[0]
                #print(U_sys.shape)
            elif len(U) == 2:  # two unitaries for ancilla and data subsystems
                U_anc, U_vel = U
                U_sys = np.kron(U_anc, U_vel)
                #print(U_anc.shape, U_vel.shape, U_sys.shape)
            else:
                raise ValueError("There are only 2 subsystems for ancilla and velocities.")
            return np.einsum('ij,...j->...i', U_sys, system_states)

        else:
            NotImplementedError("Only 'full_unitary', 'subsystem_unitary', and 'quantum_gates' operator types are implemented.")


    def collide(self, states, embed_fn: Callable):
        # Implement collision step

        # Embed states into larger space
        system_states = embed_fn(states)

        if self.apply_operators_as == 'full_unitary':
            outputs = self.apply_system_unitary(self.U_col, system_states)
        elif self.apply_operators_as == 'subsystem_unitary':
            for U in self.U_col_op_seq:
                system_states = self.apply_system_unitary(U, system_states)
            outputs = system_states
        elif self.apply_operators_as == 'quantum_gates':
            for U in self.U_col_gate_seq:
                system_states = self.apply_system_unitary(U, system_states)
            outputs = system_states
        else:
            NotImplementedError("Only 'full_unitary', 'subsystem_unitary', and 'quantum_gates' operator types are implemented.")

        return outputs


    def stream(self, states, applied_region = None):
        # Implement streaming step
        return streaming_periodic(states, lattice=self.lattice, dims=self.grid_size, applied_region=applied_region)

    def apply_boundary_conditions(self, states, obstacles, u_obstacles):
        # Implement boundary conditions
        return bounce_back_obstacles(states, obstacles, u_obstacles, lattice=self.lattice, dims=self.grid_size)


    def step(self, states: np.ndarray, obstacles: np.ndarray, u_obstacles: np.ndarray|None, embed_fn: Callable):
        # Streaming
        states = self.stream(states)
        #print("After streaming:", np.linalg.norm(states[obstacles]))

        # Boundary conditions: bounce-back
        states = self.apply_boundary_conditions(states, obstacles, u_obstacles)
        states = self.stream(states, applied_region=obstacles)

        # Collision
        norm = np.linalg.norm(states)
        system_states = self.collide(states, embed_fn)  # Embed into larger space then collide

        # post-selection |psi> from the system state |0>|psi> + |1>|junk>, then renormalize
        postselects = system_states[..., :self.Q]  # take the system part
        postselect_norm = np.linalg.norm(postselects)
        states = postselects * (norm / postselect_norm)

        return states.real

    def simulate(self,
                 F_init: np.ndarray,
                 obstacles: np.ndarray,
                 u_obstacles: np.ndarray | None,
                 num_steps: int,
                 show_every: int = 20):

        multiply_ket0 = lambda psi: np.concatenate([psi, np.zeros_like(psi)], axis=-1)  # function |psi> --> |0>|psi>

        if self.encoding_type == 'sqrt':
            init_states = F_init ** 0.5
        else: # self.encoding_type == 'full':
            init_states = F_init.copy()


        states = init_states.copy()
        init_norm = np.linalg.norm(states)
        states = states / init_norm
        print("Initial norm:", init_norm)

        for i in range(num_steps):
            states = self.step(states, obstacles=obstacles, u_obstacles=u_obstacles, embed_fn=multiply_ket0)
            if i % show_every == 0:
                print("Iteration", i)
                self.plot_field(init_norm * states, obstacles)

        plt.clf()
        plt.cla()
        plt.close('all')
        return states #* norm


    def plot_field(self, states, obstacles):
        selected_sign  = np.sign(np.sum(states, axis=-1, keepdims=True))
        states = states * selected_sign


        F = np.clip(states, a_min=0, a_max=None)**2
        F[obstacles] = 0

        rho = np.sum(F, axis=-1, keepdims=True)
        u = np.matmul(F, self.c) / (rho + 1e-12)

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
    from src.qlbm.domain_settings import axial_flow, gaussian_hill, couette_flow

    # Example usage of QLBMSimulator
    lattice = "D2Q9"
    grid_size = (401, 101)  # 2D grid
    encoding_type = 'sqrt'
    collision_model_type = 'denoising'
    eq_dist_deg = 2
    apply_operators_as = 'full_unitary'  # 'full_unitary' or 'subsystem_unitary' or 'quantum_gates'


    denoise_qlbm = QuantumLBMSimulator(lattice, grid_size, encoding_type, collision_model_type, eq_dist_deg, apply_operators_as)
    denoise_qlbm.init_collision_operator(u0=[0.15,0.], seed=0)

    # A_denoise = denoise_qlbm.U_col[:denoise_qlbm.Q, :denoise_qlbm.Q].real
    # s_denoise = np.linalg.svd(A_denoise, compute_uv=False)
    # print("Denoising collision unitary:")
    # print(A_denoise.round(4))
    # print("Singular values:", s_denoise.round(4))
    # print()

    Nx, Ny = grid_size
    obstacles = [
        ('round', (Ny // 2, Nx / 4), 13),
        #('box', (Ny * 0.7, Nx * 0.6), (14, 16)),
    ]

    #F, solid = axial_flow.setup_domain((Ny, Nx), 'D2Q9', obstacles, flow_axis=0, flow_boost=2.3)
    #F, solid = gaussian_hill.setup_domain((Ny, Nx), 'D2Q9')
    F, solid, u_solid = couette_flow.setup_domain((Ny, Nx), 'D2Q9')


    denoise_output_states = denoise_qlbm.simulate(F, obstacles=solid, u_obstacles=None, num_steps=10000)
    denoise_output_states = denoise_output_states.reshape(*grid_size, denoise_qlbm.Q)




