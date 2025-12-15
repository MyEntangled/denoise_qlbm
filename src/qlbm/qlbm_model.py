from src.lattices.lbm_lattices import get_lattice
from src.lattices.lbm_symmetries import get_symmetry
from src.qlbm.operations.boundary_conditions.bounce_back import bounce_back_obstacles

from src.qlbm.operations.streaming.streaming_by_assignment import streaming_periodic
from src.qlbm.operations.collision.denoiser import DenoisingCollision
from src.qlbm.operations.collision.ls_collision import LSCollision

from src.qlbm.data_generation.sample_distribution import sample_low_mach_data
from src.qlbm.data_generation.create_states import distributions_to_statevectors

from src.qlbm.gate_decomposition.block_encoding import schlimgen_block_encoding

from src.qlbm.gate_decomposition.clements_decomposition import unitary_to_givens_ops
from src.qlbm.gate_decomposition.givens_rotation_constructor import get_givens_angle, givens_rot_to_mat

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union, Sequence
import matplotlib.pyplot as plt

### Simulator class for Quantum Lattice Boltzmann Method
### The computation takes place on W \otimes C^2

class QuantumLBMSimulator:
    def __init__(self,
                 lattice: str,
                 grid_size: tuple,
                 encoding_type: str,
                 collision_model_type: str,
                 is_scalar_field: bool,
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

        self.is_scalar_field = is_scalar_field

        if collision_model_type == 'denoising':
            self.collision_model = DenoisingCollision(lattice, is_scalar_field)
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

    # def _init_ls_collision_(self, omega, seed: int = None):
    #     rng = np.random.default_rng(seed)
    #
    #     # Sample data & convert to statevectors
    #     F, F_target = self._sample_distributions_(num_samples=25000, omega=omega, rng=rng)
    #     X, Y = self._prepare_statevectors_(F, F_target, True)
    #
    #     B, _ = self.collision_model.train_ls_collision(X, Y)
    #
    #     # Schlimgen block encoding of B
    #     # Full unitary: U_col = (H ⊗ I) (I ⊗ U) U_Σ (I ⊗ V†) (H ⊗ I)
    #     U_col, alpha, U_svd, USigma, Vh_svd = schlimgen_block_encoding(B, rescale=True)
    #     self.U_col = U_col
    #
    #     Ia, Id = np.eye(2), np.eye(self.Q)
    #     H = np.array([[1, 1], [1,-1]]) / np.sqrt(2)
    #
    #     op_seq = [(H, Id), (Ia, Vh_svd), (USigma,), (Ia, U_svd), (H, Id)]
    #     self.U_col_op_seq = op_seq
    #
    #     # Define gate sequence
    #     gate_seq = [(H, Id)]
    #     gate_seq.extend(self._convert_operator_to_gates_(Vh_svd))
    #     gate_seq.append((USigma,))
    #     gate_seq.extend(self._convert_operator_to_gates_(U_svd))
    #     gate_seq.append((H, Id))
    #     self.U_col_gate_seq = gate_seq
    #
    #     return U_col, op_seq, gate_seq

    def init_collision_operator(self, u0 = None, seed: int = None):

        ## Train or build collision operator (many parameters are specified instead of passed for simplicity)
        if self.collision_model_type == 'denoising':
            if u0 is None:
                raise ValueError("'u0' must be provided for denoising collision.")
            U_col, op_seq, gate_seq = self._init_denoising_collision_(np.array(u0))

        # elif self.collision_model_type == 'least-square':
        #     U_col, op_seq, gate_seq = self._init_ls_collision_(omega=1, seed=seed)

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
            elif len(U) == 2:  # two unitaries for ancilla and data subsystems
                U_anc, U_vel = U
                U_sys = np.kron(U_anc, U_vel)
            else:
                raise ValueError("There are only 2 subsystems for ancilla and velocities.")
            return np.einsum('ij,...j->...i', U_sys, system_states)

        else:
            NotImplementedError("Only 'full_unitary', 'subsystem_unitary', and 'quantum_gates' operator types are implemented.")


    def collide(self, states, embed_fn: Callable, collide_operation: dict):
        # Implement collision step

        # Embed states into larger space
        system_states = embed_fn(states)

        if self.apply_operators_as == 'full_unitary':
            outputs = self.apply_system_unitary(collide_operation["full_unitary"], system_states)

        elif self.apply_operators_as == 'subsystem_unitary':
            for U in collide_operation["subsystem_unitary"]:
                system_states = self.apply_system_unitary(U, system_states)
            outputs = system_states

        elif self.apply_operators_as == 'quantum_gates':
            for U in collide_operation["quantum_gates"]:
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
        return bounce_back_obstacles(states, obstacles, u_obstacles, lattice=self.lattice, encoding_type=self.encoding_type, dims=self.grid_size)


    def step(self, states: np.ndarray, obstacles: np.ndarray, u_obstacles: np.ndarray|None, embed_fn: Callable, collide_operation: dict):
        # Streaming
        states = self.stream(states)

        # Boundary conditions: bounce-back
        states = self.apply_boundary_conditions(states, obstacles, u_obstacles)
        states = self.stream(states, applied_region=obstacles)

        # Collision
        norm = np.linalg.norm(states)
        system_states = self.collide(states, embed_fn, collide_operation)  # Embed into larger space then collide

        # post-selection |psi> from the system state |0>|psi> + |1>|junk>, then renormalize
        postselects = system_states[..., :self.Q]  # take the system part
        postselect_norm = np.linalg.norm(postselects)
        states = postselects * (norm / postselect_norm)

        return states.real

    def simulate(self,
                 F_init: np.ndarray,
                 obstacles: np.ndarray | Callable,
                 u_obstacles: np.ndarray | Callable | None,
                 u0: Union[Sequence[float], NDArray[float]] | Callable | None,
                 num_steps: int,
                 show_every: int = 20):

        if u0 is not None:
            assert self.collision_model_type == 'denoising', "Reference velocity is only supported for denoising collision model."

        multiply_ket0 = lambda psi: np.concatenate([psi, np.zeros_like(psi)], axis=-1)  # function |psi> --> |0>|psi>

        if self.encoding_type == 'sqrt':
            init_states = F_init ** 0.5
        else: # self.encoding_type == 'full':
            init_states = F_init.copy()

        obstacles_t = obstacles if isinstance(obstacles, Callable) else None
        u_obstacles_t = u_obstacles if isinstance(u_obstacles, Callable) else None

        if isinstance(u0, Callable):
            print("Reference velocity is time-dependent.")
            u0_t = u0
        elif isinstance(u0, (Sequence, np.ndarray)):
            print("Reference velocity is time-independent (fixed u0).")
            # This matches the old behavior where we called init_collision_operator(u0=...)
            self.init_collision_operator(u0=np.array(u0))
            u0_t = None
        else:
            assert u0 is None, "u0 must be either None, an array-like, or a callable."
            assert self.U_col is not None, "Collision operator must be initialized before simulation if u0 is None."
            u0_t = None

        states = init_states.copy()
        init_norm = np.linalg.norm(states)
        states = states / init_norm
        print("Initial norm:", init_norm)

        for i in range(num_steps):
            obstacles_i = obstacles_t(i) if obstacles_t is not None else obstacles
            u_obstacles_i = u_obstacles_t(i) if u_obstacles_t is not None else u_obstacles

            if u0_t is not None:
                u0_i = np.array(u0_t(i))
                U_col, op_seq, gate_seq = self._init_denoising_collision_(u0_i)
            else:
                ## Using fixed collision operator
                U_col, op_seq, gate_seq = self.U_col, self.U_col_op_seq, self.U_col_gate_seq

            if obstacles_t is not None:
                assert obstacles_i.shape == self.grid_size, "Obstacles shape mismatch."
            if u_obstacles_t is not None:
                assert u_obstacles_i.shape == self.grid_size + (self.d,), "u_obstacles shape mismatch."
            if u0_t is not None:
                assert u0_i.shape == (self.d,), "u0 shape mismatch."

            col_op = {"full_unitary": U_col, "subsystem_unitary": op_seq, "quantum_gates": gate_seq}

            states = self.step(states,
                               obstacles=obstacles_i,
                               u_obstacles=u_obstacles_i,
                               embed_fn=multiply_ket0,
                               collide_operation=col_op)

            if i % show_every == 0:
                print("Iteration", i)
                self.plot_field(init_norm * states, obstacles_i, i)

        plt.clf()
        plt.cla()
        plt.close('all')
        return init_norm * states


    def plot_field(self, states, obstacles, i: int = None):
        selected_sign  = np.sign(np.sum(states, axis=-1, keepdims=True))
        states = states * selected_sign

        if self.encoding_type == 'sqrt':
            F = np.clip(states, a_min=0, a_max=None)**2
        else:
            F = np.clip(states, a_min=0, a_max=None)

        F[obstacles] = 0

        rho = np.sum(F, axis=-1, keepdims=True)
        u = np.matmul(F, self.c) / (rho + 1e-12)
        u_norm = np.linalg.norm(u, axis=-1)
        #print("u_avg", np.mean(u, axis=(0,1)))


        plt.imshow(rho, origin='lower')
        #plt.imshow(u_norm, origin='lower', vmax=0.1/np.sqrt(3), vmin=-0.1/np.sqrt(3))
        #plt.plot(range(len(rho)), rho)

        plt.pause(.01)
        plt.cla()


if __name__ == "__main__":
    from src.testcases import taylor_green, fourier, cylinder, gaussian

    cs = 1.0 / np.sqrt(3)
    #config, F, solid, u_solid = cylinder.setup_testcase()
    config, F, solid, u_solid = gaussian.setup_testcase()
    #config, F, solid, u_solid = taylor_green.setup_testcase()
    #config, F, solid, u_solid = fourier.setup_testcase()

    # Example usage
    lattice = config["lattice"]
    grid_size = config["grid_size"]
    encoding_type = 'sqrt'  # 'sqrt' or 'full'
    collision_model_type = 'denoising'
    is_scalar_field = config["is_scalar_field"]
    apply_operators_as = 'full_unitary'  # 'full_unitary' or 'subsystem_unitary' or 'quantum_gates'

    denoise_qlbm = QuantumLBMSimulator(lattice, grid_size, encoding_type, collision_model_type, is_scalar_field, apply_operators_as)

    denoise_output_states = denoise_qlbm.simulate(F, obstacles=solid, u_obstacles=u_solid, u0=config["u0"], num_steps=10000, show_every=20)
    denoise_output_states = denoise_output_states.reshape(*grid_size, denoise_qlbm.Q)




