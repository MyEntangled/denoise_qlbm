from src.qlbm.lbm_lattices import get_lattice
from src.qlbm.operations.streaming.streaming_by_assignment import streaming_periodic
from src.qlbm.operations.boundary_conditions.bounce_back import _opposites_from_c

import numpy as np
from matplotlib import pyplot as plt

class ClassicalLBMSimulator:
    def __init__(self, lattice: str, grid_size: tuple, eq_dist_degree: int, omega: float):
        self.lattice = lattice
        self.c, self.w = get_lattice(lattice, True)
        self.Q, self.d = self.c.shape
        self.opposite_indices = _opposites_from_c(self.c)

        assert len(grid_size) == self.d, f"Grid size dimension {len(grid_size)} does not match lattice dimension {self.d}."
        self.grid_size = grid_size
        assert eq_dist_degree in [1,2], "Only eq_dist_degree 1 or 2 supported."
        self.eq_dist_degree = eq_dist_degree
        self.omega = omega  # Relaxation rate = Delta_t / tau


    def stream(self, F: np.ndarray, applied_region: np.ndarray = None) -> np.ndarray:
        return streaming_periodic(F, lattice=self.lattice, dims=self.grid_size, applied_region=applied_region)

    def apply_boundary_conditions(self, F: np.ndarray, obstacles, u_obstacles):
        #F_bndry = F[obstacles]
        #F_bndry = F_bndry[..., self.opposite_indices]  # Bounce-back mode change
        #F[obstacles] = F_bndry

        F_bndry = F[obstacles]
        F[obstacles] =  F_bndry[...,self.opposite_indices]  # Bounce-back mode change

        if u_obstacles is not None:
            # u_obs: (*grid_size, d), take only obstacle nodes -> (Nobs, d)
            u_wall = u_obstacles[obstacles, :]  # velocities of the solid nodes

            # cu: (Nobs, Q) = u_wall · c_i
            # c: (Q, d), so take transpose for matmul
            cu = u_wall @ c.T

            # Add moving-wall correction: f_i += 2 * w_i * rho0 * (c_i · u_wall) / cs^2
            rho0 = 1.
            cs2_inv = 3.
            F[obstacles, :] += (2.0 * rho0 * cs2_inv * cu * w[np.newaxis,:])  # w broadcasts to (Nobs, Q)

        return F

    def get_macros(self, F: np.ndarray, obstacles: np.ndarray):
        rho = np.sum(F, axis=-1)
        rho = np.clip(rho, 1e-12, None)

        u = np.einsum("...q,qd->...d", F, self.c) / (rho[..., np.newaxis])

        rho[obstacles] = 0.
        u[obstacles] = 0.
        return rho, u

    def collide(self, F: np.ndarray, obstacles: np.ndarray):
        # Compute macroscopic quantities
        rho, u = self.get_macros(F, obstacles)

        # Compute equilibrium distribution
        cs2_inv = 3.0
        cs4_inv = 9.0

        # Broadcast weights over batch axes: shape (1,1,...,1,Q)
        w_shape = (1,) * rho.ndim + (w.shape[0],)
        w_b = w.reshape(w_shape)

        # cu = u · c_i  -> shape (..., Q)
        cu = np.einsum('...d,qd->...q', u, c)

        if self.eq_dist_degree == 1:
            F_eq = w_b * rho[..., None] * (1.0 + cu * cs2_inv)
        else:   # self.eq_dist_degree == 2
            # uu = |u|^2 -> shape (..., 1)
            uu = np.einsum('...d,...d->...', u, u)[..., None]


            # rho[..., None] has shape (..., 1), broadcast with w_b (..., Q) ⇒ (..., Q)
            F_eq = w_b * rho[..., None] * (
                    1.0 + cu * cs2_inv + 0.5 * (cu ** 2) * cs4_inv - 0.5 * uu * cs2_inv
            )

        # BGK collision step
        F = F - omega * (F - F_eq)

        return F

    def step(self, F: np.ndarray, obstacles: np.ndarray, u_obstacles: np.ndarray):
        # Streaming
        F = self.stream(F)

        # Boundary conditions: bounce-back
        F = self.apply_boundary_conditions(F, obstacles, u_obstacles)
        F = self.stream(F, applied_region=obstacles)

        # Collision
        F = self.collide(F, obstacles)
        return F


    def simulate(self, F_init: np.ndarray, obstacles: np.ndarray, u_obstacles: np.ndarray|None, num_steps: int, show_every: int = 20):
        F = F_init.copy()

        for step in range(num_steps):
            F = self.step(F, obstacles, u_obstacles)

            if step % show_every == 0:
                print(f"Step: {step}")
                rho, u = self.get_macros(F, obstacles)
                u_sq = np.sum(u ** 2, axis=-1)

                plt.imshow(np.sqrt(u_sq))
                plt.pause(.01)
                plt.cla()

        return F

if __name__ == "__main__":
    from src.qlbm.domain_settings import axial_flow, gaussian_hill, couette_flow


    # Set up LBM lattice
    lattice = 'D2Q9'
    c, w = get_lattice(lattice, True)
    Q, d = c.shape

    eq_dist_deg = 2
    omega = 1./1 # Relaxation rate, = 1/tau

    # 2D example

    grid_size = (401, 101)
    Nx, Ny = grid_size
    obstacles = [
        ('round', (Ny / 2, Nx / 4), 13),
        #('box', (Ny * 0.7, Nx * 0.6), (4, 6)),
    ]

    #F, solid = axial_flow.setup_domain((Ny, Nx), 'D2Q9', obstacles, flow_axis=0, flow_boost=2.3)
    #F, solid = gaussian_hill.setup_domain((Ny, Nx), 'D2Q9')
    F, solid, u_solid = couette_flow.setup_domain((Ny, Nx), 'D2Q9')

    print("F shape:", F.shape)

    # Initialize simulator
    simulator = ClassicalLBMSimulator(lattice, grid_size, eq_dist_deg, omega)

    # Run simulation
    num_steps = 10000
    F_final = simulator.simulate(F, solid, u_solid, num_steps, show_every=20)

