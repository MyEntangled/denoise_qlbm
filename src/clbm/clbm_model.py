from src.qlbm.domain_settings import axial_flow
from src.qlbm.lbm_lattices import get_lattice
from src.qlbm.operations.streaming.streaming_by_assignment import streaming_periodic
from src.qlbm.operations.boundary_conditions.bounce_back import _opposites_from_c

import numpy as np
from matplotlib import pyplot as plt

class ClassicalLBMSimulator:
    def __init__(self, lattice: str, grid_size: tuple, omega: float):
        self.lattice = lattice
        self.c, self.w = get_lattice(lattice, True)
        self.Q, self.d = self.c.shape
        self.opposite_indices = _opposites_from_c(self.c)

        self.grid_size = grid_size
        self.omega = omega  # Relaxation rate = Delta_t / tau


    def stream(self, F: np.ndarray, applied_region: np.ndarray = None) -> np.ndarray:
        return streaming_periodic(F, lattice=self.lattice, dims=self.grid_size, applied_region=applied_region)

    def apply_boundary_conditions(self, F: np.ndarray, obstacles):
        #F_bndry = F[obstacles]
        #F_bndry = F_bndry[..., self.opposite_indices]  # Bounce-back mode change
        #F[obstacles] = F_bndry

        F_bndry = F[obstacles]
        F[obstacles] =  F_bndry[...,self.opposite_indices]  # Bounce-back mode change
        return F

    def get_macros(self, F: np.ndarray):
        rho = np.sum(F, axis=-1)
        u = np.einsum("...q,qd->...d", F, self.c) / (rho[..., np.newaxis] + 1e-12)
        return rho, u

    def collide(self, F: np.ndarray):
        # Compute macroscopic quantities
        rho, u = self.get_macros(F)

        # Compute equilibrium distribution
        u_sq = np.sum(u ** 2, axis=-1)
        F_eq = np.zeros_like(F)

        for q in range(self.Q):
            cu = np.einsum("...d,d->...", u, self.c[q])
            F_eq[..., q] = rho * self.w[q] * (
                1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * u_sq
            )

        # BGK collision step
        F = F - omega * (F - F_eq)

        return F

    def step(self, F: np.ndarray, obstacles: np.ndarray):
        # Streaming
        F = self.stream(F)

        # Boundary conditions: bounce-back
        F = self.apply_boundary_conditions(F, obstacles)
        #F = self.stream(F, applied_region=obstacles)   # some thing is off


        # Collision
        F = self.collide(F)
        return F


    def simulate(self, F_init: np.ndarray, obstacles: np.ndarray, num_steps: int, show_every: int = 20):
        F = F_init.copy()

        for step in range(num_steps):
            F = self.step(F, obstacles)

            if step % show_every == 0:
                print(f"Step: {step}")
                rho, u = self.get_macros(F)
                u_sq = np.sum(u ** 2, axis=-1)

                plt.imshow(np.sqrt(u_sq))
                plt.pause(.01)
                plt.cla()

        return F

if __name__ == "__main__":
    # Set up LBM lattice
    lattice = 'D2Q9'
    c, w = get_lattice(lattice, True)
    Q, d = c.shape
    omega = 1./0.53 # Relaxation rate, = 1/tau

    # 2D example

    grid_size = (400, 100)
    Nx, Ny = grid_size
    obstacles = [
        ('round', (Ny / 2, Nx / 4), 13),
        #('box', (Ny * 0.7, Nx * 0.6), (4, 6)),
    ]
    F, solid, _, _ = axial_flow.setup_domain((Ny, Nx), 'D2Q9', obstacles, flow_axis=0, flow_boost=2.3)

    F = np.ones((Ny, Nx, Q)) + 0.01 * np.random.randn(Ny, Nx, Q)
    ## The fluid is initially flowing to the right
    F[:, :, 1] = 2.3  ## Index 3 is of "to-the-right" velocity

    print("F shape:", F.shape)


    # Initialize simulator
    simulator = ClassicalLBMSimulator(lattice, grid_size, omega)

    # Run simulation
    num_steps = 10000
    F_final = simulator.simulate(F, solid, num_steps, show_every=20)

