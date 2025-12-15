from typing import Callable, Sequence

from src.lattices.lbm_lattices import get_lattice
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

        assert len(grid_size) == self.d, f"Grid size dimension {len(grid_size)} does not match lattice dimension {self.d}."
        self.grid_size = grid_size
        self.omega = omega  # Relaxation rate = Delta_t / tau


    def stream(self, F: np.ndarray, applied_region: np.ndarray = None) -> np.ndarray:
        return streaming_periodic(F, lattice=self.lattice, dims=self.grid_size, applied_region=applied_region)

    def apply_boundary_conditions(self, F: np.ndarray, obstacles, u_obstacles):
        #F_bndry = F[obstacles]
        #F_bndry = F_bndry[..., self.opposite_indices]  # Bounce-back mode change
        #F[obstacles] = F_bndry

        F_bndry = F[obstacles]


        if u_obstacles is not None:
            # u_obs: (*grid_size, d), take only obstacle nodes -> (Nobs, d)
            u_wall = u_obstacles[obstacles, :]  # velocities of the solid nodes


            # cu: (Nobs, Q) = u_wall · c_i
            # c: (Q, d), so take transpose for matmul
            cu = u_wall @ self.c.T

            # Add moving-wall correction: f_i += 2 * w_i * rho0 * (c_i · u_wall) / cs^2
            rho0 = 1.
            cs2_inv = 3.
            F_bndry -= (2.0 * rho0 * cs2_inv * cu * self.w[np.newaxis,:])  # w broadcasts to (Nobs, Q)

        F[obstacles] = F_bndry[..., self.opposite_indices]  # Bounce-back mode change
        return F

    def get_macros(self, F: np.ndarray, obstacles: np.ndarray):
        rho = np.sum(F, axis=-1)
        rho = np.clip(rho, 1e-12, None)

        u = np.einsum("...q,qd->...d", F, self.c) / (rho[..., np.newaxis])

        rho[obstacles] = 0.
        u[obstacles] = 0.
        return rho, u

    def collide(self, F: np.ndarray, obstacles: np.ndarray, u_prescribed: np.ndarray | None = None):
        # Compute macroscopic quantities
        rho, u = self.get_macros(F, obstacles)

        # Compute equilibrium distribution
        cs2_inv = 3.0
        cs4_inv = 9.0

        # Broadcast weights over batch axes: shape (1,1,...,1,Q)
        w_shape = (1,)*rho.ndim  + (self.Q,)
        w_b = self.w.reshape(w_shape)


        if u_prescribed is None:
            # cu = u · c_i  -> shape (..., Q)
            cu = np.einsum('...d,qd->...q', u, self.c)

            # uu = |u|^2 -> shape (..., 1)
            uu = np.einsum('...d,...d->...', u, u)[..., None]

            # rho[..., None] has shape (..., 1), broadcast with w_b (..., Q) ⇒ (..., Q)
            F_eq = w_b * rho[..., None] * (
                    1.0 + cu * cs2_inv + 0.5 * (cu ** 2) * cs4_inv - 0.5 * uu * cs2_inv
            )

        else:   # use prescribed velocity, advection case
            assert len(u_prescribed) == self.d, f"u_prescribed must have length equal to dimension d = {self.d}."
            # Use prescribed velocity for equilibrium

            cu = np.einsum('...d,qd->...q', u_prescribed, self.c)
            uu = np.einsum('...d,...d->...', u_prescribed, u_prescribed)[..., None]
            F_eq = w_b * rho[..., None] * (1.0 + cu * cs2_inv + 0.5 * (cu **2) * cs4_inv - 0.5 * uu * cs2_inv)


            #cu = np.einsum('qd,d->q', self.c, u_prescribed)  # shape (Q,)
            #F_eq = w_b * rho[..., None] * (1.0 + cu * cs2_inv)

        # BGK collision step
        F = F - self.omega * (F - F_eq)

        return F

    def step(self, F: np.ndarray, obstacles: np.ndarray, u_obstacles: np.ndarray, u_adv: np.ndarray | None = None):
        # Streaming
        F = self.stream(F)

        # Boundary conditions: bounce-back
        F = self.apply_boundary_conditions(F, obstacles, u_obstacles)
        F = self.stream(F, applied_region=obstacles)

        # Collision
        F = self.collide(F, obstacles, u_adv)
        return F


    def simulate(self, F_init: np.ndarray,
                 obstacles: np.ndarray,
                 u_obstacles: np.ndarray | Callable | None,
                 u_prescribed: np.ndarray | Callable | None,
                 num_steps: int,
                 show_every: int = 20):
        F = F_init.copy()

        obstacles_t = obstacles if isinstance(obstacles, Callable) else None
        u_obstacles_t = u_obstacles if isinstance(u_obstacles, Callable) else None

        if isinstance(u_prescribed, Callable):
            print("Reference velocity is time-dependent.")
            u_prescribed_t = u_prescribed
        elif isinstance(u_prescribed, (Sequence, np.ndarray)):
            print("Reference velocity is time-independent (fixed u0).")
            u_prescribed_t = None
        else:
            assert u_prescribed is None, "u0 must be either None, an array-like, or a callable."
            u_prescribed_t = None


        for i in range(num_steps):

            obstacles_i = obstacles_t(i) if obstacles_t is not None else obstacles
            u_obstacles_i = u_obstacles_t(i) if u_obstacles_t is not None else u_obstacles

            if u_prescribed_t is not None:
                u_prescribed_i = np.array(u_prescribed_t(i))
            else:
                u_prescribed_i = u_prescribed

            F = self.step(F, obstacles_i, u_obstacles_i, u_prescribed_i)

            if i % show_every == 0:
                print(f"Step: {i}")
                rho, u = self.get_macros(F, obstacles_i)
                print(np.mean(u, axis=(0,1)))

                u_norm = np.linalg.norm(u, axis=-1)
                #plt.imshow(u_norm, origin="lower", vmin=-0.1/np.sqrt(3), vmax= 0.1/np.sqrt(3))

                # u_max = 0.1 / np.sqrt(3)
                # Nx, Ny = self.grid_size
                # kx = 2 * np.pi / Nx
                # ky = 2 * np.pi / Ny
                # nu = 1./6
                # td = 1.0/(nu * (kx*kx + ky*ky))
                # print(np.max(u_norm) / u_max * np.exp(i / td))

                plt.imshow(rho, origin="lower")
                #plt.plot(range(len(rho)), rho)


                # ux, uy = u[..., 0], u[..., 1]
                # dfydx = ux[2:, 1:-1] - ux[:-2, 1:-1]
                # dfxdy = uy[1:-1, 2:] - uy[1:-1, :-2]
                # curl = dfxdy - dfydx

                # ux = u[..., 0]
                # uy = u[..., 1]
                #
                # # central differences with periodic BC
                # dvy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0)
                # dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0)
                #
                # curl = dvy_dx - dux_dy
                # plt.imshow(curl, cmap='bwr', origin='lower')

                plt.pause(.01)
                plt.cla()

        return F

if __name__ == "__main__":
    from src.testcases import taylor_green, fourier, gaussian

    # Set up LBM lattice
    lattice = 'D2Q9'
    vels, _ = get_lattice(lattice, True)
    Q, d = vels.shape

    #omega = 1./1 # Relaxation rate, = 1/tau

    # 2D example

    grid_size = (400, 100)
    Nx, Ny = grid_size
    obstacles = [
        ('round', (Ny / 2, Nx / 4), 13),
        #('box', (Ny * 0.7, Nx * 0.6), (4, 6)),
    ]

    #F, solid = axial_flow.setup_domain((Ny, Nx), 'D2Q9', obstacles, flow_axis=0, flow_boost=2.3)
    #F, solid = ade_gaussian_hill.setup_domain((Ny, Nx), 'D2Q9')
    #u_solid = None
    #F, solid, u_solid = couette_flow.setup_domain((Ny, Nx), 'D2Q9')
    #F, solid, u_solid = cavity_flow.setup_domain((Ny, Nx), 'D2Q9')

    #_, F, solid, u_solid = cylinder.setup_testcase(u_max=0.1)    # solid is time-dependent
    config, F, solid, u_solid = gaussian.setup_testcase()
    #_, F, solid, u_solid = cavity.setup_testcase(u_top=0.1)
    #config, F, solid, u_solid = couette.setup_testcase(u_top=0.1)
    #config, F, solid, u_solid = moving_cylinder.setup_testcase(u_max=0.5)    # solid is time-dependent
    #config, F, solid, u_solid = shear.setup_testcase(1.)    # solid is time-dependent
    #config, F, solid, u_solid = taylor_green.setup_testcase()
    #config, F, solid, u_solid = fourier.setup_testcase()
    grid_size = (256,256)

    print("F shape:", F.shape)

    # Initialize simulator

    simulator = ClassicalLBMSimulator(lattice, grid_size, omega=1.)

    # Run simulation
    num_steps = 10000

    #u_adv = np.array([.0, 0.])  / np.sqrt(3)
    u_adv = config["u_adv"] if "u_adv" in config else None
    F_final = simulator.simulate(F, solid, u_solid, u_adv, num_steps, show_every=20)

