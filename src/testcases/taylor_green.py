from src.lattices.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np


def setup_testcase(u_max: float = 0.1/np.sqrt(3)):
    """
    LBM initializer for the 2D Taylor–Green vortex in D2Q9 with periodic boundaries.

    Parameters
    ----------
    u_max : float
        Velocity amplitude (in lattice units).

    Returns
    -------
    config : dict
        Configuration dictionary for the solver.
    F : np.ndarray
        Initial distribution function, shape = (*grid_size, Q).
    solid : np.ndarray
        Obstacle mask, all False → fully periodic.
    u_solid : None
        No moving/no-slip obstacles.
    """
    domain_dims = (256, 256)
    grid_size = tuple(domain_dims[::-1])  # reverse for array indexing
    H,V = grid_size
    lattice = "D2Q9"

    # LBM viscosity (cs^2 = 1/3)
    omega = 1.
    tau = 1.0 / omega
    cs2 = 1.0 / 3.0
    nu = cs2 * (tau - 0.5)

    config = {
        "grid_size": grid_size,
        "lattice": lattice,
        "is_scalar_field": False,   # now a full fluid, not ADE scalar
        "u_max": u_max,
        "u0": np.array([0.0, 0.0], dtype=float),   # reference velocity scale for denoising, etc.
    }

    # Lattice
    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape
    assert d == 2, "Taylor–Green setup is implemented for 2D only."

    # X in [0, Nx-1], Y in [0, Ny-1]
    Y, X = np.indices(grid_size)

    # Wavenumbers for the periodic domain
    kx = 2.0 * np.pi / V
    ky = 2.0 * np.pi / H

    # Decay time scale
    td = 1.0/(nu * (kx**2 + ky**2))

    # --- initial density field ---
    rho0 = 1.0
    rho = rho0 - 3 * (rho0 * u_max ** 2 / 4.0) * ((ky / kx) * np.cos(2. * kx * X) + (kx / ky) * np.cos(2. * ky * Y))

    # --- initial velocity field: Taylor–Green vortex at t=0 ---
    # At t=0, decay factor = 1
    #decay_u = np.exp(-t / td)
    ux = -u_max * np.sqrt(ky / kx) * np.cos(kx * X) * np.sin(ky * Y)
    uy = u_max * np.sqrt(kx / ky) * np.sin(kx * X) * np.cos(ky * Y)

    u = np.stack([ux, uy], axis=-1)  # shape (Ny, Nx, 2)

    # --- boundary: fully periodic, no solid nodes ---
    solid = np.zeros(grid_size, dtype=bool)
    u_solid = None

    # --- initial distributions from equilibrium ---
    F = get_equilibrium(rho, u, lattice)
    F[solid] = 0.0

    assert np.all(F >= 0), "Negative values in initial distribution!"

    return config, F, solid, u_solid


def exact_solution(
    u_max: float = 0.1 / np.sqrt(3),
    rho0: float = 1.0
):
    """Analytical velocity and pressure fields for the 2D Taylor–Green vortex.

    Domain (continuous):
        x, y \in [0, 2π] with periodic boundary conditions.

    Discrete mapping:
        x = 2π * i / Nx,  i = 0, ..., Nx-1
        y = 2π * j / Ny,  j = 0, ..., Ny-1

    Velocity field (decaying Taylor–Green vortex):
        u_x(x,y,t) =  U0 * sin(x) * cos(y) * exp(-2 * ν * t)
        u_y(x,y,t) = -U0 * cos(x) * sin(y) * exp(-2 * ν * t)

    Pressure field (up to an arbitrary constant p0):
        p(x,y,t) = p0 + (rho0 * U0**2 / 4) * (cos(2x) + cos(2y)) * exp(-4 * ν * t)

    where, in lattice units with Δx = Δt = 1 and cs^2 = 1/3,
        τ   = 1 / ω,  with ω = 1 (fixed in the quantum collision),
        ν   = cs^2 * (τ - 1/2) = (1/3) * (1 - 1/2) = 1/6.

    Parameters
    ----------
    u_max : float
        Velocity amplitude in lattice units.
    rho0 : float
        Reference density.
    p0 : float
        Reference pressure offset.

    Returns
    -------
    u_exact : callable
        u_exact(t) -> ndarray of shape (Ny, Nx, 2) giving velocity field at time t.
    rho_exact : callable
        rho_exact(t) -> ndarray of shape (Ny, Nx) giving density field at time t.
    """
    domain_dims = (256, 256)
    Nx, Ny = domain_dims
    grid_size = (Ny, Nx)

    # LBM viscosity (cs^2 = 1/3) with fixed ω = 1, τ = 1
    omega = 1.0
    tau = 1.0 / omega
    cs2 = 1.0 / 3.0
    nu = cs2 * (tau - 0.5)

    # Discrete coordinates mapped to [0, 2π]
    Y, X = np.indices(grid_size)

    kx = 2 * np.pi / Nx
    ky = 2 * np.pi / Ny

    td = 1.0/(nu * (kx*kx + ky*ky))

    def u_exact(t: float) -> np.ndarray:
        decay_u = np.exp(-t/td)
        ux = -u_max * np.sqrt(ky/kx) * np.cos(kx * X) * np.sin(ky * Y) * decay_u
        uy = u_max * np.sqrt(kx/ky) * np.sin(kx * X) * np.cos(ky * Y) * decay_u
        return np.stack([ux, uy], axis=-1)  # (Ny, Nx, 2)

    def rho_exact(t: float) -> np.ndarray:
        decay_p = np.exp(-2.*t/td)

        p = - (rho0 * u_max ** 2 / 4.0) * ((ky/kx) * np.cos(2.*kx * X) + (kx/ky) * np.cos(2.*ky * Y)) * decay_p

        return rho0 + 3*p

    return u_exact, rho_exact


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    config, F, solid, u_solid = setup_testcase(u_max=0.1 / np.sqrt(3),)

    print("F.shape:", F.shape)
    print("solid.shape:", solid.shape)

    u_exact_fn, p_exact_fn = exact_solution(
        u_max=config["u_max"],
    )

    Nx, Ny = (256, 256)
    kx = 2 * np.pi / Nx
    ky = 2 * np.pi / Ny
    nu = 1./6

    td = 1.0/(nu * (kx*kx + ky*ky))
    print("td =", td)
    u_max = config["u_max"]

    for t in range(0, 10001, 20):
        print("t =", t, t/td)
        u = u_exact_fn(t)
        #rho = 3. * p

        u_norm = np.linalg.norm(u, axis=-1)
        print(np.max(u_norm) / u_max * np.exp(t/td))

        plt.imshow(u_norm, origin="lower", vmin=-config["u_max"], vmax=config["u_max"])
        plt.title(f"Taylor–Green u(t={t})")
        plt.colorbar()

        plt.pause(0.01)
        plt.cla()