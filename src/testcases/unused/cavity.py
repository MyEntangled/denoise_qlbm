from src.lattices.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np

def setup_testcase(u_top=0.1):
    """
    LBM initializer for lid-driven cavity flow in D2Q9, with no-slip boundaries on all sides and moving top lid.
    :param u_top: Velocity of the top lid in x-direction, as multiple of the speed of sound cs=1/sqrt(3).

    Returns
    -------
    F : np.ndarray      (*grid_size, Q)
    solid : np.ndarray  (*grid_size,)
    u_solid : np.ndarray (*grid_size, d) - Dirichlet BC
    """
    ## Fixed settings for this testcase
    domain_dims = (128, 129)
    grid_size = tuple(domain_dims[::-1])  # reverse for array indexing
    H,V = grid_size
    lattice = "D2Q9"


    ## Physical parameters
    omega = 1.0     # relaxation rate
    tau = 1.0   # relaxation time
    nu = (1.0 / omega - 0.5) / 3.0  # kinematic viscosity

    config = {
        "grid_size": grid_size,
        "lattice": lattice,
        "is_scalar_field": False,
        "u_top": u_top,
        "nu": nu,
        "u0": np.array([0., 0.]),  # reference velocity for denoising collision
    }

    print(f"Viscosity nu = {nu}")

    u_top = u_top * (1.0 / np.sqrt(3))  # convert to lattice units

    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape

    fluid_density = 1.
    rho = fluid_density * np.ones((*grid_size,))

    # --- Define obstacles: top and bottom planes ---
    solid = np.zeros((*grid_size,), dtype=bool)
    solid[0, :] = True     # bottom wall
    solid[-1, :] = True     # top wall
    solid[:, 0] = True     # left wall
    solid[:, -1] = True     # right wall

    # --- BCs: top wall moving with u_top and no-slip for other three walls ---
    u_solid = np.zeros((*grid_size, d))
    u_solid[-1, :, 0] = u_top  # top wall moves with velocity (u_top, 0)

    # --- initial velocity field ---
    u = np.zeros((*grid_size, d))

    # Compute equilibrium
    F = get_equilibrium(rho, u, lattice)
    F[solid] = 0.

    assert (F >= 0).all(), "Negative values in initial distribution!"

    return config, F, solid, u_solid