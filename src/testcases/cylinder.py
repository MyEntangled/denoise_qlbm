from src.lattices.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np

def setup_testcase(u_max: float = 0.1/np.sqrt(3), u_ref_factor: float = 1./3):
    """
    LBM initializer for the flow around a cylinder.

    The testcase simulates a 2D flow in a rectangular grid. It defines a solid circular
    obstacle (cylinder) and initializes the velocity field with a parabolic inflow
    profile across the y-dimension.


    Parameters
    ----------
    u_max : float, optional
        The maximum velocity of the parabolic inflow profile, by default 0.1 / sqrt(3).

    Returns
    -------
    config : dict
        Dictionary containing simulation parameters (grid size, lattice, diffusion, etc.).
    F : np.ndarray
        Initial distribution function array of shape (*domain_dims, Q).
    solid : np.ndarray
        Boolean mask for solid nodes
    u_solid : None
        Velocity boundary condition values at solid nodes.
    """
    ## Fixed settings for this testcase
    #domain_dims = (440, 82)   # (nx, ny)
    domain_dims = (512, 128)
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
        "u_max": u_max,
        "nu": nu,
        "u0": u_ref_factor * np.array([u_max, 0.0]),  # reference velocity for denoising collision
    }

    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape


    # --- base density ---
    fluid_density = 1.0
    rho = fluid_density * np.ones((*grid_size,))

    # --- Define cylinder: B_r(0.2, 0.2) with r = 0.05 ---

    center = np.array((H//2, H//2), dtype=float)
    radius = H // 8

    solid = np.zeros((*grid_size,), dtype=bool)
    coords = np.ogrid[tuple(slice(0, n) for n in grid_size)]

    dist2 = sum((coords[ax] - center[ax]) ** 2 for ax in range(d))
    solid |= dist2 <= radius ** 2

    # --- no-slip bounce-back ---
    u_solid = None

    # --- initial velocity field: inflow profile u(0,y) = (4*u_max * y*(41-y) / 41^2, 0) ---
    u = np.zeros((*grid_size, d))
    y_coords = np.arange(grid_size[0])
    u[:, :, 0] = 4 * u_max * y_coords[:, None] * (grid_size[0] - y_coords[:, None]) / (grid_size[0] ** 2)

    print(4 * u_max * y_coords * (grid_size[0] - y_coords) / (grid_size[0] ** 2))
    # Compute equilibrium
    F = get_equilibrium(rho, u, lattice)
    F[solid] = 0.

    assert (F >= 0).all(), "Negative values in initial distribution!"

    return config, F, solid, u_solid

if __name__ == "__main__":
    F, solid, u_solid = setup_testcase()
    # for row in solid:
    #     print([int(v) for v in row])

    print(F.shape)
    print(solid.shape)