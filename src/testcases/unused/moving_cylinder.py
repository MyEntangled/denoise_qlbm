from src.lattices.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np

def setup_testcase(u_max: float = 0.1):
    """
    LBM initializer for the flow around a cylinder moving in time in D2Q9. Opposite bounaries are periodic.
    There is a parabolic inflow profile in x-direction at t=0.

    :param u_max: maximum inflow velocity at y=H/2, as multiple of the speed of sound cs=1/sqrt(3).


    Returns
    -------
    F : np.ndarray      (*domain_dims, Q)
    solid : np.ndarray  (*domain_dims,)
    u_solid : np.ndarray = None (no-slip)
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
        "u0": 2./3 * np.array([u_max, 0.0]),  # reference velocity for denoising collision
    }

    u_max = u_max * (1.0 / np.sqrt(3))  # convert to lattice units

    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape


    # --- base density ---
    fluid_density = 1.0
    rho = fluid_density * np.ones((*grid_size,))

    # --- Define cylinder: B_r(0.2, 0.2) with r = 0.05 ---

    center_t = lambda t: np.array([H//2 + (H//4) * np.sin(2*np.pi*t/1000), H//2], dtype=float)
    radius = H // 8

    coords = np.ogrid[tuple(slice(0, n) for n in grid_size)]

    dist2 = lambda center: sum((coords[ax] - center[ax]) ** 2 for ax in range(d))

    solid_t = lambda t: dist2(center = center_t(t)) <= radius ** 2

    # --- boundary condition ---
    u_solid = None
    #u_solid = np.zeros((*grid_size, d))

    #y_coords = np.arange(grid_size[0])
    #u_solid[:, :, 0] = 4 * u_max * y_coords[:, None] * (grid_size[0] - y_coords[:, None]) / (grid_size[0] ** 2)

    # --- initial velocity field: inflow profile u(0,y) = (4*u_max * y*(41-y) / 41^2, 0) ---
    u = np.zeros((*grid_size, d))
    y_coords = np.arange(grid_size[0])
    u[:, :, 0] = 4 * u_max * y_coords[:, None] * (grid_size[0] - y_coords[:, None]) / (grid_size[0] ** 2)


    print(4 * u_max * y_coords * (grid_size[0] - y_coords) / (grid_size[0] ** 2))
    # Compute equilibrium
    F = get_equilibrium(rho, u, lattice)
    F[solid_t(0)] = 0.

    assert (F >= 0).all(), "Negative values in initial distribution!"

    return config, F, solid_t, u_solid

if __name__ == "__main__":
    F, solid_t, u_solid = setup_testcase()
    # for row in solid:
    #     print([int(v) for v in row])

    print(F.shape)

    for t in range(1000):
        print(np.sum(solid_t(t)))