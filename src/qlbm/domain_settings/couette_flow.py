from src.qlbm.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np

def setup_domain(domain_dims, lattice, u_top=0.1, base_density=1.0, noise_level=0.0):
    """
    Initialize 2D Couette flow.
    Bottom wall is stationary.
    Top wall moves with velocity (u_top, 0).
    """

    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape
    if d != 2:
        raise ValueError("Couette flow only supports 2D lattices")
    if len(domain_dims) != 2:
        raise ValueError(f"Expected 2D dims for lattice {lattice}, got {len(domain_dims)}D.")

    rho = base_density * np.ones((*domain_dims,))
    if noise_level:
        rho += noise_level * np.random.randn(*domain_dims, Q)

    # --- Define obstacles: top and bottom planes ---
    solid = np.zeros((*domain_dims,), dtype=bool)
    solid[0, :] = True     # bottom wall
    solid[-1, :] = True     # top wall

    # --- Moving top wall boundary condition ---
    u_solid = np.zeros((*domain_dims, d))
    u_solid[-1, :, 0] = u_top  # top wall moves with velocity (u_top, 0)


    # # Boundary velocity for top moving wall
    # u[-1, :, 0] = u_top
    # u[-1, :, 1] = 0.0

    u = np.zeros((*domain_dims, d))
    u[..., 0] = 0.2  # initial flow in x-direction
    #u[..., 1] = 0.1  # small initial flow in y-direction

    # Compute equilibrium
    F = get_equilibrium(rho, u, lattice, eq_dist_deg=2)

    F[solid] = 0.

    return F, solid, u_solid

if __name__ == "__main__":
    grid_size = (5, 3)
    F, solid, u_solid = setup_domain(grid_size, 'D2Q9', u_top=0.1)
    for row in solid:
        print([int(v) for v in row])

