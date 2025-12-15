from src.lattices.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np

# def _init_gaussian(grid_size, C0: float, sigma0: float, x0: np.ndarray):
#     """
#     Create a Gaussian hill:
#         C(x) = C0 * exp(-|x - x0|^2 / (2*sigma0^2))
#
#     Parameters
#     ----------
#     grid_size : tuple of int
#         (nx,), (nx, ny), or (nx, ny, nz)
#     C0 : float
#         Peak amplitude.
#     sigma0 : float
#         Standard deviation (lattice units).
#     x0 : None or array-like
#         Center point. Must have same dimension as grid_size.
#     Returns
#     -------
#     C : ndarray
#         Gaussian hill with shape = grid_size.
#     """
#     grid_size = tuple(grid_size)
#     dim = len(grid_size)
#
#     # Coordinates
#     coords = np.indices(grid_size)  # shape (dim, *grid_size)
#
#     x0 = np.array(x0, dtype=float)
#     if (np.any(x0 < 0) or np.any(x0 >= np.array(grid_size))):
#         raise ValueError("x0 must be within the grid.")
#
#     # Compute squared radial distance |x - x0|^2
#     r2 = np.zeros(grid_size, dtype=float)
#     for d in range(dim):
#         r2 += (coords[d] - x0[d])**2
#
#     # Gaussian amplitude
#     if sigma0 > 0:
#         C = C0 * np.exp(-r2 / (2 * sigma0**2))
#     elif sigma0 == 0:
#         C = np.zeros(grid_size, dtype=float)
#         if (x0 != np.floor(x0)).any():
#             raise ValueError("For sigma0=0, x0 must be at integer grid point.")
#         C[*np.astype(x0, int)] = C0
#     else:
#         raise ValueError("sigma0 must be non-negative.")
#
#     return C

def setup_testcase(C0: float = 1, sigma0: float = 50, x0: np.ndarray|None = None, u_adv: np.ndarray|None = np.array([.3, .2])/np.sqrt(3)):
    """
    LBM initializer for the advection-diffusion of a Gaussian hill in D2Q9, with periodic boundaries.
    :param C0: float
        Peak amplitude of the Gaussian.
    :param sigma0: float
        Standard deviation of the Gaussian (lattice units).
    :param x0: array-like, shape (2,)
        Center of the Gaussian hill. If None, defaults to center of the domain.
    :param u_adv: array-like, shape (2,)
        Constant advection velocity. If None, zero velocity is used.

    Returns
    -------
    F : np.ndarray      (*domain_dims, Q)
    solid : np.ndarray  (*domain_dims,), which is all False (hence periodic)
    u_solid = None
    """
    ## Fixed settings for this testcase
    domain_dims = (256, 256)
    grid_size = tuple(domain_dims[::-1])  # reverse for array indexing
    H,V = grid_size
    lattice = "D2Q9"

    ## Physical parameters
    omega = 1.0     # relaxation rate
    tau = 1.0   # relaxation time
    D = (tau - 0.5) / 3.0  # diffusion coefficient

    if u_adv is None:
        u_adv = np.zeros(2, dtype=float)

    if x0 is None:
        x0 = np.array([n//2 for n in grid_size], dtype=float)
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.shape != (2,):
            raise ValueError(f"x0 must have length {2}, got {x0_arr.shape}")
        x0 = x0_arr

    config = {
        "grid_size": grid_size,
        "lattice": lattice,
        "is_scalar_field": True,
        "C0": C0,
        "sigma0": sigma0,
        "D": D,
        "x0": x0,
        "u_adv": u_adv,
        "u0": u_adv,  # reference velocity for denoising collision
    }

    u_adv = np.asarray(u_adv, dtype=float)
    if u_adv.shape != (2,):
        raise ValueError(f"u_adv must have shape (2,), got {u_adv.shape}")

    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape

    # --- base density ---
    rho = exact_solution_periodic(C0=C0, sigma0=sigma0, x0=x0, u_adv=u_adv, domain_dims=domain_dims, tau=tau)(t=0)

    # --- Define boundary and boundary condition ---
    solid = np.zeros((*grid_size,), dtype=bool)
    u_solid = None  # no-slip

    # --- initial velocity field: set to u_adv ---
    u = u_adv[None, None, :].repeat(grid_size[0], axis=0).repeat(grid_size[1], axis=1)

    # Compute equilibrium
    F = get_equilibrium(rho, u, lattice)
    F[solid] = 0.

    assert (F >= 0).all(), "Negative values in initial distribution!"

    return config, F, solid, u_solid


def exact_solution_periodic(
        C0: float,
        sigma0: float,
        x0: np.ndarray,
        u_adv: np.ndarray,
        domain_dims=(256, 256),
        tau=1.0,
        n_images=4  # New parameter: 1 means sum 3x3 grid, 2 means 5x5, etc.
):
    """
    Computes the TRUE periodic solution using the Method of Images.
    Sums contributions from the main Gaussian and its periodic neighbors.
    """
    Nx, Ny = domain_dims

    # LBM Constants
    cs2 = 1.0 / 3.0
    D = cs2 * (tau - 0.5)

    # Grid generation (transposed to match matrix indexing row,col)
    # y_grid is shape (Ny, Nx), x_grid is shape (Ny, Nx)
    y_grid, x_grid = np.indices((Ny, Nx))

    def C(t: float) -> np.ndarray:
        # 1. Variance spread
        sig2 = sigma0 ** 2 + 2.0 * D * t

        # 2. Base center position (wrapped to 0..N range for consistency)
        cx_base = (x0[0] + u_adv[0] * t) % Nx
        cy_base = (x0[1] + u_adv[1] * t) % Ny

        # 3. Amplitude conservation
        # Note: We scale the individual Gaussians.
        # The sum will naturally handle the height increase due to overlap.
        pref = C0 * (sigma0 ** 2 / sig2)

        # Initialize accumulation field
        total_concentration = np.zeros((Ny, Nx), dtype=np.float64)

        # 4. Superposition Loop (Method of Images)
        # We iterate from -n_images to +n_images
        # e.g., if n_images=1, we sum offsets -1, 0, +1 (3x3 grid)
        for i in range(-n_images, n_images + 1):
            for j in range(-n_images, n_images + 1):
                # The center of the "image" Gaussian
                cx_image = cx_base + i * Nx
                cy_image = cy_base + j * Ny

                # Standard Euclidean distance from grid points to the Image Center
                # NO modulo arithmetic here! We want the real physical distance.
                dist_sq = (x_grid - cx_image) ** 2 + (y_grid - cy_image) ** 2

                # Add contribution
                total_concentration += np.exp(-dist_sq / (2.0 * sig2))

        return pref * total_concentration

    return C

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    config, F, solid, u_solid = setup_testcase(C0=1, sigma0=5, x0=np.array([100,100]), u_adv=np.array([0.1, 0.0]))
    # for row in solid:
    #     print([int(v) for v in row])

    print(F.shape)
    print(solid.shape)


    cs = 1.0 / np.sqrt(3)
    C_exact_func = exact_solution_periodic(C0=1, sigma0=20, x0=np.array([128,128]), u_adv=np.array([0.3, 0.2]))
    for t in range(1000):
        C_exact = C_exact_func(t)
        #print(f"t={t}, max C_exact={np.max(C_exact)}")

        if t % 20 == 0:
            plt.imshow(C_exact, origin='lower', vmin=-0.1, vmax=1.1, cmap='viridis')
            plt.title(f"Exact solution at t={t}")
            plt.pause(.01)
            plt.cla()