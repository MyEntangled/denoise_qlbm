from src.qlbm.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np


def _init_gaussian(domain_dims, C0=1.0, sigma0=2.0, x0=None):
    """
    Create a Gaussian hill:
        C(x) = C0 * exp(-|x - x0|^2 / (2*sigma0^2))

    Parameters
    ----------
    domain_dims : tuple of int
        (nx,), (nx, ny), or (nx, ny, nz)
    C0 : float
        Peak amplitude.
    sigma0 : float
        Standard deviation (lattice units).
    x0 : "center" or array-like
        Center point. Must have same dimension as grid_size.
        If "center", defaults to geometric center of the grid.

    Returns
    -------
    C : ndarray
        Gaussian hill with shape = grid_size.
    """
    domain_dims = tuple(domain_dims)
    dim = len(domain_dims)

    # Coordinates
    coords = np.indices(domain_dims)  # shape (dim, *grid_size)

    # Determine center x0
    if x0 is None:
        center = np.array([(n - 1) / 2 for n in domain_dims], dtype=float)
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.shape != (dim,):
            raise ValueError(f"x0 must have length {dim}, got {x0_arr.shape}")
        center = x0_arr

    center = np.array(center, dtype=float)
    if (np.any(center < 0) or np.any(center >= np.array(domain_dims))):
        raise ValueError("x0 must be within the grid.")

    # Compute squared radial distance |x - x0|^2
    r2 = np.zeros(domain_dims, dtype=float)
    for d in range(dim):
        r2 += (coords[d] - center[d])**2

    # Gaussian amplitude
    if sigma0 > 0:
        C = C0 * np.exp(-r2 / (2 * sigma0**2))
    elif sigma0 == 0:
        C = np.zeros(domain_dims, dtype=float)
        if (center != np.floor(center)).any():
            raise ValueError("For sigma0=0, x0 must be at integer grid point.")
        C[*np.astype(center, int)] = C0
    else:
        raise ValueError("sigma0 must be non-negative.")

    return C

def setup_domain(domain_dims, lattice, obstacles=()):
    rho = _init_gaussian(domain_dims, C0=0.2, sigma0=3., x0=None)
    rho += 0.1

    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape
    if len(domain_dims) != d:
        raise ValueError(f"Expected {d}-D dims for lattice {lattice}, got {len(domain_dims)}D.")

    u = np.zeros((*domain_dims, d))
    u[..., 0] = 0.2  # initial flow in x-direction
    F = get_equilibrium(rho, u, lattice, eq_dist_deg=2)

    # --- build solid mask ---
    solid = np.zeros(domain_dims, dtype=bool)
    coords = np.ogrid[tuple(slice(0, n) for n in domain_dims)]

    for obs in obstacles:
        kind = obs[0].lower()
        center = np.array(obs[1], dtype=float)
        if kind == 'round':
            radius = float(obs[2])
            dist2 = sum((coords[ax] - center[ax]) ** 2 for ax in range(d))
            solid |= dist2 <= radius ** 2
        elif kind == 'box':
            size = np.array(obs[2], dtype=float)
            half = size / 2
            mask = np.ones(domain_dims, dtype=bool)
            for ax in range(d):
                mask &= (coords[ax] >= center[ax] - half[ax]) & (coords[ax] <= center[ax] + half[ax])
            solid |= mask
        else:
            raise ValueError(f"Unknown obstacle type: {kind}")

    F[solid] = 0.0  # zero out distributions inside solid obstacles

    return F, solid


if __name__ == "__main__":
    grid_size = (21, 11)
    domain_dims = grid_size[::-1]
    C = _init_gaussian(grid_size, sigma0=0., x0=None)
    print(C.shape)
    print(C.round(3))