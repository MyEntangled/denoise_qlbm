import numpy as np
from src.lattices.lbm_lattices import get_lattice

def setup_domain(domain_dims, lattice, obstacles=(), *,
                 base_density=1.0,
                 noise_level=0.01,
                 flow_axis=0,
                 flow_boost=2.3):
    """
    Generic LBM initializer for 1D, 2D, or 3D lattices.

    Parameters
    ----------
    domain_dims : tuple[int, ...]
        Grid shape (X1, X2, ..., Xd). Output shape = (*dims, Q)
    lattice : str
        Name for get_lattice(), e.g. 'D2Q9', 'D3Q19', ...
    obstacles : list[tuple]
        Each obstacle is either:
            ('round', center, radius)
            ('box',   center, size)
        where center, size are tuples of length d
    base_density : float
        Mean initialization value.
    noise_level : float
        Standard deviation of added Gaussian noise.
    flow_axis : int
        Axis of initial flow (+ direction).
    flow_boost : float
        Value for the velocity pointing downstream.

    Returns
    -------
    F : np.ndarray      (*domain_dims, Q)
    solid : np.ndarray  (*domain_dims,)
    c : np.ndarray      (Q, d)
    w : np.ndarray      (Q,)
    """
    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape
    if len(domain_dims) != d:
        raise ValueError(f"Expected {d}-D dims for lattice {lattice}, got {len(domain_dims)}D.")

    # --- base distribution ---
    F = base_density * np.ones((*domain_dims, Q))
    if noise_level:
        F += noise_level * np.random.randn(*domain_dims, Q)

    # --- select velocity pointing along +flow_axis (+x/+y/+z) ---
    flow_dir = np.argmax(c[:, flow_axis])
    F[..., flow_dir] = flow_boost

    # --- build solid mask ---
    solid = np.zeros(domain_dims, dtype=bool)
    coords = np.ogrid[tuple(slice(0, n) for n in domain_dims)]

    for obs in obstacles:
        kind = obs[0].lower()
        center = np.array(obs[1], dtype=float)
        if kind == 'round':
            radius = float(obs[2])
            dist2 = sum((coords[ax] - center[ax])**2 for ax in range(d))
            solid |= dist2 <= radius**2
        elif kind == 'box':
            size = np.array(obs[2], dtype=float)
            half = size / 2
            mask = np.ones(domain_dims, dtype=bool)
            for ax in range(d):
                mask &= (coords[ax] >= center[ax] - half[ax]) & (coords[ax] <= center[ax] + half[ax])
            solid |= mask
        else:
            raise ValueError(f"Unknown obstacle type: {kind}")


    F[solid] = 0.  # zero out distributions inside solid obstacles

    return F, solid

if __name__ == "__main__":
    # 2D example
    #Nx, Ny = 256, 128
    Nx, Ny = 32, 32
    obstacles = [
        ('round', (Ny / 2, Nx / 4), 3),
        ('box', (Ny * 0.7, Nx * 0.6), (4, 6)),
    ]
    F, solid = setup_domain((Ny, Nx), 'D2Q9', obstacles,
                                  flow_axis=1, flow_boost=2.3)

    solid = np.asarray(solid, dtype=int)
    for r in solid:
        print(' '.join(str(v) for v in r))
