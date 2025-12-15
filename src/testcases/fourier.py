from src.lattices.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np

def setup_testcase(C0: float = 1., C1: float = 0.5, u_max = 0.1/np.sqrt(3), mu: float = 1e-3):
    """
    LBM initializer for the advection-diffusion under advection field u(x,t) = u_max cos(mu * t)

    Returns
    -------
    F : np.ndarray      (*domain_dims, Q)
    solid : np.ndarray  (*domain_dims,), which is all False (hence periodic)
    u_solid = None
    """
    ## Fixed settings for this testcase
    domain_dims = (256,)
    grid_size = tuple(domain_dims[::-1])  # reverse for array indexing
    L = grid_size[0]
    lattice = "D1Q3"

    ## Physical parameters
    omega = 1.0     # relaxation rate
    tau = 1.0   # relaxation time
    D = (tau - 0.5) / 3.0  # diffusion coefficient

    u_adv = lambda t: np.array([u_max * np.cos(mu * t)], dtype=float)

    config = {
        "grid_size": grid_size,
        "lattice": lattice,
        "is_scalar_field": True,
        "C0": C0,
        "C1": C1,
        "u_max": u_max,
        "mu": mu,
        "D": D,
        "u_adv": u_adv,
        "u0": u_adv,  # reference velocity for denoising collision
    }


    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape

    k = 2 * np.pi / L  # wave number

    # --- base density ---
    rho = C0 + C1 * np.cos(k * np.arange(L))

    # --- Define boundary and boundary condition ---
    solid = np.zeros((*grid_size,), dtype=bool)
    u_solid = None  # no-slip

    # --- initial velocity field: set to u_adv ---
    #u = u_adv[None, None, :].repeat(grid_size[0], axis=0).repeat(grid_size[1], axis=1)
    u = u_adv(t=0.0)[None, :].repeat(grid_size[0], axis=0)

    # Compute equilibrium
    F = get_equilibrium(rho, u, lattice)
    F[solid] = 0.

    assert (F >= 0).all(), "Negative values in initial distribution!"

    return config, F, solid, u_solid

def exact_solution(C0: float = 1., C1: float = 0.5, u_max = 0.1/np.sqrt(3), mu: float = 1e-3):
    """
    Exact solution for the Fourier mode advection-diffusion testcase.
    """
    ## Fixed settings for this testcase
    domain_dims = (256,)
    grid_size = tuple(domain_dims[::-1])  # reverse for array indexing
    L = grid_size[0]

    ## Physical parameters
    omega = 1.0     # relaxation rate
    tau = 1.0   # relaxation time
    D = (tau - 0.5) / 3.0  # diffusion coefficient

    k = 2 * np.pi / L  # wave number

    # Advection integral
    integral_u = lambda t: (u_max * np.sin(mu * t)) / mu

    #C = C0 + C1 * np.cos(k * (x[..., 0] - integral_u)) * np.exp(-D * k**2 * t)
    C = lambda t: C0 + C1 * np.cos(k * (np.arange(L) - integral_u(t))) * np.exp(-D * k**2 * t)

    return C

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config, F, _, _= setup_testcase()
    print("Testcase 'fourier' initialized.")
    print(config)
    print(F.sum(axis=-1))

    C_exact_fn = exact_solution(C0=config["C0"], C1=config["C1"], u_max=config["u_max"], mu=config["mu"])

    for t in range(0, 10001, 100):
        C = C_exact_fn(t)

        plt.plot(range(len(C)), C)
        plt.title(f"Fourier mode at t={t}")
        plt.pause(0.01)
        plt.cla()