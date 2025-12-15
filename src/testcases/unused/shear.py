from src.lattices.lbm_lattices import get_lattice
from src.qlbm.data_generation.sample_distribution import get_equilibrium

import numpy as np

def setup_testcase(C0: float):
    """
    Double shear layer advection–diffusion testcase (D2Q9, doubly periodic).
    Scalar field is aligned with the shear layers.
    """
    # --- basic settings ---
    H, V = 256, 256              # grid_size = (H, V)
    lattice = "D2Q9"

    # LBM diffusion (BGK ADE)
    tau = 1.0
    D = (tau - 0.5) / 3.0

    # double shear layer parameters
    u_max       = 0.01 / np.sqrt(3)
    kp       = 5.0
    delta_p  = 0.05

    c, w = get_lattice(lattice, as_array=True)
    Q, d = c.shape
    assert d == 2

    # coordinates in [0, 1]
    y = (np.arange(H) + 0.5) / H
    x = (np.arange(V) + 0.5) / V
    Yp, Xp = np.meshgrid(y, x, indexing="ij")   # (H, V)

    # velocity field
    ux = np.where(
        Yp <= 0.5,
        u_max * np.tanh(kp * (Yp - 0.25)),
        u_max * np.tanh(kp * (0.75 - Yp)),
    )
    uy = u_max * delta_p * np.sin(2.0 * np.pi * (Xp + 0.25))
    u = np.stack([ux, uy], axis=-1)  # (H, V, 2)

    config = {
        "grid_size": (H, V),
        "lattice": lattice,
        "is_scalar_field": True,
        "C0": C0,
        "D": D,
        "u_adv": u,
        "kp": kp,
        "delta_p": delta_p,
    }

    # scalar field
    rho = np.where(
        Yp <= 0.5,
        C0 * (1.0 + np.tanh(kp * (Yp - 0.25))),
        C0 * (1.0 + np.tanh(kp * (0.75 - Yp))),
    )

    solid = np.zeros((H, V), dtype=bool)
    u_solid = None

    F = get_equilibrium(rho, u, lattice)
    F[solid] = 0.0

    return config, F, solid, u_solid

if __name__ == "__main__":
    config, F, solid, u_solid = setup_testcase(C0=1.0)
    print("F shape:", F.shape)

    print(config["u_adv"][:,:2,:])

    import matplotlib.pyplot as plt
    plt.imshow(np.linalg.norm(config["u_adv"], axis=-1), origin="lower")
    plt.pause(.01)
