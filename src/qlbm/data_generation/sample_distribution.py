from src.qlbm.lbm_lattices import get_lattice
from src.qlbm.data_generation.lbm_collision import get_equilibrium, collide
from src.qlbm.data_generation.sample_noise import sample_zero_moments_noise
import numpy as np

# --- Uniform random distribution generator ---
def _sample_uniform_distribution(num_samples: int,
                                 lattice: str,
                                 weighted: bool = False,
                                 rng=None):
    """
    Generate random distributions f_i for a given lattice, normalized to sum_i f_i = 1.

    Args:
        num_samples: int, number of samples
        lattice: name of the lattice ("D1Q3","D2Q9","D3Q15","D3Q19","D3Q27")
        weighted: if True, scale by lattice weights (so mean(f_i) ∝ w_i)
        rng: optional np.random.Generator
    Returns:
        F: (num_samples, Q) random normalized distributions
    """
    if rng is None:
        rng = np.random.default_rng()

    _, w = get_lattice(lattice, as_array=True)

    w = np.asarray(w, dtype=float)
    Q = len(w)

    # Uniform positive random values
    F = rng.random((num_samples, Q))

    if weighted:
        # Weight-aware initialization: bias sampling toward higher weights
        F *= w[None, :]

    # Normalize each sample so that sum_i f_i = 1
    F /= np.sum(F, axis=1, keepdims=True)
    return F


# --- Sample valid initial distributions and their post-collision distributions ---
def sample_uniform_data(n_samples: int,
                        lattice: str,
                        omega: int =1.0,
                        weighted: bool = False,
                        rng: np.random.Generator | None = None,
                        oversample_factor: float = 1.2):
    """
    Sample random initial distributions F (normalized so sum_i F_i = 1)
    and their post-collision targets for a given lattice.

    Args:
        n_samples: number of valid samples to return
        lattice: "D1Q3" | "D2Q9" | "D3Q15" | "D3Q19" | "D3Q27"
        omega: scalar or array-like (n_samples,) for BGK relaxation
        weighted: if True, bias initial draws by lattice weights
        rng: optional numpy.random.Generator (for reproducibility)
        oversample_factor: >1, how many to draw per iteration to ensure enough valid samples

    Returns:
        F_samples: (n_samples, Q)   initial normalized distributions
        F_target:  (n_samples, Q)   post-collision distributions after applying ω
    """
    if rng is None:
        rng = np.random.default_rng()

    F_list = []
    Feq_list = []

    while len(F_list) < n_samples:
        need = n_samples - len(F_list)
        n_draw = max(1, int(np.ceil(need * oversample_factor)))

        F = _sample_uniform_distribution(n_draw, lattice=lattice, weighted=weighted, rng=rng)
        F_eq = collide(F, lattice=lattice, omega=1.0)  # equilibrium = collision with ω=1

        # Keep samples whose equilibrium is nonnegative
        mask = np.all(F_eq >= 0.0, axis=1)
        if np.any(mask):
            F_list.append(F[mask])
            Feq_list.append(F_eq[mask])

    F_samples = np.vstack(F_list)[:n_samples]
    Feq_samples = np.vstack(Feq_list)[:n_samples]

    # BGK update
    if np.isscalar(omega):
        F_target = (1.0 - omega) * F_samples + omega * Feq_samples
    else:
        omega = np.asarray(omega).reshape(-1, 1)
        F_target = (1.0 - omega) * F_samples + omega * Feq_samples

    return F_samples, F_target


# --- Sample low-Mach equilibrium states with noise ---
def sample_low_mach_data(n_samples: int,
                         lattice: str,
                         rho,                         # float or (n_samples,) array-like
                         mean_norm_u: float,
                         std_norm_u: float,
                         rel_noise_strength: float,
                         rng: np.random.Generator | None = None,
                         oversample_factor: float = 1.2,
                         weighted_noise: bool = True):
    """
    Draw low-Mach equilibrium states with |u| ~ LogNormal(mean_norm_u, std_norm_u),
    random directions in R^d, then add zero-(density,momentum) noise with magnitude
    proportional to rho (LBM weights-aware if weighted_noise=True).

    Returns:
        F_samples : (n_samples, Q)  = F_eq + eps   (nonnegative)
        F_eq_samples : (n_samples, Q)             (nonnegative)
    """
    if rng is None:
        rng = np.random.default_rng()

    c, _ = get_lattice(lattice, as_array=True)
    Q, d = c.shape

    # --- Parameterize the LogNormal(|u|) distribution
    # Given mean m and std s of |u|, the underlying normal parameters (mu,sigma):
    # s^2 = (exp(sigma^2)-1) exp(2mu + sigma^2),  m = exp(mu + sigma^2/2)
    # -> sigma = sqrt(log(1 + (s^2 / m^2))), mu = log(m^2 / sqrt(m^2 + s^2))
    m2 = mean_norm_u**2
    s2 = std_norm_u**2
    if mean_norm_u < 0:
        raise ValueError("mean_norm_u must be >= 0")
    sigma = np.sqrt(np.log(1.0 + s2 / m2))
    mu = np.log(m2 / np.sqrt(m2 + s2))

    # --- Collect valid equilibria (nonnegative)
    Feq_list = []
    while sum(len(x) for x in Feq_list) < n_samples:
        need = n_samples - sum(len(x) for x in Feq_list)
        n_draw = max(1, int(np.ceil(need * oversample_factor)))

        # magnitudes |u|
        norm_u = rng.lognormal(mean=mu, sigma=sigma, size=n_draw)  # (n_draw,)

        # random directions on S^{d-1}
        hat_u = rng.normal(size=(n_draw, d))
        # handle rare zero norms robustly
        norms = np.linalg.norm(hat_u, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        hat_u /= norms

        u = norm_u[:, None] * hat_u  # (n_draw, d)

        F_eq = get_equilibrium(rho=np.ones(n_draw), u=u, lattice=lattice)  # (n_draw, Q)

        # Keep equilibria with all populations >= 0 (stability screen)
        mask = np.all(F_eq >= 0.0, axis=1)
        if np.any(mask):
            Feq_list.append(F_eq[mask])

    F_eq_samples = np.vstack(Feq_list)[:n_samples]  # shape (n_samples, Q)

    # --- Scale by rho (allow scalar or vector)
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.ndim == 0:
        rho_arr = np.full(n_samples, float(rho_arr))
    elif rho_arr.shape[0] != n_samples:
        raise ValueError("rho must be scalar or length n_samples.")
    F_eq_samples = rho_arr[:, None] * F_eq_samples


    # Sanity: sums equal rho
    # assert np.allclose(F_eq_samples.sum(1), rho_arr, atol=1e-12)

    # --- Draw zero-(density,momentum) noise; accept only if F_eq + eps ≥ 0
    noise = np.empty((n_samples, Q))
    accepted = np.zeros(n_samples, dtype=bool)

    while not np.all(accepted):
        idx = np.where(~accepted)[0]
        k = len(idx)
        # noise strength scaled by local rho
        eps = sample_zero_moments_noise(
            n_samples=k,
            noise_strength=rel_noise_strength * rho_arr[idx],
            lattice=lattice,
            weighted=weighted_noise,
            rng=rng
        )  # (k, Q)

        # accept where nonnegative
        ok = np.all(F_eq_samples[idx] + eps >= 0.0, axis=1)
        noise[idx[ok]] = eps[ok]
        accepted[idx[ok]] = True
        # retry for remaining indices

    F_samples = F_eq_samples + noise

    # Final sanity
    if not ((F_samples >= 0.0).all() and (F_eq_samples >= 0.0).all()):
        raise AssertionError("Negative populations encountered unexpectedly.")

    return F_samples, F_eq_samples

if __name__ == "__main__":
    # Test uniform data sampling
    rng = np.random.default_rng(123)
    F, F_target = sample_uniform_data(1000, lattice="D2Q9", omega=1.4, weighted=True, rng=rng)
    print(F.shape, F_target.shape)


    # Test low-Mach data sampling

    # Parameters
    n_samples = 50000
    lattice = "D2Q9"
    rho = 1.0  # base density
    mean_norm_u = 0.05  # mean |u| (in lattice units, small => low Mach)
    std_norm_u = 0.02  # std deviation of |u|
    rel_noise_strength = 0.01  # relative magnitude of thermal/noise fluctuations

    # Generate samples
    F_samples, F_eq_samples = sample_low_mach_data(
        n_samples=n_samples,
        lattice=lattice,
        rho=rho,
        mean_norm_u=mean_norm_u,
        std_norm_u=std_norm_u,
        rel_noise_strength=rel_noise_strength,
        rng=rng,
        weighted_noise=True
    )

    print("F_samples shape:", F_samples.shape)
    print("F_eq_samples shape:", F_eq_samples.shape)

    # --- Verification of hydrodynamic consistency ---

    # Retrieve lattice velocities for D2Q9 (to compute macroscopic fields)
    c, w = get_lattice('D2Q9', as_array=True)

    # Compute density and velocity fields
    rho_samples = np.sum(F_samples, axis=1)
    ux_samples = np.sum(F_samples * c[:, 0], axis=1) / rho_samples
    uy_samples = np.sum(F_samples * c[:, 1], axis=1) / rho_samples
    speed_samples = np.sqrt(ux_samples ** 2 + uy_samples ** 2)

    print(f"mean |u|: {np.mean(speed_samples):.4f}, std |u|: {np.std(speed_samples):.4f}")
    print(f"min(F): {F_samples.min():.4e}, max(F): {F_samples.max():.4e}")