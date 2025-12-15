from src.lattices.lbm_lattices import get_lattice
import numpy as np

def _opposites_from_c(c: np.ndarray) -> np.ndarray:
    """
    Given lattice velocities c of shape (Q, d), return the opposite index map opp (Q,)
    such that c[opp[q]] == -c[q] for all q.
    """
    Q, d = c.shape
    opp = np.empty(Q, dtype=np.int64)
    for q in range(Q):
        target = -c[q]
        matches = np.where(np.all(c == target, axis=1))[0]
        if matches.size != 1:
            raise ValueError(f"Cannot identify unique opposite for direction {q} with c[q]={c[q]}.")
        opp[q] = matches[0]
    return opp


def bounce_back_obstacles(states: np.ndarray,
                          obstacle: np.ndarray,
                          u_obstacle: np.ndarray | None,
                          lattice: str,
                          encoding_type: str,
                          dims=None) -> np.ndarray:
    """
    On-site bounce-back at obstacle cells: swap distributions f_q <-> f_{opp(q)} on those cells.

    Parameters
    ----------
    states   : np.ndarray
        Either grid-shaped (X1,...,Xd, Q) or flattened (N, Q). Returned shape matches input.
    obstacle : np.ndarray (bool)
        Boolean mask of obstacle cells. If states is grid-shaped: shape (X1,...,Xd).
        If states is flattened: shape (N,) or (X1,...,Xd) with dims provided.
    lattice  : str
        Lattice name understood by get_lattice (e.g., "D2Q9", "D3Q19", ...).
    dims     : tuple[int], optional
        Required if states is flattened (N, Q): the grid dimensions (X1,...,Xd).

    Returns
    -------
    np.ndarray
        States after bounce-back, same shape as `states`.
    """
    c, w = get_lattice(lattice, as_array=True)   # c: (Q, d)
    Q, d = c.shape
    opp = _opposites_from_c(c)


    # --- Normalize shapes (mirror streaming_periodic) ---
    if states.ndim == d + 1:
        grid_shape = states.shape[:-1]
        states_grid = states.copy()
        flattened = False
    elif states.ndim == 2:
        if dims is None or len(dims) != d:
            raise ValueError(f"Flattened input detected. Provide dims=(X1,...,Xd) with d={d}.")
        N = int(np.prod(np.array(dims)))
        if states.shape != (N, Q):
            raise ValueError(f"Expected states shape (N={N}, Q={Q}); got {states.shape}.")
        states_grid = (states.copy()).reshape(*dims, Q)
        grid_shape = tuple(dims)
        flattened = True
    else:
        raise ValueError(
            f"Unsupported states shape {states.shape}. "
            "Use (X1,...,Xd,Q) or (N,Q) with dims=(X1,...,Xd)."
        )

    # --- Normalize obstacle mask to grid shape ---
    if obstacle is None:
        obstacle_grid = np.zeros(grid_shape, dtype=bool)
    elif obstacle.shape == grid_shape:
        obstacle_grid = obstacle.astype(bool, copy=False)
    elif obstacle.ndim == 1 and obstacle.size == np.prod(grid_shape):
        obstacle_grid = obstacle.reshape(grid_shape).astype(bool, copy=False)
    else:
        raise ValueError(
            f"Obstacle mask shape {obstacle.shape} is incompatible with grid {grid_shape}."
        )

    # --- Swap f_q <-> f_opp on obstacle cells ---
    obs = obstacle_grid
    if not obs.any():
        # Nothing to do
        return states_grid.reshape(states.shape) if flattened else states_grid

    assert encoding_type in ['full', 'sqrt'], f"encoding_type must be 'full' or 'sqrt', got {encoding_type}."
    if encoding_type == 'full':
        F = states_grid  # alias
    else:
        F = states_grid ** 2

    F_boundary = F[obs, :]

    # --- moving obstacle correction ---
    if u_obstacle is not None:
        assert u_obstacle.ndim >= 1, "u_obstacle must have at least one dimension (the velocity components)."
        u_dim = u_obstacle.ndim - 1  # number of spatial dimensions in u_obstacle

        if u_obstacle.shape[:u_dim] == grid_shape:
            u_obstacle_grid = u_obstacle
        elif u_dim == 1 and np.prod(u_obstacle.shape[:1]) == np.prod(grid_shape):
            rest = u_obstacle.shape[1:]
            u_obstacle_grid = u_obstacle.reshape(grid_shape, *rest)
        else:
            raise ValueError(
                f"Obstacle velocity shape {u_obstacle.shape[:-1]} is incompatible with grid {grid_shape}."
            )

        u_obs = u_obstacle_grid


        # u_obs: (*grid_size, d), take only obstacle nodes -> (Nobs, d)
        u_wall = u_obs[obs, :]  # velocities of the solid nodes

        # cu: (Nobs, Q) = u_wall · c_i
        # c: (Q, d), so take transpose for matmul
        cu = u_wall @ c.T

        # Add moving-wall correction: f_i += 2 * w_i * rho0 * (c_i · u_wall) / cs^2
        rho0 = 1.
        cs2_inv = 3.
        F_boundary -= (2.0 * rho0 * cs2_inv * cu * w[np.newaxis,:])  # w broadcasts to (Nobs, Q)

    F[obs, :] = F_boundary[:, opp]

    if encoding_type == 'sqrt':
        bb_states = np.sqrt(abs(F)) * np.sign(F)
    else:
        bb_states = F

    return bb_states.reshape(states.shape) if flattened else bb_states

if __name__ == "__main__":
    # D2Q9 test
    lattice = "D2Q9"
    W, H = 4, 3
    c, _ = get_lattice(lattice, as_array=True)
    Q = c.shape[0]

    # Each direction has unique integer value just for demonstration
    states = np.arange(H * W * Q).reshape(H, W, Q)

    # Create obstacle mask
    obstacle = np.zeros((H, W), dtype=bool)
    obstacle[1, [1,2]] = True
    obstacle[0, 1] = True
    obstacle[2, 1] = True
    print(obstacle)

    # Bounced-back states
    bounced = bounce_back_obstacles(states, obstacle, lattice)

    print("Original shape:", states.shape)
    print("bounced shape:", bounced.shape)
    print("Difference (example element):", (bounced - states).sum())

    print("Initial:")
    print(states)
    print("Bounced:")
    print(bounced)
