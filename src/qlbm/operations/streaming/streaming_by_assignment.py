from src.lattices.lbm_lattices import get_lattice
import numpy as np

# --- in-grid streaming (periodic BC) ---
def _streaming_periodic_grid_(states_grid: np.ndarray, c: np.ndarray, region: np.ndarray = None) -> np.ndarray:
    """
    states_grid: (X1,...,Xd, Q)
    c          : (Q, d)  integer lattice velocities
    """
    Q, d = c.shape

    if states_grid.ndim != d + 1:
        raise ValueError(f"Expected {d} spatial dims before Q; got shape {states_grid.shape}")
    if states_grid.shape[-1] != Q:
        raise ValueError(f"states last axis {states_grid.shape[-1]} != Q={Q}")


    spatial_shape = states_grid.shape[:-1]
    if region is None:
        region = np.ones(spatial_shape, dtype=bool)
    else:
        if region.shape != spatial_shape:
            raise ValueError(f"`region` must have shape {spatial_shape}, got {region.shape}")
        region = region.astype(bool, copy=False)


    out = np.empty_like(states_grid)

    # for q in range(Q):
    #     vel_component = states_grid[..., q]   # (X1,...,Xd)
    #     shifts = c[q]                         # (d,)
    #
    #     for ax in range(d):
    #         s = shifts[ax]
    #         vel_component = np.roll(vel_component, s, axis=d - 1 - ax)
    #     out[..., q] = vel_component

    for q in range(Q):
        v = states_grid[..., q]                 # (X1,...,Xd)
        mask_q = region                         # spatial-only

        moved = v * mask_q                      # movers (origins inside region)
        stay  = v * (~mask_q)                   # stayers (origins outside region)

        # push movers to destinations via periodic rolls
        for ax in range(d):
            s = int(c[q, ax])
            if s != 0:
                moved = np.roll(moved, s, axis=d - 1 - ax)

        out[..., q] = stay + moved

    return out


def streaming_periodic(states: np.ndarray, lattice: str, dims=None, applied_region: np.ndarray = None):
    """
    Periodic LBM streaming for any lattice.
    - If states is grid-shaped: (X1,...,Xd,Q)
    - If states is flattened  : (N,Q) and you MUST pass dims=(X1,...,Xd)

    Returns the streamed array with the SAME shape as input.
    """

    c, _ = get_lattice(lattice, as_array=True)   # c: (Q, d)
    Q, d = c.shape

    # Decide shape mode, reshape if needed
    if states.ndim == d + 1:
        # Grid input
        states_grid = states
        flattened = False
    elif states.ndim == 2:
        # Flattened input => require dims
        if dims is None or len(dims) != d:
            raise ValueError(f"Flattened input detected. Provide dims=(X1,...,Xd) with d={d}.")

        N = int(np.prod(np.array(dims)))

        if states.shape[0] != N or states.shape[1] != Q:
            raise ValueError(f"Expected states shape (N={N}, Q={Q}); got {states.shape}.")

        grid_shape = tuple(dims)
        states_grid = states.reshape(*grid_shape, Q)
        flattened = True
    else:
        raise ValueError(
            f"Unsupported states shape {states.shape}. "
            "Use (X1,...,Xd,Q) or (N,Q) with dims=(X1,...,Xd)."
        )

    # If region is provided, reshape it to grid form too
    if applied_region is not None:
        if flattened:
            if applied_region.shape != states.shape[:-1]:
                raise ValueError(f"'applied_region' must have shape {states.shape[-1]}, got {applied_region.shape}")
            region_grid = applied_region.reshape(*dims, Q)
        else:
            if applied_region.shape != states.shape[:-1]:
                raise ValueError(f"'applied_region' must have shape {states.shape[-1]}, got {applied_region.shape}")
            region_grid = applied_region
    else:
        region_grid = None

    # Stream on grid
    streamed_grid = _streaming_periodic_grid_(states_grid, c, region=region_grid)

    # Restore original shape
    return streamed_grid.reshape(states.shape) if flattened else streamed_grid

if __name__ == "__main__":

    ## D1Q3 test
    lattice = "D1Q3"
    W = 5
    c, _ = get_lattice(lattice, as_array=True)
    Q = c.shape[0]

    # Each direction has unique integer value just for demonstration
    states = np.arange(W * Q).reshape(W, Q)

    # Perform streaming (periodic)
    streamed = streaming_periodic(states, lattice=lattice)

    print("Original shape:", states.shape)
    print("Streamed shape:", streamed.shape)
    print("Difference (example element):", (streamed - states).sum())

    print(states)
    print(streamed)

    # D2Q9 test
    lattice = "D2Q9"
    W, H = 4, 3
    c, _ = get_lattice(lattice, as_array=True)
    Q = c.shape[0]

    # Each direction has unique integer value just for demonstration
    states = np.arange(H * W * Q).reshape(H, W, Q)

    # Perform streaming (periodic)
    streamed = streaming_periodic(states, lattice=lattice)

    print("Original shape:", states.shape)
    print("Streamed shape:", streamed.shape)
    print("Difference (example element):", (streamed - states).sum())

    print("Initial:")
    print(states)
    print("Streamed:")
    print(streamed)
