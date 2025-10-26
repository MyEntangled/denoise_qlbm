from src.qlbm.lbm_symmetries import permute_by_all_symmetries

import numpy as np
import jax.numpy as jnp


def distributions_to_statevectors(F: np.ndarray,
                                  take_sqrt: bool = False,
                                  normalize: bool = False,
                                  symmetries: dict[str, object] | None = None):
    """
    Convert LBM distributions F into state vectors, optionally taking square roots,
    normalizing, and applying symmetry permutations.

    Args:
        F: (batch, Q) array of distributions
        take_sqrt: if True, apply elementwise sqrt(F)
        normalize: if True, normalize each state to unit norm
        symmetries: optional dict of symmetry permutations (e.g., from get_symmetry())

    Returns:
        states: (batch', Q) array of processed state vectors
                (batch' = batch × |symmetries| if symmetries provided)
    """
    states = jnp.asarray(F, dtype=jnp.float32)

    if take_sqrt:
        states = jnp.sqrt(states)

    if normalize:
        states = states / jnp.linalg.norm(states, axis=-1, keepdims=True)

    if symmetries is not None and isinstance(symmetries, dict):
        stacked, _ = permute_by_all_symmetries(states, symmetries)
        states = jnp.asarray(stacked).reshape(-1, states.shape[-1])

    return states




def postselect_state(tensorstates: jnp.ndarray, r: int) -> jnp.ndarray:
    """
    Postselect on the ancilla register being in state |0^r⟩.

    Given a composite state |Ψ> = Σ_a |a>⊗|ψ_a>,
    this extracts the |ψ_0> unnormalized component corresponding to ancilla |a=0^r⟩.

    Args:
        tensorstates: (B, 2^r * D) array representing the full state |Ψ>.
        r: number of ancilla qubits.

    Returns:
        (B, D) array of the postselected subsystem state |ψ_0>.
    """
    factor = 2 ** r
    B, full_dim = tensorstates.shape

    if full_dim % factor != 0:
        raise ValueError(f"State dimension {full_dim} not divisible by 2**r = {factor}")

    D = full_dim // factor
    # Select the sub-block corresponding to ancilla |0^r⟩
    return tensorstates[:, :D]
