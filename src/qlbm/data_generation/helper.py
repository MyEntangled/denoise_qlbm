import numpy as np
import jax.numpy as jnp

def split_train_test(X,
                     Y,
                     train_ratio: float = 0.8,
                     shuffle: bool = True,
                     rng=None):
    """
    Split (X, Y) pairs into training and testing sets.
    Works with both NumPy and JAX arrays.

    Args:
        X: (N, D_in) array (np.ndarray or jnp.ndarray)
        Y: (N, D_out) array (np.ndarray or jnp.ndarray)
        train_ratio: fraction of data for training (0 < train_ratio < 1)
        shuffle: whether to randomly shuffle data
        rng: optional np.random.Generator or jax.random.PRNGKey

    Returns:
        X_train, Y_train, X_test, Y_test (same type as X, Y)
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    N = X.shape[0]

    if shuffle:
        perm = rng.permutation(N)
    else:
        perm = np.arange(N)

    n_train = int(np.floor(train_ratio * N))
    train_idx, test_idx = perm[:n_train], perm[n_train:]


    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    return X_train, Y_train, X_test, Y_test