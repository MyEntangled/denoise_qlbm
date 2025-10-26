from sympy.combinatorics import Permutation
import numpy as np


def _as_array_form(p: Permutation, q: int):
    """Return full-length array form of a SymPy Permutation over 0..q-1."""
    arr = list(p.array_form)
    if len(arr) < q:
        # SymPy may omit fixed points at the tail; pad identity on the rest
        missing = list(range(len(arr), q))
        arr += missing
    if len(arr) != q:
        raise ValueError(f"Bad permutation length: got {len(arr)}, expected {q}")
    return tuple(arr)

def verify_permutation_group(c, w, perm_dict, expected_size=None, check_closure=True, tol=0.0):
    """
    Verify that:
      (i) all permutations are distinct,
      (ii) each permutation preserves weights (hence symmetry of the weighted set),
      (iii) optional: group size matches expected_size,
      (iv) optional: closure under composition.

    Parameters
    ----------
    c : list[tuple[int,...]]
        Velocity tuples in the chosen ordering.
    w : list[float]
        Weights in the same ordering.
    perm_dict : dict[str, Permutation]
        Dictionary of named SymPy permutations acting on indices 0..q-1.
    expected_size : int or None
        If provided, assert len(perm_dict) == expected_size.
    check_closure : bool
        If True, verify composition of any two perms stays inside the set.
    tol : float
        Tolerance for weight-invariance check (0.0 for exact).
    """
    q = len(c)
    w = np.asarray(w, dtype=float)

    # --- shape & typing checks
    for name, p in perm_dict.items():
        if not isinstance(p, Permutation):
            raise TypeError(f"{name}: value must be sympy.combinatorics.Permutation")

    # --- (i) distinctness
    arr_forms = {name: _as_array_form(p, q) for name, p in perm_dict.items()}
    seen = set(arr_forms.values())
    if len(seen) != len(perm_dict):
        # find duplicates
        rev = {}
        dups = []
        for name, arr in arr_forms.items():
            if arr in rev:
                dups.append((rev[arr], name))
            else:
                rev[arr] = name
        dup_str = ", ".join([f"{a} == {b}" for a,b in dups])
        raise AssertionError(f"Non-distinct permutations detected: {dup_str}")

    # --- (ii) symmetry: weights invariant under each permutation
    for name, arr in arr_forms.items():
        w_perm = w[list(arr)]
        if np.max(np.abs(w_perm - w)) > tol:
            raise AssertionError(f"Permutation '{name}' does not preserve weights (max err={np.max(np.abs(w_perm-w))}).")

    # velocities themselves are just reindexed by a permutation (no need to check),
    # but for completeness we can assert it's a true permutation of indices 0..q-1:
    for name, arr in arr_forms.items():
        if sorted(arr) != list(range(q)):
            raise AssertionError(f"'{name}' is not a bijection of 0..{q-1}.")

    # --- (iii) expected size
    if expected_size is not None and len(perm_dict) != expected_size:
        raise AssertionError(f"Group size {len(perm_dict)} != expected {expected_size}.")

    # --- (iv) closure: composition stays in the set
    if check_closure:
        arr_to_name = {arr: nm for nm, arr in arr_forms.items()}
        for a_name, a_arr in arr_forms.items():
            for b_name, b_arr in arr_forms.items():
                # compose: apply b then a  (array form composition)
                comp = tuple(a_arr[i] for i in b_arr)
                if comp not in arr_to_name:
                    raise AssertionError(f"Not closed: {a_name} ∘ {b_name} not found in dict.")

    return {
        "q": q,
        "num_elements": len(perm_dict),
        "distinct": True,
        "weights_invariant": True,
        "closure_ok": check_closure,
        "expected_size_ok": (expected_size is None) or (len(perm_dict) == expected_size),
    }

if __name__ == '__main__':
    from src.qlbm.lbm_lattices import d1q3, d2q9, d3q15, d3q19, d3q27
    from src.qlbm.lbm_symmetries import permutations_D1Q3, permutations_D2Q9, permutations_D3Q15, permutations_D3Q19, permutations_D3Q27


    # Build lattice data
    c1, w1 = d1q3()
    c9, w9 = d2q9()
    c15, w15 = d3q15()
    c19, w19 = d3q19()
    c27, w27 = d3q27()

    # Build permutation groups using your constructors
    G1 = permutations_D1Q3()  # expect 2
    G9 = permutations_D2Q9()  # expect 8
    G15 = permutations_D3Q15()  # expect 48 (Oh)
    G19 = permutations_D3Q19()  # expect 48 (Oh)
    G27 = permutations_D3Q27()  # expect 48 (Oh)

    # Verify
    print(verify_permutation_group(c1, w1, G1, expected_size=2))
    print(verify_permutation_group(c9, w9, G9, expected_size=8))
    print(verify_permutation_group(c15, w15, G15, expected_size=48))
    print(verify_permutation_group(c19, w19, G19, expected_size=48))
    print(verify_permutation_group(c27, w27, G27, expected_size=48))