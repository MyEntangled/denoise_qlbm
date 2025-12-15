from src.lattices.lbm_lattices import d1q3, d2q9, d3q15, d3q19, d3q27

from functools import lru_cache
import numpy as np
from collections import deque
from sympy.combinatorics import Permutation, PermutationGroup
from typing import Dict

# -------------------------
# Helpers: build permutations from linear maps on velocity tuples
# -------------------------

def perm_from_linear_map(vels, M):
    """
    Given velocity list 'vels' (tuples) and an integer matrix M (d x d),
    return the array-form permutation induced by v -> M @ v.
    """
    v2i = {tuple(v): i for i, v in enumerate(vels)}
    d = len(vels[0])
    M = np.asarray(M, dtype=int)
    assert M.shape == (d, d)
    q = len(vels)
    p = [None]*q
    for i, v in enumerate(vels):
        vv = np.array(v, dtype=int)
        tv = tuple((M @ vv).tolist())
        if tv not in v2i:
            raise ValueError(f"Transform maps {v} -> {tv}, not in the lattice set.")
        p[i] = v2i[tv]
    return p  # array form for SymPy Permutation

def compose_perm(p, q):
    """Return composition p∘q (apply q first, then p), both in array form."""
    return [p[q[i]] for i in range(len(p))]

def compress_word(word):
    """
    Simple pretty-printer: compress consecutive generator repeats:
    e.g., 'Rz Rz Rz' -> 'Rz^3'. 'I I' -> 'I^2'.
    """
    if word in ("", "e"): return "e"
    toks = word.split()
    out = []
    i = 0
    while i < len(toks):
        g = toks[i]
        k = 1
        while i + k < len(toks) and toks[i+k] == g:
            k += 1
        out.append(f"{g}^{k}" if k > 1 else g)
        i += k
    return " ".join(out)

# -------------------------
# D1Q3: dihedral D2 (just reflection)
# -------------------------
@lru_cache
def permutations_D1Q3():
    vels, _ = d1q3()
    # reflection x -> -x
    sM = np.array([[-1]], dtype=int)
    s = Permutation(perm_from_linear_map(vels, sM))
    e = Permutation(list(range(len(vels))))
    return {"e": e, "s": s}

# -------------------------
# D2Q9: dihedral D4 with names r^k and r^k s
# -------------------------
@lru_cache
def permutations_D2Q9():
    vels, _ = d2q9()
    # rotation r: +90° (x, y) -> (-y, x)
    rM = np.array([[0, -1],
                   [1,  0]], dtype=int)
    # reflection s: across x-axis (x, y) -> (x, -y)
    sM = np.array([[1,  0],
                   [0, -1]], dtype=int)

    R = Permutation(perm_from_linear_map(vels, rM))
    S = Permutation(perm_from_linear_map(vels, sM))
    G = PermutationGroup([R, S])

    out = {}
    for g_af in G.generate(af=True):
        g = Permutation(g_af)
        named = False
        for k in range(4):
            if g == R**k:
                out[f"r^{k}"] = g
                named = True
                break
            if g == (R**k) * S:
                out[f"r^{k}s"] = g
                named = True
                break
        if not named:
            out[f"elt_{len(out)}"] = g
    return out  # 8 elements

# -------------------------
# D3Q*: cubic O_h (48 elements): generators Rz (90°), Rx (90°), I (inversion)
# -------------------------

def cubic_generators():
    # 90° about z: (x,y,z) -> (-y, x, z)
    Rz = np.array([[0, -1, 0],
                   [1,  0, 0],
                   [0,  0, 1]], dtype=int)
    # 90° about x: (x,y,z) -> (x, -z, y)
    Rx = np.array([[1,  0,  0],
                   [0,  0, -1],
                   [0,  1,  0]], dtype=int)
    # inversion: v -> -v
    I  = -np.eye(3, dtype=int)
    return Rz, Rx, I

def permutations_D3Q(lattice="D3Q27"):
    if lattice == "D3Q27":
        vels, _ = d3q27()
    elif lattice == "D3Q19":
        vels, _ = d3q19()
    elif lattice == "D3Q15":
        vels, _ = d3q15()
    else:
        raise ValueError("lattice must be one of: D3Q15, D3Q19, D3Q27")

    # Build generator permutations
    RzM, RxM, IM = cubic_generators()
    gen_names = ["Rz", "Rx", "I"]
    gen_mats  = [RzM,  RxM,  IM]
    gen_perms = [Permutation(perm_from_linear_map(vels, M)) for M in gen_mats]

    # BFS to enumerate group with shortest names
    q = len(vels)
    id_arr = tuple(range(q))
    id_perm = Permutation(list(id_arr))
    seen = {id_arr: "e"}
    queue = deque([id_arr])

    while queue:
        cur = queue.popleft()
        cur_name = seen[cur]
        for name, genP in zip(gen_names, gen_perms):
            nxtP = Permutation(list(compose_perm(list(genP.array_form), list(cur))))
            nxt_arr = tuple(nxtP.array_form)
            if nxt_arr not in seen:
                new_word = (cur_name + " " + name).strip() if cur_name != "e" else name
                seen[nxt_arr] = new_word
                queue.append(nxt_arr)

    # Convert to {pretty_name: Permutation}
    out = {}
    for arr, word in seen.items():
        pretty = compress_word(word)
        out[pretty] = Permutation(list(arr))
    return out  # size 48

# -------------------------
# Convenience wrappers for 3D lattices
# -------------------------
@lru_cache
def permutations_D3Q15(): return permutations_D3Q("D3Q15")
@lru_cache
def permutations_D3Q19(): return permutations_D3Q("D3Q19")
@lru_cache
def permutations_D3Q27(): return permutations_D3Q("D3Q27")

# Registry for convenient lookup
_registry: Dict[str, callable] = {
    "D1Q3": permutations_D1Q3, "d1q3": permutations_D1Q3,
    "D2Q9": permutations_D2Q9, "d2q9": permutations_D2Q9,
    "D3Q15": permutations_D3Q15, "d3q15": permutations_D3Q15,
    "D3Q19": permutations_D3Q19, "d3q19": permutations_D3Q19,
    "D3Q27": permutations_D3Q27, "d3q27": permutations_D3Q27,
}

def get_symmetry(name: str):
    try:
        return _registry[name]()
    except KeyError as e:
        raise KeyError(f"Unknown lattice '{name}'. "
                       f"Available: {sorted(set(_registry))}") from e



# Permutation application utilities
# -------------------------
from typing import Mapping


def _perm_to_index_array(perm, q: int) -> np.ndarray:
    """Normalize a permutation to a full index array of length q.
    Accepts:
      - SymPy Permutation
      - array-like of indices (list/tuple/np.ndarray)
    """
    if isinstance(perm, Permutation):
        arr = list(perm.array_form)
        # SymPy may omit trailing identity mappings; pad with identity indices
        if len(arr) < q:
            arr += list(range(len(arr), q))
        perm_idx = np.asarray(arr, dtype=int)
    else:
        perm_idx = np.asarray(perm, dtype=int)
        if perm_idx.ndim != 1:
            raise ValueError("Permutation must be 1D index array or SymPy Permutation")
        if len(perm_idx) != q:
            raise ValueError(f"Permutation length {len(perm_idx)} != array last-dim {q}")
    # Basic validity checks
    if sorted(perm_idx.tolist()) != list(range(q)):
        raise ValueError("Permutation is not a bijection of 0..q-1")
    return perm_idx


def permute_by_array(arr: np.ndarray, perm) -> np.ndarray:
    """Apply a permutation to the **last axis** of `arr`.
    `perm` can be a SymPy Permutation or an index array of length arr.shape[-1]."""
    if perm is None:
        return arr
    q = arr.shape[-1]
    perm_idx = _perm_to_index_array(perm, q)
    return arr[..., perm_idx]


def permute_by_symmetry(arr: np.ndarray, perm_dict: Mapping[str, Permutation], symmetry: str) -> np.ndarray:
    """Apply a named symmetry from `perm_dict` to the last axis of `arr`."""
    try:
        perm = perm_dict[symmetry]
    except KeyError as e:
        raise KeyError(f"Symmetry '{symmetry}' not found. Available: {list(perm_dict.keys())}") from e
    return permute_by_array(arr, perm)


def permute_by_all_symmetries(arr: np.ndarray, perm_dict: Mapping[str, Permutation]):
    """Apply all symmetries and stack the results along axis=-2
    Returns (stacked, names) where stacked has shape with a new axis of size |G|.
    """
    names = list(perm_dict.keys())  # preserve insertion order
    permuted = [permute_by_array(arr, perm_dict[name]) for name in names]
    stacked = np.stack(permuted, axis=-2)   # shape: (..., num_perms, dim)
    return stacked, names



# -------------------------
# Quick sanity run
# -------------------------
if __name__ == "__main__":
    D2 = permutations_D1Q3()
    print("D1Q3:", len(D2), "elements (expect 2)")
    print(D2)

    D8 = permutations_D2Q9()
    print("D2Q9:", len(D8), "elements (expect 8)")
    print(D8)

    Oh15 = permutations_D3Q15()
    Oh19 = permutations_D3Q19()
    Oh27 = permutations_D3Q27()
    print("D3Q15 O_h:", len(Oh15), "elements (expect 48)")
    print("D3Q19 O_h:", len(Oh19), "elements (expect 48)")
    print("D3Q27 O_h:", len(Oh27), "elements (expect 48)")
    print(Oh27)

    print('-----')

    rng = np.random.default_rng(0)
    F = rng.random((3, 15)).round(3)
    print("F:\n", F)

    G = get_symmetry("D3Q15")  # permutation dict
    sym = "Rz^2 I"
    F_perm = permute_by_symmetry(F, perm_dict=G, symmetry=sym)
    print(f"F permuted by {sym}:\n", F_perm)

    F_all, names = permute_by_all_symmetries(F, perm_dict=G)  # stacked along axis=-2
    print("Stacked over all symmetries, shape:", F_all.shape)
    print("Symmetry order:", names)