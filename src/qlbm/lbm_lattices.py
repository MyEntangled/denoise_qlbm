from itertools import product
from functools import lru_cache
from typing import Dict
import numpy as np

@lru_cache
def d1q3():
    c = [(0,), (1,), (-1,)]
    w = [2/3, 1/6, 1/6]
    return c, w

@lru_cache
def d2q9():
    c = [(0,0),
         (1,0), (-1,0), (0,1), (0,-1),
         (1,1), (-1,1), (-1,-1), (1,-1)]
    w = [4/9] + [1/9]*4 + [1/36]*4
    return c, w

@lru_cache
def d3q15():
    c = [(0,0,0)]
    # face-centered (6)
    c += [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    # body diagonals (8)
    c += [(sx,sy,sz) for sx,sy,sz in product([1,-1],[1,-1],[1,-1])]
    w = [2/9] + [1/9]*6 + [1/72]*8
    return c, w

@lru_cache
def d3q19():
    c = [(0,0,0)]
    # face-centered (6)
    faces = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    c += faces
    # edges (12): permutations of (±1, ±1, 0)
    edges = []
    for a,b in product([1,-1],[1,-1]):
        edges += [(a,b,0),(a,0,b),(0,a,b)]
    c += edges
    w = [1/3] + [1/18]*6 + [1/36]*12
    return c, w

@lru_cache
def d3q27():
    # all combinations of (-1,0,1)^3
    c = [(0,0,0)]
    faces = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    # edges: exactly two nonzeros
    edges = []
    for a,b in product([1,-1],[1,-1]):
        edges += [(a,b,0),(a,0,b),(0,a,b)]
    corners = [(sx,sy,sz) for sx,sy,sz in product([1,-1],[1,-1],[1,-1])]
    c = [(0,0,0)] + faces + edges + corners
    w = [8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8
    return c, w

# Registry for convenient lookup
_registry: Dict[str, callable] = {
    "D1Q3": d1q3, "d1q3": d1q3,
    "D2Q9": d2q9, "d2q9": d2q9,
    "D3Q15": d3q15, "d3q15": d3q15,
    "D3Q19": d3q19, "d3q19": d3q19,
    "D3Q27": d3q27, "d3q27": d3q27,
}

def get_lattice(name: str, as_array: bool = True):
    try:
        c,w = _registry[name]()
        if not as_array:
            return c, w
        else:
            c = np.asarray(c, dtype=int)  # (Q,d)
            w = np.asarray(w, dtype=float)  # (Q,)
            return c, w

    except KeyError as e:
        raise KeyError(f"Unknown lattice '{name}'. "
                       f"Available: {sorted(set(_registry))}") from e