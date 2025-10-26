import numpy as np
import itertools as it

def M_rank_n(c, w, n):
    """Compute rank-n moment tensor M^{(n)}."""
    c = np.asarray(c, float)
    w = np.asarray(w, float)
    d = c.shape[1]
    M = np.zeros((d,)*n)
    for i in range(len(w)):
        T = 1.0
        for _ in range(n):
            T = np.multiply.outer(T, c[i])
        M += w[i] * T
    return M

def isotropy_report(c, w, tol=1e-14):
    c = np.asarray(c, float)
    w = np.asarray(w, float)
    d = c.shape[1]

    # --- Rank 0 ---
    M0 = np.sum(w)
    e0 = abs(M0 - 1.0)

    # --- Compute cs^2 ---
    cs2 = (w * np.sum(c**2, axis=1)).sum() / d

    # --- Rank 1–5 moments ---
    M1 = M_rank_n(c, w, 1)
    M2 = M_rank_n(c, w, 2)
    M3 = M_rank_n(c, w, 3)
    M4 = M_rank_n(c, w, 4)
    M5 = M_rank_n(c, w, 5)

    # --- Targets ---
    Z1 = np.zeros_like(M1)
    Z3 = np.zeros_like(M3)
    Z5 = np.zeros_like(M5)

    # Rank 2 target
    T2 = cs2 * np.eye(d)

    # Rank 4 target
    I = np.eye(d)
    T4 = np.zeros((d,d,d,d))
    for a,b,g,dd in it.product(range(d), repeat=4):
        T4[a,b,g,dd] = (cs2**2) * (I[a,b]*I[g,dd] +
                                   I[a,g]*I[b,dd] +
                                   I[a,dd]*I[b,g])

    # --- Errors ---
    err = {
        "rank0": e0,
        "rank1": np.max(np.abs(M1 - Z1)),
        "rank2": np.max(np.abs(M2 - T2)),
        "rank3": np.max(np.abs(M3 - Z3)),
        "rank4": np.max(np.abs(M4 - T4)),
        "rank5": np.max(np.abs(M5 - Z5)),
    }

    ok = all(v <= tol for v in err.values())
    return {"cs2": cs2, "errors": err, "passes_up_to_5th": ok, "tol": tol}


if __name__ == '__main__':
    from src.qlbm.lbm_lattices import d2q9, d3q15, d3q19, d3q27, get_lattice

    lattices = {
        "D2Q9": get_lattice('d2q9'),
        "D3Q15": get_lattice('D3Q15'),
        "D3Q19": d3q19(),
        "D3Q27": d3q27(),
    }

    for name, (c, w) in lattices.items():
        report = isotropy_report(c, w, tol=1e-14)
        print(f"Lattice {name}:")
        print(f"  cs^2 = {report['cs2']}")

        for rank, err in report['errors'].items():
            print(f"  {rank} moment error: {err}")

        if report['passes_up_to_5th']:
            print("  PASSES isotropy up to 5th order.")
        else:
            print("  FAILS isotropy up to 5th order.")