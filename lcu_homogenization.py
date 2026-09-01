"""
Shift decomposition of the 2D plane-stress Q4 elasticity operator on a
PERIODIC cell with a TWO-PHASE microstructure (homogenization / RVE setting).

K = sum_{p,q,r} ( c0 * I + sum_{s=1..4} cs * R_s ) U_p^(x) (x) U_q^(y) (x) sigma_r

  U_p, U_q  in {I, S, S^dag}   cyclic shifts on the two spatial registers
  sigma_r   in {I, Z, X, iY}   Pauli on the 1-qubit DOF register
  R_s       = S^-s R_chi S^s   phase reflection of the material indicator chi,
                               at the four element positions adjacent to a node
                               R_chi = I - 2 diag(chi)

Independent re-derivation; reproduces L = 17, alpha = 6.0989 at zero contrast.
"""
import numpy as np, itertools

# ---------------------------------------------------------------- element ---
def element_stiffness(nu):
    """Plane-stress Q4 on the unit square, E = 1, unit thickness."""
    C = 1/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    K = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            dN = np.array([[-(1-eta), (1-eta), (1+eta), -(1+eta)],
                           [-(1-xi), -(1+xi), (1+xi),  (1-xi)]]) / 4
            J = np.eye(2) * 0.5
            dNx = np.linalg.solve(J, dN)
            B = np.zeros((3, 8))
            for a in range(4):
                B[0, 2*a] = dNx[0, a]; B[1, 2*a+1] = dNx[1, a]
                B[2, 2*a] = dNx[1, a]; B[2, 2*a+1] = dNx[0, a]
            K += B.T @ C @ B * np.linalg.det(J)
    return K

# --------------------------------------------------------------- assembly ---
def assemble(N, Efield, nu):
    """Periodic N x N node grid, N x N elements. dof = 2*(ix*N+iy) + d."""
    Ke = element_stiffness(nu)
    K = np.zeros((2*N*N, 2*N*N))
    for ex, ey in itertools.product(range(N), repeat=2):
        g = []
        for a, b in [(ex, ey), (ex+1, ey), (ex+1, ey+1), (ex, ey+1)]:
            a %= N; b %= N
            g += [2*(a*N+b), 2*(a*N+b)+1]
        g = np.array(g)
        K[np.ix_(g, g)] += Efield[ex, ey] * Ke
    return K

# ------------------------------------------------------------ decomposition --
ELEM_OFFSETS = [(0, 0), (-1, 0), (0, -1), (-1, -1)]   # elements touching a node

def decompose(N, chi, E1, E2, nu=0.3, tol=1e-10):
    """Return (lcu_terms, alpha, max_residual)."""
    K = assemble(N, E2 + (E1-E2)*chi, nu)
    basis = [np.ones((N, N))] + [np.roll(chi, (-a, -b), axis=(0, 1))
                                 for a, b in ELEM_OFFSETS]
    Bm = np.stack([b.ravel() for b in basis], axis=1)

    raw, residual = {}, 0.0
    for dx, dy in itertools.product([-1, 0, 1], repeat=2):
        w = {}
        for d1, d2 in itertools.product([0, 1], repeat=2):
            v = np.array([K[2*(ix*N+iy)+d1,
                            2*(((ix+dx) % N)*N + ((iy+dy) % N))+d2]
                          for ix in range(N) for iy in range(N)])
            wi, *_ = np.linalg.lstsq(Bm, v, rcond=None)
            residual = max(residual, np.abs(Bm @ wi - v).max())
            w[(d1, d2)] = wi
        for s in range(len(basis)):
            M = np.array([[w[(0, 0)][s], w[(0, 1)][s]],
                          [w[(1, 0)][s], w[(1, 1)][s]]])
            for p, val in (("I",  (M[0, 0]+M[1, 1])/2),
                           ("Z",  (M[0, 0]-M[1, 1])/2),
                           ("X",  (M[0, 1]+M[1, 0])/2),
                           ("iY", (M[0, 1]-M[1, 0])/2)):
                if abs(val) > tol:
                    raw[(dx, dy, s, p)] = val

    # diag(chi_s) = 1/2 (I - R_s)  ->  a genuine LCU over unitaries
    lcu = {}
    for (dx, dy, s, p), c in raw.items():
        if s == 0:
            lcu[(dx, dy, "I", p)] = lcu.get((dx, dy, "I", p), 0) + c
        else:
            lcu[(dx, dy, "I", p)] = lcu.get((dx, dy, "I", p), 0) + c/2
            key = (dx, dy, f"R{s}", p)
            lcu[key] = lcu.get(key, 0) - c/2
    lcu = {k: v for k, v in lcu.items() if abs(v) > tol}
    return lcu, sum(abs(v) for v in lcu.values()), residual

# -------------------------------------------------------------------- main ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'cell':<16} {'L':>4} {'alpha':>10}  residual")
    cells = [
        ("uniform",       8, np.zeros((8, 8))),
        ("random N=8",    8, rng.integers(0, 2, (8, 8)).astype(float)),
        ("random N=16",  16, rng.integers(0, 2, (16, 16)).astype(float)),
        ("single incl.",  8, np.pad(np.ones((2, 2)), ((3, 3), (3, 3)))),
    ]
    for name, N, chi in cells:
        t, a, r = decompose(N, chi, 3.0, 1.0); L = len(t)
        print(f"{name:<16} {L:>4} {a:>10.4f}  {r:.1e}")

    print("\ncontrast sweep (random N=8 cell):")
    chi = rng.integers(0, 2, (8, 8)).astype(float)
    for E1 in (1.0, 2.0, 3.0, 10.0, 100.0):
        t, a, _ = decompose(8, chi, E1, 1.0); L = len(t)
        K = assemble(8, 1.0 + (E1-1.0)*chi, 0.3)
        n2 = np.linalg.norm(K, 2)
        print(f"  E1/E2={E1:6.1f}:  L={L:3d}  alpha={a:9.4f}  "
              f"alpha/||K||_2={a/n2:.3f}")