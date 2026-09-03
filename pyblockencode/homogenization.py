"""
homogenization.py — Block encoding of the periodic two-phase Q4 elasticity
operator, for computational homogenization of a heterogeneous unit cell.

The cell is N x N elements, N = 2**m, fully periodic in both directions.  The
modulus field is two-phase,

    E_e = E2 + (E1 - E2) chi_e,        chi_e in {0, 1},

with chi the indicator of an inclusion.  Writing the assembly node by node,
node i = (ix, iy) touches four elements at offsets

    s = 1, 2, 3, 4  ->  (a, b) = (0,0), (-1,0), (0,-1), (-1,-1)

in which i sits at local node 0, 1, 3, 2 (element (ex,ey) carries local nodes
(ex,ey), (ex+1,ey), (ex+1,ey+1), (ex,ey+1)).  Node j = i + (dx,dy) belongs to
element s iff (dx-a, dy-b) is in {0,1}^2, at the local node with that offset.
Hence

    K[(i,d1),(j,d2)] = sum_s E_{i+(a,b)} T_s(dx,dy,d1,d2),
    T_s(dx,dy,d1,d2) = Ke[2 loc_i(s) + d1, 2 loc_j(s) + d2],

zero when j is not a node of element s.  The modulus field enters only
through the diagonals of the four shifted indicators.  With
diag(chi_s) = (I - R_s)/2 and R_s = I - 2 diag(chi_s) a phase reflection,

    K = sum_{dx,dy,r} ( c^I_{dx,dy,r} I + sum_s c^s_{dx,dy,r} R_s )
                      S_x^dx (x) S_y^dy (x) sigma_r

    c^I = (E1+E2)/2 * sum_s T_s        c^s = -(E1-E2)/2 * T_s

over cyclic shifts S in {I, S_c, S_c^dag} per direction and sigma in
{I, Z, X, iY} on the DOF qubit.  Both coefficient families are entries of the
element stiffness, so neither the term count nor the subnormalization can
depend on the microstructure:

    L     = 57                       every chi, every N
    alpha = min(E1,E2) A(nu) + |E1-E2| B(nu)
    A(nu) = (33 + nu) / (6 (1 - nu^2))
    B(nu) = (69 + 5 nu + 6 |1 - 3 nu|) / (12 (1 - nu^2))

At zero contrast the reflections degenerate to the identity and the minimal
count is the 17 terms of the homogeneous periodic operator.

All four R_s are the single oracle R_chi conjugated by shifts,
R_s = S^(a,b) R_chi S^(-a,-b), so the construction costs one oracle query per
SELECT branch regardless of the microstructure.

Qubits: 2m + 1 system (x, y, DOF) + 6 PREP ancilla = 2m + 7.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, Tuple

import numpy as np

# --------------------------------------------------------------------------
# geometry of the node-element incidence
# --------------------------------------------------------------------------

#: element offsets (a,b) around a node, in the order s = 1..4
ELEM_OFFSETS: Tuple[Tuple[int, int], ...] = ((0, 0), (-1, 0), (0, -1), (-1, -1))

#: local node index of a node sitting at offset -(a,b) inside element (a,b)
_LOC: Dict[Tuple[int, int], int] = {(0, 0): 0, (-1, 0): 1, (0, -1): 3,
                                    (-1, -1): 2}

#: DOF-qubit operators
PAULI: Dict[str, np.ndarray] = {
    "I": np.eye(2),
    "Z": np.diag([1.0, -1.0]),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "iY": np.array([[0.0, 1.0], [-1.0, 0.0]]),
}

#: spatial shift labels, matching bc.py.  A stencil offset dx (the operator
#: couples node ix to node ix+dx) is carried by the shift that moves amplitude
#: the other way, so dx = +1 is S_c^dagger.
_SHIFT_LABEL = {0: "I", 1: "Scd", -1: "Sc"}


def element_stiffness(nu: float, E: float = 1.0, h: float = 1.0) -> np.ndarray:
    """Plane-stress Q4 element stiffness by 2x2 Gauss quadrature."""
    D = E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0],
                                      [0, 0, (1 - nu) / 2]])
    g = 1 / np.sqrt(3)
    Ke = np.zeros((8, 8))
    for xi, eta in [(-g, -g), (g, -g), (g, g), (-g, g)]:
        dN = np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                       [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]]) / 4.0
        J = h / 2.0
        dNxy = dN / J
        B = np.zeros((3, 8))
        for a in range(4):
            B[0, 2 * a] = dNxy[0, a]
            B[1, 2 * a + 1] = dNxy[1, a]
            B[2, 2 * a] = dNxy[1, a]
            B[2, 2 * a + 1] = dNxy[0, a]
        Ke += B.T @ D @ B * (J * J)
    return Ke


# --------------------------------------------------------------------------
# inclusion geometry
# --------------------------------------------------------------------------

class Inclusion:
    """
    An inclusion in an N x N periodic cell of elements.

    Parameters
    ----------
    shape : {'square', 'rectangle', 'cross'}
    N     : int
        Elements per direction.
    vf    : float
        Target volume fraction.  The realized fraction is quantized to the
        mesh and is reported as ``volume_fraction``.
    aspect : float
        Width-to-height ratio, for 'rectangle' only.
    origin : pair of int
        Lower-left element of the inclusion bounding box.  Defaults to the
        placement that centres it; the operator does not depend on this.

    Attributes
    ----------
    chi      : (N,N) float array of 0/1
    dyadic   : True when the oracle is a plain multi-controlled Z, i.e. the
               inclusion is a square of dyadic side aligned to a dyadic
               position, so membership is a conjunction of bit equalities.
    """

    def __init__(self, shape: str = "square", N: int = 8, vf: float = 0.25,
                 aspect: float = 2.0, origin: Tuple[int, int] | None = None):
        if not 0.0 <= vf <= 1.0:
            raise ValueError("volume fraction must lie in [0,1]")
        self.shape, self.N, self.vf_target, self.aspect = shape, N, vf, aspect

        if shape == "square":
            side = int(round(N * math.sqrt(vf)))
            side = max(0, min(N, side))
            self.extent = (side, side)
        elif shape == "rectangle":
            # a x b with a/b = aspect and a*b = vf N^2
            b = math.sqrt(vf * N * N / aspect)
            self.extent = (max(0, min(N, int(round(aspect * b)))),
                           max(0, min(N, int(round(b)))))
        elif shape == "cross":
            # two orthogonal bars of width w through the cell:
            # area = 2 w N - w^2  ->  w = N - sqrt(N^2 - vf N^2)
            w = int(round(N * (1 - math.sqrt(max(0.0, 1 - vf)))))
            self.extent = (max(0, min(N, w)), max(0, min(N, w)))
        else:
            raise ValueError(f"unknown shape {shape!r}")

        self.origin = origin if origin is not None else self._default_origin()
        self.chi = self._build()

    def _default_origin(self) -> Tuple[int, int]:
        """
        Centre the inclusion, but snap a dyadic square to a dyadic position.

        The operator does not depend on where the inclusion sits, so the
        snap is free and it is what makes R_chi a plain multi-controlled Z.
        """
        N, (ax, ay) = self.N, self.extent
        if self.shape == "square" and ax and not (ax & (ax - 1)):
            o = ax * ((N // ax) // 2)
            return (o, o)
        return ((N - ax) // 2, (N - ay) // 2)

    def _build(self) -> np.ndarray:
        N = self.N
        chi = np.zeros((N, N))
        ax, ay = self.extent
        ox, oy = self.origin
        if self.shape == "cross":
            if ax:
                chi[:, oy:oy + ay] = 1.0          # horizontal bar, full width
                chi[ox:ox + ax, :] = 1.0          # vertical bar, full height
            return chi
        for i in range(ax):
            for j in range(ay):
                chi[(ox + i) % N, (oy + j) % N] = 1.0
        return chi

    @property
    def volume_fraction(self) -> float:
        return float(self.chi.mean())

    @property
    def dyadic(self) -> bool:
        """True when R_chi is a multi-controlled Z with no arithmetic."""
        if self.shape != "square":
            return False
        side = self.extent[0]
        if side == 0 or side & (side - 1):        # not a power of two
            return False
        return self.origin[0] % side == 0 and self.origin[1] % side == 0

    @property
    def control_bits(self) -> int:
        """Controls on the dyadic multi-controlled Z: 2 (m - log2 side)."""
        if not self.dyadic:
            raise ValueError("control_bits is defined for dyadic squares only")
        m = int(round(math.log2(self.N)))
        j = int(round(math.log2(self.extent[0]))) if self.extent[0] else m
        return 2 * (m - j)

    def __repr__(self) -> str:
        return (f"Inclusion({self.shape!r}, N={self.N}, "
                f"extent={self.extent}, origin={self.origin}, "
                f"vf={self.volume_fraction:.4f}"
                + (", dyadic" if self.dyadic else "") + ")")


# --------------------------------------------------------------------------
# the encoding
# --------------------------------------------------------------------------

def _pauli_split(M: np.ndarray, tol: float = 1e-12) -> Dict[str, float]:
    """2x2 real DOF block -> {I, Z, X, iY} coefficients above tol."""
    return {p: v for p, v in (("I", (M[0, 0] + M[1, 1]) / 2),
                              ("Z", (M[0, 0] - M[1, 1]) / 2),
                              ("X", (M[0, 1] + M[1, 0]) / 2),
                              ("iY", (M[0, 1] - M[1, 0]) / 2))
            if abs(v) > tol}


class HomogenizationEncoding:
    """
    Shift decomposition of the periodic two-phase plane-stress Q4 operator.

    Parameters
    ----------
    m  : int
        Qubits per spatial direction; N = 2**m elements and nodes per side.
    E1 : float
        Inclusion modulus (where chi = 1).
    E2 : float
        Matrix modulus (where chi = 0).
    nu : float
        Poisson's ratio, shared by both phases.
    inclusion : Inclusion or (N,N) array
        The microstructure.  An array is taken as chi directly.

    Attributes
    ----------
    lcu_terms()  dict {(U_x, U_y, R_label, sigma): coefficient}
    alpha        subnormalization, independent of the microstructure
    num_terms    57 at nonzero contrast, 17 at zero contrast
    num_system   2m + 1
    num_ancilla  ceil(log2 L)
    """

    def __init__(self, m: int, E1: float = 3.0, E2: float = 1.0,
                 nu: float = 0.3,
                 inclusion: "Inclusion | np.ndarray | None" = None,
                 shape: str = "square", vf: float = 0.25):
        self.m, self.N = m, 2 ** m
        self.E1, self.E2, self.nu = float(E1), float(E2), float(nu)
        if inclusion is None:
            inclusion = Inclusion(shape, self.N, vf)
        if isinstance(inclusion, Inclusion):
            if inclusion.N != self.N:
                raise ValueError("inclusion resolution does not match m")
            self.inclusion, self.chi = inclusion, inclusion.chi
        else:
            chi = np.asarray(inclusion, dtype=float)
            if chi.shape != (self.N, self.N):
                raise ValueError(f"chi must be {(self.N, self.N)}")
            self.inclusion, self.chi = None, chi
        self._terms: Dict[Tuple[str, str, str, str], float] | None = None

    # -- the T matrices -----------------------------------------------------

    def T_matrices(self) -> list:
        """T[s][(dx,dy)] = the 2x2 DOF block of Ke, per element offset s."""
        Ke = element_stiffness(self.nu)
        T = []
        for (a, b) in ELEM_OFFSETS:
            li = _LOC[(a, b)]
            d = {}
            for dx, dy in itertools.product([-1, 0, 1], repeat=2):
                if (a - dx, b - dy) not in _LOC:
                    continue
                lj = _LOC[(a - dx, b - dy)]
                d[(dx, dy)] = Ke[np.ix_([2 * li, 2 * li + 1],
                                        [2 * lj, 2 * lj + 1])]
            T.append(d)
        return T

    # -- LCU ----------------------------------------------------------------

    def lcu_terms(self, tol: float = 1e-12) -> Dict[Tuple[str, str, str, str],
                                                    float]:
        """{(x label, y label, 'I' or 'R1'..'R4', Pauli): coefficient}."""
        if self._terms is not None:
            return self._terms
        T = self.T_matrices()
        cI, cR = (self.E1 + self.E2) / 2, -(self.E1 - self.E2) / 2
        t: Dict[Tuple[str, str, str, str], float] = {}
        for dx, dy in itertools.product([-1, 0, 1], repeat=2):
            xl, yl = _SHIFT_LABEL[dx], _SHIFT_LABEL[dy]
            total = sum((d[(dx, dy)] for d in T if (dx, dy) in d),
                        np.zeros((2, 2)))
            for p, v in _pauli_split(cI * total, tol).items():
                t[(xl, yl, "I", p)] = t.get((xl, yl, "I", p), 0.0) + v
            for s, d in enumerate(T):
                if (dx, dy) not in d:
                    continue
                for p, v in _pauli_split(cR * d[(dx, dy)], tol).items():
                    t[(xl, yl, f"R{s + 1}", p)] = v
        self._terms = {k: v for k, v in t.items() if abs(v) > tol}
        return self._terms

    @property
    def alpha(self) -> float:
        return float(sum(abs(v) for v in self.lcu_terms().values()))

    def alpha_closed_form(self) -> float:
        """min(E1,E2) A(nu) + |E1-E2| B(nu)."""
        nu = self.nu
        A = (33 + nu) / (6 * (1 - nu ** 2))
        B = (69 + 5 * nu + 6 * abs(1 - 3 * nu)) / (12 * (1 - nu ** 2))
        return min(self.E1, self.E2) * A + abs(self.E1 - self.E2) * B

    @property
    def num_terms(self) -> int:
        return len(self.lcu_terms())

    @property
    def components(self) -> Tuple[str, ...]:
        present = {k[3] for k in self.lcu_terms()}
        return tuple(s for s in ("I", "Z", "X", "iY") if s in present)

    @property
    def num_system(self) -> int:
        return 2 * self.m + 1

    @property
    def num_ancilla(self) -> int:
        return math.ceil(math.log2(max(self.num_terms, 2)))

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    @property
    def volume_fraction(self) -> float:
        return float(self.chi.mean())

    @property
    def contrast(self) -> float:
        return self.E1 / self.E2

    # -- dense realizations -------------------------------------------------

    def reflection(self, s: int) -> np.ndarray:
        """R_s as a diagonal +-1 matrix on the N^2 node index."""
        a, b = ELEM_OFFSETS[s - 1]
        return np.diag(1.0 - 2.0 * np.roll(self.chi, (-a, -b),
                                           axis=(0, 1)).ravel())

    def _spatial(self, label: str) -> np.ndarray:
        """Dense matrix of one shift label, as ``bc.unitary`` defines it."""
        from . import bc as _bc
        return _bc.unitary(label, self.N)

    def dense(self) -> np.ndarray:
        """The operator implied by the LCU, assembled term by term."""
        N = self.N
        K = np.zeros((2 * N * N, 2 * N * N))
        eye = np.eye(N * N)
        for (xl, yl, rl, p), c in self.lcu_terms().items():
            U = np.kron(self._spatial(xl), self._spatial(yl))
            R = eye if rl == "I" else self.reflection(int(rl[1]))
            K += c * np.kron(R @ U, PAULI[p])
        return K

    def target(self) -> np.ndarray:
        """Direct periodic Q4 assembly with this two-phase modulus field."""
        N = self.N
        Ke = element_stiffness(self.nu)
        E = self.E2 + (self.E1 - self.E2) * self.chi
        K = np.zeros((2 * N * N, 2 * N * N))
        for ex, ey in itertools.product(range(N), repeat=2):
            g = []
            for a, b in [(ex, ey), (ex + 1, ey), (ex + 1, ey + 1), (ex, ey + 1)]:
                base = 2 * ((a % N) * N + (b % N))
                g += [base, base + 1]
            g = np.array(g)
            K[np.ix_(g, g)] += E[ex, ey] * Ke
        return K

    # -- verification -------------------------------------------------------

    def verify(self) -> dict:
        """LCU against a direct assembly, plus the closed form for alpha."""
        target = self.target()
        nrm = np.linalg.norm(target)
        out = {
            "m": self.m, "N": self.N, "nu": self.nu,
            "E1": self.E1, "E2": self.E2, "contrast": self.contrast,
            "volume_fraction": self.volume_fraction,
            "num_terms": self.num_terms, "components": self.components,
            "alpha": self.alpha, "num_qubits": self.num_qubits,
            "decomposition_rel_err": float(
                np.linalg.norm(self.dense() - target) / nrm),
            "alpha_closed_form_err": float(
                abs(self.alpha - self.alpha_closed_form())),
            "tightness": float(self.alpha / np.linalg.norm(target, 2)),
        }
        return out
