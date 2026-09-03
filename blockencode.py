"""
blockencode -- block encodings of periodic finite-element operators.

Four operators, periodic boundary conditions only:

    'poisson1d'      K = circ(-1,2,-1)                       L = 3   alpha = 4
    'poisson2d_fd'   K1 (x) I + I (x) K1                     L = 5   alpha = 8
    'poisson2d_fe'   K1 (x) M1 + M1 (x) K1                   L = 9   alpha = 16/3
    'elasticity2d'   plane-stress Q4, 2 dof per node         L = 17  alpha = A(nu)
    'poisson2d_2phase'     two-phase scalar Q4 cell
    'elasticity2d_2phase'  two-phase plane-stress Q4 cell      L = 57

The two-phase operators carry a per-element modulus E = E2 + (E1-E2) chi with
chi in {0,1}. The material enters only through the four reflections
R_s = S^(a,b) R_chi S^(-a,-b), R_chi = I - 2 diag(chi), so L and alpha do not
depend on m, on the microstructure, or on the volume fraction. Only the oracle
R_chi knows the geometry, and for a dyadic square inclusion it is a single
multi-controlled Z whose cost is flat in m.

with K1 = circ(-1,2,-1), M1 = (1/6) circ(1,4,1), G1 = (1/2) circ(-1,0,1),
and A(nu) = (33 + nu) / (6 (1 - nu^2)) for E = 1.

Usage mirrors PyEncode:

    from blockencode import blockencode, POISSON1D, POISSON2D, ELASTICITY2D
    circuit, info = blockencode(ELASTICITY2D(nu=0.3), N=4096)
    print(info)

The operator is a small parameterised object, as a pattern is in PyEncode:
POISSON1D(), POISSON2D('fe'), POISSON2D('fd'), ELASTICITY2D(nu=0.3, E=1.0).
Material parameters live on the operator, not on the call, so they cannot be
passed where they are meaningless. Plain strings are accepted too, for
scripting: 'poisson1d', 'poisson2d_fd', 'poisson2d_fe', 'elasticity2d'.

N is the number of grid points PER DIRECTION and must be a power of two, so
m = log2(N) qubits per direction. The operator itself is larger: N dofs in 1D,
N^2 in 2D, and 2 N^2 for elasticity, which carries two dofs per node. Pass
either N or m, not both.

Nothing is materialised unless you ask. `blockencode(...)` builds the circuit
and reports L, alpha, qubit counts and gate counts without ever forming a
matrix. `materialize=True` additionally assembles K densely and checks the
circuit against it, before and after transpilation.

Design
------
LCU with a FACTORISED select. Every term is a product
S_x^{dx} (x) S_y^{dy} (x) sigma, with dx, dy in {I, S, S^dag} and sigma in
{I, Z, X}, so SELECT is a product of independent multiplexers, one per
register, rather than L separately controlled blocks. The circuit therefore
contains FOUR controlled shifts in 2D regardless of L, not 2L.

Signs are carried by using two different preparations,

    U = PREP_R^dag . SELECT . PREP_L,
    PREP_L |0> = sum_i sqrt(|c_i|/alpha) |i>,
    PREP_R |0> = sum_i sign(c_i) sqrt(|c_i|/alpha) |i>,

whose top-left block is exactly sum_i c_i U_i / alpha = K / alpha. No signs
appear inside SELECT.

Shifts are ripple-carry increments with clean ancillas: 2m-2 Toffoli, m CX,
m-1 ancillas, reused by every shift in the circuit.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import ClassVar

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.synthesis import synth_mcx_noaux_v24

BASIS = ["cx", "u"]
TOL = 1e-10

# shift codes: 0 -> I, 1 -> S (increment), 2 -> S^dag (decrement)
I_, S_, SD_ = 0, 1, 2
# dof codes: 0 -> I, 1 -> Z, 2 -> X, 3 -> iY
DI_, DZ_, DX_, DIY_ = 0, 1, 2, 3

_KIND_DIM = {"poisson1d": 1, "poisson2d_fd": 2, "poisson2d_fe": 2,
             "elasticity2d": 2}


# ==========================================================================
# 1. term algebra  (no matrices)
# ==========================================================================
def _c_K1() -> dict[int, float]:
    """K1 = circ(-1,2,-1) = 2I - S - S^dag."""
    return {I_: 2.0, S_: -1.0, SD_: -1.0}


def _c_M1() -> dict[int, float]:
    """M1 = (1/6) circ(1,4,1) = (4I + S + S^dag)/6."""
    return {I_: 4 / 6, S_: 1 / 6, SD_: 1 / 6}


def _c_G1() -> dict[int, float]:
    """G1 = (1/2) circ(-1,0,1) = (S^dag - S)/2."""
    return {S_: -0.5, SD_: 0.5}


def _outer(ca: dict, cb: dict) -> dict:
    """Coefficients of A (x) B indexed by (a, b)."""
    return {(a, b): va * vb for a, va in ca.items() for b, vb in cb.items()}


def _add(dst: dict, src: dict, scale: float = 1.0) -> dict:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + scale * v
    return dst


def _prune(d: dict, tol: float = 1e-13) -> dict:
    return {k: v for k, v in d.items() if abs(v) > tol}


def terms(kind: str, nu: float = 0.3, E: float = 1.0) -> dict[tuple, float]:
    """LCU coefficients, keyed by (dx,) or (dx, dy) or (dx, dy, sigma).

    Independent of m: the decomposition is the same at every problem size.
    """
    K, M, G = _c_K1(), _c_M1(), _c_G1()

    if kind == "poisson1d":
        return _prune({(a,): v for a, v in K.items()})

    if kind == "poisson2d_fd":                     # K1 (x) I + I (x) K1
        t: dict = {}
        _add(t, _outer(K, {I_: 1.0}))
        _add(t, _outer({I_: 1.0}, K))
        return _prune(t)

    if kind == "poisson2d_fe":                     # K1 (x) M1 + M1 (x) K1
        t = {}
        _add(t, _outer(K, M))
        _add(t, _outer(M, K))
        return _prune(t)

    if kind == "elasticity2d":
        # Kxx = C ( K1(x)M1 + (1-nu)/2 M1(x)K1 )
        # Kyy = C ( (1-nu)/2 K1(x)M1 + M1(x)K1 )
        # Kxy = Kyx = -C (1+nu)/2  G1(x)G1
        # dof qubit: |0> = x, |1> = y, so with Kxy = Kyx the operator is
        #   K = I (x) (Kxx+Kyy)/2 + Z (x) (Kxx-Kyy)/2 + X (x) Kxy
        # and the iY component is absent under periodicity.
        C = E / (1.0 - nu ** 2)
        KM, MK = _outer(K, M), _outer(M, K)

        sym: dict = {}                              # (Kxx+Kyy)/2
        _add(sym, KM, C * (3 - nu) / 4)
        _add(sym, MK, C * (3 - nu) / 4)

        dif: dict = {}                              # (Kxx-Kyy)/2
        _add(dif, KM, C * (1 + nu) / 4)
        _add(dif, MK, -C * (1 + nu) / 4)

        shr: dict = {}                              # Kxy
        _add(shr, _outer(G, G), -C * (1 + nu) / 2)

        t = {}
        for comp, blk in ((DI_, sym), (DZ_, dif), (DX_, shr)):
            for (a, b), v in _prune(blk).items():
                t[(a, b, comp)] = v
        return _prune(t)

    raise ValueError(f"unknown kind {kind!r}; choose from {sorted(_KIND_DIM)}")


def alpha_closed_form(kind: str, nu: float = 0.3, E: float = 1.0) -> float:
    """Analytic subnormalization, for cross-checking sum |c_k|."""
    if kind == "poisson1d":
        return 4.0
    if kind == "poisson2d_fd":
        return 8.0
    if kind == "poisson2d_fe":
        return 16 / 3
    if kind == "elasticity2d":
        return E * (33 + nu) / (6 * (1 - nu ** 2))
    raise ValueError(kind)


# --------------------------------------------------------------------------
# operator specifications  (parameterised, like a PyEncode pattern)
# --------------------------------------------------------------------------
class _Operator:
    """Base for the operator specs. Size-independent: no m, no N."""
    kind: ClassVar[str]
    dim: ClassVar[int]
    has_dof: ClassVar[bool]
    has_refl: ClassVar[bool] = False
    nu: float
    E: float

    def terms(self) -> dict[tuple, float]:
        """LCU coefficients. Same at every problem size."""
        return terms(self.kind, self.nu, self.E)

    def alpha(self) -> float:
        """Analytic subnormalization sum |c_k|."""
        return alpha_closed_form(self.kind, self.nu, self.E)


@dataclass(frozen=True)
class POISSON1D(_Operator):
    """K = circ(-1, 2, -1) = 2I - S - S^dag.      L = 3,  alpha = 4."""
    kind: ClassVar[str] = "poisson1d"
    dim: ClassVar[int] = 1
    has_dof: ClassVar[bool] = False
    nu: ClassVar[float] = 0.0
    E: ClassVar[float] = 1.0


@dataclass(frozen=True)
class POISSON2D(_Operator):
    """disc='fe': K1(x)M1 + M1(x)K1,  L = 9,  alpha = 16/3.
    disc='fd': K1(x)I + I(x)K1,       L = 5,  alpha = 8."""
    disc: str = "fe"
    dim: ClassVar[int] = 2
    has_dof: ClassVar[bool] = False
    nu: ClassVar[float] = 0.0
    E: ClassVar[float] = 1.0

    def __post_init__(self):
        if self.disc not in ("fe", "fd"):
            raise ValueError(f"disc must be 'fe' or 'fd', got {self.disc!r}")

    @property
    def kind(self) -> str:                       # type: ignore[override]
        return f"poisson2d_{self.disc}"


@dataclass(frozen=True)
class ELASTICITY2D(_Operator):
    """Periodic plane-stress Q4, two dofs per node.
    L = 17,  alpha = E (33 + nu) / (6 (1 - nu^2))."""
    nu: float = 0.3
    E: float = 1.0
    kind: ClassVar[str] = "elasticity2d"
    dim: ClassVar[int] = 2
    has_dof: ClassVar[bool] = True

    def __post_init__(self):
        if not -1.0 < self.nu < 0.5:
            raise ValueError(f"nu must lie in (-1, 0.5), got {self.nu}")


@dataclass(frozen=True)
class POISSON2D_2PHASE(_Operator):
    """Two-phase scalar Q4 cell, modulus E2 in the matrix and E1 in a dyadic
    square inclusion of volume fraction vf."""
    vf: float = 0.25
    E1: float = 10.0
    E2: float = 1.0
    dim: ClassVar[int] = 2
    has_dof: ClassVar[bool] = False
    has_refl: ClassVar[bool] = True
    kind: ClassVar[str] = "poisson2d_2phase"
    nu: ClassVar[float] = 0.0

    def __post_init__(self):
        _check_vf(self.vf)

    @property
    def E(self) -> float:                        # type: ignore[override]
        return self.E1

    def element(self) -> np.ndarray:
        return _q4_scalar_element()

    def terms(self) -> dict:
        return element_terms(self.element(), 1, self.E1, self.E2, True)

    def chi(self, m: int) -> np.ndarray:
        return _chi_square(self.vf, m)

    def alpha(self) -> float:
        return element_alpha(self.element(), 1, self.E1, self.E2)


@dataclass(frozen=True)
class ELASTICITY2D_2PHASE(_Operator):
    """Two-phase periodic plane-stress Q4 cell. L = 57 at nonzero contrast."""
    nu: float = 0.3
    vf: float = 0.25
    E1: float = 10.0
    E2: float = 1.0
    dim: ClassVar[int] = 2
    has_dof: ClassVar[bool] = True
    has_refl: ClassVar[bool] = True
    kind: ClassVar[str] = "elasticity2d_2phase"

    def __post_init__(self):
        if not -1.0 < self.nu < 0.5:
            raise ValueError(f"nu must lie in (-1, 0.5), got {self.nu}")
        _check_vf(self.vf)

    @property
    def E(self) -> float:                        # type: ignore[override]
        return self.E1

    def element(self) -> np.ndarray:
        return _q4_element(self.nu, 1.0)

    def terms(self) -> dict:
        return element_terms(self.element(), 2, self.E1, self.E2, True)

    def chi(self, m: int) -> np.ndarray:
        return _chi_square(self.vf, m)

    def alpha(self) -> float:
        return element_alpha(self.element(), 2, self.E1, self.E2)

    def alpha_published(self) -> float:
        """Independent closed form, for cross-checking element_alpha."""
        nu = self.nu
        A = (33 + nu) / (6 * (1 - nu ** 2))
        B = (69 + 5 * nu + 6 * abs(1 - 3 * nu)) / (12 * (1 - nu ** 2))
        return min(self.E1, self.E2) * A + abs(self.E1 - self.E2) * B


def _inclusion_bits(vf: float) -> int:
    """k such that the inclusion side is 2^(m-k); vf = 4**-k."""
    return int(round(-np.log(vf) / np.log(4.0)))


def _chi_square(vf: float, m: int) -> np.ndarray:
    """Element indicator of a dyadic square at the origin. chi[ex, ey]."""
    k = _inclusion_bits(vf)
    if m < k:
        raise ValueError(f"vf = {vf} needs m >= {k}, got m = {m}")
    N, side = 2 ** m, 2 ** (m - k)
    c = np.zeros((N, N))
    c[:side, :side] = 1.0
    return c


def _check_vf(vf: float) -> None:
    """The inclusion must be a dyadic square: vf = 4^-k for integer k >= 1."""
    k = -np.log(vf) / np.log(4.0)
    if vf <= 0 or vf >= 1 or abs(k - round(k)) > 1e-12:
        raise ValueError(f"vf must be 4**-k for an integer k >= 1, got {vf}")


OPERATORS = {
    "poisson1d": POISSON1D,
    "poisson2d_2phase": POISSON2D_2PHASE,
    "elasticity2d_2phase": ELASTICITY2D_2PHASE,
    "poisson2d_fe": lambda: POISSON2D("fe"),
    "poisson2d_fd": lambda: POISSON2D("fd"),
    "elasticity2d": ELASTICITY2D,
}


def _as_operator(spec) -> _Operator:
    if isinstance(spec, _Operator):
        return spec
    if isinstance(spec, str):
        if spec not in OPERATORS:
            raise ValueError(
                f"unknown operator {spec!r}; use one of {sorted(OPERATORS)}, "
                f"or POISSON1D() / POISSON2D('fe') / ELASTICITY2D(nu=...)")
        return OPERATORS[spec]()
    raise TypeError(f"expected an operator spec or a name, got {type(spec).__name__}")

# --------------------------------------------------------------------------
# element geometry and the two-phase term engine
# --------------------------------------------------------------------------
# A node touches four elements. ELEM_OFFSETS[s] is the position of element s's
# lower-left corner relative to the node; the node sits at CORNERS.index(-off).
ELEM_OFFSETS = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]      # local nodes 0..3, ccw

# reflection selector: 3 one-bit fields g (apply R_chi), a (x offset), b (y)
R_NONE = 0


def _shift_code(d: int) -> int:
    """Displacement d maps to the operator S^{-d}, since K[i, i+d] means
    a matrix with ones at (i, i+d), which is S^{-d}."""
    return I_ if d == 0 else (S_ if d == -1 else SD_)


def _t_blocks(Ke: np.ndarray, nd: int) -> dict:
    """T[(s, dx, dy)] : the nd x nd block coupling a node to the node at
    (dx, dy), contributed by element s, at unit modulus."""
    T = {}
    for s, (a, b) in enumerate(ELEM_OFFSETS):
        li = CORNERS.index((-a, -b))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = (dx - a, dy - b)
                if c not in CORNERS:
                    continue
                lj = CORNERS.index(c)
                T[(s, dx, dy)] = Ke[nd * li:nd * li + nd,
                                    nd * lj:nd * lj + nd].copy()
    return T


def _sigma_split(M: np.ndarray) -> dict[int, float]:
    """Decompose a 2x2 real block over the dof qubit into {I, Z, X, iY}."""
    return {DI_: (M[0, 0] + M[1, 1]) / 2, DZ_: (M[0, 0] - M[1, 1]) / 2,
            DX_: (M[0, 1] + M[1, 0]) / 2, DIY_: (M[0, 1] - M[1, 0]) / 2}


def element_terms(Ke: np.ndarray, nd: int, E1: float, E2: float,
                  two_phase: bool) -> dict[tuple, float]:
    """LCU coefficients straight from the element matrix.

    Key layout: (kx, ky[, sigma][, g, a, b]).  Ke must be at unit modulus.
    With E = E2 + (E1-E2) chi and diag(chi_s) = (I - R_s)/2,

        diag(E_s) = (E1+E2)/2 I  -  (E1-E2)/2 R_s

    so the identity part sums over the four elements and each reflection part
    keeps its own s. Neither references chi, which is why L and alpha are
    independent of the microstructure.
    """
    T = _t_blocks(Ke, nd)
    cI = (E1 + E2) / 2 if two_phase else E1
    cR = -(E1 - E2) / 2

    acc: dict[tuple, np.ndarray] = {}
    for (s, dx, dy), blk in T.items():
        base = (_shift_code(dx), _shift_code(dy))
        keys = [(base, R_NONE, cI)]
        if two_phase:
            keys.append((base, s + 1, cR))
        for bk, refl, coeff in keys:
            k = (bk, refl)
            if k not in acc:
                acc[k] = np.zeros((nd, nd))
            acc[k] = acc[k] + coeff * blk

    out: dict[tuple, float] = {}
    for (bk, refl), M in acc.items():
        tail: tuple = ()
        if two_phase:
            a, b = ELEM_OFFSETS[refl - 1] if refl else (0, 0)
            tail = (1 if refl else 0, 1 if a else 0, 1 if b else 0)
        if nd == 1:
            v = float(M[0, 0])
            if abs(v) > 1e-13:
                out[bk + tail] = v
        else:
            for sig, v in _sigma_split(M).items():
                if abs(v) > 1e-13:
                    out[bk + (sig,) + tail] = float(v)
    return out


def element_alpha(Ke: np.ndarray, nd: int, E1: float, E2: float) -> float:
    """Subnormalization from the element matrix alone.

        alpha = min(E1,E2) A + |E1-E2| B,   B = A/2 + (1/2) sum_s |T_s|

    A is the homogeneous subnormalization (the four elements summed before
    taking absolute values); the sum_s |T_s| term is the price of contrast,
    each element counted separately. Neither depends on m, on chi, or on the
    volume fraction, so neither does alpha.
    """
    A = sum(abs(v) for v in element_terms(Ke, nd, 1.0, 1.0, False).values())
    T = _t_blocks(Ke, nd)
    Bs = 0.0
    for s in range(4):
        for (ss, dx, dy), blk in T.items():
            if ss != s:
                continue
            Bs += (abs(float(blk[0, 0])) if nd == 1
                   else sum(abs(v) for v in _sigma_split(blk).values()))
    return min(E1, E2) * A + abs(E1 - E2) * (A / 2 + Bs / 2)


# --------------------------------------------------------------------------
# multi-controlled helpers, explicitly synthesised
# --------------------------------------------------------------------------
def _mcx(qc, controls, target) -> None:
    k = len(controls)
    if k == 0:
        qc.x(target)
    elif k == 1:
        qc.cx(controls[0], target)
    elif k == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        qc.compose(synth_mcx_noaux_v24(k), qubits=list(controls) + [target],
                   inplace=True)


def _mcz(qc, qubits) -> None:
    """Phase flip on the all-ones state of `qubits`."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    qc.h(qubits[-1])
    _mcx(qc, qubits[:-1], qubits[-1])
    qc.h(qubits[-1])


# ==========================================================================
# 2. ripple-carry increment with clean ancillas
# ==========================================================================
def _increment(m: int, inverse: bool = False) -> QuantumCircuit:
    """Controlled cyclic increment. 2m-2 Toffoli, m CX, m-1 clean ancillas.

    Qubit order of the returned circuit: [ctrl] + x[0..m-1] + carry[0..m-2],
    with x[0] the least significant bit.
    """
    c = QuantumRegister(1, "c")
    x = QuantumRegister(m, "x")
    if m == 1:
        qc = QuantumCircuit(c, x, name="inc")
        qc.cx(c[0], x[0])
        return qc.inverse() if inverse else qc

    a = QuantumRegister(m - 1, "a")
    qc = QuantumCircuit(c, x, a, name="inc")
    qc.ccx(c[0], x[0], a[0])                        # carry into bit 1
    for k in range(2, m):
        qc.ccx(a[k - 2], x[k - 1], a[k - 1])        # carry into bit k
    for k in range(m - 1, 0, -1):
        qc.cx(a[k - 1], x[k])                       # flip bit k
        if k >= 2:                                  # uncompute that carry
            qc.ccx(a[k - 2], x[k - 1], a[k - 1])
        else:
            qc.ccx(c[0], x[0], a[0])
    qc.cx(c[0], x[0])                               # bit 0 always flips
    return qc.inverse() if inverse else qc


# ==========================================================================
# 3. the block encoding
# ==========================================================================
class PeriodicBlockEncoding:
    """Block encoding of a periodic FE operator. Builds no matrix by default."""

    def __init__(self, operator, m: int):
        if m < 1:
            raise ValueError("m must be >= 1")
        self.operator = _as_operator(operator)
        self.kind = self.operator.kind
        self.m, self.nu, self.E = m, self.operator.nu, self.operator.E
        self.dim = self.operator.dim
        self.has_dof = self.operator.has_dof
        self.has_refl = getattr(self.operator, "has_refl", False)

        self.terms = self.operator.terms()
        self.L = len(self.terms)
        self.alpha = float(sum(abs(v) for v in self.terms.values()))

        # selector fields, in key order. Two bits per shift or dof field; the
        # reflection is three one-bit fields, so its controls need no MCX.
        self.fields = [2] * self.dim
        if self.has_dof:
            self.fields.append(2)
        if self.has_refl:
            self.fields += [1, 1, 1]             # g (apply R_chi), a, b
        self.offsets = np.cumsum([0] + self.fields[:-1]).tolist()
        self.n_prep = sum(self.fields)
        self.n_system = self.dim * m + (1 if self.has_dof else 0)
        self.n_carry = max(m - 1, 0)
        self.num_ancilla = self.n_prep + 1 + self.n_carry     # +1 select scratch
        self.num_qubits = self.n_system + self.num_ancilla

    # -- reporting ---------------------------------------------------------
    def info(self) -> dict:
        return dict(kind=self.kind, m=self.m, N=2 ** self.m,
                    dofs=2 ** self.n_system, L=self.L, alpha=self.alpha,
                    alpha_closed=self.operator.alpha(),
                    system=self.n_system, prep=self.n_prep,
                    carry=self.n_carry, total_qubits=self.num_qubits)

    def _index(self, key: tuple) -> int:
        """Ancilla basis index of a term, packed by the field layout."""
        return sum(code << off for code, off in zip(key, self.offsets))

    def _prep_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        v = np.zeros(2 ** self.n_prep)
        s = np.zeros(2 ** self.n_prep)
        for key, c in self.terms.items():
            i = self._index(key)
            v[i] = np.sqrt(abs(c) / self.alpha)
            s[i] = np.sign(c) * v[i]
        return v, s

    # -- circuit -----------------------------------------------------------
    def circuit(self) -> QuantumCircuit:
        m, dim = self.m, self.dim
        regs = [QuantumRegister(m, n) for n in (["x", "y"][:dim])]
        dof = QuantumRegister(1, "d") if self.has_dof else None
        prep = QuantumRegister(self.n_prep, "p")
        sel = QuantumRegister(1, "s")
        carry = QuantumRegister(self.n_carry, "a") if self.n_carry else None

        order = list(regs) + ([dof] if dof else []) + [prep, sel]
        if carry:
            order.append(carry)
        qc = QuantumCircuit(*order, name=f"BE[{self.kind},m={m}]")

        vL, vR = self._prep_vectors()
        qc.append(StatePreparation(vL), list(prep))

        inc = _increment(m)
        dec = _increment(m, inverse=True)
        carry_q = list(carry) if carry else []

        def with_select(field: int, value: int, body) -> None:
            """Run body(ctrl) with ctrl asserting that field == value.

            A one-bit field needs no scratch: the prep qubit is the control.
            """
            w, off = self.fields[field], self.offsets[field]
            if w == 1:
                body(prep[off])
                return
            qs = [prep[off + i] for i in range(w)]
            flips = [q for i, q in enumerate(qs) if not (value >> i) & 1]
            for q in flips:
                qc.x(q)
            qc.ccx(qs[0], qs[1], sel[0])
            body(sel[0])
            qc.ccx(qs[0], qs[1], sel[0])
            for q in flips:
                qc.x(q)

        # one multiplexer per spatial register: two controlled shifts each
        for f in range(dim):
            reg = list(regs[f])
            for value, sub in ((S_, inc), (SD_, dec)):
                if not any(k[f] == value for k in self.terms):
                    continue
                with_select(f, value,
                            lambda c, sub=sub, reg=reg:
                            qc.compose(sub, qubits=[c] + reg + carry_q,
                                       inplace=True))

        # multiplexer on the dof qubit
        if self.has_dof:
            f = dim
            dof_gate = {DZ_: lambda c, t: qc.cz(c, t),
                        DX_: lambda c, t: qc.cx(c, t),
                        DIY_: lambda c, t: qc.cry(-np.pi, c, t)}  # Ry(-pi) = iY
            for value, gate in dof_gate.items():
                if not any(k[f] == value for k in self.terms):
                    continue
                with_select(f, value, lambda c, g=gate: g(c, dof[0]))

        # reflection multiplexer, applied last since R_s sits leftmost in
        # R_s . S_x^kx . S_y^ky . sigma.  R_s = (S^(a,b))^dag R_chi S^(a,b),
        # so conjugate by a conditional shift, hit the oracle, and undo.
        if self.has_refl:
            fg = dim + (1 if self.has_dof else 0)          # g, then a, then b
            shifts = [(fg + 1, list(regs[0])), (fg + 2, list(regs[1]))]
            for f, reg in shifts:
                with_select(f, 1, lambda c, reg=reg:
                            qc.compose(dec, qubits=[c] + reg + carry_q,
                                       inplace=True))
            with_select(fg, 1, lambda c: self._oracle(qc, c, regs))
            for f, reg in reversed(shifts):
                with_select(f, 1, lambda c, reg=reg:
                            qc.compose(inc, qubits=[c] + reg + carry_q,
                                       inplace=True))

        qc.append(StatePreparation(vR).inverse(), list(prep))
        return qc

    def _oracle(self, qc, ctrl, regs) -> None:
        """R_chi for a dyadic square at the origin: a phase flip when the top
        k bits of both coordinates are zero. One multi-controlled Z, and its
        width 2k+1 depends on the volume fraction alone, not on m."""
        k = _inclusion_bits(self.operator.vf)
        tops = [q for r in regs for q in list(r)[self.m - k:]]
        for q in tops:
            qc.x(q)
        _mcz(qc, [ctrl] + tops)
        for q in tops:
            qc.x(q)

    def resources(self, optimization_level: int = 1) -> dict:
        qc = self.circuit()
        raw = qc.count_ops()
        tq = transpile(qc, basis_gates=BASIS, optimization_level=optimization_level)
        t = tq.count_ops()
        return dict(qubits=self.num_qubits, L=self.L, alpha=self.alpha,
                    toffoli=raw.get("ccx", 0), depth=tq.depth(),
                    cx=t.get("cx", 0), u=t.get("u", 0),
                    total=sum(t.values()))

    # -- materialisation (opt-in) -----------------------------------------
    def matrix(self) -> np.ndarray:
        """Assemble K densely from the SAME term list the circuit uses."""
        chi = self.operator.chi(self.m) if self.has_refl else None
        return _dense_from_terms(self.terms, self.m, self.dim, self.has_dof,
                                 self.has_refl, chi)

    def reference(self) -> np.ndarray:
        """Independent assembly, not built from the term list."""
        if self.has_refl:
            op = self.operator
            return _assemble_periodic(op.element(), 2 ** self.m,
                                      2 if self.has_dof else 1,
                                      chi=op.chi(self.m), E1=op.E1, E2=op.E2)
        return _reference_matrix(self.kind, self.m, self.nu, self.E)

    def verify(self, atol: float = 1e-9, do_transpile: bool = True) -> dict:
        """Materialise K and check alpha * <0|U|0> == K, before and after
        transpilation, plus the term list against an independent assembly."""
        K = self.matrix()
        ref = self.reference()
        out = {"alpha_vs_closed": abs(self.alpha - self.operator.alpha()),}
        if hasattr(self.operator, "alpha_published"):
            out["alpha_vs_published"] = abs(
                self.alpha - self.operator.alpha_published())
        out.update({
            "terms_vs_reference": float(np.abs(K - ref).max())})

        for label, tp in (("circuit", False), ("transpiled", True)):
            if tp and not do_transpile:
                continue
            qc = self.circuit()
            if tp:
                qc = transpile(qc, basis_gates=BASIS, optimization_level=1)
            blk = _top_left_block(qc, self.n_system)
            out[f"block_err_{label}"] = float(np.abs(self.alpha * blk - K).max())
        out["ok"] = all(v <= atol for k, v in out.items() if k != "ok")
        return out


def _shift_matrix(m: int, k: int) -> np.ndarray:
    N = 2 ** m
    P = np.zeros((N, N))
    for j in range(N):
        P[(j + k) % N, j] = 1.0
    return P


_DOF_OPS = {DI_: np.eye(2), DZ_: np.diag([1.0, -1.0]),
            DX_: np.array([[0.0, 1.0], [1.0, 0.0]]),
            DIY_: np.array([[0.0, 1.0], [-1.0, 0.0]])}


# iY = [[0, 1], [-1, 0]] = Ry(-pi). It appears in the two-phase cell through
# the |1 - 3 nu| term and vanishes at nu = 1/3.


def _refl_matrix(chi: np.ndarray, m: int, s: int) -> np.ndarray:
    """R_s = diag(1 - 2 chi(i + offset_s)), ordered x low, y high."""
    a, b = ELEM_OFFSETS[s]
    N = 2 ** m
    d = np.empty(N * N)
    for y in range(N):
        for x in range(N):
            d[y * N + x] = 1.0 - 2.0 * chi[(x + a) % N, (y + b) % N]
    return np.diag(d)


def _dense_from_terms(tms: dict, m: int, dim: int, has_dof: bool,
                      has_refl: bool = False, chi=None) -> np.ndarray:
    """Sum the LCU terms densely. numpy kron puts the LEFT factor in the HIGH
    bits, and the circuit puts register x in the LOW bits, so the spatial
    factors are assembled as kron(y, x)."""
    N = 2 ** m
    size = N ** dim * (2 if has_dof else 1)
    K = np.zeros((size, size))
    ops = {I_: np.eye(N), S_: _shift_matrix(m, +1), SD_: _shift_matrix(m, -1)}
    refl = {}
    if has_refl:
        for s in range(4):
            refl[s] = _refl_matrix(chi, m, s)
    for key, c in tms.items():
        spatial = ops[key[0]]
        for f in range(1, dim):
            spatial = np.kron(ops[key[f]], spatial)     # register f above f-1
        if has_refl:
            g, a, b = key[-3:]
            if g:
                s = ELEM_OFFSETS.index((-a, -b))
                spatial = refl[s] @ spatial             # R_s applied last
        term = np.kron(_DOF_OPS[key[dim]], spatial) if has_dof else spatial
        K += c * term
    return K


def _top_left_block(qc: QuantumCircuit, n_system: int) -> np.ndarray:
    """<0_anc| U |0_anc>, one statevector simulation per system basis state."""
    n = qc.num_qubits
    N0 = 2 ** n_system
    blk = np.zeros((N0, N0), dtype=complex)
    for j in range(N0):
        out = Statevector.from_int(j, 2 ** n).evolve(qc).data
        blk[:, j] = out[:N0]        # ancillas are the high bits, so |0_anc>
    return blk


# ==========================================================================
# 4. independent reference assemblies
# ==========================================================================
def _circ(vals, N: int) -> np.ndarray:
    a, b, c = vals
    A = np.zeros((N, N))
    np.fill_diagonal(A, b)
    i = np.arange(N - 1)
    A[i + 1, i] = a
    A[i, i + 1] = c
    A[0, N - 1] += a          # += : at N = 2 both neighbours fold onto one entry
    A[N - 1, 0] += c
    return A


def _q4_scalar_element() -> np.ndarray:
    """Bilinear (Q4) Laplacian element on a unit square, 2x2 Gauss."""
    g = 1 / np.sqrt(3)
    Ke = np.zeros((4, 4))
    for xi in (-g, g):
        for eta in (-g, g):
            dN = np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                           [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]]) / 4
            J = 0.5 * np.eye(2)
            dNx = np.linalg.solve(J, dN)
            Ke += dNx.T @ dNx * np.linalg.det(J)
    return Ke


def _assemble_periodic(Ke: np.ndarray, N: int, ndof_node: int,
                       chi=None, E1: float = 1.0, E2: float = 1.0) -> np.ndarray:
    """Periodic assembly on an N x N grid of unit squares.

    Global index = d * N^2 + y * N + x, matching the circuit's register order
    (x low, then y, dof highest).
    """
    ndof = ndof_node * N * N
    K = np.zeros((ndof, ndof))
    offs = [(0, 0), (1, 0), (1, 1), (0, 1)]          # ccw from lower left
    for ey in range(N):
        for ex in range(N):
            g = [((ex + ox) % N, (ey + oy) % N) for ox, oy in offs]
            scale = 1.0 if chi is None else E2 + (E1 - E2) * chi[ex, ey]
            for li, (xi, yi) in enumerate(g):
                for di in range(ndof_node):
                    gi = di * N * N + yi * N + xi
                    for lj, (xj, yj) in enumerate(g):
                        for dj in range(ndof_node):
                            gj = dj * N * N + yj * N + xj
                            K[gi, gj] += scale * Ke[ndof_node * li + di,
                                                    ndof_node * lj + dj]
    return K


def _q4_element(nu: float, E: float) -> np.ndarray:
    """Plane-stress Q4 stiffness for a unit square, 2x2 Gauss, unit thickness."""
    D = E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0],
                                      [0, 0, (1 - nu) / 2]])
    g = 1 / np.sqrt(3)
    nodes = [(-1, -1), (1, -1), (1, 1), (-1, 1)]       # ccw, local 0..3
    Ke = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            dN = np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                           [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]]) / 4
            J = 0.5 * np.eye(2)                        # unit square, h = 1
            dNx = np.linalg.solve(J, dN)
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2 * i] = dNx[0, i]
                B[1, 2 * i + 1] = dNx[1, i]
                B[2, 2 * i] = dNx[1, i]
                B[2, 2 * i + 1] = dNx[0, i]
            Ke += B.T @ D @ B * np.linalg.det(J)
    _ = nodes
    return Ke


def _reference_matrix(kind: str, m: int, nu: float, E: float) -> np.ndarray:
    """Assembled without using the LCU term list.

    For the two finite-element operators this is a direct element-by-element
    periodic assembly from Gauss quadrature: no Kronecker identity, no shift
    algebra, no LCU coefficient. It is the check that the decomposition is
    right, not merely self-consistent."""
    N = 2 ** m
    K1, M1 = _circ((-1, 2, -1), N), _circ((1, 4, 1), N) / 6
    G1 = _circ((-1, 0, 1), N) / 2
    I = np.eye(N)

    if kind == "poisson1d":
        return K1
    if kind == "poisson2d_fd":
        return np.kron(I, K1) + np.kron(K1, I)
    if kind == "poisson2d_fe":
        return _assemble_periodic(_q4_scalar_element(), N, 1)
    if kind == "elasticity2d":
        _ = (M1, G1)
        return _assemble_periodic(_q4_element(nu, E), N, 2)
    raise ValueError(kind)


# ==========================================================================
# 5. user-facing entry point
# ==========================================================================
@dataclass
class EncodingInfo:
    """Everything known about the encoding without materialising K."""
    operator: object            # the spec that was encoded
    kind: str
    m: int
    N: int                      # grid points per spatial direction
    dofs: int                   # rows of K: N, N^2, or 2 N^2
    L: int                      # number of LCU terms
    alpha: float                # subnormalization, sum |c_k|
    alpha_closed: float         # analytic value, for cross-checking
    qubits: int
    system: int
    prep: int
    carry: int
    toffoli: int | None = None
    cx: int | None = None
    depth: int | None = None
    verification: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        w = []
        w.append(f"{self.operator!r}   N = {self.N} per direction   "
                 f"m = {self.m} qubits/direction   dofs = {self.dofs:,}")
        w.append(f"  L      = {self.L}")
        w.append(f"  alpha  = {self.alpha:.6f}   "
                 f"(closed form {self.alpha_closed:.6f})")
        w.append(f"  qubits = {self.qubits}  (system {self.system}, "
                 f"prep {self.prep}, select 1, carry {self.carry})")
        if self.toffoli is not None:
            w.append(f"  Toffoli = {self.toffoli}   CX = {self.cx}   "
                     f"depth = {self.depth}      [transpiled to {BASIS}]")
        v = self.verification
        if v is not None:
            w.append(f"  verified: alpha vs closed form  "
                     f"{v['alpha_vs_closed']:.2e}")
            if "alpha_vs_published" in v:
                w.append(f"            alpha vs published    "
                         f"{v['alpha_vs_published']:.2e}")
            w.append(f"            terms vs assembly     "
                     f"{v['terms_vs_reference']:.2e}")
            w.append(f"            block vs K            "
                     f"{v['block_err_circuit']:.2e}")
            w.append(f"            block vs K transpiled "
                     f"{v['block_err_transpiled']:.2e}")
            w.append(f"            OK = {v['ok']}")
        return "\n".join(w)


def _resolve_size(N: int | None, m: int | None) -> int:
    """Accept either N (points per direction) or m (qubits), return m."""
    if (N is None) == (m is None):
        raise ValueError("give exactly one of N or m")
    if m is not None:
        if m < 1:
            raise ValueError("m must be >= 1")
        return int(m)
    if N < 2 or (N & (N - 1)):
        raise ValueError(f"N must be a power of two and at least 2, got {N}")
    return int(N).bit_length() - 1


def blockencode(operator, N: int | None = None, *, m: int | None = None,
                materialize: bool = False, costs: bool = True,
                atol: float = 1e-9) -> tuple[QuantumCircuit, EncodingInfo]:
    """Block encode a periodic FE operator.

    Parameters
    ----------
    operator
            POISSON1D(), POISSON2D('fe'|'fd'), ELASTICITY2D(nu=, E=), or the
            equivalent name as a string
    N       grid points PER DIRECTION, a power of two. The operator has N dofs
            in 1D, N**2 in 2D, and 2 N**2 for elasticity.
    m       qubits per direction, N = 2**m. Give N or m, not both.
    materialize
            assemble K densely and check alpha * <0|U|0> against it, before
            and after transpilation, and check the term list against an
            independent element-by-element assembly. Costs O(4^n); use for
            small m only.
    costs   transpile once to report Toffoli / CX / depth. Set False to skip.

    Returns
    -------
    (circuit, info)
    """
    m = _resolve_size(N, m)
    be = PeriodicBlockEncoding(operator, m)
    qc = be.circuit()
    i = be.info()
    info = EncodingInfo(
        operator=be.operator, kind=be.kind, m=m, N=i["N"], dofs=i["dofs"], L=i["L"],
        alpha=i["alpha"], alpha_closed=i["alpha_closed"],
        qubits=i["total_qubits"], system=i["system"], prep=i["prep"],
        carry=i["carry"])
    if costs:
        r = be.resources()
        info.toffoli, info.cx, info.depth = r["toffoli"], r["cx"], r["depth"]
    if materialize:
        info.verification = be.verify(atol=atol)
    return qc, info


if __name__ == "__main__":
    for op, N in ((POISSON1D(), 16), (POISSON2D("fd"), 8),
                  (POISSON2D("fe"), 8), (ELASTICITY2D(nu=0.3), 4),
                  (POISSON2D_2PHASE(vf=0.25, E1=10, E2=1), 4),
                  (ELASTICITY2D_2PHASE(nu=0.3, vf=0.25, E1=10, E2=1), 4)):
        circuit, info = blockencode(op, N, materialize=True)
        print(info)
        print()

    print("scaling, no materialisation")
    print(f"{'kind':>20} {'m':>3} {'L':>4} {'alpha':>9} {'qubits':>7} "
          f"{'Toffoli':>8} {'CX':>7} {'depth':>7}")
    for op in (POISSON1D(), POISSON2D("fd"), POISSON2D("fe"), ELASTICITY2D(),
               POISSON2D_2PHASE(), ELASTICITY2D_2PHASE()):
        for m in (4, 8, 12):
            _, info = blockencode(op, m=m)
            print(f"{info.kind:>20} {m:>3} {info.L:>4} {info.alpha:>9.4f} "
                  f"{info.qubits:>7} {info.toffoli:>8} {info.cx:>7} "
                  f"{info.depth:>7}")
