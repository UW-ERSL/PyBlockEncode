"""
bc.py — Boundary conditions for shift-decomposition block encodings.

The 1D finite element factors are Laurent polynomials in the cyclic shift
S_c.  A cyclic shift wraps in exactly one row -- S_c in row 0, S_c^dagger in
row N-1 -- and *which* row it wraps in is a property of the shift alone, not
of the coefficient multiplying it.  Boundary conditions are therefore imposed
by adjoining reflection operators

    R_A = I - 2 sum_{j in A} |j><j|

to the unitary set, which costs no ancilla and no extra circuit depth.  This
is the construction of Kharazi et al., Quantum 9, 1764 (2025), extended here
to the finite element mass and gradient factors and to mixed conditions.

Three treatments
----------------
periodic   the raw cyclic operator; not a boundary-value problem, but the
           starting point for the other two.

essential  (Dirichlet / clamped)  De-periodize by deleting the wrap-around
           coupling.  Using  (R_j + I)/2 = I - |j><j|,

               S_c    ->  (1/2)(R_0    + I) S_c
               S_c^d  ->  (1/2)(R_{N-1}+ I) S_c^dagger

           Each shift splits into two terms of half the coefficient, so the
           term count grows and alpha does not.

free       (traction-free / zero Neumann)  A boundary node belongs to one
           element rather than two, so each factor picks up a correction
           supported on the boundary nodes:

               dK = -(Pi_0 + Pi_{N-1})
               dM = -(1/3)(Pi_0 + Pi_{N-1})
               dG = +(1/2)(Pi_{N-1} - Pi_0)

           with Pi_j = |j><j| = (1/2)(I - R_j), so every correction is again
           a combination of reflections.

Mixed conditions are specified per end.  The two ends of a direction need not
agree: a cantilever is clamped at one end and free at the other, and its
correction is a single Pi_j rather than the symmetric pair.  Note that the
identity terms in dG cancel between the two ends only when *both* are free;
a one-sided free correction keeps its identity term.

Unitary set
-----------
    I, S_c, R_0 S_c, S_c^dagger, R_{N-1} S_c^dagger, R_0, R_{N-1}

Seven labels, of which the periodic case uses three, the essential case five,
a one-sided case six, and the traction-free case all seven.

Transpose
---------
The set is closed under transposition, by the label permutation in
``TRANSPOSE``.  The only non-obvious pair is

    (R_0 S_c)^T = S_c^dagger R_0 = R_{N-1} S_c^dagger,

which holds because Pi_{N-1} S_c^dagger and S_c^dagger Pi_0 are the same
one-entry matrix.  This is what lets the gradient factor's transpose stay in
the set once the free correction has destroyed its antisymmetry.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# The unitary set
# ---------------------------------------------------------------------------

LABELS: Tuple[str, ...] = ("I", "Sc", "R0.Sc", "Scd", "RN.Scd", "R0", "RN")

#: label permutation induced by matrix transposition
TRANSPOSE: Dict[str, str] = {
    "I": "I",
    "Sc": "Scd",
    "Scd": "Sc",
    "R0.Sc": "RN.Scd",
    "RN.Scd": "R0.Sc",
    "R0": "R0",
    "RN": "RN",
}

#: net grid offset of each label, used to place terms on a stencil
OFFSET: Dict[str, int] = {
    "I": 0, "R0": 0, "RN": 0,
    "Sc": +1, "R0.Sc": +1,
    "Scd": -1, "RN.Scd": -1,
}


def shift(N: int, k: int = 1) -> np.ndarray:
    """Cyclic shift S_c^k : |j> -> |(j+k) mod N>."""
    S = np.zeros((N, N))
    for j in range(N):
        S[(j + k) % N, j] = 1.0
    return S


def reflection(N: int, idx) -> np.ndarray:
    """R_A = I - 2 sum_{j in A} |j><j|, a diagonal +-1 unitary."""
    R = np.eye(N)
    for j in np.atleast_1d(idx):
        R[int(j), int(j)] = -1.0
    return R


def unitary(label: str, N: int) -> np.ndarray:
    """Dense matrix for one label of the unitary set."""
    if label not in TRANSPOSE:
        raise KeyError(f"unknown label {label!r}; expected one of {LABELS}")
    I = np.eye(N)
    if label == "I":
        return I
    if label == "R0":
        return reflection(N, 0)
    if label == "RN":
        return reflection(N, N - 1)
    if label == "Sc":
        return shift(N, +1)
    if label == "Scd":
        return shift(N, -1)
    if label == "R0.Sc":
        return reflection(N, 0) @ shift(N, +1)
    return reflection(N, N - 1) @ shift(N, -1)      # "RN.Scd"


def basis(N: int) -> Dict[str, np.ndarray]:
    """The seven unitaries as a dict, for building dense operators."""
    return {lbl: unitary(lbl, N) for lbl in LABELS}


# ---------------------------------------------------------------------------
# Boundary-condition specification
# ---------------------------------------------------------------------------

#: the value each end may take
ENDS = ("clamped", "free")


def parse_ends(spec) -> Tuple[str, str] | str:
    """
    Normalize a per-direction boundary specification.

    Accepts
        'periodic'                     -> 'periodic'
        'essential' | 'clamped'        -> ('clamped', 'clamped')
        'free' | 'neumann'             -> ('free', 'free')
        ('clamped', 'free')            -> as given
        (False, True)                  -> ('clamped', 'free')
    """
    if isinstance(spec, str):
        s = spec.lower()
        if s == "periodic":
            return "periodic"
        if s in ("essential", "clamped", "dirichlet"):
            return ("clamped", "clamped")
        if s in ("free", "neumann", "traction-free"):
            return ("free", "free")
        raise ValueError(f"unknown boundary specification {spec!r}")
    if isinstance(spec, Sequence) and len(spec) == 2:
        out = []
        for e in spec:
            if isinstance(e, str):
                if e.lower() not in ENDS:
                    raise ValueError(f"end must be one of {ENDS}; got {e!r}")
                out.append(e.lower())
            else:
                out.append("free" if e else "clamped")
        return (out[0], out[1])
    raise ValueError(f"cannot interpret boundary specification {spec!r}")


def parse_bc(spec, ndim: int) -> Tuple:
    """
    Normalize a boundary specification for an ndim-dimensional problem.

    Accepts a single specification applied to every direction, a sequence of
    ndim specifications, or a dict keyed by 'x', 'y', 'z' (or 0, 1, 2).
    """
    axis_names = ("x", "y", "z")
    if isinstance(spec, dict):
        out = []
        for d in range(ndim):
            if d in spec:
                out.append(parse_ends(spec[d]))
            elif axis_names[d] in spec:
                out.append(parse_ends(spec[axis_names[d]]))
            else:
                raise ValueError(f"boundary specification missing direction "
                                 f"{axis_names[d]!r}")
        return tuple(out)
    if isinstance(spec, (list, tuple)) and len(spec) == ndim \
            and not (len(spec) == 2 and all(
                isinstance(e, str) and e.lower() in ENDS for e in spec)):
        return tuple(parse_ends(s) for s in spec)
    return tuple(parse_ends(spec) for _ in range(ndim))


# ---------------------------------------------------------------------------
# The 1D factors
# ---------------------------------------------------------------------------

def _clean(d: Dict[str, float], tol: float = 1e-14) -> Dict[str, float]:
    return {k: v for k, v in d.items() if abs(v) > tol}


#: periodic factors, over {I, Sc, Scd}
_PERIODIC: Dict[str, Dict[str, float]] = {
    "K": {"I": 2.0, "Sc": -1.0, "Scd": -1.0},
    "M": {"I": 4.0 / 6.0, "Sc": 1.0 / 6.0, "Scd": 1.0 / 6.0},
    "G": {"Scd": 0.5, "Sc": -0.5},
    "Iop": {"I": 1.0},
}

#: free-surface correction at one end, as {label: coefficient}
#:   -Pi_j        = -1/2 I + 1/2 R_j                      (stiffness)
#:   -(1/3) Pi_j  = -1/6 I + 1/6 R_j                      (mass)
#:   -1/2 Pi_0    = -1/4 I + 1/4 R_0     (gradient, left end)
#:   +1/2 Pi_{N-1}= +1/4 I - 1/4 R_{N-1} (gradient, right end)
_FREE_CORRECTION: Dict[str, Dict[str, Tuple[float, float]]] = {
    #        left  (I coeff, R0 coeff)      right (I coeff, RN coeff)
    "K": {"left": (-0.5, +0.5), "right": (-0.5, +0.5)},
    "M": {"left": (-1 / 6, +1 / 6), "right": (-1 / 6, +1 / 6)},
    "G": {"left": (-0.25, +0.25), "right": (+0.25, -0.25)},
    "Iop": {"left": (0.0, 0.0), "right": (0.0, 0.0)},
}


def factor(kind: str, ends) -> Dict[str, float]:
    """
    One 1D factor as a dict over the unitary set.

    Parameters
    ----------
    kind : {'K', 'M', 'G', 'Iop'}
        stiffness, consistent mass, gradient coupling, or identity.
    ends : 'periodic' or a pair of 'clamped' / 'free'
        The boundary treatment at the two ends of this direction.

    Returns
    -------
    dict {label: coefficient}
    """
    if kind not in _PERIODIC:
        raise ValueError(f"kind must be one of {tuple(_PERIODIC)}; got {kind!r}")
    ends = parse_ends(ends) if not isinstance(ends, tuple) or \
        ends == "periodic" else ends
    if ends == "periodic":
        return dict(_PERIODIC[kind])

    # de-periodize: every shift splits into a bare and a reflected half
    d: Dict[str, float] = {}
    for lbl, c in _PERIODIC[kind].items():
        if lbl == "I":
            d["I"] = d.get("I", 0.0) + c
        elif lbl == "Sc":
            d["Sc"] = d.get("Sc", 0.0) + c / 2
            d["R0.Sc"] = d.get("R0.Sc", 0.0) + c / 2
        elif lbl == "Scd":
            d["Scd"] = d.get("Scd", 0.0) + c / 2
            d["RN.Scd"] = d.get("RN.Scd", 0.0) + c / 2

    # free-surface corrections, applied per end
    corr = _FREE_CORRECTION[kind]
    for side, refl_lbl in (("left", "R0"), ("right", "RN")):
        if ends[0 if side == "left" else 1] == "free":
            ci, cr = corr[side]
            d["I"] = d.get("I", 0.0) + ci
            d[refl_lbl] = d.get(refl_lbl, 0.0) + cr

    return _clean(d)


def transpose(d: Dict[str, float]) -> Dict[str, float]:
    """Transpose a 1D factor, exactly, by permuting labels."""
    out: Dict[str, float] = {}
    for lbl, c in d.items():
        t = TRANSPOSE[lbl]
        out[t] = out.get(t, 0.0) + c
    return _clean(out)


def dense(d: Dict[str, float], N: int) -> np.ndarray:
    """Realize a 1D factor as a dense N x N matrix."""
    A = np.zeros((N, N))
    for lbl, c in d.items():
        A += c * unitary(lbl, N)
    return A


def alpha(d) -> float:
    """Subnormalization sum |c_k| of a term dict (any key type)."""
    return float(sum(abs(v) for v in d.values()))


# ---------------------------------------------------------------------------
# Symbolic algebra on multi-dimensional term dicts
# ---------------------------------------------------------------------------

def kron(a: Dict, b: Dict) -> Dict:
    """Kronecker product of two term dicts; keys concatenate as tuples."""
    out: Dict = {}
    for la, ca in a.items():
        for lb, cb in b.items():
            ta = la if isinstance(la, tuple) else (la,)
            tb = lb if isinstance(lb, tuple) else (lb,)
            key = ta + tb
            out[key] = out.get(key, 0.0) + ca * cb
    return _clean(out)


def add(a: Dict, b: Dict, s: float = 1.0) -> Dict:
    """a + s*b, merging coefficients that land on the same key."""
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + s * v
    return _clean(out)


def scale(d: Dict, s: float) -> Dict:
    return _clean({k: v * s for k, v in d.items()})


def dense_nd(terms: Dict[Tuple[str, ...], float], N: int,
             extra: Dict[str, np.ndarray] | None = None) -> np.ndarray:
    """
    Realize a multi-dimensional term dict as a dense matrix.

    Each key is a tuple of labels, one per spatial direction, optionally
    followed by one key of ``extra`` (used for the DOF-qubit Pauli).
    """
    first = next(iter(terms))
    ndim = len(first) - (1 if extra else 0)
    dim = N ** ndim * (list(extra.values())[0].shape[0] if extra else 1)
    A = np.zeros((dim, dim))
    for key, c in terms.items():
        M = unitary(key[0], N)
        for lbl in key[1:ndim]:
            M = np.kron(M, unitary(lbl, N))
        if extra:
            M = np.kron(M, extra[key[-1]])
        A += c * M
    return A


def basis_size(ends) -> int:
    """Number of distinct unitaries a direction with these ends requires."""
    ends = parse_ends(ends) if not isinstance(ends, tuple) else ends
    if ends == "periodic":
        return 3
    return 5 + sum(1 for e in ends if e == "free")
