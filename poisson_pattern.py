"""
poisson_pattern.py — Pattern-compression block encoding of the Poisson operator.

Supports:
    dim  ∈ {1, 2, 3}          spatial dimension
    disc ∈ {'fdm', 'fem'}     discretization
    m    ∈ {1, 2, ...}        qubits per spatial dimension; N = 2**m nodes

The 1D building blocks
----------------------
Every 1D factor is a cyclic-shift LCU:

    K1 = 2I - Sc - Sc†          (FDM/FEM stiffness)
    M1 = (4I + Sc + Sc†) / 6   (FEM consistent mass;  → I for FDM)

where Sc : |j⟩ → |(j+1) mod N⟩.  A single flag ancilla converts the cyclic
boundary to Dirichlet (boundary correction).

Multi-dimensional encoding via Kronecker separability
------------------------------------------------------
2D FDM:  K = I⊗K1 + K1⊗I             → 2×3 = 6  LCU terms, α = 8
2D FEM:  K = M1⊗K1 + K1⊗M1           → 2×9 = 18 LCU terms (before cancel)
3D FDM:  K = I⊗I⊗K1 + I⊗K1⊗I + K1⊗I⊗I → 3×3 = 9 terms, α = 12
3D FEM:  K = M1⊗M1⊗K1 + ...           → 3×27 terms (before cancel)

Coefficients are computed analytically; α is always constant in N.

Qubit layout (MSB → LSB)
-------------------------
    [PREP ancilla | flag | dim-1 | dim-2 | ... | dim-d]
System register = d*m qubits (d spatial dimensions, m qubits each).
Ancilla = ⌈log₂ L⌉ PREP qubits + 1 flag qubit.
Total qubits = d*m + ⌈log₂ L⌉ + 1.

Convention: system register on least-significant qubits; encoded block is
the top-left U[:N0, :N0] sub-matrix, N0 = 2**(d*m).
"""
from __future__ import annotations

import math
import numpy as np
from typing import Dict, Tuple, List

from . import operators

# ---------------------------------------------------------------------------
# Symbolic 1D shift dictionaries  {label: coefficient}
# ---------------------------------------------------------------------------

_K1_sym = {"I": 2.0, "Sc": -1.0, "Scd": -1.0}   # tridiag(-1,2,-1)
_M1_sym = {"I": 4.0 / 6.0, "Sc": 1.0 / 6.0, "Scd": 1.0 / 6.0}  # tridiag(1,4,1)/6
_I_sym  = {"I": 1.0}                               # identity

# For FDM, the non-differentiated factor is I not M1
_MASS = {"fem": _M1_sym, "fdm": _I_sym}


def _kron_sym(a: Dict, b: Dict) -> Dict:
    """Symbolic Kronecker product of two shift dicts.

    Keys are always flat tuples of strings, e.g. ('I', 'Sc', 'Scd').
    """
    out: Dict = {}
    for la, ca in a.items():
        for lb, cb in b.items():
            ta = la if isinstance(la, tuple) else (la,)
            tb = lb if isinstance(lb, tuple) else (lb,)
            key = ta + tb
            out[key] = out.get(key, 0.0) + ca * cb
    return {k: v for k, v in out.items() if abs(v) > 1e-15}


def _scale_sym(d: Dict, s: float) -> Dict:
    return {k: v * s for k, v in d.items()}


def _add_sym(a: Dict, b: Dict) -> Dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return {k: v for k, v in out.items() if abs(v) > 1e-15}


def _build_lcu_terms(dim: int, disc: str
                     ) -> Dict[Tuple[str, ...], float]:
    """
    Compute the LCU term dictionary for a d-dimensional Poisson operator.

    Returns
    -------
    dict mapping (label_dim1, label_dim2, ..., label_dimD) → coefficient
    where each label ∈ {'I', 'Sc', 'Scd'}.
    """
    stiff = _K1_sym
    mass  = _MASS[disc]

    if dim == 1:
        # K = K1  (single term sum, 3 entries)
        raw = stiff
        return {(k,): v for k, v in raw.items()}

    # Build each Kronecker term  mass^(d-1) ⊗ K1  summed over d dimensions
    # In d dimensions: K = Σ_i  ( ⊗_{j≠i} mass ) ⊗_i K1
    combined: Dict[Tuple, float] = {}
    for active in range(dim):
        # Build the tensor product for this term
        factors = []
        for j in range(dim):
            factors.append(stiff if j == active else mass)
        # Kronecker the factors together
        term: Dict = factors[0]
        for f in factors[1:]:
            term = _kron_sym(term, f)
        # Convert flat keys → tuples and add
        for key, val in term.items():
            tkey = key if isinstance(key, tuple) else (key,)
            combined[tkey] = combined.get(tkey, 0.0) + val
    return {k: v for k, v in combined.items() if abs(v) > 1e-15}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PoissonPatternEncoding:
    """
    Pattern-compression block encoding of the Poisson stiffness operator.

    Parameters
    ----------
    m    : int   — qubits per spatial dimension; grid has N = 2**m interior points
    dim  : int   — spatial dimension (1, 2, or 3)
    disc : str   — discretization: 'fdm' or 'fem'

    Key properties
    --------------
    lcu_terms()  — dict {(labels,): coefficient} of the LCU decomposition
    alpha        — subnormalization Σ|c_k|  (constant in N)
    num_terms    — number of LCU terms
    num_system   — system qubits = dim * m
    num_ancilla  — ancilla qubits = ⌈log₂ L⌉ + 1  (PREP + flag)
    num_qubits   — total qubits

    target()     — classically assembled reference matrix (dense, trimmed)
    verify()     — dict with decomposition error and (optionally) unitary check
    """

    def __init__(self, m: int, dim: int = 1, disc: str = "fdm"):
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3; got {dim}")
        if disc not in ("fdm", "fem"):
            raise ValueError(f"disc must be 'fdm' or 'fem'; got {disc}")
        self.m    = m
        self.dim  = dim
        self.disc = disc
        self.N    = 2 ** m
        self._terms: Dict[Tuple, float] | None = None

    # ------------------------------------------------------------------
    # LCU decomposition
    # ------------------------------------------------------------------

    def lcu_terms(self) -> Dict[Tuple[str, ...], float]:
        """Return the LCU term dictionary (cached)."""
        if self._terms is None:
            self._terms = _build_lcu_terms(self.dim, self.disc)
        return self._terms

    @property
    def alpha(self) -> float:
        """Subnormalization α = Σ|c_k|, constant in N."""
        return float(sum(abs(v) for v in self.lcu_terms().values()))

    @property
    def num_terms(self) -> int:
        return len(self.lcu_terms())

    @property
    def num_system(self) -> int:
        return self.dim * self.m

    @property
    def num_ancilla(self) -> int:
        L = self.num_terms
        return math.ceil(math.log2(max(L, 2))) + 1  # PREP + flag

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    # ------------------------------------------------------------------
    # Classical reference
    # ------------------------------------------------------------------

    def target(self) -> np.ndarray:
        """Classically assembled Dirichlet interior stiffness (dense)."""
        m, dim, disc = self.m, self.dim, self.disc
        if dim == 1:
            return (operators.poisson_1d_fdm(m) if disc == "fdm"
                    else operators.poisson_1d_fem(m))
        if dim == 2:
            return (operators.poisson_2d_fdm(m) if disc == "fdm"
                    else operators.poisson_2d_fem(m))
        # dim == 3
        return (operators.poisson_3d_fdm(m) if disc == "fdm"
                else operators.poisson_3d_fem(m))

    # ------------------------------------------------------------------
    # Classical decomposition (for verification without building unitary)
    # ------------------------------------------------------------------

    def decomposition(self) -> np.ndarray:
        """
        Reconstruct the operator from the LCU terms classically.

        Uses dense cyclic-shift matrices (periodic), then applies the
        Dirichlet boundary correction analytically: rows/columns
        corresponding to wrap-around are zeroed out.
        """
        N = self.N
        dim = self.dim

        # Cyclic shift matrix Sc  (N×N)
        Sc  = np.eye(N, k=1); Sc[-1, 0] = 1.0   # forward
        Scd = np.eye(N, k=-1); Scd[0, -1] = 1.0  # backward (Sc†)
        mats = {"I": np.eye(N), "Sc": Sc, "Scd": Scd}

        A = np.zeros((self.N ** dim, self.N ** dim))
        for key, coeff in self.lcu_terms().items():
            # key is always a flat tuple of strings ('I', 'Sc', ...) of length dim
            labels = key if isinstance(key, tuple) else (key,)
            M = mats[labels[0]]
            for lbl in labels[1:]:
                M = np.kron(M, mats[lbl])
            A = A + coeff * M

        # Dirichlet boundary correction:
        # For Dirichlet BCs the cyclic entries at wrap-around positions
        # should be zero.  Apply boundary trimming: zero out rows/cols
        # that correspond to wrap-around connections.
        # The correction matrix = target - (cyclic version of target).
        # Equivalently: target() was assembled without wrap-around.
        # So the mismatch = A - target = boundary term.  We just return
        # A here; verify() computes the error against target().
        return A

    # ------------------------------------------------------------------
    # Full unitary (small m only, dense, for verification)
    # ------------------------------------------------------------------

    def _shift(self, lbl: str, j: int) -> Tuple[int, bool]:
        """Apply one cyclic shift to index j; return (new_j, wrapped)."""
        N = self.N
        if lbl == "I":
            return j, False
        if lbl == "Sc":
            return (j + 1) % N, (j == N - 1)
        return (j - 1) % N, (j == 0)   # Scd

    def block_encoding(self) -> np.ndarray:
        """
        Build the full block-encoding unitary (dense; small m only).

        Returns U such that  alpha * U[:N0, :N0] ≈ K_target,
        where N0 = N**dim is the system dimension.

        Qubit layout (MSB → LSB): [PREP ancilla | flag | dim registers]
        The encoded block is the top-left U[:N0, :N0] sub-matrix
        (PREP and flag ancilla both in |0⟩).
        """
        N    = self.N
        dim  = self.dim
        N0   = N ** dim        # system dimension
        terms   = self.lcu_terms()
        labels  = list(terms)
        coeffs  = np.array([terms[k] for k in labels])
        alpha   = float(np.sum(np.abs(coeffs)))
        L       = len(labels)
        K2      = 2 ** math.ceil(math.log2(max(L, 2)))   # PREP ancilla span
        Dflag   = 2 * N0       # flag qubit × system

        def idx(flag: int, *js: int) -> int:
            """Flatten (flag, j_0, j_1, ..., j_{d-1}) → linear index."""
            n = flag
            for j in js:
                n = n * N + j
            return n  # flag is MSB among the system+flag register

        SEL = np.zeros((K2 * Dflag, K2 * Dflag))
        for i in range(K2):
            if i < L:
                key_i = labels[i]
                lbls = key_i if isinstance(key_i, tuple) else (key_i,)
                sgn  = float(np.sign(coeffs[i]))
                # Iterate over all system states
                for flag in range(2):
                    for state in np.ndindex(*([N] * dim)):
                        new_js = []
                        any_wrap = False
                        for d, lbl in enumerate(lbls):
                            nj, wrapped = self._shift(lbl, state[d])
                            new_js.append(nj)
                            if wrapped:
                                any_wrap = True
                        new_flag = int(any_wrap) if flag == 0 else int(not any_wrap)
                        row = i * Dflag + idx(int(new_flag), *new_js)
                        col = i * Dflag + idx(flag, *state)
                        SEL[row, col] = sgn
            else:
                SEL[i * Dflag:(i + 1) * Dflag,
                    i * Dflag:(i + 1) * Dflag] = np.eye(Dflag)

        # PREP: load sqrt(|c_k|/alpha) into ancilla register
        amps = np.zeros(K2)
        for i, c in enumerate(coeffs):
            amps[i] = np.sqrt(abs(c) / alpha)
        P = np.eye(K2)
        P[:, 0] = amps
        Q, _ = np.linalg.qr(P)
        if np.dot(Q[:, 0], amps) < 0:
            Q[:, 0] *= -1

        U = (np.kron(Q.conj().T, np.eye(Dflag))
             @ SEL
             @ np.kron(Q, np.eye(Dflag)))
        return U

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, build_unitary: bool | None = None) -> dict:
        """
        Verify the encoding.

        Returns a dict with:
            dim, disc, m, alpha, num_terms, num_qubits
            decomposition_rel_err  — ‖A_cyclic − K_target‖ / ‖K_target‖
            block_encoding         — (if built) ‖alpha·U[:N0,:N0] − K‖ / ‖K‖
        """
        target = self.target()
        A_cyc  = self.decomposition()
        # The cyclic decomposition differs from target by the boundary correction.
        # For the block encoding the flag ancilla enforces this correction,
        # so we report the decomposition error as is (should match the
        # boundary-correction magnitude, not zero).
        dec_err = float(np.linalg.norm(A_cyc - target) / np.linalg.norm(target))

        out = {
            "dim": self.dim, "disc": self.disc, "m": self.m,
            "alpha": self.alpha, "num_terms": self.num_terms,
            "num_qubits": self.num_qubits,
            "decomposition_rel_err (cyclic vs Dirichlet)": dec_err,
        }

        if build_unitary is None:
            build_unitary = self.num_qubits <= 12

        if build_unitary:
            N0 = self.N ** self.dim
            U  = self.block_encoding()
            block = self.alpha * U[:N0, :N0]
            be_err = float(np.linalg.norm(block - target) / np.linalg.norm(target))
            uu_err = float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
            out["block_encoding_rel_err"] = be_err
            out["unitarity_err"]          = uu_err

        return out
