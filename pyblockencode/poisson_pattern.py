"""
poisson_pattern.py — Shift-decomposition block encoding of the Poisson
stiffness operator in 1, 2 and 3 dimensions, FDM or FEM.

Each 1D factor is a Laurent polynomial in the cyclic shift S_c.  Boundary
conditions are imposed by adjoining reflection operators to the unitary set
(see ``bc.py``), which costs no ancilla: there is no flag qubit anywhere in
this construction.

    K_dD = sum_i ( kron_{j != i} M_j ) kron_i K_i

with M = I for 'fdm' and the consistent mass matrix for 'fem'.  The
per-direction 1D bases have sizes

    |B_d| = 3 periodic, 5 clamped, 6 one-sided, 7 free.

In two dimensions with 'fem', every factor spans its direction's full basis
and the products merge onto it exactly, so L = |B_x| |B_y| -- 9, 25 and 49
for the three matched treatments.  Two things break that identity elsewhere:
with 'fdm' the non-differentiated factor is the identity, whose basis is a
single label (L = 5, 9, 13 in 2D); and in three dimensions some coefficients
cancel on merging, so L falls below the product (21, 113, 235 for 'fem').
In every case L is a constant, independent of N, and alpha is unchanged by
the boundary treatment.

The boundary treatment is chosen per *direction* and applied to *every*
factor in that direction's slot -- stiffness and mass alike.  Applying the
clamped forms in a direction whose boundary rows survive the solve is an
O(1) error (4/3 for 2D Poisson); see ``bc.py`` and Appendix B of the paper.

Resource summary (2D FEM, clamped)
----------------------------------
System qubits : dim * m
PREP ancilla  : ceil(log2 L)          (no flag qubit)
Total qubits  : dim * m + ceil(log2 L)
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from . import bc as _bc
from . import operators


class PoissonEncoding:
    """
    Shift-decomposition block encoding of the Poisson stiffness operator.

    Parameters
    ----------
    m    : int
        Qubits per spatial direction; the grid has N = 2**m nodes/direction.
    dim  : int
        Spatial dimension, 1, 2 or 3.
    disc : {'fdm', 'fem'}
        Discretization.  'fdm' uses the identity in the non-differentiated
        directions; 'fem' uses the consistent mass matrix.
    bc   : str, pair, sequence or dict
        Boundary treatment.  A single value applies to every direction:

            'periodic'                    the raw cyclic operator
            'essential' (or 'clamped')    Dirichlet, via reflections
            'free' (or 'neumann')         traction-free / zero Neumann

        A pair such as ``('clamped', 'free')`` sets the two *ends* of every
        direction.  A sequence of length ``dim``, or a dict keyed by
        ``'x'``/``'y'``/``'z'``, sets each direction independently, e.g.

            bc={'x': ('clamped', 'free'), 'y': 'free'}

    Attributes
    ----------
    lcu_terms()   dict {(label_x, label_y, ...): coefficient}
    alpha         subnormalization sum |c_k|, constant in N
    num_terms     number of LCU terms, constant in N
    num_system    dim * m
    num_ancilla   ceil(log2 L)  -- PREP only
    num_qubits    total
    target()      the operator this LCU encodes, assembled densely
    block_encoding()  the full unitary (dense; small m only)
    verify()      error metrics
    """

    _MASS_KIND = {"fem": "M", "fdm": "Iop"}

    def __init__(self, m: int, dim: int = 1, disc: str = "fdm",
                 bc: str = "essential"):
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2 or 3; got {dim}")
        if disc not in ("fdm", "fem"):
            raise ValueError(f"disc must be 'fdm' or 'fem'; got {disc!r}")
        self.m = m
        self.dim = dim
        self.disc = disc
        self.N = 2 ** m
        self.bc = _bc.parse_bc(bc, dim)
        self._terms: Dict[Tuple[str, ...], float] | None = None

    # ------------------------------------------------------------------
    # LCU decomposition
    # ------------------------------------------------------------------

    def _factor(self, kind: str, d: int) -> Dict[str, float]:
        return _bc.factor(kind, self.bc[d])

    def lcu_terms(self) -> Dict[Tuple[str, ...], float]:
        """The LCU term dictionary, keyed by one label per direction."""
        if self._terms is not None:
            return self._terms
        mass_kind = self._MASS_KIND[self.disc]
        total: Dict[Tuple[str, ...], float] = {}
        for active in range(self.dim):
            term: Dict = {(): 1.0}
            for d in range(self.dim):
                kind = "K" if d == active else mass_kind
                term = _bc.kron(term, self._factor(kind, d))
            total = _bc.add(total, term)
        self._terms = total
        return self._terms

    @property
    def alpha(self) -> float:
        return _bc.alpha(self.lcu_terms())

    @property
    def num_terms(self) -> int:
        return len(self.lcu_terms())

    @property
    def num_system(self) -> int:
        return self.dim * self.m

    @property
    def num_ancilla(self) -> int:
        """PREP qubits only; the reflection construction needs no flag."""
        return math.ceil(math.log2(max(self.num_terms, 2)))

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    @property
    def basis_sizes(self) -> Tuple[int, ...]:
        """
        Size of the 1D unitary basis each direction's boundary treatment
        requires.  This bounds the term count; see the module docstring for
        when the bound is attained.
        """
        return tuple(_bc.basis_size(e) for e in self.bc)

    # ------------------------------------------------------------------
    # Classical reference
    # ------------------------------------------------------------------

    def target(self) -> np.ndarray:
        """
        The operator this LCU encodes, assembled from dense 1D factors.

        For clamped conditions this coincides with the classical interior
        Dirichlet assembly in ``operators`` (checked by ``verify``); for the
        other treatments it is the corresponding periodic or open-mesh
        operator.
        """
        N, mass_kind = self.N, self._MASS_KIND[self.disc]
        A = np.zeros((N ** self.dim,) * 2)
        for active in range(self.dim):
            M = None
            for d in range(self.dim):
                kind = "K" if d == active else mass_kind
                F = _bc.dense(self._factor(kind, d), N)
                M = F if M is None else np.kron(M, F)
            A += M
        return A

    def reference(self) -> np.ndarray | None:
        """Classical assembly from ``operators``, when one applies."""
        if any(e != ("clamped", "clamped") for e in self.bc):
            return None
        fn = {
            (1, "fdm"): operators.poisson_1d_fdm, (1, "fem"): operators.poisson_1d_fem,
            (2, "fdm"): operators.poisson_2d_fdm, (2, "fem"): operators.poisson_2d_fem,
            (3, "fdm"): operators.poisson_3d_fdm, (3, "fem"): operators.poisson_3d_fem,
        }[(self.dim, self.disc)]
        return fn(self.m)

    # ------------------------------------------------------------------
    # Block encoding
    # ------------------------------------------------------------------

    def block_encoding(self) -> np.ndarray:
        """
        The full block-encoding unitary (dense; practical for small m).

        Layout: PREP ancilla most significant, system least significant, so
        ``alpha * U[:N0, :N0]`` is the encoded operator with N0 = N**dim.
        """
        terms = self.lcu_terms()
        keys = list(terms)
        coeffs = np.array([terms[k] for k in keys])
        a = float(np.abs(coeffs).sum())
        L = len(keys)
        na = self.num_ancilla
        K2, N0 = 2 ** na, self.N ** self.dim

        amps = np.zeros(K2)
        amps[:L] = np.sqrt(np.abs(coeffs) / a)
        P = np.eye(K2)
        P[:, 0] = amps
        Q, R = np.linalg.qr(P)
        Q = Q * np.sign(np.diag(R))

        # SELECT is block diagonal, so build U blockwise:
        #   U[a, b] = sum_i Q[i, a] Q[i, b] * SEL_i
        # avoiding the full-size Kronecker temporaries.
        blocks = []
        for i in range(K2):
            if i < L:
                U = _bc.unitary(keys[i][0], self.N)
                for lbl in keys[i][1:]:
                    U = np.kron(U, _bc.unitary(lbl, self.N))
                blocks.append(float(np.sign(coeffs[i])) * U)
            else:
                blocks.append(None)                      # identity padding

        out = np.zeros((K2 * N0, K2 * N0))
        for ai in range(K2):
            for bi in range(K2):
                acc = np.zeros((N0, N0))
                for i in range(K2):
                    w = Q[i, ai] * Q[i, bi]
                    if abs(w) < 1e-15:
                        continue
                    if blocks[i] is None:
                        acc[np.diag_indices(N0)] += w
                    else:
                        acc += w * blocks[i]
                out[ai * N0:(ai + 1) * N0, bi * N0:(bi + 1) * N0] = acc
        return out

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> dict:
        """Reconstruct the operator from the LCU and check the unitary."""
        N0 = self.N ** self.dim
        target = self.target()
        nrm = np.linalg.norm(target)

        recon = _bc.dense_nd(self.lcu_terms(), self.N)
        dec_err = float(np.linalg.norm(recon - target) / nrm)

        out = {
            "dim": self.dim, "disc": self.disc, "m": self.m,
            "bc": self.bc, "alpha": self.alpha, "num_terms": self.num_terms,
            "num_qubits": self.num_qubits,
            "decomposition_rel_err": dec_err,
        }

        ref = self.reference()
        if ref is not None:
            out["vs_classical_assembly"] = float(
                np.linalg.norm(target - ref) / np.linalg.norm(ref))

        if self.num_qubits <= 12:
            U = self.block_encoding()
            out["block_encoding_rel_err"] = float(
                np.linalg.norm(self.alpha * U[:N0, :N0] - target) / nrm)
            out["unitarity_err"] = float(
                np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
        return out

#: Deprecated name kept for the published API; "pattern compression" was
#: renamed "shift decomposition" and the class name followed.
PoissonPatternEncoding = PoissonEncoding
