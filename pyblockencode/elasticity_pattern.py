"""
elasticity_pattern.py — Shift-decomposition block encoding of the 2D
plane-stress Q4 finite element stiffness operator, including full shear.

The 2x2 displacement-block structure is absorbed into a single DOF qubit
d in {0 = x, 1 = y}.  Writing the four blocks in the Pauli basis of that
qubit, with Exy = (X + iY)/2 and Eyx = (X - iY)/2,

    K = (Kxx+Kyy)/2 kron I  +  (Kxx-Kyy)/2 kron Z
      + (Kxy+Kyx)/2 kron X  +  (Kxy-Kyx)/2 kron iY

gives four **Pauli components**.  With C = E/(1-nu^2) and the 1D factors
K, M, G of ``bc.py`` carried in each direction's slot,

    I  :  C(3-nu)/4  ( K_x kron M_y  +  M_x kron K_y )
    Z  :  C(1+nu)/4  ( K_x kron M_y  -  M_x kron K_y )
    X  :  C(1+nu)/4  ( G_x kron G_y^T  +  G_x^T kron G_y )
    iY :  C(1-3nu)/4 ( G_x kron G_y^T  -  G_x^T kron G_y )

Why a fourth component
----------------------
On a periodic grid G is antisymmetric, G^T = -G, so the two Kronecker
products in the shear block collapse onto one, Kxy comes out symmetric, and
the iY coefficient vanishes identically -- three components suffice.  The
same holds for clamped conditions, whose corrections do not touch G's
antisymmetry.

A traction-free correction is diagonal and destroys that antisymmetry.  Kxy
is then no longer symmetric, X alone cannot carry it, and iY appears.  One
free direction is enough: a domain clamped on two opposite edges and free on
the other two already needs four components.

iY = [[0, 1], [-1, 0]] is real, unitary, and renders its term Hermitian, so
it is a legitimate member of a real LCU.

The value nu = 1/3
------------------
The iY coefficient carries (1-3nu)/4, which vanishes at nu = 1/3.  There the
traction-free shear block is symmetric again and the term count drops from
109 to 93.  This is where the transverse coupling nu and the shear term
(1-nu)/2 are equal.

Term counts (m-independent)
---------------------------
    periodic                        17   {I, Z, X}
    all edges clamped               49   {I, Z, X}
    clamped left and right          75   {I, Z, X, iY}
    clamped left only               98   {I, Z, X, iY}
    traction-free                  109   {I, Z, X, iY}   (93 at nu = 1/3)

Boundary conditions are imposed by reflections adjoined to the unitary set
(see ``bc.py``); there is no flag qubit in this construction.

Convention: system register least significant, DOF qubit innermost, so the
linear index is (jx * N + jy) * 2 + d and the encoded block is
alpha * U[:N0, :N0] with N0 = 2 N^2.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from . import bc as _bc
from . import operators

#: the DOF-qubit operators
PAULI: Dict[str, np.ndarray] = {
    "I": np.eye(2),
    "Z": np.diag([1.0, -1.0]),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "iY": np.array([[0.0, 1.0], [-1.0, 0.0]]),
}


def q4_assemble(N: int, nu: float, E: float = 1.0, h: float = 1.0,
                periodic: Tuple[bool, bool] = (False, False)) -> np.ndarray:
    """
    Direct plane-stress Q4 assembly by 2x2 Gauss quadrature.

    DOF ordering (jx * N + jy) * 2 + d, matching the encoding convention.
    ``periodic`` selects wrap-around per direction; an open mesh in a
    direction uses N-1 elements there.
    """
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

    K = np.zeros((2 * N * N, 2 * N * N))
    offs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    nx = N if periodic[0] else N - 1
    ny = N if periodic[1] else N - 1
    for ix in range(nx):
        for iy in range(ny):
            gd = []
            for dx, dy in offs:
                base = 2 * (((ix + dx) % N) * N + ((iy + dy) % N))
                gd += [base, base + 1]
            K[np.ix_(gd, gd)] += Ke
    return K


class ElasticityEncoding:
    """
    Shift-decomposition block encoding of the 2D plane-stress Q4 stiffness.

    Parameters
    ----------
    m  : int
        Qubits per spatial direction; N = 2**m nodes per direction.
    E  : float
        Young's modulus.
    nu : float
        Poisson's ratio in [0, 0.5).
    bc : str, pair, sequence or dict
        Boundary treatment, per direction.  See ``bc.parse_bc``; e.g.
        ``'essential'``, ``'free'``, ``('clamped', 'free')``, or
        ``{'x': ('clamped', 'free'), 'y': 'free'}`` for a cantilever.

    Attributes
    ----------
    lcu_terms()  dict {(U_x, U_y, sigma): coefficient}
    components   the Pauli components actually present
    alpha        subnormalization, constant in N
    num_system   2m + 1
    num_ancilla  ceil(log2 L)  -- PREP only, no flag qubit
    """

    def __init__(self, m: int, E: float = 1.0, nu: float = 0.3,
                 bc: str = "essential"):
        self.m = m
        self.N = 2 ** m
        self.E = E
        self.nu = nu
        self.C = E / (1.0 - nu ** 2)
        self.bc = _bc.parse_bc(bc, 2)
        self._terms: Dict[Tuple[str, str, str], float] | None = None

    # ------------------------------------------------------------------
    # 1D factors
    # ------------------------------------------------------------------

    def factors(self) -> Dict[str, Dict[str, float]]:
        """The six 1D factors, three per direction, under this bc."""
        return {
            "Kx": _bc.factor("K", self.bc[0]), "Ky": _bc.factor("K", self.bc[1]),
            "Mx": _bc.factor("M", self.bc[0]), "My": _bc.factor("M", self.bc[1]),
            "Gx": _bc.factor("G", self.bc[0]), "Gy": _bc.factor("G", self.bc[1]),
        }

    # ------------------------------------------------------------------
    # LCU decomposition
    # ------------------------------------------------------------------

    def lcu_terms(self) -> Dict[Tuple[str, str, str], float]:
        """LCU dictionary {(U_x, U_y, sigma): coefficient}."""
        if self._terms is not None:
            return self._terms
        C, nu = self.C, self.nu
        f = self.factors()
        Gxt, Gyt = _bc.transpose(f["Gx"]), _bc.transpose(f["Gy"])

        KM = _bc.kron(f["Kx"], f["My"])
        MK = _bc.kron(f["Mx"], f["Ky"])
        GGt = _bc.kron(f["Gx"], Gyt)
        GtG = _bc.kron(Gxt, f["Gy"])

        chan = {
            "I": _bc.scale(_bc.add(KM, MK), C * (3 - nu) / 4),
            "Z": _bc.scale(_bc.add(KM, MK, -1.0), C * (1 + nu) / 4),
            "X": _bc.scale(_bc.add(GGt, GtG), C * (1 + nu) / 4),
            "iY": _bc.scale(_bc.add(GGt, GtG, -1.0), C * (1 - 3 * nu) / 4),
        }
        t: Dict[Tuple[str, str, str], float] = {}
        for sigma, d in chan.items():
            for (xl, yl), c in d.items():
                key = (xl, yl, sigma)
                t[key] = t.get(key, 0.0) + c
        self._terms = {k: v for k, v in t.items() if abs(v) > 1e-13}
        return self._terms

    @property
    def components(self) -> Tuple[str, ...]:
        """Pauli components actually present, in the order I, Z, X, iY."""
        present = {k[2] for k in self.lcu_terms()}
        return tuple(s for s in ("I", "Z", "X", "iY") if s in present)

    @property
    def alpha(self) -> float:
        return _bc.alpha(self.lcu_terms())

    @property
    def num_terms(self) -> int:
        return len(self.lcu_terms())

    @property
    def num_system(self) -> int:
        return 2 * self.m + 1          # x-register, y-register, DOF qubit

    @property
    def num_ancilla(self) -> int:
        """PREP qubits only; the reflection construction needs no flag."""
        return math.ceil(math.log2(max(self.num_terms, 2)))

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    def alpha_closed_form(self) -> float:
        """E(33+nu) / (6(1-nu^2)); valid for the periodic and clamped cases."""
        return self.E * (33 + self.nu) / (6 * (1 - self.nu ** 2))

    # ------------------------------------------------------------------
    # Classical reference
    # ------------------------------------------------------------------

    def target(self) -> np.ndarray:
        """
        The operator this LCU encodes, assembled from the displacement
        blocks -- an independent path from ``lcu_terms``.
        """
        N, C, nu = self.N, self.C, self.nu
        f = self.factors()
        d = {k: _bc.dense(v, N) for k, v in f.items()}
        KM = np.kron(d["Kx"], d["My"])
        MK = np.kron(d["Mx"], d["Ky"])
        Kxx = C * (KM + (1 - nu) / 2 * MK)
        Kyy = C * ((1 - nu) / 2 * KM + MK)
        Kxy = C * (nu * np.kron(d["Gx"].T, d["Gy"])
                   + (1 - nu) / 2 * np.kron(d["Gx"], d["Gy"].T))
        Exx = np.array([[1.0, 0], [0, 0]])
        Eyy = np.array([[0.0, 0], [0, 1]])
        Exy = np.array([[0.0, 1], [0, 0]])
        Eyx = np.array([[0.0, 0], [1, 0]])
        return (np.kron(Kxx, Exx) + np.kron(Kyy, Eyy)
                + np.kron(Kxy, Exy) + np.kron(Kxy.T, Eyx))

    def free_indices(self) -> np.ndarray:
        """Degrees of freedom that survive the essential constraints."""
        N = self.N
        keep_x = [j for j in range(N)
                  if not ((j == 0 and self.bc[0] != "periodic"
                           and self.bc[0][0] == "clamped")
                          or (j == N - 1 and self.bc[0] != "periodic"
                              and self.bc[0][1] == "clamped"))]
        keep_y = [j for j in range(N)
                  if not ((j == 0 and self.bc[1] != "periodic"
                           and self.bc[1][0] == "clamped")
                          or (j == N - 1 and self.bc[1] != "periodic"
                              and self.bc[1][1] == "clamped"))]
        return np.array([2 * (a * N + b) + d
                         for a in keep_x for b in keep_y for d in (0, 1)],
                        dtype=int)

    def reference(self) -> np.ndarray:
        """Direct Q4 assembly on the mesh this boundary treatment implies."""
        per = tuple(e == "periodic" for e in self.bc)
        return q4_assemble(self.N, self.nu, self.E, periodic=per)

    # ------------------------------------------------------------------
    # Block encoding
    # ------------------------------------------------------------------

    def block_encoding(self) -> np.ndarray:
        """
        Full block-encoding unitary (dense; small m only).

        Layout: PREP ancilla most significant, then x, y and the DOF qubit,
        so ``alpha * U[:N0, :N0]`` is the operator, N0 = 2 N^2.
        """
        terms = self.lcu_terms()
        keys = list(terms)
        coeffs = np.array([terms[k] for k in keys])
        a = float(np.abs(coeffs).sum())
        L, N = len(keys), self.N
        K2, N0 = 2 ** self.num_ancilla, 2 * N * N

        amps = np.zeros(K2)
        amps[:L] = np.sqrt(np.abs(coeffs) / a)
        P = np.eye(K2)
        P[:, 0] = amps
        Q, R = np.linalg.qr(P)
        Q = Q * np.sign(np.diag(R))

        blocks = []
        for i in range(K2):
            if i < L:
                xl, yl, sl = keys[i]
                blocks.append(float(np.sign(coeffs[i])) * np.kron(
                    _bc.unitary(xl, N),
                    np.kron(_bc.unitary(yl, N), PAULI[sl])))
            else:
                blocks.append(None)

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
        """
        Three independent checks:
        the LCU against the block formula, the block formula against a direct
        Q4 assembly on the free degrees of freedom, and the unitary.
        """
        N0 = 2 * self.N ** 2
        target = self.target()
        nrm = np.linalg.norm(target)

        recon = _bc.dense_nd(self.lcu_terms(), self.N, extra=PAULI)
        out = {
            "m": self.m, "nu": self.nu, "E": self.E, "bc": self.bc,
            "alpha": self.alpha, "num_terms": self.num_terms,
            "components": self.components, "num_qubits": self.num_qubits,
            "decomposition_rel_err": float(
                np.linalg.norm(recon - target) / nrm),
        }

        # On the smallest grid every node can be constrained, leaving no
        # free degrees of freedom and nothing to compare.
        idx = self.free_indices()
        if idx.size:
            ii = np.ix_(idx, idx)
            ref = self.reference()
            out["vs_q4_assembly"] = float(
                np.linalg.norm(target[ii] - ref[ii])
                / np.linalg.norm(ref[ii]))

        if self.num_qubits <= 12:
            U = self.block_encoding()
            out["block_encoding_rel_err"] = float(
                np.linalg.norm(self.alpha * U[:N0, :N0] - target) / nrm)
            out["unitarity_err"] = float(
                np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
        return out

#: Deprecated name kept for the published API; "pattern compression" was
#: renamed "shift decomposition" and the class name followed.
ElasticityPatternEncoding = ElasticityEncoding
