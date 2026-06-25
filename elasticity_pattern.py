"""
elasticity_pattern.py — Exact pattern-compression block encoding of the
2D plane-stress Q4 finite element stiffness operator, including full shear.

This supersedes the preliminary axial-only version.  The encoding is exact
to machine precision for all Poisson ratios and all grid sizes.

Mathematical background
-----------------------
With  C = E/(1−ν²)  and the 1D interior operators

    K1 = tridiag(−1, 2, −1)           (stiffness / FDM Laplacian)
    M1 = tridiag( 1, 4,  1) / 6       (consistent mass)
    G1 = tridiag(−1, 0,  1) / 2       (antisymmetric gradient coupling)

the Q4 global stiffness has the 2×2 displacement-DOF block form

    Kxx = C ( K1⊗M1 + (1−ν)/2 · M1⊗K1 )
    Kyy = C ( (1−ν)/2 · K1⊗M1 + M1⊗K1 )
    Kxy = Kyx = −C(1+ν)/2 · G1⊗G1

Rewriting with the displacement qubit d ∈ {0=x, 1=y}

    K = [(Kxx+Kyy)/2] ⊗ I  +  [(Kxx−Kyy)/2] ⊗ Z  +  Kxy ⊗ X

gives three DOF channels (I / Z / X), each a sum of tensor products of
cyclic-shift operators over the x- and y-grid registers.

Substituting the cyclic-shift expansions of K1, M1, G1 yields a 17-term LCU

    K = Σ_{(p,q,r)} c_{pqr}  S_p^(x) ⊗ S_q^(y) ⊗ σ_r^(dof)

where  S_p, S_q ∈ {I, Sc, Sc†}  and  σ_r ∈ {I, Z, X}.

The 17 terms and subnormalization α = Σ|c_{pqr}| depend only on ν, not N.

Resource summary
----------------
System qubits   : 2m + 1  (m x-qubits + m y-qubits + 1 DOF qubit)
PREP ancilla    : ⌈log₂ 17⌉ = 5  qubits
Flag ancilla    : 1  qubit  (Dirichlet boundary correction)
Total qubits    : 2m + 7

α  (ν=0.00) = 5.500  (E=1)
α  (ν=0.30) = 6.099
α  (ν=0.45) = 6.991
α / ‖K‖   ≈ 1.6   for all tested ν and m

Convention: system register on least-significant qubits; the encoded block
is  alpha · U[:N0, :N0]  with  N0 = 2·N²  (N = 2**m interior nodes/dim).
"""
from __future__ import annotations

import math
import numpy as np
from typing import Dict, Tuple

from . import operators

# ---------------------------------------------------------------------------
# 1-D cyclic-shift dictionaries  {label: coefficient}
# ---------------------------------------------------------------------------

_K1 = {"I": 2.0,        "Sc": -1.0,      "Scd": -1.0}
_M1 = {"I": 4.0 / 6.0, "Sc":  1.0 / 6.0, "Scd":  1.0 / 6.0}
_G1 = {                  "Sc": -0.5,       "Scd":  0.5}


def _kron(a: Dict, b: Dict) -> Dict:
    out: Dict = {}
    for la, ca in a.items():
        for lb, cb in b.items():
            key = (la, lb)
            out[key] = out.get(key, 0.0) + ca * cb
    return out


def _scale(d: Dict, s: float) -> Dict:
    return {k: v * s for k, v in d.items()}


def _add(a: Dict, b: Dict) -> Dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ElasticityPatternEncoding:
    """
    Exact pattern-compression block encoding of the 2D plane-stress Q4
    finite element stiffness matrix.

    Parameters
    ----------
    m  : int   — qubits per spatial dimension; N = 2**m interior nodes/dim
    E  : float — Young's modulus (default 1.0)
    nu : float — Poisson's ratio ∈ [0, 0.5)  (default 0.3)

    Key properties
    --------------
    lcu_terms()  — dict {(Vx, Vy, dof): coefficient}  (17 entries)
    alpha        — subnormalization Σ|c| (constant in N)
    num_terms    — 17 (always)
    num_system   — 2m + 1
    num_ancilla  — 6  (5 PREP + 1 flag)
    num_qubits   — 2m + 7

    target()      — classically assembled Q4 stiffness (dense, Dirichlet)
    decomposition()— Kronecker-block reconstruction (classical, any m)
    block_encoding()— full unitary U (dense, small m only)
    verify()      — dict with error metrics
    """

    def __init__(self, m: int, E: float = 1.0, nu: float = 0.3):
        self.m  = m
        self.N  = 2 ** m
        self.E  = E
        self.nu = nu
        self.C  = E / (1.0 - nu ** 2)
        self._terms: Dict | None = None

    # ------------------------------------------------------------------
    # LCU decomposition
    # ------------------------------------------------------------------

    def lcu_terms(self) -> Dict[Tuple[str, str, str], float]:
        """
        Return the 17-term LCU dictionary {(Vx, Vy, dof): coefficient}.

        Vx, Vy ∈ {'I', 'Sc', 'Scd'}  (cyclic shifts on x / y registers)
        dof    ∈ {'I', 'Z', 'X'}     (Pauli on the DOF qubit)
        """
        if self._terms is not None:
            return self._terms

        C, nu = self.C, self.nu
        KM = _kron(_K1, _M1)
        MK = _kron(_M1, _K1)
        GG = _kron(_G1, _G1)

        # Three DOF channels from the DOF-qubit rewriting:
        #   (Kxx+Kyy)/2 ⊗ I :  coefficient  C*(3-nu)/4 * (KM + MK)
        #   (Kxx-Kyy)/2 ⊗ Z :  coefficient  C*(1+nu)/4 * (KM - MK)
        #   Kxy         ⊗ X :  coefficient  -C*(1+nu)/2 * GG
        chanI = _scale(_add(KM, MK),          C * (3 - nu) / 4.0)
        chanZ = _scale(_add(KM, _scale(MK, -1)), C * (1 + nu) / 4.0)
        chanX = _scale(GG,                    -C * (1 + nu) / 2.0)

        t: Dict[Tuple[str, str, str], float] = {}
        for (xl, yl), c in chanI.items():
            k = (xl, yl, "I");  t[k] = t.get(k, 0.0) + c
        for (xl, yl), c in chanZ.items():
            k = (xl, yl, "Z");  t[k] = t.get(k, 0.0) + c
        for (xl, yl), c in chanX.items():
            k = (xl, yl, "X");  t[k] = t.get(k, 0.0) + c

        self._terms = {k: v for k, v in t.items() if abs(v) > 1e-15}
        return self._terms

    @property
    def alpha(self) -> float:
        """Subnormalization α = Σ|c_{pqr}|, constant in N."""
        return float(sum(abs(v) for v in self.lcu_terms().values()))

    @property
    def num_terms(self) -> int:
        return len(self.lcu_terms())

    @property
    def num_system(self) -> int:
        return 2 * self.m + 1   # m x-qubits + m y-qubits + 1 DOF qubit

    @property
    def num_ancilla(self) -> int:
        L = self.num_terms
        return math.ceil(math.log2(max(L, 2))) + 1   # PREP + flag

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    # ------------------------------------------------------------------
    # Classical reference (Dirichlet trimmed)
    # ------------------------------------------------------------------

    def target(self) -> np.ndarray:
        """
        Classically assembled Q4 plane-stress stiffness matrix (dense).
        Shape: (2·N², 2·N²) where N = 2**m.
        DOF ordering: all x-displacements first, then all y-displacements.
        """
        return operators.elasticity_q4(self.m, self.m, self.E, self.nu)

    # ------------------------------------------------------------------
    # Classical Kronecker-block decomposition (for any m)
    # ------------------------------------------------------------------

    def decomposition(self) -> np.ndarray:
        """
        Reconstruct K from the Kronecker decomposition (classical, dense).

        Uses the exact formula with standard (non-cyclic) tridiagonal
        operators, so the result matches target() to machine precision.
        """
        N = self.N
        C, nu = self.C, self.nu

        def tri(a, b, c, n=N):
            M = b * np.eye(n)
            M += a * np.diag(np.ones(n-1), -1)
            M += c * np.diag(np.ones(n-1),  1)
            return M

        K1 = tri(-1,  2, -1)
        M1 = tri( 1,  4,  1) / 6.0
        G1 = tri(-1,  0,  1) / 2.0

        Kxx = C * (np.kron(M1, K1) + (1 - nu) / 2.0 * np.kron(K1, M1))
        Kyy = C * ((1 - nu) / 2.0 * np.kron(M1, K1) + np.kron(K1, M1))
        Kxy = -C * (1 + nu) / 2.0 * np.kron(G1, G1)

        # DOF-innermost assembly: kron(K_spatial, E_dof)
        # Matches the block_encoding unitary convention:
        # index = (jx*N + jy)*2 + d, with d ∈ {0=x, 1=y}
        Exx = np.array([[1., 0.], [0., 0.]])
        Eyy = np.array([[0., 0.], [0., 1.]])
        Exy = np.array([[0., 1.], [0., 0.]])
        Eyx = np.array([[0., 0.], [1., 0.]])
        return (np.kron(Kxx, Exx) + np.kron(Kxy, Exy)
              + np.kron(Kxy.T, Eyx) + np.kron(Kyy, Eyy))

    # ------------------------------------------------------------------
    # Full block-encoding unitary (dense, small m only)
    # ------------------------------------------------------------------

    @staticmethod
    def _cyclic_shift(lbl: str, j: int, N: int) -> Tuple[int, bool]:
        """Apply one cyclic shift; return (new_j, wrapped)."""
        if lbl == "I":
            return j, False
        if lbl == "Sc":
            return (j + 1) % N, (j == N - 1)
        return (j - 1) % N, (j == 0)   # Scd

    @staticmethod
    def _dof_action(dl: str, d: int) -> Tuple[int, float]:
        """Apply Pauli to DOF qubit; return (new_d, sign)."""
        if dl == "I":
            return d, 1.0
        if dl == "X":
            return 1 - d, 1.0
        # Z
        return d, (1.0 if d == 0 else -1.0)

    def block_encoding(self) -> np.ndarray:
        """
        Build the full block-encoding unitary (dense, for small m).

        Qubit layout (MSB → LSB):  [PREP ancilla | flag | x-reg | y-reg | dof]
        System dimension: N0 = 2·N²  (N = 2**m).
        The encoded block is  alpha · U[:N0, :N0].

        Recommended: m ≤ 3  (total qubits ≤ 13).
        """
        N = self.N
        terms  = self.lcu_terms()
        labels = list(terms)
        coeffs = np.array([terms[k] for k in labels])
        alpha  = float(np.sum(np.abs(coeffs)))
        L      = len(labels)
        K2     = 2 ** math.ceil(math.log2(max(L, 2)))   # PREP ancilla span

        # System block = flag(1) × x(N) × y(N) × dof(2)
        Dflag = 2 * N * N * 2

        def flat(flag, jx, jy, d):
            return ((flag * N + jx) * N + jy) * 2 + d

        SEL = np.zeros((K2 * Dflag, K2 * Dflag))
        for i in range(K2):
            if i < L:
                xl, yl, dl = labels[i]
                sgn = float(np.sign(coeffs[i]))
                for flag in range(2):
                    for jx in range(N):
                        for jy in range(N):
                            for d in range(2):
                                sjx, wx = self._cyclic_shift(xl, jx, N)
                                sjy, wy = self._cyclic_shift(yl, jy, N)
                                sd,  ds = self._dof_action(dl, d)
                                # Boundary flag: set if any shift wraps
                                new_flag = (int(wx or wy) if flag == 0
                                            else 1 - int(wx or wy))
                                row = i * Dflag + flat(new_flag, sjx, sjy, sd)
                                col = i * Dflag + flat(flag, jx, jy, d)
                                SEL[row, col] = sgn * ds
            else:
                SEL[i*Dflag:(i+1)*Dflag, i*Dflag:(i+1)*Dflag] = np.eye(Dflag)

        # PREP: load sqrt(|c_k|/alpha) → QR-orthogonalise
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
        Verify the encoding at two levels.

        Level 1 (always):   ‖decomposition() − target()‖ / ‖target()‖
        Level 2 (if small): ‖alpha · U[:N0,:N0] − target()‖ / ‖target()‖
                             ‖U†U − I‖

        Returns
        -------
        dict with keys:
            nu, E, m, alpha, num_terms, num_qubits
            decomposition_rel_err
            block_encoding_rel_err   (if unitary built)
            unitarity_err            (if unitary built)
        """
        target = self.target()
        dec    = self.decomposition()
        dec_err = float(np.linalg.norm(dec - target) / np.linalg.norm(target))

        out = {
            "nu": self.nu, "E": self.E, "m": self.m,
            "alpha": self.alpha, "num_terms": self.num_terms,
            "num_qubits": self.num_qubits,
            "decomposition_rel_err": dec_err,
        }

        if build_unitary is None:
            build_unitary = self.num_qubits <= 13

        if build_unitary:
            N0 = 2 * self.N ** 2
            U  = self.block_encoding()
            block = self.alpha * U[:N0, :N0]
            be_err = float(np.linalg.norm(block - target) / np.linalg.norm(target))
            uu_err = float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
            out["block_encoding_rel_err"] = be_err
            out["unitarity_err"]          = uu_err

        return out
