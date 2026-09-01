"""
linear_circuits.py - block-encoding circuits with O(m) Toffoli depth per shift.

Drop this file into pyblockencode/ alongside qiskit_encoding.py.  It provides

    LinearPoissonCircuit(m, dim, disc)      Poisson / Laplacian, 1D-3D, FDM+FEM
    LinearElasticityCircuit(m, E, nu)       2D plane-stress Q4 elasticity

with the same public API as the existing classes (alpha, num_terms,
num_system, num_ancilla, num_qubits, target, circuit, unitary, verify), so
they are drop-in replacements.

Why this exists
---------------
The cyclic shift S_c is the only part of the encoding whose cost grows with
m, so it alone sets the complexity of the whole block encoding.  Implementing
it as a ladder of multi-controlled X gates with 0..m-1 controls -- the
construction in qiskit_encoding._shift_gate, and the one in the reference
implementation of Kharazi et al. (arXiv:2407.18347) -- costs O(m^3) Toffolis
once Qiskit synthesizes each MCX ancilla-free, or O(m^2) with dirty ancillas.
Both papers nonetheless quote O(m), citing Gidney's borrowed-ancilla
incrementer.  This module actually implements a linear-Toffoli incrementer,
so the O(log N) claim is measured rather than asserted.

The construction
----------------
Bit x[k] of an increment flips iff x[0] & ... & x[k-1].  Compute those prefix
ANDs up a ladder into m-1 clean ancillas, then unwind, applying each flip on
the way back down so that every Toffoli sees unmodified inputs:

    p[0] := x[0];   p[k] := p[k-1] AND x[k]        (ancilla, k = 1..m-1)

The wrap predicate comes free: p[m-1] = AND of all m bits is exactly the
condition "j = 2^m - 1", i.e. the increment wraps.  So the Dirichlet boundary
flag costs one extra Toffoli instead of a separate m-controlled X.

Decrement is the same circuit conjugated by X on every qubit, since
X^(x)m . inc . X^(x)m maps j -> j-1 and turns "all ones" into "all zeros".

Cost per controlled shift-with-flag: 3m - 2 Toffolis, m-1 clean ancillas
(returned to |0> and reused by every term and every register).

SELECT is built by hand rather than through Gate.control(), because
.control(k) on a composite gate makes Qiskit control every gate inside,
including the prefix ladder.  Instead each term computes a single control
line ctrl = (anc == i) with one MCX, and only the flip gates are controlled
on it; an uncomputed prefix ladder is the identity when ctrl = 0, so it does
not need controlling.  The sign of c_i is a Z on ctrl.

Qubit layout (system on the least significant qubits, so the encoded block is
the contiguous corner alpha * U[:N0, :N0]):

    Poisson     : [d0 .. d_{dim-1}] [flag x dim] [ctrl] [prefix x (m-1)] [prep]
    Elasticity  : [dof] [y] [x]     [flag x 2]   [ctrl] [prefix x (m-1)] [prep]
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator

from .poisson_pattern import PoissonPatternEncoding
from .elasticity_pattern import ElasticityPatternEncoding


# ---------------------------------------------------------------------------
# the primitive: controlled cyclic shift, with wrap flag, in O(m) Toffolis
# ---------------------------------------------------------------------------

def _shift_with_flag(qc: QuantumCircuit, lbl: str, x, p, flag, ctrl) -> None:
    """Append  ctrl-controlled  S_c^{+-1}  on register x, flagging wraps.

    x    : list of m qubits, x[0] the LSB
    p    : list of >= m-1 clean ancillas (restored to |0>)
    flag : qubit XORed with the wrap predicate (conditioned on ctrl)
    ctrl : single control qubit

    Cost: 3m - 2 Toffolis.  No-op for lbl == 'I'.
    """
    if lbl == 'I':
        return
    m = len(x)
    dec = (lbl == 'Scd')

    if dec:                       # decrement = X-conjugated increment
        for q in x:
            qc.x(q)

    if m == 1:
        qc.ccx(ctrl, x[0], flag)  # wraps iff the single bit is 1
        qc.cx(ctrl, x[0])
    else:
        # p_full[k] holds AND(x[0..k]); p_full[0] is x[0] itself, no ancilla
        pf = [x[0]] + [p[k - 1] for k in range(1, m)]

        for k in range(1, m - 1):             # ladder up
            qc.ccx(pf[k - 1], x[k], pf[k])

        qc.ccx(pf[m - 2], x[m - 1], pf[m - 1])  # AND of all m bits = wrap
        qc.ccx(ctrl, pf[m - 1], flag)           # flag ^= ctrl & wrap
        qc.ccx(pf[m - 2], x[m - 1], pf[m - 1])  # uncompute (x[m-1] still clean)

        qc.ccx(ctrl, pf[m - 2], x[m - 1])       # flip the top bit

        for k in range(m - 2, 0, -1):           # unwind, flipping on the way
            qc.ccx(pf[k - 1], x[k], pf[k])
            qc.ccx(ctrl, pf[k - 1], x[k])

        qc.cx(ctrl, x[0])

    if dec:
        for q in x:
            qc.x(q)


def _set_ctrl(qc: QuantumCircuit, anc, ctrl, i: int) -> None:
    """ctrl ^= (anc == i).  Self-inverse, so the same call uncomputes it."""
    na = len(anc)
    bits = [(i >> b) & 1 for b in range(na)]
    for b, v in enumerate(bits):
        if not v:
            qc.x(anc[b])
    qc.mcx(list(anc), ctrl)
    for b, v in enumerate(bits):
        if not v:
            qc.x(anc[b])


def _prep_amps(coeffs: np.ndarray, width: int) -> np.ndarray:
    amps = np.zeros(2 ** width)
    alpha = float(np.sum(np.abs(coeffs)))
    for i, c in enumerate(coeffs):
        amps[i] = math.sqrt(abs(c) / alpha)
    return amps


# ---------------------------------------------------------------------------
# Poisson / Laplacian
# ---------------------------------------------------------------------------

class LinearPoissonCircuit:
    """O(m)-Toffoli block encoding of the d-dimensional Laplacian (FDM or FEM).

    Parameters mirror PoissonPatternEncoding: m qubits per dimension
    (N = 2**m interior nodes), dim in {1,2,3}, disc in {'fdm','fem'}.
    """

    def __init__(self, m: int, dim: int = 1, disc: str = 'fdm'):
        self.m = m
        self.dim = dim
        self.disc = disc
        self.enc = PoissonPatternEncoding(m=m, dim=dim, disc=disc)

    @property
    def alpha(self) -> float:
        return self.enc.alpha

    @property
    def num_terms(self) -> int:
        return self.enc.num_terms

    @property
    def num_system(self) -> int:
        return self.dim * self.m

    @property
    def num_ancilla(self) -> int:
        na_prep = int(math.ceil(math.log2(max(self.num_terms, 2))))
        return na_prep + self.dim + 1 + max(self.m - 1, 0)

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    def target(self) -> np.ndarray:
        return self.enc.target()

    def circuit(self) -> QuantumCircuit:
        m, dim = self.m, self.dim
        terms = self.enc.lcu_terms()
        labels = list(terms)
        coeffs = np.array([terms[k] for k in labels])
        na_prep = int(math.ceil(math.log2(max(len(labels), 2))))
        npre = max(m - 1, 0)

        qr_dims = [QuantumRegister(m, f'd{i}') for i in range(dim)]
        qr_flag = QuantumRegister(dim, 'flag')
        qr_ctrl = QuantumRegister(1, 'ctrl')
        qr_pre = QuantumRegister(npre, 'pre') if npre else None
        qr_anc = QuantumRegister(na_prep, 'anc')

        regs = qr_dims + [qr_flag, qr_ctrl]
        if qr_pre is not None:
            regs.append(qr_pre)
        regs.append(qr_anc)
        qc = QuantumCircuit(*regs, name=f'LinPoisson{dim}D_{self.disc.upper()}')

        prep = StatePreparation(_prep_amps(coeffs, na_prep), label='Prep')
        qc.append(prep, qr_anc)

        pre = list(qr_pre) if qr_pre is not None else []
        ctrl = qr_ctrl[0]
        for i, key in enumerate(labels):
            _set_ctrl(qc, qr_anc, ctrl, i)
            if coeffs[i] < 0:
                qc.z(ctrl)
            for d, lbl in enumerate(key):
                _shift_with_flag(qc, lbl, list(qr_dims[d]), pre,
                                 qr_flag[d], ctrl)
            _set_ctrl(qc, qr_anc, ctrl, i)

        qc.append(prep.inverse(), qr_anc)
        return qc

    def unitary(self) -> np.ndarray:
        return Operator(self.circuit()).data

    def verify(self) -> dict:
        N0 = 2 ** self.num_system
        target = self.target()
        U = self.unitary()
        block = self.alpha * U[:N0, :N0]
        return {
            'dim': self.dim, 'disc': self.disc, 'm': self.m,
            'alpha': self.alpha, 'num_terms': self.num_terms,
            'num_qubits': self.num_qubits,
            'block_encoding_rel_err':
                float(np.linalg.norm(block - target) / np.linalg.norm(target)),
            'unitarity_err':
                float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))),
        }


# ---------------------------------------------------------------------------
# 2D plane-stress Q4 elasticity
# ---------------------------------------------------------------------------

class LinearElasticityCircuit:
    """O(m)-Toffoli block encoding of the 2D plane-stress Q4 stiffness."""

    def __init__(self, m: int, E: float = 1.0, nu: float = 0.3):
        self.m = m
        self.E = E
        self.nu = nu
        self.enc = ElasticityPatternEncoding(m=m, E=E, nu=nu)

    @property
    def alpha(self) -> float:
        return self.enc.alpha

    @property
    def num_terms(self) -> int:
        return self.enc.num_terms

    @property
    def num_system(self) -> int:
        return 2 * self.m + 1

    @property
    def num_ancilla(self) -> int:
        na_prep = int(math.ceil(math.log2(max(self.num_terms, 2))))
        return na_prep + 2 + 1 + max(self.m - 1, 0)

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    def target(self) -> np.ndarray:
        return self.enc.target()

    def circuit(self) -> QuantumCircuit:
        m = self.m
        terms = self.enc.lcu_terms()
        labels = list(terms)
        coeffs = np.array([terms[k] for k in labels])
        na_prep = int(math.ceil(math.log2(max(len(labels), 2))))
        npre = max(m - 1, 0)

        # target() indexes as (jx*N + jy)*2 + d, so dof is the LSB and jx the
        # MSB of the system block: declare dof, y, x in that order.
        qr_dof = QuantumRegister(1, 'dof')
        qr_y = QuantumRegister(m, 'y')
        qr_x = QuantumRegister(m, 'x')
        qr_flag = QuantumRegister(2, 'flag')
        qr_ctrl = QuantumRegister(1, 'ctrl')
        qr_pre = QuantumRegister(npre, 'pre') if npre else None
        qr_anc = QuantumRegister(na_prep, 'anc')

        regs = [qr_dof, qr_y, qr_x, qr_flag, qr_ctrl]
        if qr_pre is not None:
            regs.append(qr_pre)
        regs.append(qr_anc)
        qc = QuantumCircuit(*regs, name='LinElasticityQ4')

        prep = StatePreparation(_prep_amps(coeffs, na_prep), label='Prep')
        qc.append(prep, qr_anc)

        pre = list(qr_pre) if qr_pre is not None else []
        ctrl = qr_ctrl[0]
        for i, (xl, yl, dl) in enumerate(labels):
            _set_ctrl(qc, qr_anc, ctrl, i)
            if coeffs[i] < 0:
                qc.z(ctrl)
            _shift_with_flag(qc, xl, list(qr_x), pre, qr_flag[0], ctrl)
            _shift_with_flag(qc, yl, list(qr_y), pre, qr_flag[1], ctrl)
            if dl == 'X':
                qc.cx(ctrl, qr_dof[0])
            elif dl == 'Z':
                qc.cz(ctrl, qr_dof[0])
            _set_ctrl(qc, qr_anc, ctrl, i)

        qc.append(prep.inverse(), qr_anc)
        return qc

    def unitary(self) -> np.ndarray:
        return Operator(self.circuit()).data

    def verify(self) -> dict:
        N0 = 2 ** self.num_system
        target = self.target()
        U = self.unitary()
        block = self.alpha * U[:N0, :N0]
        return {
            'nu': self.nu, 'E': self.E, 'm': self.m,
            'alpha': self.alpha, 'num_terms': self.num_terms,
            'num_qubits': self.num_qubits,
            'block_encoding_rel_err':
                float(np.linalg.norm(block - target) / np.linalg.norm(target)),
            'unitarity_err':
                float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))),
        }


# ---------------------------------------------------------------------------
# column-wise verification (no dense Operator, so it reaches far larger m)
# ---------------------------------------------------------------------------

def verify_columns(enc, atol: float = 1e-9) -> dict:
    """Check alpha * U[:N0, :N0] == target() one column at a time.

    Operator(circuit) costs 4^(#qubits) and becomes impractical around 12
    qubits.  Each column here is a single statevector simulation, 2^(#qubits),
    which reaches roughly twice the m.  Works for either Linear*Circuit class.
    """
    from qiskit.quantum_info import Statevector
    N0 = 2 ** enc.num_system
    qc = enc.circuit()
    nq = qc.num_qubits
    target = enc.target()
    block = np.zeros((N0, N0), dtype=complex)
    for j in range(N0):
        init = np.zeros(2 ** nq, dtype=complex)
        init[j] = 1.0                      # ancillas are the high bits => |0>
        sv = Statevector(init).evolve(qc).data
        block[:, j] = enc.alpha * sv[:N0]
    err = float(np.linalg.norm(block - target) / np.linalg.norm(target))
    return {'m': enc.m, 'num_qubits': nq, 'N0': N0,
            'block_encoding_rel_err': err, 'passed': err < atol}
