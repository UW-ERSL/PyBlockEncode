"""
homogenization_circuit.py — Qiskit circuit for the periodic two-phase
elasticity block encoding of ``homogenization.py``.

Layout follows ``linear_circuits.py``: system registers are declared first, so
the encoded block is the contiguous top-left corner

    alpha * U[:N0, :N0]  =  K,        N0 = 2 N^2 = 2^(2m+1),

and every ancilla is restored to |0>.  Registers, in declaration order:

    [dof (1)] [y (m)] [x (m)] [ctrl (1)] [pre (m-1)] [orc (0 or 2)] [prep (6)]

SELECT is built by hand rather than through ``Gate.control(6)``, which would
control every gate inside a composite branch and synthesize tens of thousands
of CX.  Each term instead computes one control line ctrl = (anc == i) with a
single MCX and conditions only the gates that need it.

The oracle
----------
Every term carries either the identity or one of

    R_s = S^(a,b) R_chi S^(-a,-b),     (a,b) = (0,0), (-1,0), (0,-1), (-1,-1),

so the microstructure enters through the single phase oracle

    R_chi |ix,iy> = (-1)^{chi(ix,iy)} |ix,iy>

conjugated by cyclic shifts.  The conjugating shifts are applied
uncontrolled: on the ctrl = 0 branch they cancel against their inverses, so
only the phase itself is controlled.  Three constructions:

``mcz``     A dyadic square of side 2^j aligned to a multiple of its side.
            Membership is a conjunction of bit equalities on the top m-j bits
            of each coordinate, so R_chi is one 2(m-j)-controlled Z: no
            arithmetic, no comparator, no ancilla, and a cost set by the
            volume fraction alone rather than by m.

``compare`` A general a x b rectangle.  Two constant comparators per
            coordinate mark the interval and a controlled phase fires when
            both hold.  O(m) Toffolis and two clean ancillas, buying
            continuous volume fraction.

``diagonal``Any chi at all, as an explicit +-1 diagonal.  A verification
            device, not a circuit to cost: it carries N^2 bits.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector

from .homogenization import ELEM_OFFSETS, HomogenizationEncoding, Inclusion
from .linear_circuits import _ladder_up, _prefix, _shift_ctrl


# ---------------------------------------------------------------------------
# multi-controlled gates that survive transpilation
# ---------------------------------------------------------------------------
#
# Qiskit 2.5 synthesizes ``QuantumCircuit.mcx`` with four or more controls by
# borrowing spare circuit qubits as ancillas, and the result does not agree
# with the multi-controlled X unless those qubits happen to be |0>.  In a
# block encoding they are not: the prefix ladder and the coordinate registers
# are live.  The transpiled circuit is then silently wrong -- it is not a
# rounding difference but an order-one error -- and any gate count taken from
# it counts a circuit that computes something else.
#
# Synthesizing the multi-controlled gates here, ancilla-free, leaves the
# transpiler nothing to guess.

def _mcx(qc: QuantumCircuit, controls, target) -> None:
    """Ancilla-free multi-controlled X."""
    from qiskit.synthesis import synth_mcx_noaux_v24
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


def _mcz(qc: QuantumCircuit, qubits) -> None:
    """Phase -1 on the all-ones state of ``qubits``; ancilla-free."""
    qs = list(qubits)
    if len(qs) == 1:
        qc.z(qs[0])
        return
    qc.h(qs[-1])
    _mcx(qc, qs[:-1], qs[-1])
    qc.h(qs[-1])


def _set_ctrl(qc: QuantumCircuit, anc, ctrl, i: int) -> None:
    """ctrl ^= (anc == i).  Self-inverse, so the same call uncomputes it."""
    zeros = [anc[b] for b in range(len(anc)) if not (i >> b) & 1]
    if zeros:
        qc.x(zeros)
    _mcx(qc, list(anc), ctrl)
    if zeros:
        qc.x(zeros)


# ---------------------------------------------------------------------------
# uncontrolled cyclic shift, same O(m) prefix ladder
# ---------------------------------------------------------------------------

def _shift_plain(qc: QuantumCircuit, k: int, x, p) -> None:
    """S_c^{+-1} on register x (x[0] the LSB), using clean prefix ancillas."""
    if k == 0:
        return
    m = len(x)
    dec = (k < 0)
    if dec:                                   # decrement = X-conjugated inc
        qc.x(x)
    if m == 1:
        qc.x(x[0])
    else:
        pf = _prefix(x, p)
        _ladder_up(qc, x, pf)
        qc.cx(pf[m - 2], x[m - 1])
        for j in range(m - 2, 0, -1):
            qc.ccx(pf[j - 1], x[j], pf[j])    # unwind
            qc.cx(pf[j - 1], x[j])            # flip on the way down
        qc.x(x[0])
    if dec:
        qc.x(x)


def _less_than(m: int, c: int) -> QuantumCircuit:
    """
    Flip the target when the m-bit register (LSB first) holds a value < c.

    Constant comparator: a register value is below c exactly when, at the
    highest bit where the two differ, c has a 1 and the register a 0.  Each
    such bit contributes one multi-controlled X conditioned on agreement
    above it.
    """
    qc = QuantumCircuit(m + 1)
    reg, tgt = list(range(m)), m
    if c <= 0:
        return qc
    if c >= 2 ** m:
        qc.x(tgt)
        return qc
    higher: List[int] = []
    for b in range(m - 1, -1, -1):
        if (c >> b) & 1:
            ctrl = higher + [reg[b]]
            zeros = [reg[b]] + [reg[i] for i in higher if not (c >> i) & 1]
            if zeros:
                qc.x(zeros)
            _mcx(qc, ctrl, tgt)
            if zeros:
                qc.x(zeros)
        higher.append(reg[b])
    return qc


def _in_window(m: int, lo: int, length: int) -> QuantumCircuit:
    """Flip the target when the register lies in [lo, lo+length) mod 2^m."""
    qc = QuantumCircuit(m + 1)
    hi = lo + length
    if length <= 0:
        return qc
    if length >= 2 ** m:
        qc.x(m)
        return qc
    if hi <= 2 ** m:
        qc.compose(_less_than(m, hi), inplace=True)
        qc.compose(_less_than(m, lo), inplace=True)
    else:                                     # wrapped window
        qc.x(m)
        qc.compose(_less_than(m, lo), inplace=True)
        qc.compose(_less_than(m, hi - 2 ** m), inplace=True)
    return qc


# ---------------------------------------------------------------------------
# the reflection oracle
# ---------------------------------------------------------------------------

class ReflectionOracle:
    """R_chi as a phase oracle on the coordinate registers."""

    def __init__(self, inclusion: Inclusion, method: str = "auto"):
        self.inc = inclusion
        self.m = int(round(math.log2(inclusion.N)))
        if method == "auto":
            method = ("mcz" if inclusion.dyadic else
                      "compare" if inclusion.shape in ("square", "rectangle")
                      else "diagonal")
        self.method = method

    @property
    def num_ancilla(self) -> int:
        return 2 if self.method == "compare" else 0

    # -- application --------------------------------------------------------

    def apply(self, qc: QuantumCircuit, qy, qx, orc, ctrl) -> None:
        """Append ctrl-controlled R_chi.  Ancillas are restored to |0>."""
        getattr(self, f"_{self.method}")(qc, qy, qx, orc, ctrl)

    def _mcz(self, qc, qy, qx, orc, ctrl) -> None:
        m, inc = self.m, self.inc
        side = inc.extent[0]
        if side == 0:
            return
        j = int(round(math.log2(side)))
        nb = m - j
        if nb == 0:                            # inclusion fills the cell
            qc.z(ctrl)
            return
        vx, vy = inc.origin[0] >> j, inc.origin[1] >> j
        ctrls = ([qx[j + i] for i in range(nb)]
                 + [qy[j + i] for i in range(nb)] + [ctrl])
        zeros = ([qx[j + i] for i in range(nb) if not (vx >> i) & 1]
                 + [qy[j + i] for i in range(nb) if not (vy >> i) & 1])
        if zeros:
            qc.x(zeros)
        _mcz(qc, ctrls)
        if zeros:
            qc.x(zeros)

    def _compare(self, qc, qy, qx, orc, ctrl) -> None:
        m, inc = self.m, self.inc
        ax, ay = inc.extent
        ox, oy = inc.origin
        fx, fy = orc[0], orc[1]
        wx, wy = _in_window(m, ox, ax), _in_window(m, oy, ay)
        qc.compose(wx, qubits=list(qx) + [fx], inplace=True)
        qc.compose(wy, qubits=list(qy) + [fy], inplace=True)
        _mcz(qc, [ctrl, fx, fy])
        qc.compose(wy.inverse(), qubits=list(qy) + [fy], inplace=True)
        qc.compose(wx.inverse(), qubits=list(qx) + [fx], inplace=True)

    def _diagonal(self, qc, qy, qx, orc, ctrl) -> None:
        from qiskit.circuit.library import DiagonalGate
        diag = list(1.0 - 2.0 * self.inc.chi.ravel())
        qc.append(DiagonalGate(diag).control(1), [ctrl] + list(qy) + list(qx))

    # -- standalone form, for testing and costing ---------------------------

    def circuit(self) -> QuantumCircuit:
        """R_chi alone, on [y, x, ancilla, ctrl] with ctrl forced to |1>."""
        m = self.m
        n = 2 * m + self.num_ancilla
        qc = QuantumCircuit(n + 1, name="R_chi")
        qy, qx = list(range(m)), list(range(m, 2 * m))
        orc = list(range(2 * m, n))
        qc.x(n)
        self.apply(qc, qy, qx, orc, n)
        qc.x(n)
        return qc

    def cost(self) -> dict:
        from qiskit import transpile
        qc = self.circuit()
        t = transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
        return {"method": self.method, "ancilla": self.num_ancilla,
                "cx": int(t.count_ops().get("cx", 0)), "depth": int(t.depth())}


# ---------------------------------------------------------------------------
# the block encoding circuit
# ---------------------------------------------------------------------------

class HomogenizationCircuit:
    """
    PREP - SELECT - PREP^dagger block encoding of the periodic two-phase
    plane-stress Q4 operator.

    Parameters
    ----------
    m, E1, E2, nu, inclusion, shape, vf : as for ``HomogenizationEncoding``.
    oracle : {'auto', 'mcz', 'compare', 'diagonal'}
    """

    def __init__(self, m: int, E1: float = 3.0, E2: float = 1.0,
                 nu: float = 0.3, inclusion=None, shape: str = "square",
                 vf: float = 0.25, oracle: str = "auto"):
        self.enc = HomogenizationEncoding(m, E1, E2, nu, inclusion=inclusion,
                                          shape=shape, vf=vf)
        if self.enc.inclusion is None:
            raise ValueError("the circuit path needs an Inclusion, not a raw "
                             "chi array")
        self.m = m
        self.oracle = ReflectionOracle(self.enc.inclusion, oracle)

    alpha = property(lambda self: self.enc.alpha)
    num_terms = property(lambda self: self.enc.num_terms)
    num_system = property(lambda self: self.enc.num_system)

    @property
    def num_prep(self) -> int:
        return math.ceil(math.log2(max(self.num_terms, 2)))

    @property
    def num_ancilla(self) -> int:
        """PREP + the SELECT control line + the prefix ladder + the oracle."""
        return (self.num_prep + 1 + max(self.m - 1, 0)
                + self.oracle.num_ancilla)

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    def target(self) -> np.ndarray:
        return self.enc.target()

    # -- circuit ------------------------------------------------------------

    def circuit(self) -> QuantumCircuit:
        m = self.m
        terms = self.enc.lcu_terms()
        keys = list(terms)
        coeffs = np.array([terms[k] for k in keys])
        alpha = float(np.abs(coeffs).sum())
        npre, no = max(m - 1, 0), self.oracle.num_ancilla

        qr_d = QuantumRegister(1, "dof")
        qr_y = QuantumRegister(m, "y")
        qr_x = QuantumRegister(m, "x")
        qr_c = QuantumRegister(1, "ctrl")
        qr_p = QuantumRegister(npre, "pre") if npre else None
        qr_o = QuantumRegister(no, "orc") if no else None
        qr_a = QuantumRegister(self.num_prep, "prep")

        regs = [qr_d, qr_y, qr_x, qr_c]
        if qr_p is not None:
            regs.append(qr_p)
        if qr_o is not None:
            regs.append(qr_o)
        regs.append(qr_a)
        qc = QuantumCircuit(*regs, name="HomogenizationQ4")

        amps = np.zeros(2 ** self.num_prep)
        amps[:len(keys)] = np.sqrt(np.abs(coeffs) / alpha)
        prep = StatePreparation(amps, label="Prep")
        qc.append(prep, qr_a)

        pre = list(qr_p) if qr_p is not None else []
        orc = list(qr_o) if qr_o is not None else []
        ctrl, dof = qr_c[0], qr_d[0]
        xs, ys = list(qr_x), list(qr_y)

        for i, (xl, yl, rl, dl) in enumerate(keys):
            _set_ctrl(qc, qr_a, ctrl, i)
            if coeffs[i] < 0:
                qc.z(ctrl)
            # DOF Pauli; iY = Z @ X, so X first
            if dl in ("X", "iY"):
                qc.cx(ctrl, dof)
            if dl in ("Z", "iY"):
                qc.cz(ctrl, dof)
            # spatial shifts, controlled
            _shift_ctrl(qc, yl, ys, pre, ctrl)
            _shift_ctrl(qc, xl, xs, pre, ctrl)
            # reflection: uncontrolled conjugating shifts, controlled phase
            if rl != "I":
                # R_s = T_v^dag R_chi T_v with v = (a,b): as a gate sequence
                # the shift goes first, the oracle second, the inverse last
                a, b = ELEM_OFFSETS[int(rl[1]) - 1]
                _shift_plain(qc, +a, xs, pre)
                _shift_plain(qc, +b, ys, pre)
                self.oracle.apply(qc, ys, xs, orc, ctrl)
                _shift_plain(qc, -b, ys, pre)
                _shift_plain(qc, -a, xs, pre)
            _set_ctrl(qc, qr_a, ctrl, i)

        qc.append(prep.inverse(), qr_a)
        return qc

    def unitary(self) -> np.ndarray:
        return Operator(self.circuit()).data

    # -- verification -------------------------------------------------------

    def verify_columns(self, atol: float = 1e-9) -> dict:
        """
        alpha * U[:N0,:N0] against the direct assembly, one column at a time.

        Each column is a single statevector simulation, so this reaches
        roughly twice the m that a dense ``Operator`` would.
        """
        from qiskit import transpile
        N0 = 2 ** self.num_system
        qc = transpile(self.circuit(), basis_gates=["u", "cx"],
                       optimization_level=1)
        nq = qc.num_qubits
        block = np.zeros((N0, N0), dtype=complex)
        for j in range(N0):
            init = np.zeros(2 ** nq, dtype=complex)
            init[j] = 1.0                      # ancillas are the high bits
            block[:, j] = self.alpha * Statevector(init).evolve(qc).data[:N0]
        target = self.target()
        err = float(np.linalg.norm(block - target) / np.linalg.norm(target))
        return {
            "m": self.m, "oracle": self.oracle.method,
            "volume_fraction": self.enc.volume_fraction,
            "contrast": self.enc.contrast, "alpha": self.alpha,
            "num_terms": self.num_terms, "num_qubits": nq,
            "cx": int(qc.count_ops().get("cx", 0)), "depth": int(qc.depth()),
            "block_encoding_rel_err": err, "passed": err < atol,
        }

    def verify(self) -> dict:
        """Dense check, including unitarity.  Small m only."""
        N0 = 2 ** self.num_system
        U = self.unitary()
        target = self.target()
        return {
            "m": self.m, "oracle": self.oracle.method,
            "alpha": self.alpha, "num_terms": self.num_terms,
            "num_qubits": self.num_qubits,
            "block_encoding_rel_err": float(
                np.linalg.norm(self.alpha * U[:N0, :N0] - target)
                / np.linalg.norm(target)),
            "unitarity_err": float(
                np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))),
        }

    def cost(self) -> dict:
        """Transpiled gate counts for the whole block encoding."""
        from qiskit import transpile
        t = transpile(self.circuit(), basis_gates=["u", "cx"],
                      optimization_level=1)
        return {"m": self.m, "num_qubits": t.num_qubits,
                "oracle": self.oracle.method,
                "cx": int(t.count_ops().get("cx", 0)),
                "depth": int(t.depth())}
