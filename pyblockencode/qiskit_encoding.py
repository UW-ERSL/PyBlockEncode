"""
qiskit_encoding.py — Qiskit circuit generation for pyblockencode.

Provides three levels of block-encoding circuits:

  1. Pauli LCU  (general-purpose, expensive)
       PauliBlockEncoding(A)

  2. Pattern-compression — Poisson 1D/2D/3D FDM+FEM
       PoissonCircuit(m, dim, disc)

  3. Pattern-compression — Plane-stress Q4 elasticity
       ElasticityCircuit(m, E, nu)

All circuits use the MSB-ancilla convention (system register declared first,
ancilla declared second), so the encoded block A/alpha lives in the contiguous
top-left corner U[:N0, :N0] of the unitary matrix.

  QuantumCircuit(qr_sys, qr_anc)   ← system = LSB, ancilla = MSB
  Operator(qc).data[:N0, :N0]  ≈  A / alpha    ✓

Each class exposes:
    circuit()        → QuantumCircuit  (PREP–SELECT–UNPREP, no measurements)
    unitary()        → np.ndarray      (full unitary matrix, small m only)
    verify()         → dict            (‖alpha·U[:N0,:N0] − A‖ / ‖A‖ etc.)

Standalone function:
    apply_Ax(enc, x) → dict           (quantum A|x> vs classical, for checking)

Helper imported from your existing codebase:
    simulate_statevector(circuit)     (from Chapter08_QuantumGates_functions)
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, SparsePauliOp

from .poisson_pattern import PoissonPatternEncoding
from .elasticity_pattern import ElasticityPatternEncoding
from . import operators


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prep_gate(amps: np.ndarray, label: str = 'Prep') -> StatePreparation:
    """StatePreparation gate for the amplitude vector (padded to power-of-2)."""
    return StatePreparation(amps, label=label)


def _shift_circ(m: int, k: int) -> QuantumCircuit:
    """Cyclic increment (k=+1) or decrement (k=-1) on an m-qubit register."""
    qc = QuantumCircuit(m)
    if k == +1:
        for j in range(m - 1, 0, -1):
            qc.mcx(list(range(j)), j)
        qc.x(0)
    else:
        qc.x(0)
        for j in range(1, m):
            qc.mcx(list(range(j)), j)
    return qc


def _reflection_circ(m: int, which: str) -> QuantumCircuit:
    """
    R_j = I - 2|j><j| on an m-qubit register.

    R_{N-1} is a multi-controlled Z (phase -1 on |11...1>); R_0 is the same
    conjugated by X on every qubit.  Both are O(m) and use no ancilla.
    """
    qc = QuantumCircuit(m)
    flip = (which == "R0")
    if flip:
        qc.x(range(m))
    if m == 1:
        qc.z(0)
    else:
        qc.mcp(np.pi, list(range(m - 1)), m - 1)
    if flip:
        qc.x(range(m))
    return qc


def _label_gate(lbl: str, m: int) -> QuantumCircuit:
    """
    Circuit for one label of the unitary set of ``bc.py``:

        I, Sc, R0.Sc, Scd, RN.Scd, R0, RN

    A composite label such as 'R0.Sc' denotes the matrix product R0 @ Sc, so
    the shift is applied first and the reflection second.
    """
    qc = QuantumCircuit(m, name=lbl)
    if lbl == "I":
        return qc
    if lbl in ("R0", "RN"):
        qc.compose(_reflection_circ(m, lbl), inplace=True)
        return qc
    shift_part = "Sc" if lbl.endswith("Sc") else "Scd"
    qc.compose(_shift_circ(m, +1 if shift_part == "Sc" else -1), inplace=True)
    if "." in lbl:
        qc.compose(_reflection_circ(m, lbl.split(".")[0]), inplace=True)
    return qc


# ---------------------------------------------------------------------------
# 1. General-purpose Pauli LCU block encoding
# ---------------------------------------------------------------------------

class PauliBlockEncoding:
    """
    Block-encoding of a Hermitian operator A via full Pauli decomposition.

    This is the general-purpose (expensive) baseline — $L = O(4^m)$ terms,
    $\\alpha = \\sum_k |c_k|$.  Use the pattern-compression classes for
    FEM/FDM operators.

    Convention: MSB ancilla.
        QuantumCircuit(qr_sys, qr_anc)
        A/alpha  =  Operator(qc).data[:N, :N]

    Parameters
    ----------
    A : np.ndarray
        Hermitian operator, size must be a power of 2.
    """

    def __init__(self, A: np.ndarray):
        assert A.shape[0] == A.shape[1], "A must be square"
        self.A = A
        self.N = A.shape[0]
        self.num_system = int(np.ceil(np.log2(self.N)))

        pauli_op = SparsePauliOp.from_operator(A)
        self.paulis = pauli_op.paulis
        self.coeffs = pauli_op.coeffs
        self.alpha  = float(np.sum(np.abs(self.coeffs)))

        L = len(self.coeffs)
        self.num_ancilla = int(np.ceil(np.log2(max(L, 2))))
        self._K2 = 2 ** self.num_ancilla

    @property
    def num_terms(self) -> int:
        return len(self.coeffs)

    @property
    def num_qubits(self) -> int:
        return self.num_system + self.num_ancilla

    def circuit(self) -> QuantumCircuit:
        """
        Return the PREP–SELECT–UNPREP circuit (no measurements).

        Qubit layout (Qiskit declaration order):
            qr_sys  (num_system qubits)   — declared first → LSB in statevector
            qr_anc  (num_ancilla qubits)  — declared second → MSB in statevector

        Post-selection on ancilla = |0⟩ (MSB) extracts the top-left block
            alpha * U[:N, :N]  ≈  A
        """
        ns = self.num_system
        na = self.num_ancilla
        qr_sys = QuantumRegister(ns, 'sys')
        qr_anc = QuantumRegister(na, 'anc')
        # MSB ancilla: system first, ancilla second
        qc = QuantumCircuit(qr_sys, qr_anc, name='PauliLCU')

        # PREP
        amps = np.zeros(self._K2)
        for i, c in enumerate(self.coeffs):
            amps[i] = np.sqrt(abs(c) / self.alpha)
        prep = _prep_gate(amps, label='Prep')
        qc.append(prep, qr_anc)

        # SELECT: for each Pauli P_k, apply P_k on sys controlled on anc=|k⟩
        for i, (pauli, coeff) in enumerate(zip(self.paulis, self.coeffs)):
            phase = float(np.angle(coeff))
            p_circ = QuantumCircuit(ns, global_phase=phase, name=str(pauli))
            p_circ.append(pauli.to_instruction(), range(ns))
            ctrl_gate = p_circ.to_gate().control(
                na,
                ctrl_state=format(i, f'0{na}b')
            )
            qc.append(ctrl_gate, [*qr_anc, *qr_sys])

        # UNPREP
        qc.append(prep.inverse(), qr_anc)
        return qc

    def unitary(self) -> np.ndarray:
        """Full unitary matrix (dense). Practical for small systems only."""
        return Operator(self.circuit()).data

    def verify(self) -> dict:
        """Check alpha * U[:N, :N] ≈ A."""
        U = self.unitary()
        N = self.N
        block = self.alpha * U[:N, :N]
        err = float(np.linalg.norm(block - self.A) / np.linalg.norm(self.A))
        uu  = float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
        return {
            'alpha': self.alpha,
            'num_terms': len(self.coeffs),
            'num_qubits': self.num_qubits,
            'block_encoding_rel_err': err,
            'unitarity_err': uu,
        }


# ---------------------------------------------------------------------------
# 2. Pattern-compression — Poisson
# ---------------------------------------------------------------------------

class PoissonCircuit:
    """
    Qiskit block-encoding circuit for the Poisson operator, via shift
    decomposition with reflection-based boundary conditions.

    There is no flag qubit: the ancilla register is PREP only.

    Parameters
    ----------
    m, dim, disc, bc : as for ``PoissonPatternEncoding``.
    """

    def __init__(self, m: int, dim: int = 1, disc: str = 'fdm',
                 bc: str = 'essential'):
        self.enc = PoissonPatternEncoding(m=m, dim=dim, disc=disc, bc=bc)
        self.m, self.dim, self.disc = m, dim, disc

    alpha = property(lambda self: self.enc.alpha)
    num_terms = property(lambda self: self.enc.num_terms)
    num_system = property(lambda self: self.enc.num_system)
    num_ancilla = property(lambda self: self.enc.num_ancilla)
    num_qubits = property(lambda self: self.enc.num_qubits)

    def target(self) -> np.ndarray:
        return self.enc.target()

    def circuit(self) -> QuantumCircuit:
        """
        PREP - SELECT - PREP^dagger, no measurements.

        Qubit layout: one m-qubit register per spatial direction (declared
        last-direction-first so the linear index is (j_0 * N + j_1) * ...),
        then the PREP ancilla.  System registers are declared first, so the
        encoded block is the contiguous ``U[:N0, :N0]``.
        """
        m, dim = self.m, self.dim
        terms = self.enc.lcu_terms()
        keys = list(terms)
        coeffs = np.array([terms[k] for k in keys])
        alpha = float(np.abs(coeffs).sum())
        L = len(keys)
        na = self.enc.num_ancilla

        qr_dims = [QuantumRegister(m, f'd{i}') for i in range(dim)]
        qr_anc = QuantumRegister(na, 'anc')
        # declare the last direction first so direction 0 is most significant
        qc = QuantumCircuit(*reversed(qr_dims), qr_anc,
                            name=f'Poisson{dim}D_{self.disc.upper()}')

        amps = np.zeros(2 ** na)
        amps[:L] = np.sqrt(np.abs(coeffs) / alpha)
        prep = _prep_gate(amps, label='Prep')
        qc.append(prep, qr_anc)

        sys_qubits = []
        for qr in reversed(qr_dims):
            sys_qubits.extend(list(qr))

        for i, key in enumerate(keys):
            sgn = float(np.sign(coeffs[i]))
            sel = QuantumCircuit(dim * m,
                                 global_phase=(0.0 if sgn > 0 else np.pi),
                                 name=f'SEL_{i}')
            # sub-circuit qubit blocks follow the declaration order above
            for d, lbl in enumerate(reversed(key)):
                if lbl == 'I':
                    continue
                sel.compose(_label_gate(lbl, m),
                            qubits=list(range(d * m, (d + 1) * m)),
                            inplace=True)
            ctrl = sel.to_gate().control(na, ctrl_state=i)
            qc.append(ctrl, [*qr_anc, *sys_qubits])

        qc.append(prep.inverse(), qr_anc)
        return qc

    def unitary(self) -> np.ndarray:
        return Operator(self.circuit()).data

    def verify(self) -> dict:
        N0 = 2 ** (self.dim * self.m)
        target = self.target()
        U = self.unitary()
        return {
            'dim': self.dim, 'disc': self.disc, 'm': self.m,
            'bc': self.enc.bc, 'alpha': self.alpha,
            'num_terms': self.num_terms, 'num_qubits': self.num_qubits,
            'block_encoding_rel_err': float(
                np.linalg.norm(self.alpha * U[:N0, :N0] - target)
                / np.linalg.norm(target)),
            'unitarity_err': float(
                np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))),
        }


# ---------------------------------------------------------------------------
# 3. Shift decomposition -- elasticity
# ---------------------------------------------------------------------------

class ElasticityCircuit:
    """
    Qiskit block-encoding circuit for the 2D plane-stress Q4 elasticity
    operator, via shift decomposition with reflection-based boundary
    conditions.

    System register : 2m + 1 qubits (x, y, and the DOF qubit)
    Ancilla         : ceil(log2 L) PREP qubits -- no flag qubit
    """

    #: DOF-qubit gates; iY = Z @ X is real and needs no global phase
    _DOF_GATES = {'I': (), 'X': ('x',), 'Z': ('z',), 'iY': ('x', 'z')}

    def __init__(self, m: int, E: float = 1.0, nu: float = 0.3,
                 bc: str = 'essential'):
        self.enc = ElasticityPatternEncoding(m=m, E=E, nu=nu, bc=bc)
        self.m, self.E, self.nu = m, E, nu

    alpha = property(lambda self: self.enc.alpha)
    num_terms = property(lambda self: self.enc.num_terms)
    num_system = property(lambda self: self.enc.num_system)
    num_ancilla = property(lambda self: self.enc.num_ancilla)
    num_qubits = property(lambda self: self.enc.num_qubits)
    components = property(lambda self: self.enc.components)

    def target(self) -> np.ndarray:
        return self.enc.target()

    def circuit(self) -> QuantumCircuit:
        """
        PREP - SELECT - PREP^dagger, no measurements.

        ``target`` uses the index (jx * N + jy) * 2 + d, so the DOF qubit is
        least significant and the x register most significant among the
        system qubits: declare dof, y, x, then the PREP ancilla.
        """
        m = self.m
        terms = self.enc.lcu_terms()
        keys = list(terms)
        coeffs = np.array([terms[k] for k in keys])
        alpha = float(np.abs(coeffs).sum())
        L = len(keys)
        na = self.enc.num_ancilla

        qr_x = QuantumRegister(m, 'x')
        qr_y = QuantumRegister(m, 'y')
        qr_d = QuantumRegister(1, 'dof')
        qr_anc = QuantumRegister(na, 'anc')
        qc = QuantumCircuit(qr_d, qr_y, qr_x, qr_anc, name='ElasticityQ4')

        amps = np.zeros(2 ** na)
        amps[:L] = np.sqrt(np.abs(coeffs) / alpha)
        prep = _prep_gate(amps, label='Prep')
        qc.append(prep, qr_anc)

        sys_qubits = [*qr_d, *qr_y, *qr_x]

        for i, key in enumerate(keys):
            xl, yl, dl = key
            sgn = float(np.sign(coeffs[i]))
            sel = QuantumCircuit(2 * m + 1,
                                 global_phase=(0.0 if sgn > 0 else np.pi),
                                 name=f'SEL_{i}')
            # qubit 0 = dof, 1..m = y, m+1..2m = x
            for g in self._DOF_GATES[dl]:
                getattr(sel, g)(0)
            if yl != 'I':
                sel.compose(_label_gate(yl, m),
                            qubits=list(range(1, m + 1)), inplace=True)
            if xl != 'I':
                sel.compose(_label_gate(xl, m),
                            qubits=list(range(m + 1, 2 * m + 1)), inplace=True)
            ctrl = sel.to_gate().control(na, ctrl_state=i)
            qc.append(ctrl, [*qr_anc, *sys_qubits])

        qc.append(prep.inverse(), qr_anc)
        return qc

    def unitary(self) -> np.ndarray:
        return Operator(self.circuit()).data

    def verify(self) -> dict:
        N0 = 2 * (2 ** self.m) ** 2
        target = self.target()
        U = self.unitary()
        return {
            'm': self.m, 'nu': self.nu, 'bc': self.enc.bc,
            'alpha': self.alpha, 'num_terms': self.num_terms,
            'components': self.components, 'num_qubits': self.num_qubits,
            'block_encoding_rel_err': float(
                np.linalg.norm(self.alpha * U[:N0, :N0] - target)
                / np.linalg.norm(target)),
            'unitarity_err': float(
                np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))),
        }


# ---------------------------------------------------------------------------
# apply_Ax — verify A|x⟩ using a block-encoding circuit
# ---------------------------------------------------------------------------

def apply_Ax(enc, x: np.ndarray,
             simulate_fn=None) -> dict:
    """
    Compute A|x⟩ using statevector simulation of the block-encoding circuit
    and compare against the classical result.

    The LCU circuit implements:
        |0⟩_anc |x⟩_sys  →  (1/alpha) A|x⟩ |0⟩_anc  +  |garbage⟩

    Post-selecting the statevector on ancilla = |0⟩ (all ancilla/flag qubits
    in the MSB positions) and rescaling by alpha recovers A|x⟩.

    Parameters
    ----------
    enc : PauliBlockEncoding | PoissonCircuit | ElasticityCircuit
        Any encoding object with .circuit(), .alpha, .target(), .num_system,
        .num_ancilla attributes.
    x   : np.ndarray
        Input state vector, length N = 2**num_system.  Will be normalised.
    simulate_fn : callable, optional
        simulate_statevector(circuit) from Chapter08_QuantumGates_functions.
        If None, uses Qiskit's Statevector directly (no Aer dependency).

    Returns
    -------
    dict with keys:
        'Ax_quantum'   : np.ndarray  — quantum result (length N)
        'Ax_classical' : np.ndarray  — classical A @ x
        'rel_err'      : float       — ‖Ax_quantum − Ax_classical‖ / ‖Ax_classical‖
        'success_prob' : float       — P(ancilla = |0⟩)
        'alpha'        : float
    """
    # Normalise input
    x = np.asarray(x, dtype=complex)
    x = x / np.linalg.norm(x)
    N = len(x)

    # Determine subspace sizes
    if hasattr(enc, 'num_system'):
        ns = enc.num_system
        na = enc.num_ancilla
    else:
        raise ValueError("enc must have num_system and num_ancilla attributes")

    alpha = enc.alpha

    # Build circuit: prepend state preparation on system register
    base_qc = enc.circuit()

    # We need to initialise |x⟩ on the system qubits.
    # The system register is declared first in all our circuits → qubits 0..ns-1
    # Insert initialisation at the front.
    ns_circ = QuantumCircuit(base_qc.num_qubits, name='Ax_circuit')
    ns_circ.append(StatePreparation(x, label='|x⟩'), list(range(ns)))
    ns_circ.compose(base_qc, inplace=True)

    # Simulate statevector
    if simulate_fn is not None:
        sv = np.asarray(simulate_fn(ns_circ))
    else:
        from qiskit.quantum_info import Statevector
        sv = np.asarray(Statevector(ns_circ))

    # Total statevector dimension = 2**(ns + na)
    total = 2 ** (ns + na)
    assert len(sv) == total, f"Statevector length {len(sv)} ≠ {total}"

    # MSB ancilla convention (QuantumCircuit(qr_sys, qr_anc)):
    #   qr_sys declared first  → sys = LSB → varies fastest
    #   qr_anc declared second → anc = MSB → varies slowest
    #   statevector index = anc_index * 2**ns + sys_index
    #   ancilla=|0⟩ → anc_index = 0 → indices 0 .. N-1  (contiguous) ✓
    #
    # This is consistent with the block encoding convention:
    #   alpha * U[:N, :N]  ≈  A   (top-left block)

    anc_zero_slice = sv[:N]   # ancilla=|0⟩ subspace: contiguous top-left
    success_prob   = float(np.real(np.dot(anc_zero_slice.conj(), anc_zero_slice)))

    # Rescale: anc_zero_slice = (1/alpha) * A|x⟩  (unnormalised)
    Ax_quantum = alpha * anc_zero_slice

    # Classical reference
    A          = enc.target()
    Ax_classical = A @ x

    rel_err = float(np.linalg.norm(Ax_quantum - Ax_classical)
                    / np.linalg.norm(Ax_classical))

    return {
        'Ax_quantum':   Ax_quantum,
        'Ax_classical': Ax_classical,
        'rel_err':      rel_err,
        'success_prob': success_prob,
        'alpha':        alpha,
    }