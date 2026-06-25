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


def _shift_gate(lbl: str, m: int) -> QuantumCircuit:
    """
    Return a small QuantumCircuit implementing one cyclic shift on m qubits.

    'I'   → identity (no gates)
    'Sc'  → cyclic increment  |j⟩ → |(j+1) mod 2^m⟩
    'Scd' → cyclic decrement  |j⟩ → |(j-1) mod 2^m⟩

    Implemented as a multi-controlled incrementer / decrementer using the
    standard X-ladder construction (Vedral et al. style):
      q0 (MSB) ... q_{m-1} (LSB)
      CNOT q_{m-2} → q_{m-1}
      CNOT q_{m-3} → q_{m-2} (controlled on q_{m-2} being |0⟩ after prev step)
      ... cascading upward
    For simplicity we use the QFT-adder approach via Qiskit's add_to_int,
    but to keep the circuit self-contained we build it from X + MCX gates.
    """
    qc = QuantumCircuit(m, name=lbl)
    if lbl == 'I':
        return qc
    # Cyclic increment using X + multi-controlled-X ladder (ripple carry style)
    # For Sc (increment): flip qubit k if all lower qubits are |1⟩
    # For Scd (decrement): flip qubit k if all lower qubits are |0⟩
    if lbl == 'Sc':
        # Increment: apply X gates from LSB upward with controls
        for k in range(m - 1, -1, -1):
            if k == m - 1:
                qc.x(k)          # LSB: always flip
            else:
                # flip qubit k if qubits k+1..m-1 are all |1⟩
                qc.mcx(list(range(k + 1, m)), k)
    else:  # Scd — decrement: flip from LSB upward if lower bits are all |0⟩
        for k in range(m - 1, -1, -1):
            if k == m - 1:
                qc.x(k)
            else:
                # X the lower qubits to convert |0⟩ control to |1⟩ control
                for j in range(k + 1, m):
                    qc.x(j)
                qc.mcx(list(range(k + 1, m)), k)
                for j in range(k + 1, m):
                    qc.x(j)
    return qc


def _flag_correction(m: int, lbl: str) -> QuantumCircuit:
    """
    Return a circuit that sets a flag qubit when the shift wraps around.

    For Sc  (increment): flag ← 1  iff  j = 2^m − 1  (all |1⟩ before shift)
    For Scd (decrement): flag ← 1  iff  j = 0         (all |0⟩ before shift)
    Flag qubit is the last qubit; the first m qubits are the shift register.

    Layout: qc has m+1 qubits — [shift_reg(0..m-1), flag(m)]
    """
    qc = QuantumCircuit(m + 1, name=f'flag_{lbl}')
    if lbl == 'I':
        return qc
    if lbl == 'Sc':
        # Flag ← 1 when all m shift bits are |1⟩
        qc.mcx(list(range(m)), m)
    else:  # Scd
        # Flag ← 1 when all m shift bits are |0⟩  (flip→control→flip)
        for j in range(m):
            qc.x(j)
        qc.mcx(list(range(m)), m)
        for j in range(m):
            qc.x(j)
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
    Qiskit block-encoding circuit for the Poisson stiffness operator via
    pattern compression (cyclic-shift LCU + flag ancilla).

    Parameters
    ----------
    m    : int  — qubits per spatial dimension; N = 2**m interior nodes
    dim  : int  — spatial dimension (1, 2, or 3)
    disc : str  — 'fdm' or 'fem'
    """

    def __init__(self, m: int, dim: int = 1, disc: str = 'fdm'):
        self.enc  = PoissonPatternEncoding(m=m, dim=dim, disc=disc)
        self.m    = m
        self.dim  = dim
        self.disc = disc

    @property
    def alpha(self) -> float:
        return self.enc.alpha

    @property
    def num_terms(self) -> int:
        return self.enc.num_terms

    @property
    def num_system(self) -> int:
        return self.enc.num_system   # dim * m

    @property
    def num_ancilla(self) -> int:
        return self.enc.num_ancilla  # ceil(log2 L) + 1 (flag)

    @property
    def num_qubits(self) -> int:
        return self.enc.num_qubits

    def target(self) -> np.ndarray:
        return self.enc.target()

    def circuit(self) -> QuantumCircuit:
        """
        Return the pattern-compression LCU circuit (no measurements).

        Qubit layout:
            qr_sys  = d*m system qubits  (d spatial registers of m qubits each)
            qr_flag = 1 flag qubit       (Dirichlet boundary correction)
            qr_anc  = ceil(log2 L) PREP ancilla qubits

        Declaration order: QuantumCircuit(qr_sys, qr_flag, qr_anc)
        → system = LSB, ancilla (PREP+flag) = MSB
        → encoded block is U[:N0, :N0]  where N0 = 2**(dim*m)
        """
        m   = self.m
        dim = self.dim
        N0  = 2 ** (dim * m)
        terms   = self.enc.lcu_terms()
        labels  = list(terms)
        coeffs  = np.array([terms[k] for k in labels])
        alpha   = float(np.sum(np.abs(coeffs)))
        L       = len(labels)
        na_prep = int(np.ceil(np.log2(max(L, 2))))
        K2      = 2 ** na_prep

        # Registers: one m-qubit register per spatial dimension
        qr_dims = [QuantumRegister(m, f'd{i}') for i in range(dim)]
        qr_flag = QuantumRegister(1, 'flag')
        qr_anc  = QuantumRegister(na_prep, 'anc')

        # MSB ancilla: system registers first, then flag, then PREP ancilla
        all_regs = qr_dims + [qr_flag, qr_anc]
        qc = QuantumCircuit(*all_regs, name=f'Poisson{dim}D_{self.disc.upper()}')

        # ── PREP ──────────────────────────────────────────────────────────
        amps = np.zeros(K2)
        for i, c in enumerate(coeffs):
            amps[i] = np.sqrt(abs(c) / alpha)
        prep = _prep_gate(amps, label='Prep')
        qc.append(prep, qr_anc)

        # ── SELECT ────────────────────────────────────────────────────────
        # Each LCU term is a tuple of shift labels, one per spatial dimension
        sys_qubits = []
        for qr in qr_dims:
            sys_qubits.extend(list(qr))
        flag_qubit = list(qr_flag)

        for i, key in enumerate(labels):
            # key is a tuple of length dim, e.g. ('Sc', 'I') for 2D
            lbls = key if isinstance(key, tuple) else (key,)
            sgn  = float(np.sign(coeffs[i]))

            # Build a circuit for this SELECT term on (sys + flag) qubits
            # dim*m system + 1 flag = dim*m+1 qubits
            n_sel = dim * m + 1
            sel_circ = QuantumCircuit(n_sel, global_phase=(0 if sgn > 0 else np.pi),
                                      name=f'SEL_{i}')

            # Apply shift on each dimension register + accumulate flag
            offset = 0
            for d, lbl in enumerate(lbls):
                shift_circ = _shift_gate(lbl, m)
                flag_circ  = _flag_correction(m, lbl)

                if lbl != 'I':
                    # Apply shift on qubits offset..offset+m-1
                    sel_circ.append(shift_circ.to_gate(),
                                    list(range(offset, offset + m)))
                    # Update flag: flag qubit is qubit n_sel-1
                    # flag_circ uses qubits [shift_reg(m), flag]
                    # here the shift reg for dim d is offset..offset+m-1
                    sel_circ.append(flag_circ.to_gate(),
                                    list(range(offset, offset + m)) + [n_sel - 1])
                offset += m

            # Control this whole SELECT block on ancilla = |i⟩
            ctrl_gate = sel_circ.to_gate().control(
                na_prep,
                ctrl_state=format(i, f'0{na_prep}b')
            )
            qc.append(ctrl_gate, [*qr_anc, *sys_qubits, *flag_qubit])

        # ── UNPREP ────────────────────────────────────────────────────────
        qc.append(prep.inverse(), qr_anc)
        return qc

    def unitary(self) -> np.ndarray:
        """Full unitary (dense). Practical for m ≤ 2, dim ≤ 2."""
        return Operator(self.circuit()).data

    def verify(self) -> dict:
        """
        Check alpha * U[:N0, :N0] ≈ K_target.

        The flag qubit doubles the system dimension, so the full system+flag
        space has 2*N0 states; the encoded block is at U[:N0, :N0]
        (ancilla=|0⟩, flag=|0⟩ subspace).
        """
        N0     = 2 ** (self.dim * self.m)
        target = self.target()
        U      = self.unitary()
        block  = self.alpha * U[:N0, :N0]
        err    = float(np.linalg.norm(block - target) / np.linalg.norm(target))
        uu     = float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
        return {
            'dim': self.dim, 'disc': self.disc, 'm': self.m,
            'alpha': self.alpha, 'num_terms': self.num_terms,
            'num_qubits': self.num_qubits,
            'block_encoding_rel_err': err,
            'unitarity_err': uu,
        }


# ---------------------------------------------------------------------------
# 3. Pattern-compression — Elasticity
# ---------------------------------------------------------------------------

class ElasticityCircuit:
    """
    Qiskit block-encoding circuit for the 2D plane-stress Q4 elasticity
    operator via pattern compression.

    System register: 2m+1 qubits (m x-qubits + m y-qubits + 1 DOF qubit)
    Ancilla:         6 qubits  (5 PREP + 1 flag)
    Total:           2m+7 qubits

    Convention: MSB ancilla — system declared first.
        alpha * U[:N0, :N0]  ≈  K     where N0 = 2 * N^2

    Parameters
    ----------
    m  : int   — qubits per spatial dimension; N = 2**m interior nodes/dim
    E  : float — Young's modulus (default 1.0)
    nu : float — Poisson's ratio (default 0.3)
    """

    def __init__(self, m: int, E: float = 1.0, nu: float = 0.3):
        self.enc = ElasticityPatternEncoding(m=m, E=E, nu=nu)
        self.m   = m
        self.E   = E
        self.nu  = nu

    @property
    def alpha(self) -> float:
        return self.enc.alpha

    @property
    def num_terms(self) -> int:
        return self.enc.num_terms

    @property
    def num_system(self) -> int:
        return self.enc.num_system   # 2m + 1

    @property
    def num_ancilla(self) -> int:
        return self.enc.num_ancilla  # 6  (5 PREP + 1 flag)

    @property
    def num_qubits(self) -> int:
        return self.enc.num_qubits   # 2m + 7

    def target(self) -> np.ndarray:
        return self.enc.target()

    def circuit(self) -> QuantumCircuit:
        """
        Return the elasticity pattern-compression LCU circuit (no measurements).

        Qubit layout (declaration order → Qiskit statevector ordering):
            qr_x    (m qubits)   x spatial register
            qr_y    (m qubits)   y spatial register
            qr_dof  (1 qubit)    displacement DOF  (0=x-disp, 1=y-disp)
            qr_flag (1 qubit)    Dirichlet boundary flag
            qr_anc  (5 qubits)   PREP ancilla  (ceil(log2 17) = 5)

        System = [qr_x, qr_y, qr_dof] → N0 = 2*N^2 states
        Ancilla = [qr_flag, qr_anc]   → MSB, post-select on all zeros
        Encoded block: alpha * U[:N0, :N0] ≈ K
        """
        m  = self.m
        N  = 2 ** m
        N0 = 2 * N * N   # system dimension

        terms   = self.enc.lcu_terms()
        labels  = list(terms)
        coeffs  = np.array([terms[k] for k in labels])
        alpha   = float(np.sum(np.abs(coeffs)))
        L       = len(labels)
        na_prep = int(np.ceil(np.log2(max(L, 2))))   # = 5 for L=17
        K2      = 2 ** na_prep

        # Registers
        qr_x    = QuantumRegister(m, 'x')
        qr_y    = QuantumRegister(m, 'y')
        qr_dof  = QuantumRegister(1, 'dof')
        qr_flag = QuantumRegister(1, 'flag')
        qr_anc  = QuantumRegister(na_prep, 'anc')

        # MSB ancilla: system (x, y, dof) first; flag + PREP ancilla last
        qc = QuantumCircuit(qr_x, qr_y, qr_dof, qr_flag, qr_anc,
                            name='ElasticityQ4')

        # ── PREP ──────────────────────────────────────────────────────────
        amps = np.zeros(K2)
        for i, c in enumerate(coeffs):
            amps[i] = np.sqrt(abs(c) / alpha)
        prep = _prep_gate(amps, label='Prep')
        qc.append(prep, qr_anc)

        # ── SELECT ────────────────────────────────────────────────────────
        # Each term: (x_label, y_label, dof_label)
        # x/y labels  ∈ {'I', 'Sc', 'Scd'}  → cyclic shift on m-qubit register
        # dof label   ∈ {'I', 'Z', 'X'}     → Pauli on 1 DOF qubit

        # Pauli gates on DOF qubit
        dof_circuits = {
            'I': QuantumCircuit(1, name='I'),
            'X': QuantumCircuit(1, name='X'),
            'Z': QuantumCircuit(1, name='Z'),
        }
        dof_circuits['X'].x(0)
        dof_circuits['Z'].z(0)

        x_qubits   = list(qr_x)
        y_qubits   = list(qr_y)
        dof_qubit  = list(qr_dof)
        flag_qubit = list(qr_flag)

        for i, key in enumerate(labels):
            xl, yl, dl = key
            sgn = float(np.sign(coeffs[i]))

            # Build SELECT sub-circuit on (x + y + dof + flag) = 2m+2 qubits
            n_sel = 2 * m + 2
            sel_circ = QuantumCircuit(
                n_sel,
                global_phase=(0 if sgn > 0 else np.pi),
                name=f'SEL_{i}'
            )
            # Qubit layout in sel_circ:
            #   0..m-1          → x register
            #   m..2m-1         → y register
            #   2m              → DOF qubit
            #   2m+1            → flag qubit

            x_idx   = list(range(m))
            y_idx   = list(range(m, 2 * m))
            dof_idx = [2 * m]
            flg_idx = [2 * m + 1]

            # x shift
            if xl != 'I':
                sel_circ.append(_shift_gate(xl, m).to_gate(), x_idx)
                sel_circ.append(_flag_correction(m, xl).to_gate(),
                                x_idx + flg_idx)
            # y shift
            if yl != 'I':
                sel_circ.append(_shift_gate(yl, m).to_gate(), y_idx)
                sel_circ.append(_flag_correction(m, yl).to_gate(),
                                y_idx + flg_idx)
            # DOF Pauli
            if dl != 'I':
                sel_circ.append(dof_circuits[dl].to_gate(), dof_idx)

            # Control on ancilla = |i⟩
            ctrl_gate = sel_circ.to_gate().control(
                na_prep,
                ctrl_state=format(i, f'0{na_prep}b')
            )
            qc.append(ctrl_gate,
                      [*qr_anc, *x_qubits, *y_qubits, *dof_qubit, *flag_qubit])

        # ── UNPREP ────────────────────────────────────────────────────────
        qc.append(prep.inverse(), qr_anc)
        return qc

    def unitary(self) -> np.ndarray:
        """Full unitary (dense). Practical for m = 1 only (9 qubits total)."""
        return Operator(self.circuit()).data

    def verify(self) -> dict:
        """Check alpha * U[:N0, :N0] ≈ K_target."""
        N0     = 2 * (2 ** self.m) ** 2
        target = self.target()
        U      = self.unitary()
        block  = self.alpha * U[:N0, :N0]
        err    = float(np.linalg.norm(block - target) / np.linalg.norm(target))
        uu     = float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
        return {
            'nu': self.nu, 'E': self.E, 'm': self.m,
            'alpha': self.alpha, 'num_terms': self.num_terms,
            'num_qubits': self.num_qubits,
            'block_encoding_rel_err': err,
            'unitarity_err': uu,
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