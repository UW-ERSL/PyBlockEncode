from pyencode import encode, SPARSE, SUM, STEP, POLYNOMIAL,SQUARE,PARTITION
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.compiler import transpile
from qiskit import transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
BASIS = ['cx', 'u', 'x', 'h', 'ry', 'rz', 'rx', 'p']   # transpilation basis

# PyEncode is not on PyPI; the sections below need it:
#   pip install git+https://github.com/UW-ERSL/PyEncode
from IPython.display import display


def estimateCircuitGatesNISQ(circuit, basis_gates=None, optimization_level=1):
    """
    NISQ complexity metrics: width, depth, and size.

    Transpiles the circuit to a continuous (arbitrary-angle) basis --- the
    same kind of native gate set that today's NISQ hardware executes
    directly --- and reports the metrics relevant to *running* the circuit on
    such hardware:

        * width  = number of qubits
        * depth  = length of the critical path
        * size   = total gate count (split into single-qubit and two-qubit)

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to analyse. It is decomposed first, so high-level gates
        (MCX, controlled-U, PhaseOracleGate, ...) are unrolled.
    basis_gates : list[str], optional
        Target basis. If None (default), the simulator's own basis is used.
        Pass e.g. ``['u3', 'cx']`` to count against that gate set.
    optimization_level : int, optional
        Pinned at 1 by default. The transpiler's default moved from 1 to 2 in
        Qiskit 2.x, which changes every count below; pin it so the numbers are
        reproducible.

    Note: this basis is continuous, so it does NOT report a meaningful
    T-count (arbitrary-angle rotations absorb T gates). For the
    fault-tolerant T-count metric, use estimateCircuitGatesFTC().
    """

    # Transpile to decompose MCX and adapt to the target basis gates
    decomposedCircuit = circuit.decompose(reps = 10)
    if basis_gates is None:
        transpiled_circuit = transpile(decomposedCircuit, AerSimulator(),
                                       optimization_level=optimization_level)
    else:
        transpiled_circuit = transpile(decomposedCircuit,
                                       basis_gates=basis_gates,
                                       optimization_level=optimization_level)

    # Extract key metrics
    gate_counts = transpiled_circuit.count_ops()
    depth = transpiled_circuit.depth()

    # Count two-qubit gates by arity, not by name: cp, rzz, crx, cry, ... are
    # all two-qubit gates and were previously booked as single-qubit.
    total_gates = sum(gate_counts.values())
    cx_gates = sum(1 for instruction in transpiled_circuit.data
                   if len(instruction.qubits) == 2
                   and instruction.operation.name not in ('barrier',))
    singleGateCount = total_gates - cx_gates

    result = {
         'num_qubits': transpiled_circuit.num_qubits,
        'single_gate_count': singleGateCount,
        'cx_gates': cx_gates,
        'total_gates': total_gates,
        'depth': depth,
        'transpiled_circuit': transpiled_circuit,
    }

    print("--- Circuit Analysis (NISQ: continuous basis) ---")
    print(f"Qubits (width):     {result['num_qubits']}")
    print(f"Depth:              {result['depth']}")
    print(f"Size (total gates): {result['total_gates']}")
    print(f"  Single-qubit:     {result['single_gate_count']}")
    print(f"  Two-qubit:        {result['cx_gates']}")

    return result

example = 4

if (example == 1):
    m = 5
    k = 23
    N = 2**m
    b = [0] * N
    b[k] = 1.0

    qc = QuantumCircuit(m)
    qc.prepare_state(b, range(m))
    circuitData = estimateCircuitGatesNISQ(qc, basis_gates=BASIS)
    print(circuitData) 
    transpiled_circuit = transpile(qc, basis_gates=BASIS)
    transpiled_circuit.draw("mpl")
    plt.show()
    pyEncodeCircuit, info = encode(SPARSE([(k, 1.0)]), N=N, validate=True)

    print(info)

    transpiled_circuit = transpile(pyEncodeCircuit, basis_gates=BASIS)
    transpiled_circuit.draw("mpl")
    plt.show()
elif (example == 2):
    m = 12
    k = 65
    N = 2**m
    b = [0] * N
    b[k] = 1.0

    qc = QuantumCircuit(m)
    qc.prepare_state(b, range(m))
    circuitData = estimateCircuitGatesNISQ(qc)
    print(circuitData) 

   
    pyEncodeCircuit, info = encode(SPARSE([(k, 1.0)]), N=N, validate=True)
    circuitData = estimateCircuitGatesNISQ(pyEncodeCircuit, basis_gates=BASIS)
    print(circuitData)

    transpiled_circuit = transpile(pyEncodeCircuit, basis_gates=BASIS)
    transpiled_circuit.draw("mpl")
    plt.show()
elif (example == 3):
    m = 12
    N = 2**m

    load1 = STEP(k_e = N//4,c = 1.0)
    load2 = SPARSE([(N//2, 1.0)])
    load3 = SQUARE(3*N//4, N, 1.0)
    load = SUM([(1,load1),(-1.25,load2),(0.25,load3)])
    
    pyEncodeCircuit, info = encode(load, N=N, validate=True)
    
    circuitData = estimateCircuitGatesNISQ(pyEncodeCircuit, basis_gates=BASIS)
    print(circuitData['total_gates'])
elif (example == 4):
    qiskitGates = []
    pyencodeGates = []
    mRange = range(8, 16)
    for m in mRange:
        N = 2**m
        b = [0] * N
        b[0:N//4] = [1.0] * (N//4)
        b[N//2] = -1.25
        b[3*N//4:N] = [0.25] * (N//4)
        b = b/np.linalg.norm(b)
        qc = QuantumCircuit(m)
        qc.prepare_state(b, range(m))
        print("Qiskit")
        circuitData = estimateCircuitGatesNISQ(qc, basis_gates=BASIS)
        qiskitGates.append(circuitData['total_gates'])
        load1 = STEP(k_e = N//4,c = 1.0)
        load2 = SPARSE([(N//2, 1.0)])
        load3 = SQUARE(3*N//4, N, 1.0)
        load = SUM([(1,load1),(-1.25,load2),(0.25,load3)])
        pyEncodeCircuit, info = encode(load, N=N, validate=True)
        print("PyEncode")
        circuitData = estimateCircuitGatesNISQ(pyEncodeCircuit, basis_gates=BASIS)
        pyencodeGates.append(circuitData['total_gates']) 
    plt.xlabel('Qubits')
    plt.ylabel('Gates')
    plt.title('Circuit Size vs Qubits')
    plt.semilogy(mRange,qiskitGates, label='Qiskit', color='orange', marker='o')
    plt.semilogy(mRange,pyencodeGates, label='PyEncode', color='blue', marker='o')
    plt.legend()
    plt.xticks(mRange)

    plt.grid()
    plt.show()