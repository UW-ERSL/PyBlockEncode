#!/usr/bin/env python3
"""
scaling_study.py - resource scaling of the PyBlockEncode block encodings.

Run from the root of your PyBlockEncode clone, AFTER applying the circuit
fixes:

    python scaling_study.py                 # elasticity, m = 1..6
    python scaling_study.py --mmax 8        # push further
    python scaling_study.py --poisson       # Poisson 2D FEM instead

Why this works at m far beyond what verify() can reach
------------------------------------------------------
verify() calls Operator(circuit), which densely simulates: cost 4^(#qubits).
That caps you at m = 1 or 2.  Resource counting needs no simulation at all,
only transpilation, which is polynomial.  So correctness and cost are two
separate studies:

    correctness -> ElasticityPatternEncoding.verify()   (dense, m <= 3)
                   ElasticityCircuit.verify()           (gate level, m = 1)
    cost        -> this script                          (transpile, m >> 3)

What is counted
---------------
The circuit is transpiled to a Clifford+T basis and T + Tdg is reported,
since T-count is the fault-tolerant currency.  A Toffoli is 7 T gates, so
the ~Toffoli column is T/7, comparable to the Toffoli counts quoted by
Kharazi et al. and by most of the block-encoding literature.

Caveat on the numbers
---------------------
These reflect Qiskit's DEFAULT synthesis of multi-controlled X gates, which
is ancilla-free and costs O(k^2) per k-controlled gate.  _shift_gate builds
an increment as a ladder of MCX gates with 0..m-1 controls, so one shift
costs O(m^3) rather than the O(m) claimed in the paper.  The claimed O(m)
requires the Gidney borrowed-ancilla incrementer, which is a different
construction, not an optimization of this one.  See the notes printed at
the end of the run.
"""
import argparse
import sys
import time

from qiskit import transpile

# Clifford+T. rz is kept so StatePreparation's rotations survive; they are a
# constant-size contribution (5-qubit PREP) and do not affect the scaling.
BASIS = ['h', 't', 'tdg', 's', 'sdg', 'x', 'z', 'cx', 'rz']


def row(m, qc, t_build):
    t0 = time.time()
    tq = transpile(qc, basis_gates=BASIS, optimization_level=1)
    t_tr = time.time() - t0
    ops = tq.count_ops()
    ncx = ops.get('cx', 0)
    nt = ops.get('t', 0) + ops.get('tdg', 0)
    print(f"{m:>2} {2**m:>6} {qc.num_qubits:>7} {tq.depth():>9} "
          f"{ncx:>9} {nt:>9} {nt/7:>10.0f} {t_build:>8.2f} {t_tr:>9.1f}",
          flush=True)
    return dict(m=m, qubits=qc.num_qubits, depth=tq.depth(), cx=ncx, t=nt)


def fit(rows, key):
    """Least-squares exponent of rows[key] ~ m^p, using the largest 3 points."""
    import math
    pts = rows[-3:]
    xs = [math.log(r['m']) for r in pts]
    ys = [math.log(r[key]) for r in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmax', type=int, default=6)
    ap.add_argument('--mmin', type=int, default=1)
    ap.add_argument('--nu', type=float, default=0.3)
    ap.add_argument('--poisson', action='store_true',
                    help='Poisson 2D FEM instead of elasticity')
    args = ap.parse_args()

    if args.poisson:
        from pyblockencode.qiskit_encoding import PoissonCircuit
        make = lambda m: PoissonCircuit(m=m, dim=2, disc='fem')
        title = "Poisson 2D FEM"
    else:
        from pyblockencode.qiskit_encoding import ElasticityCircuit
        make = lambda m: ElasticityCircuit(m=m, nu=args.nu)
        title = f"plane-stress Q4 elasticity (nu = {args.nu})"

    print(f"\n{title}\n")
    print(f"{'m':>2} {'N':>6} {'qubits':>7} {'depth':>9} "
          f"{'CX':>9} {'T+Tdg':>9} {'~Toffoli':>10} {'t_build':>8} {'t_transp':>9}")

    rows = []
    for m in range(args.mmin, args.mmax + 1):
        enc = make(m)
        t0 = time.time()
        qc = enc.circuit()
        t_build = time.time() - t0
        try:
            rows.append(row(m, qc, t_build))
        except MemoryError:
            print(f"{m:>2}  out of memory during transpile; stopping.")
            break

    if len(rows) >= 3:
        print("\nFitted exponents (largest three points, cost ~ m^p):")
        for key in ('t', 'cx', 'depth'):
            print(f"  {key:>5}  ~ m^{fit(rows, key):.2f}")
        print(f"\n  m = log2(N), so m^3.7 means ~log^3.7(N), NOT the O(log N)")
        print("  claimed in the paper.  The gap is entirely in _shift_gate:")
        print("  it builds an increment as a ladder of MCX gates with")
        print("  0..m-1 controls, and Qiskit synthesizes each ancilla-free at")
        print("  O(k^2).  To reach O(m) the increment must be the Gidney")
        print("  borrowed-ancilla construction (algassert.com/circuits/2015/")
        print("  06/12/Constructing-Large-Increment-Gates.html), which is")
        print("  O(m) Toffolis for the WHOLE incrementer, not per carry.")
        print("\n  Second, independent constant factor: SELECT is currently")
        print("  17 blocks each controlled on all 5 PREP qubits.  The term")
        print("  index factorizes (2 bits pick Vx, 2 pick Vy, 2 pick sigma),")
        print("  so SELECT can be three small multiplexers, roughly 4")
        print("  controlled incrementers instead of 34.")
    print()


if __name__ == '__main__':
    main()