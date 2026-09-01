"""
smoke_test.py — Verification harness for pyblockencode.

Covers the Poisson operator in 1-3 dimensions (FDM and FEM) and the 2D
plane-stress Q4 elasticity operator, under every boundary treatment:
periodic, essential (clamped), traction-free, one-sided and mixed.

Each case is checked along independent paths:

  decomposition   the LCU reproduces the operator assembled from dense
                  1D factors
  vs assembly     that operator agrees with a direct finite element
                  assembly on the degrees of freedom that survive the
                  essential constraints
  block encoding  alpha * U[:N0, :N0] equals the operator
  unitarity       U is unitary

Boundary conditions use the reflection construction of ``bc.py``.  There is
no flag qubit anywhere; a failure of the ``num_ancilla`` checks below would
mean one had crept back in.

Usage:
    python -m pyblockencode.smoke_test

Exit code 0 = all pass, 1 = any failure.
"""
from __future__ import annotations

import math
import sys

TOL = 1e-11


def _check(label: str, value: float, results: list, tol: float = TOL) -> bool:
    ok = value < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<46} {value:.2e}")
    results.append(ok)
    return ok


def _check_eq(label: str, got, want, results: list) -> bool:
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<46} {got!r}"
          f"{'' if ok else f'  (expected {want!r})'}")
    results.append(ok)
    return ok


# ---------------------------------------------------------------------------

def test_bc(results: list) -> None:
    """The 1D factor algebra and the transpose permutation."""
    import numpy as np
    from pyblockencode import bc

    print("\n-- 1D factors and reflections ------------------------------")
    N = 8
    worst = 0.0
    for lbl in bc.LABELS:
        A = bc.unitary(lbl, N)
        worst = max(worst, float(np.abs(A @ A.T - np.eye(N)).max()),
                    float(np.abs(A.T - bc.unitary(bc.TRANSPOSE[lbl], N)).max()))
    _check("unitarity and the transpose permutation", worst, results)

    # alpha is unchanged by any boundary treatment
    for kind, want in (("K", 4.0), ("M", 1.0)):
        vals = [bc.alpha(bc.factor(kind, e))
                for e in ("periodic", "essential", "free",
                          ("clamped", "free"), ("free", "clamped"))]
        _check(f"alpha({kind}) constant across treatments",
               float(max(vals) - min(vals)), results)
        _check(f"alpha({kind}) == {want}", abs(vals[0] - want), results)

    # transposing a factor whose antisymmetry the free correction destroyed
    for ends in ("free", ("clamped", "free")):
        G = bc.factor("G", ends)
        err = float(np.abs(bc.dense(bc.transpose(G), N)
                           - bc.dense(G, N).T).max())
        _check(f"transpose(G, {ends}) stays in the unitary set", err, results)


def test_poisson(results: list) -> None:
    from pyblockencode.poisson_pattern import PoissonPatternEncoding

    print("\n-- Poisson -------------------------------------------------")
    for dim in (1, 2, 3):
        for disc in ("fdm", "fem"):
            for b in ("periodic", "essential", "free"):
                enc = PoissonPatternEncoding(m=2, dim=dim, disc=disc, bc=b)
                r = enc.verify()
                tag = f"{dim}D {disc} {b}"
                _check(f"{tag}: decomposition",
                       r["decomposition_rel_err"], results)
                if "vs_classical_assembly" in r:
                    _check(f"{tag}: vs classical assembly",
                           r["vs_classical_assembly"], results)
                if "block_encoding_rel_err" in r:
                    _check(f"{tag}: block encoding",
                           r["block_encoding_rel_err"], results)
                    _check(f"{tag}: unitarity", r["unitarity_err"], results)

    print("\n-- Poisson, mixed conditions -------------------------------")
    for label, spec in [("x clamped, y free", {"x": "essential", "y": "free"}),
                        ("clamped left only", ("clamped", "free")),
                        ("per-direction list", ["essential", "free"])]:
        enc = PoissonPatternEncoding(m=2, dim=2, disc="fem", bc=spec)
        r = enc.verify()
        _check(f"{label}: decomposition", r["decomposition_rel_err"], results)
        _check_eq(f"{label}: L == |Bx|*|By|", enc.num_terms,
                  enc.basis_sizes[0] * enc.basis_sizes[1], results)

    print("\n-- Poisson, L and alpha constant in N -----------------------")
    for b in ("periodic", "essential", "free"):
        Ls = [PoissonPatternEncoding(m=m, dim=2, disc="fem", bc=b).num_terms
              for m in (2, 3, 4, 5)]
        als = [PoissonPatternEncoding(m=m, dim=2, disc="fem", bc=b).alpha
               for m in (2, 3, 4, 5)]
        _check_eq(f"{b}: L constant", len(set(Ls)), 1, results)
        _check(f"{b}: alpha constant", float(max(als) - min(als)), results)

    print("\n-- Poisson, no flag qubit ----------------------------------")
    enc = PoissonPatternEncoding(m=3, dim=2, disc="fem", bc="essential")
    _check_eq("ancilla == ceil(log2 L)", enc.num_ancilla,
              math.ceil(math.log2(enc.num_terms)), results)


def test_elasticity(results: list) -> None:
    from pyblockencode.elasticity_pattern import ElasticityPatternEncoding

    print("\n-- Elasticity ----------------------------------------------")
    cases = [
        ("periodic", "periodic", 17, ("I", "Z", "X")),
        ("all clamped", "essential", 49, ("I", "Z", "X")),
        ("clamped left+right", [("clamped", "clamped"), ("free", "free")],
         75, ("I", "Z", "X", "iY")),
        ("clamped left only", [("clamped", "free"), ("free", "free")],
         98, ("I", "Z", "X", "iY")),
        ("traction-free", "free", 109, ("I", "Z", "X", "iY")),
    ]
    for label, spec, want_L, want_comp in cases:
        enc = ElasticityPatternEncoding(m=2, nu=0.3, bc=spec)
        r = enc.verify()
        _check_eq(f"{label}: L", enc.num_terms, want_L, results)
        _check_eq(f"{label}: components", enc.components, want_comp, results)
        _check(f"{label}: decomposition", r["decomposition_rel_err"], results)
        _check(f"{label}: vs Q4 assembly", r["vs_q4_assembly"], results)
        if "block_encoding_rel_err" in r:
            _check(f"{label}: block encoding",
                   r["block_encoding_rel_err"], results)
            _check(f"{label}: unitarity", r["unitarity_err"], results)

    print("\n-- Elasticity, alpha ---------------------------------------")
    for nu in (0.0, 0.3, 0.45):
        enc = ElasticityPatternEncoding(m=2, nu=nu, bc="essential")
        _check(f"nu={nu}: alpha matches E(33+nu)/(6(1-nu^2))",
               abs(enc.alpha - enc.alpha_closed_form()), results)

    print("\n-- Elasticity, the fourth component at nu = 1/3 -------------")
    enc = ElasticityPatternEncoding(m=2, nu=1 / 3, bc="free")
    _check_eq("iY vanishes", "iY" in enc.components, False, results)
    _check_eq("L drops to 93", enc.num_terms, 93, results)

    print("\n-- Elasticity, no flag qubit -------------------------------")
    enc = ElasticityPatternEncoding(m=3, nu=0.3, bc="essential")
    _check_eq("ancilla == ceil(log2 L)", enc.num_ancilla,
              math.ceil(math.log2(enc.num_terms)), results)
    _check_eq("total qubits == 2m+7", enc.num_qubits, 2 * 3 + 7, results)


def test_circuits(results: list) -> None:
    """Qiskit circuits, if qiskit is installed."""
    try:
        from pyblockencode.qiskit_encoding import (PoissonCircuit,
                                                   ElasticityCircuit, apply_Ax)
    except ImportError:
        print("\n-- Circuits: qiskit not installed, skipping ----------------")
        return
    import numpy as np

    print("\n-- Circuits ------------------------------------------------")
    for b in ("periodic", "essential", "free", ("clamped", "free")):
        r = PoissonCircuit(m=2, dim=1, disc="fdm", bc=b).verify()
        _check(f"Poisson 1D {b}: block encoding",
               r["block_encoding_rel_err"], results)

    r = PoissonCircuit(m=2, dim=2, disc="fdm",
                       bc={"x": "essential", "y": "free"}).verify()
    _check("Poisson 2D mixed: block encoding",
           r["block_encoding_rel_err"], results)

    r = ElasticityCircuit(m=1, nu=0.3, bc="periodic").verify()
    _check("Elasticity periodic: block encoding",
           r["block_encoding_rel_err"], results)

    # the larger term counts are checked end to end instead of by
    # building the dense operator, which is prohibitive past ~10 qubits
    rng = np.random.default_rng(0)
    for label, b in [("essential", "essential"),
                     ("cantilever", [("clamped", "free"), ("free", "free")]),
                     ("traction-free", "free")]:
        ec = ElasticityCircuit(m=1, nu=0.3, bc=b)
        n = 2 * ec.enc.N ** 2
        x = rng.random(n) + 1j * rng.random(n)
        x /= np.linalg.norm(x)
        _check(f"Elasticity {label}: A|x> by simulation",
               apply_Ax(ec, x)["rel_err"], results)


def main() -> int:
    results: list = []
    try:
        test_bc(results)
        test_poisson(results)
        test_elasticity(results)
        test_circuits(results)
    except Exception as exc:              # noqa: BLE001
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        return 1
    n, total = sum(results), len(results)
    print("\n" + "=" * 62)
    print(f"{n}/{total} checks passed")
    print("=" * 62)
    return 0 if n == total else 1


if __name__ == "__main__":
    sys.exit(main())
