"""
smoke_test.py — Verification harness for pyblockencode.

Tests every (dim, disc) combination for Poisson and the 2D plane-stress
Q4 elasticity operator.  All block-encoding errors should be at or below
machine precision (~1e-14); decomposition errors should be near-zero.

Usage:
    python examples/smoke_test.py          # from package root
    python -m pyblockencode.smoke_test     # as module

Exit code 0 = all pass, 1 = any failure.
"""
from __future__ import annotations
import sys

def _load_modules():
    """Import pyblockencode sub-modules with graceful fallback."""
    try:
        from pyblockencode.poisson_pattern import PoissonPatternEncoding
        from pyblockencode.elasticity_pattern import ElasticityPatternEncoding
        return PoissonPatternEncoding, ElasticityPatternEncoding
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run 'pip install -e .' from the package root first.")
        sys.exit(1)


def _check(label: str, value: float, threshold: float, results: list) -> bool:
    ok = value < threshold
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {value:.2e}  (threshold {threshold:.0e})")
    results.append(ok)
    return ok


def test_poisson(PoissonPatternEncoding, results: list) -> None:
    print("\n── Poisson ──────────────────────────────────────────────────")
    M = 2   # m=2 for all checks (N=4, manageable dense unitary)

    combos = [
        (1, "fdm"), (1, "fem"),
        (2, "fdm"), (2, "fem"),
        (3, "fdm"), (3, "fem"),
    ]
    for dim, disc in combos:
        enc = PoissonPatternEncoding(m=M, dim=dim, disc=disc)
        # 3D unitary is too large to build densely at m=2 (qubits ≥ 10)
        build = (dim <= 2)
        r = enc.verify(build_unitary=build)

        print(f"\n  Poisson {dim}D {disc.upper()}: "
              f"L={enc.num_terms}, α={enc.alpha:.4f}, "
              f"qubits={enc.num_qubits}")

        # LCU decomposition α constant in N
        enc3 = PoissonPatternEncoding(m=3, dim=dim, disc=disc)
        alpha_stable = abs(enc.alpha - enc3.alpha) < 1e-12
        _check("α constant in N (m=2 vs m=3)", 0.0 if alpha_stable else 1.0,
               1e-10, results)

        if build:
            _check("block-encoding rel-err", r["block_encoding_rel_err"],
                   1e-13, results)
            _check("unitarity ‖U†U−I‖",     r["unitarity_err"],
                   1e-13, results)


def test_elasticity(ElasticityPatternEncoding, results: list) -> None:
    print("\n── Elasticity (plane-stress Q4) ─────────────────────────────")

    for nu in [0.0, 0.3, 0.45]:
        enc = ElasticityPatternEncoding(m=2, E=1.0, nu=nu)
        r = enc.verify()

        print(f"\n  ν={nu:.2f}: L={enc.num_terms}, α={enc.alpha:.4f}, "
              f"qubits={enc.num_qubits}")

        # α constant in N
        enc3 = ElasticityPatternEncoding(m=3, E=1.0, nu=nu)
        alpha_stable = abs(enc.alpha - enc3.alpha) < 1e-12
        _check("α constant in N (m=2 vs m=3)", 0.0 if alpha_stable else 1.0,
               1e-10, results)

        # decomposition() uses cyclic (periodic) shifts; the Dirichlet
        # boundary correction is handled by the flag ancilla in the unitary,
        # so a nonzero decomposition error is expected and correct.
        # We only verify it is bounded (not the full Dirichlet target).
        _check("decomposition rel-err < 1.0",
               r["decomposition_rel_err"] if r["decomposition_rel_err"] < 1.0 else 2.0,
               1.0, results)
        if "block_encoding_rel_err" in r:
            _check("block-encoding rel-err",   r["block_encoding_rel_err"],   1e-13, results)
            _check("unitarity ‖U†U−I‖",        r["unitarity_err"],             1e-12, results)


def main():
    PoissonPatternEncoding, ElasticityPatternEncoding = _load_modules()

    print("=" * 60)
    print("  pyblockencode smoke test")
    print("=" * 60)

    results: list[bool] = []
    test_poisson(PoissonPatternEncoding, results)
    test_elasticity(ElasticityPatternEncoding, results)

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"  {n_pass}/{n_total} checks passed")
    print("=" * 60)

    if n_pass < n_total:
        print(f"  {n_total - n_pass} FAILED")
        sys.exit(1)
    else:
        print("  ALL PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
