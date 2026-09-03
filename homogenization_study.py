"""
homogenization_study.py — tables for the periodic two-phase block encoding.

Run: python homogenization_study.py
"""
import time

import numpy as np

from pyblockencode.homogenization import HomogenizationEncoding, Inclusion
from pyblockencode.homogenization_circuit import (HomogenizationCircuit,
                                                  ReflectionOracle)


def rule(title):
    print(f"\n{title}\n" + "-" * len(title))


# ---------------------------------------------------------------------------
rule("1. mesh independence: square inclusion, vf = 1/4, E1/E2 = 3, nu = 0.3")
print(f"{'m':>2} {'N':>4} {'dofs':>7} {'L':>4} {'alpha':>9} {'|K|_2':>9} "
      f"{'alpha/|K|':>9} {'qubits':>7} {'rel err':>9}")
for m in range(2, 7):
    e = HomogenizationEncoding(m, 3.0, 1.0, 0.3, shape="square", vf=0.25)
    K = e.target()
    n2 = np.linalg.norm(K, 2)
    err = (np.linalg.norm(e.dense() - K) / np.linalg.norm(K)) if m <= 5 else np.nan
    print(f"{m:>2} {e.N:>4} {2*e.N**2:>7} {e.num_terms:>4} {e.alpha:>9.4f} "
          f"{n2:>9.4f} {e.alpha/n2:>9.4f} {e.num_qubits:>7} {err:>9.1e}")

# ---------------------------------------------------------------------------
rule("2. volume fraction and shape, m = 4 (N = 16), E1/E2 = 3")
print(f"{'shape':<10} {'vf target':>9} {'vf':>7} {'L':>4} {'alpha':>9} "
      f"{'alpha/|K|':>9} {'oracle':>9} {'dyadic':>7}")
for shape in ("square", "rectangle", "cross"):
    for vf in (1 / 64, 1 / 16, 1 / 4, 1 / 2):
        inc = Inclusion(shape, 16, vf)
        e = HomogenizationEncoding(4, 3.0, 1.0, 0.3, inclusion=inc)
        orc = ReflectionOracle(inc)
        n2 = np.linalg.norm(e.target(), 2)
        print(f"{shape:<10} {vf:>9.4f} {e.volume_fraction:>7.4f} "
              f"{e.num_terms:>4} {e.alpha:>9.4f} {e.alpha/n2:>9.4f} "
              f"{orc.method:>9} {str(inc.dyadic):>7}")

# ---------------------------------------------------------------------------
rule("3. contrast, square vf = 1/4 at m = 4;  alpha = Emin A(nu) + |E1-E2| B(nu)")
print(f"{'E1/E2':>8} {'L':>4} {'alpha':>10} {'closed form':>12} {'err':>9} "
      f"{'alpha/|K|':>9}")
for E1 in (1.0, 2.0, 3.0, 10.0, 100.0, 0.1):
    e = HomogenizationEncoding(4, E1, 1.0, 0.3, shape="square", vf=0.25)
    n2 = np.linalg.norm(e.target(), 2)
    print(f"{E1:>8.2f} {e.num_terms:>4} {e.alpha:>10.4f} "
          f"{e.alpha_closed_form():>12.4f} "
          f"{abs(e.alpha-e.alpha_closed_form()):>9.1e} {e.alpha/n2:>9.4f}")

# ---------------------------------------------------------------------------
rule("4. Poisson ratio;  the iY component vanishes at nu = 1/3")
print(f"{'nu':>7} {'L':>4} {'components':>18} {'alpha':>10} {'closed form':>12}")
for nu in (0.0, 0.15, 0.25, 0.3, 1 / 3, 0.4, 0.45):
    e = HomogenizationEncoding(3, 3.0, 1.0, nu, shape="square", vf=0.25)
    print(f"{nu:>7.4f} {e.num_terms:>4} {','.join(e.components):>18} "
          f"{e.alpha:>10.4f} {e.alpha_closed_form():>12.4f}")

# ---------------------------------------------------------------------------
rule("5. oracle cost;  a dyadic square is set by the volume fraction, not by m")
print(f"{'m':>2} {'shape':<10} {'vf':>7} {'method':>9} {'anc':>4} {'cx':>7} "
      f"{'depth':>7}")
for m in range(2, 8):
    for shape, vf in (("square", 0.25), ("square", 1 / 64), ("rectangle", 0.25)):
        N = 2 ** m
        if shape == "square" and vf < 1 / N ** 2:
            continue
        inc = Inclusion(shape, N, vf)
        c = ReflectionOracle(inc).cost()
        print(f"{m:>2} {shape:<10} {inc.volume_fraction:>7.4f} "
              f"{c['method']:>9} {c['ancilla']:>4} {c['cx']:>7} "
              f"{c['depth']:>7}")

# ---------------------------------------------------------------------------
rule("6. full block encoding circuit")
print(f"{'m':>2} {'qubits':>7} {'oracle':>9} {'cx':>9} {'depth':>9} {'t (s)':>7}")
for m in range(1, 6):
    t0 = time.time()
    c = HomogenizationCircuit(m, 3.0, 1.0, 0.3, shape="square", vf=0.25)
    d = c.cost()
    print(f"{m:>2} {d['num_qubits']:>7} {d['oracle']:>9} {d['cx']:>9} "
          f"{d['depth']:>9} {time.time()-t0:>7.1f}")

# ---------------------------------------------------------------------------
rule("7. circuit verification, alpha U[:N0,:N0] against a direct assembly")
print(f"{'m':>2} {'shape':<10} {'vf':>7} {'E1/E2':>7} {'oracle':>9} "
      f"{'qubits':>7} {'rel err':>10} {'t (s)':>7}")
for m, shape, vf, E1, E2, orc in [
        (1, "square", 0.25, 3.0, 1.0, "auto"),
        (2, "square", 0.25, 3.0, 1.0, "auto"),
        (2, "square", 1 / 16, 3.0, 1.0, "auto"),
        (2, "square", 0.25, 1.0, 10.0, "auto"),
        (2, "rectangle", 0.25, 3.0, 1.0, "compare"),
        (2, "cross", 0.25, 3.0, 1.0, "diagonal")]:
    t0 = time.time()
    c = HomogenizationCircuit(m, E1, E2, 0.3, shape=shape, vf=vf, oracle=orc)
    v = c.verify_columns()
    print(f"{m:>2} {shape:<10} {v['volume_fraction']:>7.4f} {E1/E2:>7.2f} "
          f"{v['oracle']:>9} {v['num_qubits']:>7} "
          f"{v['block_encoding_rel_err']:>10.1e} {time.time()-t0:>7.1f}")
