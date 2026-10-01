# PyBlockEncode

Block encodings of periodic finite element operators, built without assembling the matrix.

`pyblockencode` constructs a linear combination of unitaries (LCU) block encoding $\mathcal{U}$ of a periodic stiffness matrix $K$ such that the top-left block of $\mathcal{U}$ equals $K/\alpha$. You supply an operator and a grid size; it returns a Qiskit circuit and a report of $L$ (number of LCU terms), $\alpha$ (subnormalization), qubit counts, and gate counts. The matrix is formed only when you request verification.

This repository accompanies the paper *Block Encoding for 2-Phase Periodic Poisson and Elasticity* (K. Suresh, University of Wisconsin–Madison) and regenerates every table and figure in it.

## Operators

All operators are periodic. $K_1 = \mathrm{circ}(-1,2,-1)$ and $M_1 = \tfrac16\,\mathrm{circ}(1,4,1)$.

| Operator spec | String name | Operator | $L$ | $\alpha$ |
|---|---|---|---|---|
| `POISSON1D()` | `poisson1d` | $K_1$ | 3 | 4 |
| `POISSON2D('fd')` | `poisson2d_fd` | $K_1 \otimes I + I \otimes K_1$ | 5 | 8 |
| `POISSON2D('fe')` | `poisson2d_fe` | $K_1 \otimes M_1 + M_1 \otimes K_1$ (scalar Q4) | 9 | 16/3 |
| `ELASTICITY2D(nu, E)` | `elasticity2d` | plane-stress Q4, 2 dofs per node | 17 | $E(33+\nu)/(6(1-\nu^2))$ |
| `POISSON2D_2PHASE(vf, E1, E2)` | `poisson2d_2phase` | two-phase scalar Q4 cell | 25 | $\tfrac{16}{3}\max(E_1,E_2)$ |
| `ELASTICITY2D_2PHASE(nu, vf, E1, E2)` | `elasticity2d_2phase` | two-phase plane-stress Q4 cell | 57 | see below |

For the two-phase elasticity cell,

$$\alpha = \frac{\tfrac12(E_1+E_2)(33+\nu) + |E_1-E_2|\,(18+2\nu+3|1-3\nu|)}{6(1-\nu^2)}.$$

In every case $L$ and $\alpha$ are independent of the grid size. For the two-phase cells they are also independent of the volume fraction and of the microstructure. The term count of the two-phase elasticity cell drops from 57 to 49 at $\nu = 1/3$, where the $iY$ component vanishes.

The two-phase cells carry a per-element modulus $E = E_2 + (E_1 - E_2)\chi$, with $\chi \in \{0,1\}$ the indicator of a centred dyadic square inclusion of volume fraction `vf` $= 4^{-k}$, $k \ge 1$ integer. Defaults are `vf=0.25`, `E1=10`, `E2=1`, `nu=0.3`.

## Installation

```bash
git clone https://github.com/UW-ERSL/PyBlockEncode.git
cd PyBlockEncode
python -m pip install -r requirements.txt
```

`requirements.txt` pins the environment used for the paper (Qiskit 2.4.2, NumPy 2.4.6) and includes Matplotlib and `pylatexenc` for the figure scripts. It requires Python 3.11 or newer. To install the module alone, with only NumPy and Qiskit as dependencies:

```bash
python -m pip install .
```

## Quick start

```python
from pyblockencode import blockencode, ELASTICITY2D

circuit, info = blockencode(ELASTICITY2D(nu=0.3), N=4096)
print(info)
```

```
----- output --------------
ELASTICITY2D(nu=0.3, E=1.0)   N = 4096 per direction   m = 12 qubits/direction   dofs = 33,554,432
  L      = 17
  alpha  = 6.098901   (closed form 6.098901)
  qubits = 43  (system 25, prep 6, select 1, carry 11)
  Toffoli = 100   CX = 719   depth = 1105      [transpiled to ['cx', 'u']]
```

A system of $33.5$ million degrees of freedom is encoded on 43 qubits; no matrix is formed.

### Size arguments

`N` is the number of grid points **per direction** and must be a power of two; `m = log2(N)` is the number of qubits per direction. Pass either `N` or `m`, not both. The operator has $N$ rows in 1D, $N^2$ in 2D, and $2N^2$ for elasticity.

### Verification

`materialize=True` assembles $K$ densely and runs four checks: $\alpha$ against its closed form, the LCU term list against an independent element-by-element assembly from Gauss quadrature, and $\alpha\langle 0|\mathcal{U}|0\rangle$ against $K$ before and after transpilation. The cost is $O(4^n)$ in the total qubit count $n$, so use it at small `m` only.

```python
from pyblockencode import blockencode, ELASTICITY2D_2PHASE

circuit, info = blockencode(ELASTICITY2D_2PHASE(nu=0.3, vf=0.25, E1=10, E2=1),
                            N=4, materialize=True)
print(info)
```

```
----- output --------------
ELASTICITY2D_2PHASE(nu=0.3, vf=0.25, E1=10, E2=1)   N = 4 per direction   m = 2 qubits/direction   dofs = 32
  L      = 57
  alpha  = 64.697802   (closed form 64.697802)
  qubits = 16  (system 5, prep 9, select 1, carry 1)
  Toffoli = 31   CX = 1204   depth = 2204      [transpiled to ['cx', 'u']]
  verified: alpha vs closed form  1.42e-14
            alpha vs published    0.00e+00
            terms vs assembly     3.55e-15
            block vs K            6.76e-12
            block vs K transpiled 1.17e-11
            OK = True
```

All residuals are at floating-point level, so the circuit encodes the assembled two-phase stiffness matrix.

### Other entry points

- **`blockencode(op, N, *, m, materialize, costs, atol)`:** returns `(circuit, EncodingInfo)`. Set `costs=False` to skip the transpilation used for gate counts.
- **`PeriodicBlockEncoding(op, m)`:** the underlying class, with `.circuit()`, `.terms`, `.L`, `.alpha`, `.resources()`, `.matrix()`, `.reference()`, and `.verify()`.
- **`op.terms()`, `op.alpha()`:** the LCU coefficients and the analytic subnormalization of an operator spec, without a size.
- **`increment_circuit(m, inverse=False)`:** the controlled cyclic shift used throughout, for inspection.

String names are accepted in place of operator specs, for example `blockencode("poisson2d_fe", m=6)`.

## Construction

- **Factorised SELECT:** every LCU term is a product $S_x^{d_x} \otimes S_y^{d_y} \otimes \sigma$ with $d_x, d_y \in \{I, S, S^\dagger\}$ and $\sigma \in \{I, Z, X, iY\}$ on the dof qubit. SELECT is a product of one multiplexer per register, so a homogeneous 2D cell contains four controlled shifts regardless of $L$.
- **Signs in the preparation:** $\mathcal{U} = P_R^\dagger \cdot \mathrm{SELECT} \cdot P_L$, with $P_L|0\rangle = \sum_k \sqrt{|c_k|/\alpha}\,|k\rangle$ and $P_R|0\rangle = \sum_k \mathrm{sgn}(c_k)\sqrt{|c_k|/\alpha}\,|k\rangle$. No signs appear inside SELECT.
- **Shifts:** ripple-carry increments with $2m-2$ Toffoli gates, $m$ CNOT gates, and $m-1$ clean ancillas, shared by every shift in the circuit.
- **Material reflections:** the two-phase material enters only through $R_s = S^{(a,b)\dagger} R_\chi S^{(a,b)}$ with $R_\chi = I - 2\,\mathrm{diag}(\chi)$. For a centred dyadic square, the oracle $R_\chi$ costs $4k$ CNOT gates and one multi-controlled Z of width $2k+1$, independent of $m$.
- **Qubit order:** the system register occupies the least-significant qubits (x lowest, then y, then the dof qubit), so the encoded block is the contiguous top-left block of the unitary.

## Resource summary

Counts from `blockencode(op, m=m)`, transpiled to `['cx', 'u']` at Qiskit optimization level 2, with default operator parameters.

| Operator | $L$ | $\alpha$ | Qubits ($m=4$ / $12$) | Toffoli ($m=4$ / $12$) | CX ($m=4$ / $12$) |
|---|---|---|---|---|---|
| `poisson1d` | 3 | 4.000 | 10 / 26 | 16 / 48 | 99 / 291 |
| `poisson2d_fd` | 5 | 8.000 | 16 / 40 | 32 / 96 | 217 / 601 |
| `poisson2d_fe` | 9 | 5.333 | 16 / 40 | 32 / 96 | 217 / 601 |
| `elasticity2d` | 17 | 6.099 | 19 / 43 | 36 / 100 | 335 / 719 |
| `poisson2d_2phase` | 25 | 53.333 | 19 / 43 | 57 / 185 | 593 / 1361 |
| `elasticity2d_2phase` | 57 | 64.698 | 22 / 46 | 63 / 191 | 1396 / 2164 |

$L$ and $\alpha$ are constant in $m$; qubit and gate counts grow linearly in $m = \log_2 N$. The two 2D Poisson discretizations have identical gate counts because the circuit contains the same four controlled shifts in both; they differ only in the preparation.

Run `python pyblockencode.py` to reproduce the verified examples and a scaling table for $m \in \{4, 8, 12\}$.

## Repository contents

| File | Purpose |
|---|---|
| `pyblockencode.py` | The library: term algebra, operator specs, circuit construction, verification. |
| `PyBlockEncode_Tutorial.ipynb` | Walkthrough: operator catalogue, circuit drawings for each cell, verification, two-phase cells, cost scaling. |
| `makeTables.py` | Writes every table in the paper to `tables/` as a LaTeX `tabular`. |
| `makeFigures.py` | Writes every figure in the paper to `figs/` as PDF. |
| `requirements.txt` | Pinned environment used for the paper. |
| `setup.py` | Minimal install of the module. |

## Reproducing the paper

```bash
python makeTables.py            # all tables  -> tables/*.tex
python makeFigures.py           # all figures -> figs/*.pdf
python makeTables.py --list     # names and the paper section each serves
python makeTables.py tightness  # a single table
```

`makeTables.py` asserts each closed form in the paper against the measured value before writing the table. `makeFigures.py` verifies each circuit in the same run before drawing it and prints the residuals.

## Scope

1. **Periodic boundary conditions only:** there are no Dirichlet, traction, or mixed boundary conditions. The reflections in the two-phase cells are material reflections, not boundary reflections.
2. **Uniform grids with power-of-two size per direction:** the shift construction requires $N = 2^m$.
3. **Two-phase microstructure:** the implemented oracle is a centred dyadic square inclusion with volume fraction $4^{-k}$. Other geometries enter only through $R_\chi$; $L$ and $\alpha$ are unchanged.
4. **Block encoding only:** the library does not implement QSVT, a linear solver, or state preparation for a right-hand side.

## Citation

If you use this code, please cite:

```bibtex
@misc{suresh_pyblockencode,
  author = {Krishnan Suresh},
  title  = {Block Encoding for 2-Phase Periodic Poisson and Elasticity},
  note   = {Code: \url{https://github.com/UW-ERSL/PyBlockEncode}},
  year   = {2026}
}
```

## Contact

Krishnan Suresh, Engineering Representations and Simulation Laboratory (ERSL), University of Wisconsin–Madison, ksuresh@wisc.edu.
