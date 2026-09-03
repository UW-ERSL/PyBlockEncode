#!/usr/bin/env python3
"""
makePauliCountFigure.py - Pauli term counts for the two-phase Poisson cell.

    python makePauliCountFigure.py          # figs/fig_pauli_count.pdf

Linear (1D) and Q4 bilinear (2D) discretizations of -div(k grad u) on a
periodic cell of 2^m elements per direction, the scalar twin of the elasticity
operator: k = k1 on an inclusion at v_f = 1/4 and k2 = r k1 outside.  The 1D
panel also carries the classical Dirichlet tridiagonal, tridiag(-1, 2, -1),
as the reference every treatment of this operator starts from.

Counts come from a fast tensor transform, one pass per qubit over the whole
array, rather than 4^n separate trace inner products; that is what makes
n = 12 qubits reachable.  Every count below is exact, and each fits a closed
form over the whole range tested:

    1D, n = m qubits            2D, n = 2m qubits
    ------------------------    ---------------------------
    tridiagonal   2^m           r = 1    (9/16) 4^m
    r = 1     (3/4) 2^m         r = 10   (2^(m+1) - 1)^2
    r = 10    (9/4) 2^m - 2

The 1D contrast law holds for m >= 3 and the others for m >= 2; below that
the inclusion is one or two elements and the cell is degenerate.

The shape of the result is the same in both dimensions.  L is proportional to
the number of unknowns, 2^(dm), not to the 4^n of a dense Hermitian operator,
so a Pauli expansion of this operator is far cheaper than the worst case and
still exponential in m.  Contrast costs a constant factor -- three in 1D,
about seven in 2D -- and does not touch the growth rate.  That factor is the
whole argument for carrying the material field in an oracle rather than
enumerating it.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import matplotlib.pyplot as plt

from makeFigures import (FULL_W, C_AQUA, C_BLUE, C_ORANGE, INK2, despine,
                         save, style)

R = 10.0             # contrast ratio r = k2/k1
VF = 0.25            # inclusion volume fraction
MMAX_1D = 12         # 12 qubits
MMAX_2D = 6          # also 12 qubits, the transform being O(4^n) in memory
MMAX_PLOT = 12

#: element stiffness: linear on a segment, Q4 bilinear on the unit square
KE_1D = np.array([[1.0, -1.0], [-1.0, 1.0]])
KE_2D = np.array([[4, -1, -2, -1], [-1, 4, -1, -2],
                  [-2, -1, 4, -1], [-1, -2, -1, 4]]) / 6.0


def pauli_count(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Number of nonzero Pauli coefficients of a 2^n-square matrix.

    One pass per qubit over the whole array, splitting every block into its
    I, X, Y, Z parts, rather than 4^n separate trace inner products.
    """
    n = int(np.log2(A.shape[0]))
    B = A.astype(complex).reshape(1, A.shape[0], A.shape[0])
    for _ in range(n):
        M, D, _ = B.shape
        B = B.reshape(M, 2, D // 2, 2, D // 2)
        a00, a01 = B[:, 0, :, 0, :], B[:, 0, :, 1, :]
        a10, a11 = B[:, 1, :, 0, :], B[:, 1, :, 1, :]
        B = np.stack([(a00 + a11) / 2, (a01 + a10) / 2,
                      1j * (a01 - a10) / 2, (a00 - a11) / 2],
                     axis=1).reshape(M * 4, D // 2, D // 2)
    return int(np.count_nonzero(np.abs(B.ravel()) > tol))


def _dyadic_origin(N: int, side: int) -> int:
    """Centre the inclusion, snapping a dyadic one to a dyadic position."""
    if side and not (side & (side - 1)):
        return side * ((N // side) // 2)
    return (N - side) // 2


def tridiagonal(m: int) -> np.ndarray:
    """The classical 1D Poisson operator, tridiag(-1, 2, -1), Dirichlet."""
    N = 2 ** m
    return 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)


def poisson_1d(m: int, r: float, vf: float = VF) -> np.ndarray:
    """Periodic two-phase 1D Laplacian; k = 1 on the inclusion, r outside."""
    N = 2 ** m
    side = max(int(round(N * vf)), 1)
    o = _dyadic_origin(N, side)
    chi = np.zeros(N)
    chi[o:o + side] = 1.0
    k = r + (1.0 - r) * chi
    K = np.zeros((N, N))
    for e in range(N):
        g = np.array([e % N, (e + 1) % N])
        K[np.ix_(g, g)] += k[e] * KE_1D
    return K


def poisson_2d(m: int, r: float, vf: float = VF) -> np.ndarray:
    """Periodic two-phase Q4 Laplacian; k = 1 on the inclusion, r outside."""
    N = 2 ** m
    side = int(round(N * np.sqrt(vf)))
    o = _dyadic_origin(N, side)
    chi = np.zeros((N, N))
    chi[o:o + side, o:o + side] = 1.0
    k = r + (1.0 - r) * chi
    K = np.zeros((N * N, N * N))
    for ex, ey in itertools.product(range(N), repeat=2):
        g = np.array([((ex + a) % N) * N + ((ey + b) % N)
                      for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]])
        K[np.ix_(g, g)] += k[ex, ey] * KE_2D
    return K


def _panel(ax, ms, series, dense, title, xlabel):
    """One panel: a dense-operator bound plus measured series with their law."""
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.plot(ms, dense[1], ls='--', color=C_ORANGE, zorder=3, label=dense[0])
    for label, law, valid_from, got, col, mk, ls in series:
        keep = ms >= valid_from
        ax.plot(ms[keep], law[keep], ls=ls, color=col, zorder=4, label=label)
        ax.plot(range(1, len(got) + 1), got, ls='none', marker=mk, ms=3.5,
                mfc='white', mec=col, mew=1.0, zorder=5)
    ax.set_yscale('log')
    ax.set_xticks(ms[::2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Pauli terms $L$')
    ax.legend(frameon=False, loc='upper left', handlelength=2.0, fontsize=6)
    ax.set_title(title, loc='left', color=INK2)
    despine(ax)


def fig_pauli_count():
    ms = np.arange(1, MMAX_PLOT + 1)

    print("1D:")
    g1_het, g1_hom, g1_tri = [], [], []
    for m in range(1, MMAX_1D + 1):
        g1_het.append(pauli_count(poisson_1d(m, R)))
        g1_hom.append(pauli_count(poisson_1d(m, 1.0)))
        g1_tri.append(pauli_count(tridiagonal(m)))
        print(f"  m={m:>2}  tridiag={g1_tri[-1]:>5}  r=1: {g1_hom[-1]:>5}"
              f"  r={R:g}: {g1_het[-1]:>5}")

    print("2D:")
    g2_het, g2_hom = [], []
    for m in range(1, MMAX_2D + 1):
        t0 = time.time()
        g2_het.append(pauli_count(poisson_2d(m, R)))
        g2_hom.append(pauli_count(poisson_2d(m, 1.0)))
        print(f"  m={m:>2}  r=1: {g2_hom[-1]:>7}  r={R:g}: {g2_het[-1]:>7}"
              f"   ({time.time()-t0:.1f}s)")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL_W, 2.7))

    _panel(
        axL, ms,
        [(rf'$r={R:g}$:  $\frac{{9}}{{4}}2^{{m}}-2$',
          (9 / 4) * 2.0 ** ms - 2, 3, g1_het, C_BLUE, 'o', '-'),
         (r'$r=1$:  $\frac{3}{4}2^{m}$',
          (3 / 4) * 2.0 ** ms, 2, g1_hom, C_AQUA, '^', '-.'),
         (r'tridiagonal:  $2^{m}$',
          2.0 ** ms, 1, g1_tri, INK2, 'D', ':')],
        (r'dense Hermitian:  $4^{n}=4^{m}$', 4.0 ** ms),
        '(a) 1D,  $n=m$ qubits',
        r'qubits per dimension $m=\log_2 N$')

    _panel(
        axR, ms,
        [(rf'$r={R:g}$:  $(2^{{m+1}}-1)^2$',
          (2.0 ** (ms + 1) - 1) ** 2, 2, g2_het, C_BLUE, 'o', '-'),
         (r'$r=1$:  $\frac{9}{16}4^{m}$',
          (9 / 16) * 4.0 ** ms, 2, g2_hom, C_AQUA, '^', '-.')],
        (r'dense Hermitian:  $4^{n}=16^{m}$', 4.0 ** (2 * ms)),
        '(b) 2D,  $n=2m$ qubits',
        r'qubits per dimension $m=\log_2 N$')

    fig.suptitle(r'Periodic Poisson cell, inclusion at $v_f=1/4$;  '
                 r'marks counted directly, lines the closed form',
                 fontsize=7.5, x=0.005, ha='left', y=1.03, color=INK2)
    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_pauli_count.pdf')


if __name__ == '__main__':
    style()
    fig_pauli_count()