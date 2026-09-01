#!/usr/bin/env python3
"""
makeFigures.py - generate the PDF figures for the elasticity block-encoding paper.

Place in the repository root (next to pyblockencode/) and run:

    python makeFigures.py                # figs/*.pdf, gate counts to m = 6
    python makeFigures.py --mmax 8       # push the scaling figure further
    python makeFigures.py --mcx          # add the MCX-ladder comparison series

The MCX-ladder series is opt-in because it is the expensive half of the run:
its T-count is superlinear in m where the linear incrementer is affine, so it
costs 46s at m = 4 and grows fast.  Pass --mcx when regenerating the figure
for the manuscript; without it fig_scaling shows the linear series alone.

Writes into ./figs/ :

    fig_scaling.pdf   measured T-count vs m, MCX ladder vs linear incrementer
    fig_alpha.pdf     alpha(nu) and the convergence of alpha/||K||_2 to
                      (33+nu)/24
    fig_boundary.pdf  term count and subnormalization under each boundary
                      treatment, and where the fourth Pauli component appears
    fig_shear.pdf     the elasticity operator in component-major ordering,
                      at nu = 0 and nu = 0.3, showing that the shear blocks
                      do NOT vanish at nu = 0

Figures are vector PDF at column width for a two-column article, in a serif
face that matches the body text.  Every series is distinguished by marker and
line style as well as hue, so the figures survive grayscale printing and
colour-vision deficiency; the categorical hues are slots 1-3 of a palette
validated for all-pairs CVD separation.

Nothing is cached: every run transpiles from scratch.  A cache keyed only on
m silently survives a change to the encoders and hands back gate counts for
circuits that no longer exist, which is a worse failure than a slow run.  Use
--mmax and --mcx to bound the cost.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

FIGS = "figs"
BASIS = ['h', 't', 'tdg', 's', 'sdg', 'x', 'z', 'cx', 'rz']

# Categorical slots 1-3 of the validated palette (all-pairs CVD dE 9.2,
# normal-vision dE 24.0 on a light surface).
C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRIDC = '#0b0b0b', '#52514e', '#d8d7d2'

# Diverging pair for the operator heatmap: blue <-> red with a gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    'blue_gray_red',
    ['#104281', '#2a78d6', '#9ec5f4', '#f0efec', '#f0a3a3', '#d03b3b', '#7d1f1f'])

COL_W = 3.35     # single column, inches
FULL_W = 6.9     # full text width, inches


def style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.edgecolor': INK2,
        'axes.linewidth': 0.6,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK2,
        'ytick.color': INK2,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'grid.color': GRIDC,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.4,
        'lines.markersize': 4,
        'figure.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# data collection
# ---------------------------------------------------------------------------

def collect_gatecounts(mmax: int, mcx: bool) -> dict:
    from qiskit import transpile
    from pyblockencode.linear_circuits import LinearElasticityCircuit
    from pyblockencode.qiskit_encoding import ElasticityCircuit

    gc: dict = {}
    for m in range(2, mmax + 1):
        rec: dict = {}
        print(f"  transpiling m={m} ...", end='', flush=True)
        t0 = time.time()

        n = LinearElasticityCircuit(m=m, nu=0.3, bc='essential')
        tq = transpile(n.circuit(), basis_gates=BASIS, optimization_level=1)
        ops = tq.count_ops()
        rec.update(new_t=ops.get('t', 0) + ops.get('tdg', 0),
                   new_cx=ops.get('cx', 0), new_depth=tq.depth(),
                   new_qubits=n.num_qubits)

        if mcx:
            o = ElasticityCircuit(m=m, nu=0.3, bc='essential')
            tq = transpile(o.circuit(), basis_gates=BASIS, optimization_level=1)
            ops = tq.count_ops()
            rec.update(old_t=ops.get('t', 0) + ops.get('tdg', 0),
                       old_cx=ops.get('cx', 0), old_depth=tq.depth(),
                       old_qubits=o.num_qubits)

        gc[str(m)] = rec
        print(f" {time.time()-t0:.1f}s")
    return gc


# ---------------------------------------------------------------------------
# Figure 1 - measured scaling
# ---------------------------------------------------------------------------

def fig_scaling(gc: dict, mcx: bool):
    ms = sorted(int(k) for k in gc)
    new = np.array([gc[str(m)]['new_t'] for m in ms], float)
    have_old = mcx and all('old_t' in gc[str(m)] for m in ms)
    old = np.array([gc[str(m)]['old_t'] for m in ms], float) if have_old else None

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis='both')

    x = np.array(ms, float)
    if have_old:
        # Fit log T = log a + p log m over the top of the range.  Annotate
        # the whole fitted law, not just the exponent: the linear series is
        # labelled with a value, so a bare m^p beside it on a log axis reads
        # as one too.
        p_old, log_a = np.polyfit(np.log(x[-3:]), np.log(old[-3:]), 1)
        a_old = np.exp(log_a)
        mant, expo = f"{a_old:.1e}".split('e')
        ax.plot(x, old, marker='s', ls='--', color=C_ORANGE,
                label='MCX ladder', zorder=3)
        ax.annotate(rf'$\approx {mant}\times 10^{{{int(expo)}}}\,m^{{{p_old:.1f}}}$',
                    xy=(x[-1], old[-1]), xytext=(-4, -13),
                    textcoords='offset points', ha='right', va='top',
                    color=C_ORANGE, fontsize=7.5)

    slope = (new[-1] - new[-2]) / (x[-1] - x[-2])
    icept = new[-1] - slope * x[-1]
    ax.plot(x, new, marker='o', ls='-', color=C_BLUE,
            label='linear incrementer', zorder=4)
    ax.annotate(rf'${slope:.0f}\,m {icept:+.0f}$',
                xy=(x[-1], new[-1]), xytext=(-4, 8),
                textcoords='offset points', ha='right', va='bottom',
                color=C_BLUE, fontsize=7.5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel(r'qubits per dimension $m=\log_2 N$')
    ax.set_ylabel(r'$T$ count')
    ax.legend(frameon=False, loc='upper left', handlelength=2.0)
    despine(ax)
    save(fig, 'fig_scaling.pdf')


# ---------------------------------------------------------------------------
# Figure 2 - subnormalization
# ---------------------------------------------------------------------------

def _spec_norm(m: int, nu: float) -> float:
    """Largest eigenvalue of the elasticity operator, via a sparse assembly.

    K is symmetric positive definite, so ||K||_2 is its largest eigenvalue.
    Assembling densely and calling np.linalg.norm(.,2) would be an SVD of a
    2N^2-square matrix -- 8192 square already at m=6.  Sparse Kronecker
    products plus a single Lanczos pass costs a fraction of a second.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from pyblockencode import bc

    N = 2 ** m
    C = 1.0 / (1.0 - nu ** 2)
    # the same 1D factors the encoder uses, so the two cannot drift apart
    K1 = sp.csr_matrix(bc.dense(bc.factor('K', 'essential'), N))
    M1 = sp.csr_matrix(bc.dense(bc.factor('M', 'essential'), N))
    G1 = sp.csr_matrix(bc.dense(bc.factor('G', 'essential'), N))

    Kxx = C * (sp.kron(K1, M1) + (1 - nu) / 2 * sp.kron(M1, K1))
    Kyy = C * ((1 - nu) / 2 * sp.kron(K1, M1) + sp.kron(M1, K1))
    Kxy = C * (nu * sp.kron(G1.T, G1)
               + (1 - nu) / 2 * sp.kron(G1, G1.T))
    K = sp.bmat([[Kxx, Kxy], [Kxy.T, Kyy]], format='csr')
    return float(spla.eigsh(K, k=1, which='LA',
                            return_eigenvectors=False, tol=1e-10)[0])


def fig_alpha(mmax_norm: int = 8):
    from pyblockencode.elasticity_pattern import ElasticityPatternEncoding

    ratios: dict = {}
    nus = [0.0, 0.3, 0.45]
    for nu in nus:
        have = {}
        for m in range(2, mmax_norm + 1):
            print(f"  ||K||_2 for nu={nu}, m={m} ...", end='', flush=True)
            nk = _spec_norm(m, nu)
            have[str(m)] = nk
            print(f" {nk:.4f}")
        ratios[f"{nu}"] = have

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL_W, 2.4))

    # (a) alpha(nu): closed form against the LCU 1-norm
    nn = np.linspace(0, 0.49, 200)
    closed = (33 + nn) / (6 * (1 - nn ** 2))
    axL.set_axisbelow(True); axL.grid(True)
    axL.plot(nn, closed, color=C_BLUE, lw=1.4, zorder=3,
             label=r'$E(33+\nu)\,/\,6(1-\nu^{2})$')
    pts = [ElasticityPatternEncoding(m=3, nu=v).alpha for v in nus]
    axL.plot(nus, pts, ls='none', marker='o', mfc='white', mec=C_BLUE,
             mew=1.2, zorder=4, label=r'$\sum_k |c_k|$ from the circuit')
    for v, a in zip(nus, pts):
        axL.annotate(f'{a:.3f}', xy=(v, a), xytext=(7, -2),
                     textcoords='offset points', fontsize=6.5, color=INK2,
                     va='center')
    axL.set_xlabel(r"Poisson ratio $\nu$")
    axL.set_ylabel(r'subnormalization $\alpha$')
    axL.legend(frameon=False, loc='upper left', handlelength=1.8)
    axL.set_title('(a) closed form vs. computed', loc='left', color=INK2)
    despine(axL)

    # (b) alpha/||K|| -> (33+nu)/24
    axR.set_axisbelow(True); axR.grid(True)
    for nu, col, mk in zip(nus, (C_BLUE, C_ORANGE, C_AQUA), ('o', 's', '^')):
        ms = sorted(int(k) for k in ratios[f"{nu}"])
        a = ElasticityPatternEncoding(m=3, nu=nu).alpha
        y = [a / ratios[f"{nu}"][str(m)] for m in ms]
        axR.plot(ms, y, marker=mk, color=col, label=rf'$\nu={nu}$', zorder=4)
        axR.axhline((33 + nu) / 24, color=col, lw=0.8, ls=':', zorder=2)
    axR.set_xticks(ms)
    axR.set_xlabel(r'qubits per dimension $m=\log_2 N$')
    axR.set_ylabel(r'$\alpha\,/\,\|\mathbf{K}\|_2$')
    axR.set_xlim(ms[0] - 0.2, ms[-1] + 0.2)
    axR.legend(frameon=False, loc='upper right', handlelength=1.8)
    axR.set_title(r'(b) convergence to $(33+\nu)/24$ (dotted)',
                  loc='left', color=INK2)
    despine(axR)

    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_alpha.pdf')


# ---------------------------------------------------------------------------
# Figure - boundary treatments
# ---------------------------------------------------------------------------

def fig_boundary(nu: float = 0.3):
    """Term count and subnormalization across the boundary treatments.

    Both panels make the same point from different sides: imposing boundary
    conditions costs terms and never costs subnormalization, and the fourth
    Pauli component appears as soon as one direction carries a diagonal
    correction.
    """
    from pyblockencode.elasticity_pattern import ElasticityPatternEncoding

    cases = [
        ("periodic", "periodic"),
        ("all four\nclamped", "essential"),
        ("left and\nright", [("clamped", "clamped"), ("free", "free")]),
        ("left\nonly", [("clamped", "free"), ("free", "free")]),
        ("traction-\nfree", "free"),
    ]
    labels, Ls, alphas, four = [], [], [], []
    for label, spec in cases:
        e = ElasticityPatternEncoding(m=3, nu=nu, bc=spec)
        labels.append(label)
        Ls.append(e.num_terms)
        alphas.append(e.alpha)
        four.append("iY" in e.components)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    x = np.arange(len(cases))

    # (a) term count, coloured by how many Pauli components are needed
    axL.set_axisbelow(True); axL.grid(True, axis='y')
    cols = [C_ORANGE if f else C_BLUE for f in four]
    axL.bar(x, Ls, color=cols, width=0.62, zorder=3)
    for xi, L in zip(x, Ls):
        axL.annotate(str(L), xy=(xi, L), xytext=(0, 3),
                     textcoords='offset points', ha='center',
                     fontsize=6.5, color=INK2)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=6.5)
    axL.set_ylabel('LCU terms $L$')
    axL.set_ylim(0, max(Ls) * 1.18)
    axL.set_title(r'(a) term count; orange needs the fourth component $iY$',
                  loc='left', color=INK2)
    despine(axL)

    # (b) alpha, against the two bounds it lies between
    axR.set_axisbelow(True); axR.grid(True, axis='y')
    axR.plot(x, alphas, marker='o', color=C_BLUE, zorder=4)
    axR.axhline(alphas[0], color=INK2, lw=0.8, ls=':', zorder=2)
    axR.axhline(max(alphas), color=INK2, lw=0.8, ls=':', zorder=2)
    for xi, a in zip(x, alphas):
        axR.annotate(f'{a:.3f}', xy=(xi, a), xytext=(0, 5),
                     textcoords='offset points', ha='center',
                     fontsize=6.5, color=INK2)
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=6.5)
    axR.set_ylabel(r'subnormalization $\alpha$')
    span = max(alphas) - min(alphas)
    axR.set_ylim(min(alphas) - 0.35 * span, max(alphas) + 0.55 * span)
    axR.set_title(rf'(b) $\alpha$ at $\nu={nu}$, bounded by the two extremes',
                  loc='left', color=INK2)
    despine(axR)

    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_boundary.pdf')


# ---------------------------------------------------------------------------
# Figure 3 - the shear coupling does not vanish at nu = 0
# ---------------------------------------------------------------------------

def _component_major(m, nu):
    """Assemble K with all x-DOFs first, then all y-DOFs (Eq. 11 ordering)."""
    from pyblockencode import operators as op
    N = 2 ** m
    C = 1.0 / (1.0 - nu ** 2)
    K1, M1, G1 = op._K1(N), op._M1(N), op._G1(N)
    Kxx = C * (np.kron(K1, M1) + (1 - nu) / 2 * np.kron(M1, K1))
    Kyy = C * ((1 - nu) / 2 * np.kron(K1, M1) + np.kron(M1, K1))
    Kxy = -C * (1 + nu) / 2 * np.kron(G1, G1)
    return np.block([[Kxx, Kxy], [Kxy.T, Kyy]]), Kxy


def fig_shear(m: int = 2):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0))
    mats = [(0.0, 'ν = 0'), (0.3, 'ν = 0.3')]
    K0, _ = _component_major(m, 0.0)
    vmax = np.abs(K0).max()
    for ax, (nu, _lbl) in zip(axes, mats):
        K, Kxy = _component_major(m, nu)
        n = K.shape[0] // 2
        im = ax.imshow(K, cmap=DIVERGING, interpolation='nearest',
                       norm=SymLogNorm(linthresh=0.05, linscale=0.6,
                                       vmin=-vmax, vmax=vmax, base=10))
        ax.axhline(n - 0.5, color=INK, lw=0.8)
        ax.axvline(n - 0.5, color=INK, lw=0.8)
        for (r, c, t) in ((0.25, 0.25, r'$\mathbf{K}_{xx}$'),
                          (0.25, 0.75, r'$\mathbf{K}_{xy}$'),
                          (0.75, 0.25, r'$\mathbf{K}_{yx}$'),
                          (0.75, 0.75, r'$\mathbf{K}_{yy}$')):
            ax.text(c * 2 * n, r * 2 * n, t, ha='center', va='center',
                    fontsize=8, color=INK,
                    bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.2))
        ax.set_title(rf'$\nu={nu}$:  $\max|\mathbf{{K}}_{{xy}}| = '
                     rf'{np.abs(Kxy).max():.3f}$', loc='left', color=INK2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(INK2); s.set_linewidth(0.6)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6.5, width=0.5)
    fig.suptitle('The shear blocks are non-zero at every $\\nu > -1$'
                 '   (symmetric-log colour scale)',
                 fontsize=8, x=0.02, ha='left', y=1.02)
    save(fig, 'fig_shear.pdf')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmax', type=int, default=6)
    ap.add_argument('--mcx', action='store_true',
                    help='include the MCX-ladder comparison series; it is the '
                         'slow half of the run (46s at m=4, superlinear)')
    args = ap.parse_args()

    try:
        import pyblockencode  # noqa: F401
    except ImportError:
        sys.exit("Run this from the repository root (the folder holding "
                 "pyblockencode/).")

    style()
    mcx = args.mcx

    print("scaling ...")
    if not mcx:
        print("  (MCX-ladder series omitted; rerun with --mcx for the "
              "manuscript figure)")
    fig_scaling(collect_gatecounts(args.mmax, mcx), mcx)
    print("subnormalization ...")
    fig_alpha()
    print("boundary treatments ...")
    fig_boundary()
    print("shear structure ...")
    fig_shear()

    print("\nDone.  \\includegraphics{figs/...} from the manuscript.")


if __name__ == '__main__':
    main()