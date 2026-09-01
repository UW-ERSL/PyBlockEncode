#!/usr/bin/env python3
"""
makeFigures.py - generate the PDF figures for the elasticity block-encoding paper.

Place in the repository root (next to pyblockencode/) and run:

    python makeFigures.py                # figs/*.pdf, gate counts to m = 6
    python makeFigures.py --mmax 8       # push the scaling figure further
    python makeFigures.py --no-legacy    # skip the slow MCX-ladder series

Writes into ./figs/ :

    fig_scaling.pdf   measured T-count vs m, MCX ladder vs linear incrementer
    fig_alpha.pdf     alpha(nu) and the convergence of alpha/||K|| to (33+nu)/24
    fig_shear.pdf     the elasticity operator in component-major ordering,
                      at nu = 0 and nu = 0.3, showing that the shear blocks
                      do NOT vanish at nu = 0

Figures are vector PDF at column width for a two-column article, in a serif
face that matches the body text.  Every series is distinguished by marker and
line style as well as hue, so the figures survive grayscale printing and
colour-vision deficiency; the categorical hues are slots 1-3 of a palette
validated for all-pairs CVD separation.

Expensive transpilation results are cached in .paper_cache.json and shared
with makeTables.py.  Delete that file to force recomputation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

FIGS = "figs"
CACHE = ".paper_cache.json"
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


def cache_load() -> dict:
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def cache_save(d: dict) -> None:
    with open(CACHE, 'w') as f:
        json.dump(d, f, indent=1, sort_keys=True)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# data collection (shared cache with makeTables.py)
# ---------------------------------------------------------------------------

def collect_gatecounts(mmax: int, legacy: bool, cache: dict) -> dict:
    from qiskit import transpile
    from pyblockencode.linear_circuits import LinearElasticityCircuit
    from pyblockencode.qiskit_encoding import ElasticityCircuit

    gc = cache.get('gatecounts', {})
    for m in range(2, mmax + 1):
        rec = gc.get(str(m), {})
        need_new = 'new_t' not in rec
        need_old = legacy and 'old_t' not in rec
        if not (need_new or need_old):
            continue
        print(f"  transpiling m={m} ...", end='', flush=True)
        t0 = time.time()
        if need_new:
            n = LinearElasticityCircuit(m=m, nu=0.3)
            tq = transpile(n.circuit(), basis_gates=BASIS, optimization_level=1)
            ops = tq.count_ops()
            rec.update(new_t=ops.get('t', 0) + ops.get('tdg', 0),
                       new_cx=ops.get('cx', 0), new_depth=tq.depth(),
                       new_qubits=n.num_qubits)
        if need_old:
            o = ElasticityCircuit(m=m, nu=0.3)
            tq = transpile(o.circuit(), basis_gates=BASIS, optimization_level=1)
            ops = tq.count_ops()
            rec.update(old_t=ops.get('t', 0) + ops.get('tdg', 0),
                       old_cx=ops.get('cx', 0), old_depth=tq.depth(),
                       old_qubits=o.num_qubits)
        gc[str(m)] = rec
        print(f" {time.time()-t0:.1f}s")
    cache['gatecounts'] = gc
    cache_save(cache)
    return gc


# ---------------------------------------------------------------------------
# Figure 1 - measured scaling
# ---------------------------------------------------------------------------

def fig_scaling(gc: dict, legacy: bool):
    ms = sorted(int(k) for k in gc)
    new = np.array([gc[str(m)]['new_t'] for m in ms], float)
    have_old = legacy and all('old_t' in gc[str(m)] for m in ms)
    old = np.array([gc[str(m)]['old_t'] for m in ms], float) if have_old else None

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis='both')

    x = np.array(ms, float)
    if have_old:
        p_old = np.polyfit(np.log(x[-3:]), np.log(old[-3:]), 1)[0]
        ax.plot(x, old, marker='s', ls='--', color=C_ORANGE,
                label='MCX ladder', zorder=3)
        ax.annotate(rf'$\sim m^{{{p_old:.1f}}}$',
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

    N = 2 ** m
    C = 1.0 / (1.0 - nu ** 2)
    K1 = sp.diags([-1., 2., -1.], [-1, 0, 1], shape=(N, N), format='csr')
    M1 = sp.diags([1., 4., 1.], [-1, 0, 1], shape=(N, N), format='csr') / 6.0
    G1 = sp.diags([-1., 1.], [-1, 1], shape=(N, N), format='csr') / 2.0

    Kxx = C * (sp.kron(K1, M1) + (1 - nu) / 2 * sp.kron(M1, K1))
    Kyy = C * ((1 - nu) / 2 * sp.kron(K1, M1) + sp.kron(M1, K1))
    Kxy = -C * (1 + nu) / 2 * sp.kron(G1, G1)
    K = sp.bmat([[Kxx, Kxy], [Kxy.T, Kyy]], format='csr')
    return float(spla.eigsh(K, k=1, which='LA',
                            return_eigenvectors=False, tol=1e-10)[0])


def fig_alpha(cache: dict, mmax_norm: int = 8):
    from pyblockencode.elasticity_pattern import ElasticityPatternEncoding

    ratios = cache.get('alpha_ratio', {})
    nus = [0.0, 0.3, 0.45]
    for nu in nus:
        key = f"{nu}"
        have = ratios.get(key, {})
        for m in range(2, mmax_norm + 1):
            if str(m) in have:
                continue
            print(f"  ||K|| for nu={nu}, m={m} ...", end='', flush=True)
            nk = _spec_norm(m, nu)
            have[str(m)] = nk
            print(f" {nk:.4f}")
        ratios[key] = have
    cache['alpha_ratio'] = ratios
    cache_save(cache)

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
    axR.set_ylabel(r'$\alpha\,/\,\|\mathbf{K}\|$')
    axR.set_xlim(ms[0] - 0.2, ms[-1] + 0.2)
    axR.legend(frameon=False, loc='upper right', handlelength=1.8)
    axR.set_title(r'(b) convergence to $(33+\nu)/24$ (dotted)',
                  loc='left', color=INK2)
    despine(axR)

    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_alpha.pdf')


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
    ap.add_argument('--no-legacy', action='store_true')
    args = ap.parse_args()

    try:
        import pyblockencode  # noqa: F401
    except ImportError:
        sys.exit("Run this from the repository root (the folder holding "
                 "pyblockencode/).")

    style()
    cache = cache_load()
    legacy = not args.no_legacy

    print("scaling ...")
    fig_scaling(collect_gatecounts(args.mmax, legacy, cache), legacy)
    print("subnormalization ...")
    fig_alpha(cache)
    print("shear structure ...")
    fig_shear()

    print("\nDone.  \\includegraphics{figs/...} from the manuscript.")


if __name__ == '__main__':
    main()
