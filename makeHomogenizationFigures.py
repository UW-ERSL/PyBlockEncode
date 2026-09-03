#!/usr/bin/env python3
"""
makeHomogenizationFigures.py - figures for the periodic two-phase cell.

Place in the repository root (next to pyblockencode/) and run:

    python makeHomogenizationFigures.py              # figs/fig_homog_*.pdf
    python makeHomogenizationFigures.py --mmax 8     # push the cost figure
    python makeHomogenizationFigures.py --quick      # skip the slow panels

Writes into ./figs/ :

    fig_homog_invariance.pdf  alpha and L are flat in the mesh, the volume
                              fraction and the geometry, while ||K||_2 is not
    fig_homog_alpha.pdf       alpha against contrast and against nu, with the
                              closed form and the kink at nu = 1/3
    fig_homog_oracle.pdf      measured oracle cost: a dyadic square is set by
                              the volume fraction alone and is flat in m; the
                              comparator that buys continuous volume fraction
                              is not
    fig_homog_terms.pdf       where the 57 terms sit, over the nine stencil
                              offsets and the five material operators

Style, palette and page widths come from makeFigures, so the two figure sets
cannot drift apart.  Every series is distinguished by marker and line style as
well as hue, so the figures survive grayscale printing and colour-vision
deficiency.

The expensive quantity is ||K||_2.  It is taken from a single Lanczos pass on
the sparse assembly (``HomogenizationEncoding.spectral_norm``) rather than a
dense SVD, which is what lets the invariance figure reach m = 7 instead of
stopping at m = 5.  Nothing is cached between runs.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from makeFigures import (COL_W, FULL_W, C_AQUA, C_BLUE, C_ORANGE, GRIDC,
                         INK, INK2, despine, save, style)

# Sequential ramp for the term map: one hue family, light to dark, so the
# cells read as magnitude rather than as identity.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'blue_seq', ['#dce9fb', '#9ec5f4', '#5b9ae4', '#2a78d6', '#104281'])

NU = 0.3
E1, E2 = 3.0, 1.0


# ---------------------------------------------------------------------------
# Figure 1 - what the microstructure does not change
# ---------------------------------------------------------------------------

def fig_invariance(mmax: int = 7):
    """
    Both panels say the same thing from opposite sides.  alpha is a sum of
    element-stiffness entries and never references chi, so it is flat in the
    mesh and in the volume fraction; ||K||_2 is a property of the assembled
    operator and is neither.  The ratio is therefore not flat, and the second
    panel is where the benchmark geometry gets chosen.
    """
    from pyblockencode.homogenization import HomogenizationEncoding, Inclusion

    ms = list(range(2, mmax + 1))
    alphas, norms = [], []
    for m in ms:
        e = HomogenizationEncoding(m, E1, E2, NU, shape='square', vf=0.25)
        print(f"  ||K||_2 at m={m} ...", end='', flush=True)
        t0 = time.time()
        alphas.append(e.alpha)
        norms.append(e.spectral_norm())
        print(f" {norms[-1]:.4f}  ({time.time()-t0:.1f}s)")
    alphas, norms = np.array(alphas), np.array(norms)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL_W, 2.4))

    # (a) alpha flat, ||K||_2 rising.  Both are operator norms in the same
    # units, so they share one axis; a second scale here would be a lie.
    axL.set_axisbelow(True)
    axL.grid(True)
    axL.plot(ms, alphas, marker='o', ls='-', color=C_BLUE, zorder=4,
             label=r'$\alpha$')
    axL.plot(ms, norms, marker='s', ls='--', color=C_ORANGE, zorder=3,
             label=r'$\|\mathbf{K}\|_2$')
    axL.annotate(rf'$\alpha={alphas[0]:.4f}$ at every $m$',
                 xy=(ms[-1], alphas[-1]), xytext=(-4, 6),
                 textcoords='offset points', ha='right', va='bottom',
                 fontsize=7, color=C_BLUE)
    axL.annotate(rf'$\to {norms[-1]:.3f}$',
                 xy=(ms[-1], norms[-1]), xytext=(-4, -10),
                 textcoords='offset points', ha='right', va='top',
                 fontsize=7, color=C_ORANGE)
    axL.set_xticks(ms)
    axL.set_xlabel(r'qubits per dimension $m=\log_2 N$')
    axL.set_ylabel('operator norm')
    axL.set_ylim(0, max(alphas.max(), norms.max()) * 1.22)
    axL.legend(frameon=False, loc='lower right', handlelength=2.0)
    axL.set_title(r'(a) square inclusion, $v_f=1/4$, $E_1/E_2=3$',
                  loc='left', color=INK2)
    despine(axL)

    # (b) tightness against volume fraction, one curve per mesh
    axR.set_axisbelow(True)
    axR.grid(True)
    targets = [1 / 256, 1 / 64, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4]
    for m, col, mk, ls in zip((4, 5, 6), (C_BLUE, C_ORANGE, C_AQUA),
                              ('o', 's', '^'), ('-', '--', '-.')):
        xs, ys = [], []
        for vf in targets:
            inc = Inclusion('square', 2 ** m, vf)
            if not 0.0 < inc.volume_fraction < 1.0:
                continue
            e = HomogenizationEncoding(m, E1, E2, NU, inclusion=inc)
            xs.append(e.volume_fraction)
            ys.append(e.alpha / e.spectral_norm())
        axR.plot(xs, ys, marker=mk, ls=ls, color=col, zorder=4,
                 label=rf'$m={m}$')
    axR.axvline(0.25, color=INK2, lw=0.8, ls=':', zorder=2)
    axR.annotate('benchmark $v_f=1/4$', xy=(0.25, axR.get_ylim()[1]),
                 xytext=(-5, -4), textcoords='offset points',
                 ha='right', va='top', fontsize=6.5, color=INK2)
    axR.set_xscale('log', base=2)
    axR.set_xlabel(r'volume fraction $v_f$')
    axR.set_ylabel(r'$\alpha\,/\,\|\mathbf{K}\|_2$')
    axR.legend(frameon=False, loc='lower left', handlelength=2.0)
    axR.set_title('(b) dilution loosens the bound, refinement tightens it',
                  loc='left', color=INK2)
    despine(axR)

    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_homog_invariance.pdf')


# ---------------------------------------------------------------------------
# Figure 2 - the closed form for alpha
# ---------------------------------------------------------------------------

def _A(nu):
    return (33 + nu) / (6 * (1 - nu ** 2))


def _B(nu):
    return (69 + 5 * nu + 6 * np.abs(1 - 3 * nu)) / (12 * (1 - nu ** 2))


def fig_alpha_law(m: int = 4):
    """
    alpha = min(E1,E2) A(nu) + |E1-E2| B(nu), exactly.  The left panel is the
    contrast dependence, affine with slope B; the right is the nu dependence
    of the two coefficients, whose kink at nu = 1/3 is the iY component
    switching off.
    """
    from pyblockencode.homogenization import HomogenizationEncoding

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL_W, 2.4))

    # (a) alpha against contrast, closed form as the line, encoding as marks.
    # Linear axes: the claim is that alpha is affine in |E1-E2|, and a log
    # axis would turn every affine law into the same near-straight line.
    axL.set_axisbelow(True)
    axL.grid(True)
    marks = np.array([0.0, 1.0, 2.0, 4.0, 9.0, 14.0, 19.0])
    curve = np.linspace(0.0, 20.0, 200)
    for nu, col, mk, ls in zip((0.0, 0.3, 0.45),
                               (C_BLUE, C_ORANGE, C_AQUA),
                               ('o', 's', '^'), ('-', '--', '-.')):
        axL.plot(curve, _A(nu) + curve * _B(nu),
                 ls=ls, color=col, lw=1.2, zorder=3, label=rf'$\nu={nu}$')
        got = [HomogenizationEncoding(m, 1.0 + d, 1.0, nu, shape='square',
                                      vf=0.25).alpha for d in marks]
        axL.plot(marks, got, ls='none', marker=mk, mfc='white', mec=col,
                 mew=1.1, zorder=4)
    axL.annotate(rf'$\nu=0.3$:  slope $B={_B(0.3):.3f}$,' '\n'
                 rf'intercept $A={_A(0.3):.3f}$',
                 xy=(0.97, 0.04), xycoords='axes fraction',
                 ha='right', va='bottom', fontsize=6.5, color=C_ORANGE)
    axL.set_xlabel(r'contrast $|E_1-E_2|$   (with $\min(E_1,E_2)=1$)')
    axL.set_ylabel(r'subnormalization $\alpha$')
    axL.legend(frameon=False, loc='upper left', handlelength=2.2,
               title='line: closed form\nmarks: encoding',
               title_fontsize=6.5)
    axL.set_title('(a) affine in the contrast', loc='left', color=INK2)
    despine(axL)

    # (b) the two coefficients, and where the fourth Pauli component goes
    axR.set_axisbelow(True)
    axR.grid(True)
    nn = np.linspace(0.0, 0.49, 400)
    axR.plot(nn, _A(nn), ls='-', color=C_BLUE, zorder=4, label=r'$A(\nu)$')
    axR.plot(nn, _B(nn), ls='--', color=C_ORANGE, zorder=3, label=r'$B(\nu)$')
    axR.axvline(1 / 3, color=INK2, lw=0.8, ls=':', zorder=2)

    L_gen = HomogenizationEncoding(3, E1, E2, 0.3, shape='square',
                                   vf=0.25).num_terms
    L_third = HomogenizationEncoding(3, E1, E2, 1 / 3, shape='square',
                                     vf=0.25).num_terms
    axR.annotate(rf'$\nu=1/3$: $iY$ vanishes,' '\n' rf'$L={L_gen}\to{L_third}$',
                 xy=(1 / 3, _B(1 / 3)), xytext=(-6, 10),
                 textcoords='offset points', ha='right', va='bottom',
                 fontsize=6.5, color=INK2)
    axR.plot([1 / 3], [_B(1 / 3)], marker='o', ms=4, mfc='white',
             mec=INK2, mew=1.0, ls='none', zorder=5)
    axR.set_xlabel(r"Poisson ratio $\nu$")
    axR.set_ylabel('coefficient')
    axR.legend(frameon=False, loc='upper left', handlelength=2.2)
    axR.set_title(r'(b) $\alpha=\min(E_1,E_2)\,A+|E_1-E_2|\,B$',
                  loc='left', color=INK2)
    despine(axR)

    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_homog_alpha.pdf')


# ---------------------------------------------------------------------------
# Figure 3 - oracle cost
# ---------------------------------------------------------------------------

def fig_oracle(mmax: int = 7):
    """
    The whole microstructure dependence of the encoding sits in one phase
    oracle, so this is the only part of the circuit whose cost the geometry
    can touch.  For a dyadic square it is a single multi-controlled Z on the
    top bits of each coordinate: flat in m, set by the volume fraction alone.
    A comparator buys arbitrary volume fraction and gives that up.
    """
    from pyblockencode.homogenization import Inclusion
    from pyblockencode.homogenization_circuit import ReflectionOracle

    series = [
        (r'dyadic square, $v_f=1/4$',   'square',    0.25,     C_BLUE, 'o', '-'),
        (r'dyadic square, $v_f=1/64$',  'square',    1 / 64,   C_AQUA, '^', '-.'),
        (r'rectangle, comparator',      'rectangle', 0.25,     C_ORANGE, 's', '--'),
    ]

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.set_axisbelow(True)
    ax.grid(True)
    ms = list(range(2, mmax + 1))
    for label, shape, vf, col, mk, ls in series:
        xs, ys = [], []
        for m in ms:
            N = 2 ** m
            inc = Inclusion(shape, N, vf)
            if inc.volume_fraction <= 0.0:
                continue
            print(f"  oracle {shape} vf={vf:.4f} m={m} ...",
                  end='', flush=True)
            c = ReflectionOracle(inc).cost()
            print(f" {c['cx']} CX")
            xs.append(m)
            ys.append(max(c['cx'], 1))
        ax.plot(xs, ys, marker=mk, ls=ls, color=col, zorder=4, label=label)
        ax.annotate(f'{ys[-1]}', xy=(xs[-1], ys[-1]), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=6.5, color=col)

    ax.set_yscale('log')
    ax.set_xticks(ms)
    ax.set_xlim(ms[0] - 0.2, ms[-1] + 0.9)
    ax.set_xlabel(r'qubits per dimension $m=\log_2 N$')
    ax.set_ylabel('CX in the reflection oracle')
    ax.legend(frameon=False, loc='upper left', handlelength=2.2)
    despine(ax)
    save(fig, 'fig_homog_oracle.pdf')


# ---------------------------------------------------------------------------
# Figure 4 - where the terms are
# ---------------------------------------------------------------------------

def fig_terms(m: int = 3):
    """
    The 57 terms laid out over the nine stencil offsets and the five material
    operators.  The identity column block is the operator a homogeneous cell
    would need; the four reflection blocks are what the two-phase field adds,
    and they are the same element-stiffness entries halved.
    """
    from pyblockencode.homogenization import HomogenizationEncoding

    e = HomogenizationEncoding(m, E1, E2, NU, shape='square', vf=0.25)
    terms = e.lcu_terms()

    offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    off_lbl = {0: 'I', 1: 'Scd', -1: 'Sc'}
    rlabels = ['I', 'R1', 'R2', 'R3', 'R4']
    paulis = ['I', 'Z', 'X', 'iY']

    M = np.zeros((len(offs), len(rlabels) * len(paulis)))
    for i, (dx, dy) in enumerate(offs):
        for j, rl in enumerate(rlabels):
            for k, p in enumerate(paulis):
                key = (off_lbl[dx], off_lbl[dy], rl, p)
                M[i, j * len(paulis) + k] = abs(terms.get(key, 0.0))
    Mm = np.ma.masked_where(M == 0.0, M)

    n_id = sum(1 for k in terms if k[2] == 'I')
    n_rf = len(terms) - n_id

    fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
    # Log colour scale: the diagonal identity term is an order of magnitude
    # above the rest, and on a linear ramp it would flatten every other cell
    # to the same pale step.  The figure is about where the terms are.
    im = ax.imshow(Mm, cmap=SEQUENTIAL, interpolation='nearest',
                   aspect='auto',
                   norm=LogNorm(vmin=Mm.min(), vmax=Mm.max()))
    im.cmap.set_bad('#f6f5f2')

    for j in range(1, len(rlabels)):
        ax.axvline(j * len(paulis) - 0.5, color=INK, lw=0.9)
    for i in range(1, len(offs)):
        ax.axhline(i - 0.5, color=GRIDC, lw=0.4)

    ax.set_xticks(range(len(rlabels) * len(paulis)))
    ax.set_xticklabels(paulis * len(rlabels), fontsize=6)
    ax.set_yticks(range(len(offs)))
    ax.set_yticklabels([rf'$({dx:+d},{dy:+d})$' for dx, dy in offs],
                       fontsize=6.5)
    ax.set_ylabel('stencil offset $(d_x,d_y)$')

    sec = ax.secondary_xaxis('top')
    sec.set_xticks([j * len(paulis) + 1.5 for j in range(len(rlabels))])
    sec.set_xticklabels([r'$\mathbf{I}$', r'$R_1$', r'$R_2$', r'$R_3$',
                         r'$R_4$'], fontsize=7.5)
    sec.tick_params(length=0)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)

    ax.set_xlabel(r'DOF-qubit component $\sigma_r$, grouped by material '
                  r'operator')
    ax.set_title(rf'$L={len(terms)}$ at every microstructure:  '
                 rf'{n_id} carry the identity, {n_rf} a reflection'
                 rf'   ($\nu={NU}$, $E_1/E_2={E1/E2:.0f}$)',
                 loc='left', color=INK2)

    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cb.set_label(r'$|c|$', fontsize=7)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6.5, width=0.5)
    save(fig, 'fig_homog_terms.pdf')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmax', type=int, default=7,
                    help='largest m for the invariance and oracle figures')
    ap.add_argument('--quick', action='store_true',
                    help='stop the invariance figure at m=5 and the oracle '
                         'figure at m=5')
    args = ap.parse_args()

    try:
        import pyblockencode  # noqa: F401
    except ImportError:
        sys.exit("Run this from the repository root (the folder holding "
                 "pyblockencode/).")

    style()
    mmax = 5 if args.quick else args.mmax

    print("invariance ...")
    fig_invariance(mmax)
    print("closed form for alpha ...")
    fig_alpha_law()
    print("oracle cost ...")
    fig_oracle(mmax)
    print("term structure ...")
    fig_terms()

    print("\nDone.  \\includegraphics{figs/fig_homog_...} from the manuscript.")


if __name__ == '__main__':
    main()
