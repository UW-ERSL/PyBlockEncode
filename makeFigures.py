"""makeFigures.py -- every figure in the paper, generated from pyblockencode.py.

    python makeFigures.py                 all figures
    python makeFigures.py increment       just that one
    python makeFigures.py --list          names and destinations

Each circuit figure is drawn from a circuit that has been verified in the same
run, so a figure is never a picture of an unchecked construction. The residuals
are printed as the figures are written.

Everything lands in FIGDIR as .pdf, for \\includegraphics.
"""
from __future__ import annotations

import sys
import os

import numpy as np
from qiskit import QuantumCircuit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pyblockencode import (blockencode, increment_circuit,
                         POISSON2D, ELASTICITY2D,
                         POISSON2D_2PHASE, ELASTICITY2D_2PHASE)

FIGDIR = "figs"
MPL = {"name": "bw", "creglinecolor": "#000000",
       "fontsize": 14, "subfontsize": 11}
EDGE, RED = "#3c3c3c", "#b03030"

# the four cells of the paper, in the order they are derived, each with the
# names of its SELECT stages in circuit order: the spatial multiplexers, then
# the displacement multiplexer, then the reflection stage (as `tail`).
SPATIAL = ["$S_x$", "$S_x^\\dagger$", "$S_y$", "$S_y^\\dagger$"]
REFL = ["$S_{ab}$", "$R_\\chi$", "$S_{ab}^\\dagger$"]

CELLS = [
    ("sc_hom", POISSON2D("fe"), "2D scalar, homogeneous",
     SPATIAL, None),
    ("sc_2ph", POISSON2D_2PHASE(vf=0.25, E1=3.0, E2=1.0), "2D scalar, two-phase",
     SPATIAL, REFL),
    ("el_hom", ELASTICITY2D(nu=0.3), "2D elasticity, homogeneous",
     SPATIAL + ["$Z$", "$X$"], None),
    ("el_2ph", ELASTICITY2D_2PHASE(nu=0.3, vf=0.25, E1=3.0, E2=1.0),
     "2D elasticity, two-phase", SPATIAL + ["$Z$", "$X$", "$iY$"], REFL),
]
# CELLS stays in derivation order, which is the order the circuit figures
# appear in the paper. The scaling legend reads by physics instead.
LEGEND = ["2D scalar, homogeneous", "2D elasticity, homogeneous",
          "2D scalar, two-phase", "2D elasticity, two-phase"]

BETWEEN = "reindex"          # the gap where the prep index is recoded
M_FIG = 2               # circuits are drawn at this size
LABEL_STAGES = True     # bracket each SELECT stage with a labelled barrier
ROWS = {"sc_hom": 1, "sc_2ph": 2, "el_hom": 2, "el_2ph": 2}   # rows per figure
WIRE_SIZE = 1.8         # wire labels, relative to Qiskit's size
STAGE_SIZE = 1.6        # barrier labels, relative to Qiskit's size
STAGE_LIFT = 0.40       # barrier label baseline above the top wire, in wires



def label_select_stages(qc, labels, tail=None, between=None, flag="s"):
    """Bracket each SELECT stage with a labelled barrier.

    The pair of Toffolis that sets and clears the select flag delimits one
    multiplexer, and what sits between them is the operator it applies. A
    stage driven by a one-bit prep field has no such pair; `tail` labels
    everything after the last flagged stage, which is where those sit. Three
    tail labels split it into the conjugating shift, the oracle, and the shift
    undone. `between` labels the gaps, where the flag is cleared, the prep
    index is recoded for the next term, and the flag is set again.

    Ported from BlockEncode_Tutorial.ipynb.
    """
    reg = lambda q: qc.find_bit(q).registers[0][0].name
    ccx = [k for k, ins in enumerate(qc.data)
           if ins.operation.name == "ccx" and reg(ins.qubits[2]) == flag]
    pairs = list(zip(ccx[0::2], ccx[1::2]))[:len(labels)]
    opens = {a + 1: labels[j] for j, (a, _) in enumerate(pairs)}
    closes = {b: between for _, b in pairs}

    if tail:
        tail = [tail] if isinstance(tail, str) else list(tail)
        t0, end = pairs[-1][1] + 1, len(qc.data) - 1      # end = closing prep
        if len(tail) == 1:
            opens[t0] = tail[0]
        else:
            # Each tail stage is driven by a one-bit prep flag. The oracle
            # flag fires once, between the two conjugating shifts, so its span
            # is the one nested inside the others.
            span = {}
            for k, ins in enumerate(qc.data[t0:end], start=t0):
                for q in ins.qubits:
                    if reg(q) == "p":
                        lo, hi = span.get(q, (k, k))
                        span[q] = (min(lo, k), max(hi, k))
            nested = [q for q, (lo, hi) in span.items()
                      if any(l < lo and h > hi for r, (l, h) in span.items()
                             if r is not q)]
            g = min(nested, key=lambda q: span[q][1] - span[q][0])
            conj = [k for k, ins in enumerate(qc.data[t0:end], start=t0)
                    if any(q in span and q is not g and
                           span[q][0] < span[g][0] < span[q][1]
                           for q in ins.qubits if reg(q) == "p")]
            gk = span[g][0]
            opens[min(conj)] = tail[0]
            opens[max(k for k in conj if k < gk) + 1] = tail[1]
            opens[min(k for k in conj if k > gk)] = tail[2]
        closes[end] = None

    out = QuantumCircuit(*qc.qregs, name=qc.name)
    for k, ins in enumerate(qc.data):
        if k in opens:
            out.barrier(label=opens[k])
        if k in closes:
            out.barrier(label=closes[k])
        out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def labels_for_print(fig) -> None:
    """Enlarge the wire and barrier labels, and lift the barrier labels.

    Gate text stays at Qiskit's size because the gate boxes are sized for it.
    Wire labels are the only right-aligned text and barrier labels the only
    text hung top-and-centre, which is how each is found. Qiskit hangs a
    barrier label from 0.4225 above the top wire, so it overlaps the gate boxes
    there, which reach 0.325; it is set instead on a baseline STAGE_LIFT above
    the wire. Clipping is dropped so labels past the axes limits survive.
    """
    for t in fig.axes[0].texts:
        if t.get_ha() == "right":
            t.set_fontsize(WIRE_SIZE * t.get_fontsize())
        elif t.get_va() == "top" and t.get_ha() == "center":
            x, y = t.get_position()
            t.set_position((x, y - 0.65 * 0.65 + STAGE_LIFT))
            t.set_va("bottom")
            t.set_fontsize(STAGE_SIZE * t.get_fontsize())
        else:
            continue
        t.set_clip_on(False)


def save(fig, stem: str) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(f"{FIGDIR}/fig_{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    {FIGDIR}/fig_{stem}.pdf")


def fig_inclusion(m: int = 5) -> None:
    """The two-phase cell, drawn from the SAME chi the circuit's oracle uses.

    Matrix shaded, inclusion left white, one line per element, as in the
    reference figure.
    """
    op = POISSON2D_2PHASE(vf=0.25)
    chi = op.chi(m)                        # chi[ex, ey], 1 in the inclusion
    N = 2 ** m
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    shade = np.where(chi.T > 0.5, 1.0, 0.52)   # inclusion white, matrix grey
    ax.imshow(shade, origin="lower", cmap="gray", vmin=0.0, vmax=1.0,
              extent=(0, N, 0, N), interpolation="nearest")
    for t in range(N + 1):
        ax.axhline(t, color="black", lw=0.5)
        ax.axvline(t, color="black", lw=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_aspect("equal")
    save(fig, "inclusion")
    xs = np.where(chi.any(axis=1))[0]
    print(f"       chi from pyblockencode: centred, span [{xs[0]},{xs[-1] + 1}) "
          f"of {N}, side {len(xs)}, vf = {chi.mean():.4f}")


def fig_mapping(ix: int = 8, iy: int = 3) -> None:
    """The four elements incident on one node. Pure geometry, no circuit."""
    from pyblockencode import ELEM_OFFSETS, CORNERS
    colours = {(0, 0): "#e2dbf0", (-1, 0): "#f9e6d2",
               (0, -1): "#dbead6", (-1, -1): "#d7e5f2"}
    white = dict(facecolor="white", edgecolor="none", alpha=0.85,
                 boxstyle="round,pad=0.06")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for (a, b) in ELEM_OFFSETS:
        ex, ey = ix + a, iy + b
        ax.add_patch(Rectangle((ex, ey), 1, 1, facecolor=colours[(a, b)],
                               edgecolor=EDGE, lw=1.7, zorder=1))
        ax.text(ex + .5, ey + .5, f"$e = ({ex},{ey})$", ha="center",
                va="center", fontsize=14, color="#1c1c1c", zorder=3)
    for nx in (ix - 1, ix, ix + 1):
        for ny in (iy - 1, iy, iy + 1):
            mid = (nx, ny) == (ix, iy)
            ax.plot([nx], [ny], "o", markersize=14 if mid else 14,
                    color=RED if mid else EDGE, zorder=5)
    for (a, b) in ELEM_OFFSETS:            # each element is named by this node
        ax.plot([ix + a], [iy + b], marker="s", markersize=15,
                markerfacecolor="none", markeredgecolor=EDGE, mew=1.6, zorder=7)
    D = 0.09
    for nx in (ix - 1, ix, ix + 1):
        for ny in (iy - 1, iy, iy + 1):
            sx, sy = nx - ix, ny - iy
            if sx == 0 and sy == 0:
                dx, dy, ha, va = 0.0, D + .04, "center", "bottom"
            elif sx and sy:
                dx, dy = D * sx, D * sy
                ha = "left" if sx > 0 else "right"
                va = "bottom" if sy > 0 else "top"
            elif sx:
                dx, dy, ha, va = D * sx, 0.0, \
                    ("left" if sx > 0 else "right"), "center"
            else:
                dx, dy, ha, va = 0.0, D * sy, "center", \
                    ("bottom" if sy > 0 else "top")
            mid = sx == 0 and sy == 0
            ax.text(nx + dx, ny + dy, f"({nx}, {ny})", ha=ha, va=va,
                    fontsize=12 if mid else 10.5,
                    color=RED if mid else "#404040",
                    fontweight="bold" if mid else "normal",
                    bbox=white, zorder=8)
    ax.set_xlim(ix - 1.32, ix + 1.32)
    ax.set_ylim(iy - 1.28, iy + 1.28)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "mapping")
    _ = CORNERS


def fig_element(a: int = -1, b: int = 0, cp=(0, 1)) -> None:
    """One element seen from node i: the local corners c', and one delta.

    Node i sits at c = -(a,b); the element's own index node is its lower-left
    corner, c' = (0,0). The arrow runs to the node at c', a separation of
    delta = (a,b) + c'. Drawn for the worked instance of Sec. 5, and the
    indices are checked against pyblockencode's CORNERS ordering.
    """
    from pyblockencode import CORNERS, ELEM_OFFSETS
    assert (a, b) in ELEM_OFFSETS and tuple(cp) in CORNERS
    c = (-a, -b)                                # node i's own local corner
    delta = (a + cp[0], b + cp[1])              # nodal separation
    tint = {(0, 0): "#e2dbf0", (-1, 0): "#f9e6d2",
            (0, -1): "#dbead6", (-1, -1): "#d7e5f2"}[(a, b)]
    white = dict(facecolor="white", edgecolor="none", alpha=0.9,
                 boxstyle="round,pad=0.08")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"

    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    ax.add_patch(Rectangle((a, b), 1, 1, facecolor=tint, edgecolor=EDGE,
                           lw=1.8, zorder=1))

    pos = lambda q: (a + q[0], b + q[1])        # node at local corner q
    for q in CORNERS:                           # every corner of the element
        px, py = pos(q)
        here = (q == c)
        ax.plot([px], [py], "o", markersize=13 if here else 8,
                color=RED if here else EDGE, zorder=5)
        ox = 0.13 if q[0] == 1 else -0.13
        oy = 0.13 if q[1] == 1 else -0.13
        ax.text(px + ox, py + oy, f"$c' = ({q[0]},{q[1]})$",
                ha="left" if q[0] == 1 else "right",
                va="bottom" if q[1] == 1 else "top",
                fontsize=11, color="#404040", bbox=white, zorder=8)
    ax.plot([a], [b], marker="s", markersize=16, markerfacecolor="none",
            markeredgecolor=EDGE, mew=1.7, zorder=7)      # the index node

    ax.annotate("", xy=pos(cp), xytext=pos(c), zorder=6,
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=RED,
                                shrinkA=11, shrinkB=11))
    mx, my = (pos(c)[0] + pos(cp)[0]) / 2, (pos(c)[1] + pos(cp)[1]) / 2
    ax.text(mx - 0.14, my + 0.14, f"$\\delta = ({delta[0]},{delta[1]})$",
            ha="right", va="bottom", fontsize=12.5, color=RED,
            bbox=white, zorder=9)
    ax.text(pos(c)[0] + 0.14, pos(c)[1] - 0.34,
            f"$i$, at $c = ({c[0]},{c[1]})$", ha="left", va="top",
            fontsize=12.5, color=RED, fontweight="bold", bbox=white, zorder=9)
    ax.text(a + 0.5, b + 1.30, f"$(a,b) = ({a},{b})$", ha="center",
            va="center", fontsize=12.5, color="#1c1c1c", zorder=3)

    ax.set_xlim(a - 0.88, a + 1.72)
    ax.set_ylim(b - 0.62, b + 1.52)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "element")
    print(f"       (a,b) = {(a, b)}, c = {c}, c' = {tuple(cp)}, "
          f"delta = {delta};  k^e[c, c'] entry, operator S^-delta")


# ==========================================================================
#  Section 7: the circuit
# ==========================================================================
def fig_increment(m: int = 4) -> None:
    """The one primitive whose cost grows with the mesh."""
    qc = increment_circuit(m)
    save(qc.draw("mpl", style=MPL, fold=-1), "increment")
    print(f"       m = {m}: {qc.count_ops().get('ccx', 0)} Toffoli, "
          f"{qc.count_ops().get('cx', 0)} CNOT, {max(m - 1, 0)} clean ancillas")


def _fold_for(qc, rows: int) -> int:
    """Fewest gate columns per row that draw `qc` in `rows` rows.

    Balanced rows make the figure as narrow as the row count allows, so it
    shrinks least at \\textwidth. Rows are counted from the wire labels,
    which the drawer repeats once per row.
    """
    def n_rows(fold):
        fig = qc.draw("mpl", style=MPL, fold=fold, scale=0.85)
        n = sum(t.get_ha() == "right" for t in fig.axes[0].texts)
        plt.close(fig)
        return n // qc.num_qubits
    lo, hi = 2, 4 * len(qc.data)          # hi draws in one row
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if n_rows(mid) <= rows else (mid + 1, hi)
    return lo


def _one_circuit(stem, op, label, labels=None, tail=None) -> None:
    qc, info = blockencode(op, m=M_FIG, materialize=True)
    v = info.verification
    drawn = (label_select_stages(qc, labels, tail, BETWEEN)
             if labels and LABEL_STAGES else qc)
    fig = drawn.draw("mpl", style=MPL, fold=_fold_for(drawn, ROWS[stem]),
                     scale=0.85)
    labels_for_print(fig)
    save(fig, stem)
    print(f"       {label:<26} {info.qubits} qubits, L = {info.L}, "
          f"alpha = {info.alpha:.4f}")
    print(f"       {'':<26} verified: block {v['block_err_circuit']:.1e}, "
          f"transpiled {v['block_err_transpiled']:.1e}, "
          f"terms vs assembly {v['terms_vs_reference']:.1e}")
    if not v["ok"]:
        raise SystemExit(f"VERIFICATION FAILED for {stem}: {v}")


def fig_circuits() -> None:
    """The four encodings at m = 2, each verified before it is drawn."""
    for stem, op, label, labels, tail in CELLS:
        _one_circuit(stem, op, label, labels, tail)


# ==========================================================================
#  Section 8: resources
# ==========================================================================
def _affine(ms, ys):
    """(slope, intercept, first m) of the exact affine law the tail obeys.

    Fitted on two consecutive points and then required to hold exactly at
    every larger m, so a reported law is a verified one. The two-phase cells
    depart at m = 2, where the oracle overlaps the incrementer, and the fit
    starts at m = 3 there; returns None if no tail is affine.
    """
    for k in range(len(ms) - 2):
        a = (ys[k + 1] - ys[k]) / (ms[k + 1] - ms[k])
        b = ys[k] - a * ms[k]
        if all(abs(a * m + b - y) < 1e-9 for m, y in zip(ms[k:], ys[k:])):
            return a, b, ms[k]
    return None


def _law(fit, unit: str) -> str:
    if fit is None:
        return f"{unit}: no affine law"
    a, b, m0 = fit
    return f"{unit} = {a:.0f}m{b:+.0f} from m = {m0}"


def fig_scaling(ms=range(2, 13)) -> None:
    """Two-qubit gate counts against m, measured not fitted."""
    ms = list(ms)
    tof, cx = {}, {}
    for stem, op, label, _lb, _tl in CELLS:
        tof[label], cx[label] = [], []
        for m in ms:
            _, info = blockencode(op, m=m)
            tof[label].append(info.toffoli)
            cx[label].append(info.cx)
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    marks = ["o", "s", "^", "D"]
    cols = ["#3b6ea5", "#b03030", "#3f7a45", "#6b4c9a"]
    assert set(LEGEND) == set(cx), "LEGEND and CELLS labels disagree"
    for label, mk, c in zip(LEGEND, marks, cols):
        ax.plot(ms, cx[label], marker=mk, ms=4.5, lw=1.4, color=c, label=label)
    ax.set_xlabel("$m$, qubits per direction")
    ax.set_ylabel("two-qubit gates")
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "scaling")
    for label in tof:
        print(f"       {label:<26} "
              f"{_law(_affine(ms, tof[label]), 'Toffoli')}, "
              f"{_law(_affine(ms, cx[label]), 'CX')}")


# ==========================================================================
FIGURES = {
    #"inclusion": (fig_inclusion, "Sec. 4, the two-phase cell"),
    "mapping":   (fig_mapping,   "Sec. 4, node-to-element incidence"),
    #"element":   (fig_element,   "Sec. 5, local corners and delta"),
    "increment": (fig_increment, "Sec. 7, the shift primitive"),
    "circuits":  (fig_circuits,  "Sec. 7, the four encodings at m = 2"),
    "scaling":   (fig_scaling,   "Sec. 8, counts against m"),
}


def main(argv) -> None:
    if "--list" in argv:
        for k, (_, d) in FIGURES.items():
            print(f"  {k:<12} {d}")
        return
    wanted = [a for a in argv if not a.startswith("-")] or list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; "
                         f"choose from {sorted(FIGURES)}")
    for name in wanted:
        fn, desc = FIGURES[name]
        print(f"{name}  ({desc})")
        fn()
    print("\ndone")


if __name__ == "__main__":
    main(sys.argv[1:])