"""Node-to-element mapping figure: the 2 x 2 element patch around node (8,3).

Each element is identified with its own lower-left node, so the element index
is that node's index. The four elements touching node i are then i + (a,b) for
(a,b) in E = {0,-1}^2.

Typography: element indices are set in math italic serif, node indices in
upright sans, so the two kinds of pair are distinguishable at a glance.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams["font.family"] = "DejaVu Sans"      # nodes, legend
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"  # element indices

IX, IY = 8, 3                                  # the node under discussion

ELEMS = {                                      # (a, b) -> face color
    (0, 0):   "#e2dbf0",     # up-right
    (-1, 0):  "#f9e6d2",     # up-left
    (0, -1):  "#dbead6",     # down-right
    (-1, -1): "#d7e5f2",     # down-left
}

EDGE, RED = "#3c3c3c", "#b03030"
WHITE = dict(facecolor="white", edgecolor="none", alpha=0.85,
             boxstyle="round,pad=0.06")

fig, ax = plt.subplots(figsize=(6.4, 6.4))

# ---- elements -------------------------------------------------------------
for (a, b), col in ELEMS.items():
    ex, ey = IX + a, IY + b                    # element index = its LL node
    ax.add_patch(Rectangle((ex, ey), 1, 1, facecolor=col,
                           edgecolor=EDGE, linewidth=1.7, zorder=1))
    ax.text(ex + .5, ey + .5, f"$e = ({ex},{ey})$", ha="center", va="center",
            fontsize=13, color="#1c1c1c", zorder=3)

# ---- nodes ----------------------------------------------------------------
for nx in (IX - 1, IX, IX + 1):
    for ny in (IY - 1, IY, IY + 1):
        mid = (nx, ny) == (IX, IY)
        ax.plot([nx], [ny], "o", markersize=12 if mid else 7,
                color=RED if mid else EDGE, zorder=5)

# the four element indices are themselves nodes: mark them
for (a, b) in ELEMS:
    ax.plot([IX + a], [IY + b], marker="s", markersize=15,
            markerfacecolor="none", markeredgecolor=EDGE,
            markeredgewidth=1.6, zorder=7)

# ---- node labels, set close to their nodes --------------------------------
D = 0.09
for nx in (IX - 1, IX, IX + 1):
    for ny in (IY - 1, IY, IY + 1):
        sx, sy = nx - IX, ny - IY
        mid = sx == 0 and sy == 0
        if mid:
            dx, dy, ha, va = 0.0, D + .04, "center", "bottom"
        elif sx and sy:
            dx, dy, ha, va = D * sx, D * sy, \
                ("left" if sx > 0 else "right"), ("bottom" if sy > 0 else "top")
        elif sx:
            dx, dy, ha, va = D * sx, 0.0, \
                ("left" if sx > 0 else "right"), "center"
        else:
            dx, dy, ha, va = 0.0, D * sy, "center", \
                ("bottom" if sy > 0 else "top")
        ax.text(nx + dx, ny + dy, f"({nx}, {ny})", ha=ha, va=va,
                fontsize=13 if mid else 13,
                color=RED if mid else "#404040",
                fontweight="bold" if mid else "normal",
                bbox=WHITE, zorder=8)

ax.set_xlim(IX - 1.32, IX + 1.32)
ax.set_ylim(IY - 1.28, IY + 1.28)
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figs/fig_mapping.{ext}", dpi=200, bbox_inches="tight")

print(f"node i = ({IX},{IY})")
for (a, b) in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
    print(f"  (a,b) = {str((a,b)):>9}   e = i+(a,b) = {(IX + a, IY + b)}")