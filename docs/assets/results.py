"""The README's results figure: how often a generated graph has the requested property AND the requested concept.

    python docs/assets/results.py            # reads results.json, writes results.png and results_dark.png

``results.json`` holds the numbers of the paper's two joint experiments (Tables 3 and 4: synthetic colored graphs and
molecules steered towards Crippen logP), each the mean joint match over 30 target cells of 1024 graphs, with their
protocol and where they come from; this script only draws them, so the chart cannot drift from the data it shows. What
they do and do not support is stated in the README next to the chart.

Form (after the chart of defog-adapt): one arm is the point, both adapters with Feynman-Kac steering, the end of the
progression, so it is drawn in the accent and the rest in the de-emphasis gray. The two experiments are two groups of
six bars side by side, which share one column of arm labels and one scale, with a hairline between them, and every bar is
labelled with its value, so there is no legend and no axis. The protocol is in the footer, so the figure does not claim
more than it shows when it is seen on its own.

Colours and ink are the dataviz reference palette on its light surface and, for readers on a dark page, its dark
surface (``results_dark.png``, chosen by the README's <picture> element).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

HEADLINE = "joint_fk"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "accent": "#2a78d6", "context": "#908f87",
              "hairline": "#e1e0d9", "baseline": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "accent": "#3987e5", "context": "#6c6b65",
             "hairline": "#2c2c2a", "baseline": "#383835"},
}
FONT = "Roboto Condensed"

W = 760                                              # canvas width in "1x" pixels; the file is written at 2x
LEFT = 24
LABEL_X = 206                                        # the arm labels end here: one column for both groups
COL_X = (220, 478)                                   # where each group's bars start (its baseline), left to right
MAX_BAR = 200                                        # the longest bar, in px
VALUE_GAP, VALUE_W = 10, 34                          # a bar's value label: the gap after the bar and its width
PITCH, THICK, RADIUS = 33, 18, 4
HEADER_TO_ROW = 26                                   # the groups' header baseline to the centre of the first row


def layout(n_rows):
    """Vertical positions in "1x" pixels: the groups' header baseline, every row's centre (the rows are shared by
    the groups), the footer's baseline and the canvas height."""
    header = 88
    centres = [header + HEADER_TO_ROW + PITCH * i for i in range(n_rows)]
    footer = centres[-1] + THICK / 2 + 34
    return header, centres, footer, int(footer + 38)


def rounded_bar(ax, x0, y_mid, length, color):
    """A bar that grows from a square baseline end to a 4 px rounded data end."""
    y0, y1, x1, r = y_mid - THICK / 2, y_mid + THICK / 2, x0 + length, RADIUS
    r = min(r, max(length, 0.01) / 2)
    verts = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r), (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", lw=0))


def main(data_path: Path, out: Path, theme: str = "light") -> None:
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}; choose one of {sorted(THEMES)}")
    c = THEMES[theme]
    data = json.loads(data_path.read_text(encoding="utf-8"))
    groups = data["groups"]
    arms = [row["arm"] for row in groups[0]["rows"]]
    n_rows = len(arms)
    if len(groups) != len(COL_X) or any([row["arm"] for row in g["rows"]] != arms for g in groups):
        raise SystemExit(f"the chart draws {len(COL_X)} groups with the same arms in the same order")
    proto = data["protocol"]
    top = max(row["value"] for g in groups for row in g["rows"])
    scale = MAX_BAR / top
    header, centres, footer, H = layout(n_rows)

    plt.rcParams["font.family"] = FONT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200, facecolor=c["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    px = 72 / 100                                    # one "1x" pixel in points
    ax.text(LEFT, 34, "Graphs with the requested property and the requested concept (higher is better)", fontsize=16.5 * px,
            fontweight="medium", color=c["ink"], va="baseline")
    ax.text(LEFT, 56, "Joint match: the share of generated graphs that reach the requested property and are a member of "
                      "the requested concept.", fontsize=12.5 * px, color=c["ink2"], va="baseline")
    for row, y in zip(groups[0]["rows"], centres):                  # the arm labels, once, for both groups
        headline = row["arm"] == HEADLINE
        ax.text(LABEL_X, y + 4.6, row["label"], fontsize=13.5 * px, ha="right", va="baseline",
                color=c["ink"] if headline else c["ink2"], fontweight="medium" if headline else "regular")

    for x0, group in zip(COL_X, groups):
        ax.text(x0, header, group["name"], fontsize=12 * px, fontweight="medium", color=c["ink2"], va="baseline")
        ax.plot([x0, x0], [centres[0] - THICK / 2 - 6, centres[-1] + THICK / 2 + 6], color=c["baseline"], lw=0.8,
                solid_capstyle="butt")                                # the one baseline this group's bars grow from
        for row, y in zip(group["rows"], centres):
            headline = row["arm"] == HEADLINE
            length = row["value"] * scale
            rounded_bar(ax, x0, y, length, c["accent"] if headline else c["context"])
            ax.text(x0 + length + VALUE_GAP, y + 4.8, f"{row['value']:.3f}", fontsize=13.5 * px, ha="left",
                    va="baseline", color=c["ink"], fontweight="medium" if headline else "regular")

    widest = COL_X[0] + max(row["value"] for row in groups[0]["rows"]) * scale + VALUE_GAP + VALUE_W
    divider = (widest + COL_X[1]) / 2                                 # a hairline between the two groups
    ax.plot([divider, divider], [header - 14, centres[-1] + THICK / 2 + 6], color=c["hairline"], lw=0.8,
            solid_capstyle="butt")

    tol = proto["property_tolerance"]
    ax.text(LEFT, footer - 8, f"Means over {proto['cells']} target cells of {proto['graphs_per_cell']} graphs each. Joint match needs the property "
                          f"within {tol['colored graphs']:g} (colored graphs) or {tol['molecules']:g} logP (molecules)",
            fontsize=11.5 * px, color=c["ink2"], va="baseline")
    ax.text(LEFT, footer + 10, f"and core membership in the requested concept. Feynman-Kac steering uses {proto['particles']} particles. "
                           "Numbers of the paper's Tables 3 and 4.",
            fontsize=11.5 * px, color=c["ink2"], va="baseline")

    fig.savefig(out, dpi=200, facecolor=c["surface"])
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    here = Path(__file__).parent
    data_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results.json"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else here
    main(data_arg, out_dir / "results.png", "light")
    main(data_arg, out_dir / "results_dark.png", "dark")
