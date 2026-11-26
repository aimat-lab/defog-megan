"""Ball-and-stick molecules in the house style, shared by the banner and its contact sheet (after the one of defog-adapt).

A discrete state of the sampler is a class per node (an element) and a class per node pair (no bond, single,
double, triple). ``draw_state`` draws exactly that: nodes filled by element in muted colours (pale enough for
black letters), the element letter inside hetero atoms, and bond order as 1, 2 or 3 parallel strokes. The same
function draws a half-denoised state, so every flip in the animation is a flip the model made.

Node POSITIONS carry no meaning in the model (a graph has no coordinates); they are a drawing choice. The final
molecule uses an RDKit 2D layout, and the noise uses a scatter, which the banner interpolates between.
"""
import itertools
import math

import numpy as np
from figlib import BLACK, ORANGE, WHITE, shadow, tint

#: muted element colours, pale enough for black letters. Carbon is a neutral grey and carries no letter.
ELEMENT_FILL = {
    'C': '#a9a9a9', 'N': '#8fb1f1', 'O': '#f19a93', 'S': '#efd96c', 'P': '#f4b27c',
    'F': '#a3dca3', 'Cl': '#7dd08e', 'Br': '#d3a276', 'I': '#c4a1e1',
}
BOND_CLASSES = ('none', 'single', 'double', 'triple')


# ------------------------------------------------------------------------------ layouts
def molecule_layout(X, bonds, atom_types):
    """2D coordinates for a final state, in units of one mean bond length, centred on the origin and rotated so
    that the wider side is horizontal. Returns [(x, y)] per node (page orientation: y down)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rw = Chem.RWMol()
    for c in X:
        rw.AddAtom(Chem.Atom(atom_types[int(c)]))
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for i, j, c in bonds:
        rw.AddBond(int(i), int(j), order[int(c)])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    mean_bond = np.mean([np.linalg.norm(pts[i] - pts[j]) for i, j, _ in bonds])
    pts = (pts - pts.mean(0)) / mean_bond
    w, v = np.linalg.eigh(np.cov(pts.T))                       # principal axes: the long one horizontal
    pts = pts @ v[:, ::-1]
    pts[:, 1] *= -1 if np.corrcoef(pts[:, 0], pts[:, 1])[0, 1] < 0 else 1   # keep a stable handedness
    return [(float(x), float(y)) for x, y in pts]


def fit_layout(unit_pts, box_w, box_h, bond_max, cx=0.0, cy=0.0):
    """Scale a unit-bond-length layout to fit a box (centred on (cx, cy)); returns points in pt and the bond
    length used."""
    xs, ys = [p[0] for p in unit_pts], [p[1] for p in unit_pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    b = min(bond_max, box_w / max(w, 1e-6), box_h / max(h, 1e-6))
    mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [(cx + (x - mx) * b, cy + (y - my) * b) for x, y in unit_pts], b


def scatter_layout(n, w, h, seed, min_dist, cx=0.0, cy=0.0):
    """A deterministic scatter of n points in an ellipse of half-axes (w/2, h/2), no two closer than min_dist: the
    layout of pure noise (a graph with no structure yet)."""
    rng = np.random.default_rng(seed)
    pts = []
    tries = 0
    while len(pts) < n:
        tries += 1
        x, y = rng.uniform(-1, 1, 2)
        if x * x + y * y > 1:
            continue
        p = (cx + x * w / 2, cy + y * h / 2)
        if tries > 5000 or all(math.dist(p, q) >= min_dist for q in pts):
            pts.append(p)
    return pts


def assign_slots(slots, targets):
    """Which scatter slot each node sits in. The scatter of the noise is arbitrary, so choose the assignment that
    moves the nodes least on their way to ALL the final layouts in ``targets`` (a list of per-node point lists,
    already in the slots' frame): a linear assignment on the summed squared distances. Straight-line glides
    between unrelated layouts contract into a clump; between matched ones they stay spread out."""
    from scipy.optimize import linear_sum_assignment

    n = len(slots)
    cost = np.zeros((n, n))
    for k in range(n):
        for s_i in range(n):
            cost[k, s_i] = sum((slots[s_i][0] - t[k][0]) ** 2 + (slots[s_i][1] - t[k][1]) ** 2 for t in targets)
    _, cols = linear_sum_assignment(cost)
    return [slots[c] for c in cols]


def keep_apart(pts, dmin, iters=16):
    """Push apart nodes that are closer than dmin (a few passes; deterministic), so that atoms never sit on top of
    each other mid-glide. Leaves a layout with all distances >= dmin untouched."""
    p = [[x, y] for x, y in pts]
    for _ in range(iters):
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                dx, dy = p[j][0] - p[i][0], p[j][1] - p[i][1]
                d = math.hypot(dx, dy)
                if d >= dmin:
                    continue
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0
                push = (dmin - d) / 2 / d
                p[i][0] -= dx * push
                p[i][1] -= dy * push
                p[j][0] += dx * push
                p[j][1] += dy * push
    return [(x, y) for x, y in p]


def lerp_points(a, b, k):
    return [(ax + (bx - ax) * k, ay + (by - ay) * k) for (ax, ay), (bx, by) in zip(a, b)]


# ------------------------------------------------------------------------------ drawing
def pct(x):
    """An opacity in percent, kept off figlib's ambiguous value 1 (read as 100 %) and off 0."""
    return max(2.0, min(100.0, float(x)))


def _offsets(c):
    return {1: (0.0,), 2: (-1.25, 1.25), 3: (-2.0, 0.0, 2.0)}[c]


def draw_state(fig, pos, X, bonds, atom_types, r=3.3, r_hetero=4.1, glow=0.0, fade=1.0, letters=True,
               name='molecule', bond_sw=1.15, ink=BLACK, shadow_scale=0.5, glow_atoms=None, glow_color=ORANGE, glow_r=None):
    """Draw one discrete state.

    pos      [(x, y)] per node, in pt
    r        radius of a carbon node; r_hetero the (larger) radius of every other element, so that the letter
             fits inside it while the carbon-rich skeleton keeps visible bonds
    X        class (element) per node
    bonds    [(i, j, class)] with class 1, 2 or 3
    glow     0..1: strength of the halo behind the explained atoms
    glow_atoms  the atoms the halo marks (the explanation of a concept); None marks every atom
    glow_color  the colour of the halo (the colour of the concept)
    fade     0..1: opacity of the whole drawing, for cross-fades
    """
    if fade <= 0.01:
        return
    op = None if fade >= 0.995 else pct(100 * fade)
    if glow * fade > 0.03:                                      # below this figlib would misread the opacity
        marked = set(range(len(pos))) if glow_atoms is None else {int(a) for a in glow_atoms}
        segs = [((pos[i][0], pos[i][1]), (pos[j][0], pos[j][1])) for i, j, _ in bonds if i in marked and j in marked]
        fig.glow([pos[a] for a in sorted(marked)], glow_color, glow_r or r, bonds=segs, strength=glow * fade, name=f'{name} glow')
    for i, j, c in bonds:
        (x0, y0), (x1, y1) = pos[i], pos[j]
        d = math.hypot(x1 - x0, y1 - y0) or 1.0
        nx, ny = -(y1 - y0) / d, (x1 - x0) / d
        sw = bond_sw if c == 1 else bond_sw * 0.85
        for o in _offsets(c):
            fig.line(x0 + nx * o, y0 + ny * o, x1 + nx * o, y1 + ny * o, stroke=ink, sw=sw, opacity=op,
                     name=f'{name} bond')
    for k, (x, y) in enumerate(pos):
        sym = atom_types[int(X[k])]
        rr = r if sym == 'C' else r_hetero
        fig.ellipse(x, y, rr, fill=ELEMENT_FILL[sym], stroke=ink, sw=0.85, opacity=op,
                    effects=[shadow(shadow_scale * rr / 3.6, opacity=30 * fade)], name=f'{name} atom')
        if letters and sym != 'C':
            size = (5.6 if len(sym) == 1 else 4.9) * r_hetero / 4.1
            fig.text(x, y + 0.36 * size, sym, size=size, font='rcm', color=tint(ink, 1 - fade) if fade < 0.995 else ink,
                     justify='center')
