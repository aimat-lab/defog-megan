"""Contact sheet of candidate pairs for the banner: the same noise denoised with the logP adapter alone (left) and
with the concept adapter and the logP adapter together (right), drawn exactly as the banner draws molecules, with the
explainer's explanation as the glow.

Reads ``candidates.json`` and ``states.npz`` from the folder in $BANNER_WORK (written by record_trajectories.py
record). Build:  BANNER_WORK=... python3 ~/.claude/skills/figure/scripts/build.py candidates.py
"""
import json
import os

import numpy as np
from figlib import *  # noqa: F401,F403
from molecules import fit_layout, molecule_layout, draw_state

NAME = 'candidates'
COLS, ROWS = 2, 6
CELL_W, CELL_H = 228, 62
SIZE = (468, 18 + ROWS * CELL_H + 4)

MOL_W, MOL_H, BOND_MAX = 88, 40, 12.0
X_PROP, X_ARROW0, X_ARROW1, X_JOINT = 56, 106, 126, 176     # inside a cell
GLOW_REQUESTED, GLOW_OTHER = PURPLE, PINK


def final_state(states, kind, run):
    X = states[f'{kind}_X'][run][-1]
    E = states[f'{kind}_E'][run][-1]
    bonds = [(i, j, int(E[i, j])) for i in range(len(X)) for j in range(i + 1, len(X)) if E[i, j]]
    return X, bonds


def build(fig):
    work = os.environ['BANNER_WORK']
    with open(os.path.join(work, 'candidates.json')) as fh:
        data = json.load(fh)
    states = np.load(os.path.join(work, 'states.npz'))
    meta = data['meta']
    atom_types = meta['atom_types']
    fig.text(6, 9, 'Same noise: logP adapter alone (left) and with the concept %s (right, logP %.1f). '
                   'Glow = the explainer\'s explanation. Ranked automatically; pick by number.'
             % (meta['concept'], meta['level']), size=7, color=GREY)
    start = int(os.environ.get('BANNER_FIRST', '0'))
    for k, cand in enumerate(data['candidates'][start:start + COLS * ROWS]):
        rank = start + k
        col, row = k % COLS, k // COLS
        x0, y0 = 6 + col * (CELL_W + 4), 18 + row * CELL_H
        cy = y0 + 25
        fig.text(x0, y0 + 8, f'#{rank}', size=7.5, font='rcm')
        for kind, cx, color in (('property', x0 + X_PROP, GLOW_OTHER), ('joint', x0 + X_JOINT, GLOW_REQUESTED)):
            X, bonds = final_state(states, kind, cand['run'])
            unit = molecule_layout(X, bonds, atom_types)
            pos, _ = fit_layout(unit, MOL_W, MOL_H, BOND_MAX, cx, cy)
            draw_state(fig, pos, X, bonds, atom_types, glow=0.8, glow_atoms=cand[kind]['atoms'], glow_color=color,
                       name=f'{kind} {rank}')
            fig.text(cx, y0 + 57, f'logP = {cand[kind]["logp"]:.1f}  {cand[kind]["concept"]}', size=6.5, color=GREY,
                     justify='center')
        fig.arrow([(x0 + X_ARROW0, cy), (x0 + X_ARROW1, cy)], sw=0.9, head=3.0, name='adapters')
