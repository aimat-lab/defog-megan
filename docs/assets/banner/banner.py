"""README banner, one frame: what a concept is (above) and a REAL recorded run of the frozen DeFoG base (below).

Above, the concepts. MEGAN, a self-explaining graph network, explains molecules: the glow marks the substructure that
drives the property it predicts. Explanations that look alike gather into one cluster, a concept, which has a colour and
a letter (A, B, C). Three small molecules per concept are drawn for the banner to show this. They are illustrative, not
data.

Below, the generator, from a recording of the sampler (trajectory.json, see record_trajectories.py). The animation runs
twice from the SAME noise. Run 1: the frozen DeFoG base with a logP adapter docked on it turns the noise into a
molecule at the requested logP, and MEGAN reads the molecule back: its explanation lands in a concept, B as it happens.
Run 2: a concept adapter docks next to the logP adapter and takes the request phi*, the membership vector of the concept
that is asked for (A, one filled cell, a line from the cluster); the same noise now ends in a molecule at the same logP
whose explanation lands in A. Every atom and bond flip is a flip the model made. Only the node POSITIONS are a drawing
choice, because a graph has no coordinates: the noise is scattered and the finished molecule uses an RDKit 2D layout,
and the drawing glides from one to the other while the categories flip.

While a run goes on, small pulses travel from the noise, disappear behind the model and come out of it along the arrow
to the sample; one pulse stands for one application of the model, and where an adapter is docked each pulse passes
under it and the adapter's ring fires. A faint time axis under the stations runs from the noise (t = 0) to the sample
(t = 1); its knob carries the flow time the model was given at the step on screen, and it rewinds when the noise comes
back.

This script draws ONE frame, chosen by $BANNER_FRAME; render_banner.py renders them all and assembles the GIF.

Build one frame:  BANNER_FRAME=120 python3 ~/.claude/skills/figure/scripts/build.py banner.py --scale 4
"""
import bisect
import json
import math
import os
from types import SimpleNamespace

import numpy as np
from figlib import *  # noqa: F401,F403
from figlib import _svg_file
from molecules import (assign_slots, draw_state, fit_layout, keep_apart, lerp_points, molecule_layout, pct,
                       scatter_layout)
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

NAME = 'banner'
SIZE = (460, 208)                  # about 2.2:1; the animation is the whole banner; type sizes as in the thesis figures

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('BANNER_DATA', os.path.join(HERE, 'trajectory.json'))
FRAME = int(os.environ.get('BANNER_FRAME', '120'))

# ------------------------------------------------------------------------------ colours
CONCEPT = {'A': ORANGE, 'B': PINK, 'C': PURPLE}        # categories without a good or bad reading: the paper's colours
LANES = ('A', 'B', 'C')                                # the three concepts, one lane each, top to bottom
FROST, FROST_SW, FROST_GAP = '#b9dcff', 2.0, 5.0       # light blue stripes: frozen weights (as in the visual abstract)
ICE_D = '#2275c9'                  # the frozen model's blue, deeper than the pale stripes so that a snowflake can be seen
ADAPTER_FILL = tint(ORANGE, 0.80)                      # a trained adapter, as in the paper's figures
PULSE = '#3d3d3d'                                      # a pulse of the model working (orange once an adapter acted)
AX_TRACK, AX_TICK_OFF, AX_PASSED = '#ececec', '#dedede', '#a3a3a3'     # the faint time axis
AX_KNOB, AX_TEXT = '#8c8c8c', '#808080'
ATOMS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
LETTER = {'property': 'B', 'joint': 'A'}               # the concept each recorded run lands in

# ------------------------------------------------------------------------------ layout (pt): the upper part
ROW = (26, 49, 72)                 # centre lines of the three lanes
X_IN, X_OUT = 38, 162              # centres of the molecules that go into MEGAN and of their explanations
MOL_W, MOL_H = 52, 22
X_MEGAN, MEGAN_W, MEGAN_H = 88, 36, 64
CARD_X, CARD_Y, CARD_W, CARD_H = 212, 9, 124, 80
RING_GAP = 3.5                     # a lit cluster gets a ring this far outside its glow
# the clusters on a staircase, one per lane, so that no line crosses another: A (the requested concept) sits straight
# above the concept adapter. The dots are offsets from the centre: one per wave of molecules, and one for the sample
CLUSTERS = {
    'A': dict(x=0.0, y=33, r=10.5, wave=((-5, -2), (-1, 5), (1, -5)), new=(6, 1), token=(-1, -1)),
    'B': dict(x=0.0, y=51, r=10.0, wave=((-5, 1), (1, -5), (0, 5)), new=(6, 0), token=(-1, -1)),
    'C': dict(x=0.0, y=71, r=10.0, wave=((-4, -3), (-4, 4), (2, -4)), new=(6, 2), token=(1, -1)),
}

# ------------------------------------------------------------------------------ layout (pt): the lower part
MID = 151                          # centre line of the stations
X_NOISE, NOISE_W, NOISE_H = 44, 50, 46           # noise graph: centre and extent of its scatter
X_MODEL, MW, MH, TAPER = 186, 64, 42, 0.38       # frozen DeFoG: left edge, width, height
X_SAMPLE, SAMPLE_W, SAMPLE_H, BOND_MAX = 392, 72, 38, 11.0
X_ARROW = 322                      # where the arrow from the model ends
LABEL = MID + 33                   # baseline of the station labels
AXIS_Y, TICKS = MID + 43, 10       # the time axis under the stations, and its ticks (every 0.1)
T_BASE = MID + 52.5                # baseline of the t counter that rides under the knob
AP_X = X_MODEL + 0.20 * MW         # where the logP adapter docks, along the model's top edge
AC_X = X_MODEL + 0.80 * MW         # where the concept adapter docks
ADAPTER_W, ADAPTER_H, DOCK_GAP = 34, 22, 5
FLOW_X0, FLOW_X1 = X_NOISE + NOISE_W / 2 + 4, X_ARROW    # the path of the pulses: line into the model, model, arrow
FLOW_TAU = (0.0, 0.30, 0.55, 1.0)                        # time fractions at which a pulse reaches the model, leaves it
for _i, _L in enumerate(LANES):
    CLUSTERS[_L]['x'] = AC_X + 34.0 * _i

# ------------------------------------------------------------------------------ timeline (15 fps)
SLOW = 1.3                         # pace of the lower part's machinery (pulses, rings, rewinds): the first draft / this


def fr(n):
    """A duration of the first draft (in frames) at the current pace."""
    return round(n * SLOW)


# the explain phase: three waves of molecules, three lanes, each lane a few frames after the one above
WAVES = 3
E_START, WAVE_T, LANE_T = 4, 18, 3
U_IN, U_PULSE, N_PULSE = 5, 3, 6   # a wave's molecule fades in over U_IN frames; its pulse runs into MEGAN
U_EXPL, U_DOT = 10, 15             # the explanation fades in (6 frames); the dot leaves for the concepts
U_OUT = 13                         # the molecule fades out (5 frames) so that the next wave's can come
DOT_SPEED = 10.0                   # pt per frame
SETTLE = 6                         # frames from the last landing to the end of the phase

EXAMPLES = {                       # per concept and wave: a designed molecule and the atoms its explanation marks
    'A': [('CNCc1ccc(Br)cc1', [3, 4, 5, 6, 7, 8, 9]),
          ('OCc1ccc(Br)cc1', [2, 3, 4, 5, 6, 7, 8]),
          ('NCCc1ccc(Br)cc1', [3, 4, 5, 6, 7, 8, 9])],             # the bromophenyl ring, not the chain
    'B': [('COc1ccc(C#N)cc1', [2, 3, 4, 5, 6, 7, 8, 9]),
          ('CCCc1ccc(C#N)cc1', [3, 4, 5, 6, 7, 8, 9, 10]),
          ('CCOc1ccc(C#N)cc1', [3, 4, 5, 6, 7, 8, 9, 10])],        # the cyano-aryl ring, not the chain
    'C': [('NCc1cccs1', [2, 3, 4, 5, 6]),
          ('CCCc1cccs1', [3, 4, 5, 6, 7]),
          ('OCCc1cccs1', [3, 4, 5, 6, 7])],                        # the thiophene ring, not the chain
}


def lane_t0(w, i):
    """Frame at which wave w starts in lane i."""
    return E_START + w * WAVE_T + i * LANE_T


def dot_ends(i, w):
    """Start and end of the explanation dot of wave w in lane i: from the explained molecule to its place in the cluster."""
    c = CLUSTERS[LANES[i]]
    return (X_OUT + MOL_W / 2 + 5.0, float(ROW[i])), (c['x'] + c['wave'][w][0], c['y'] + c['wave'][w][1])


def dot_len(i, w):
    a, b = dot_ends(i, w)
    return math.hypot(b[0] - a[0], b[1] - a[1])


def t_land(i, w):
    """Frame at which the explanation of wave w in lane i reaches its cluster."""
    return lane_t0(w, i) + U_DOT + math.ceil(dot_len(i, w) / DOT_SPEED)


E_END = max(t_land(i, WAVES - 1) for i in range(len(LANES))) + SETTLE

T_DIM, T_DOCK = 10, 14             # the upper part dims while the logP adapter docks
B_DOCK1 = E_END
B_BEAT1 = B_DOCK1 + T_DOCK
N_DEN = fr(40)                                         # frames of visible denoising per run
T_READ = 42                        # a result is shown, read back and held
T_SWAP = 26                        # the result goes, the noise comes, the concept adapter docks
T_RESET = 20
B_HOLD1 = B_BEAT1 + N_DEN
B_SWAP = B_HOLD1 + T_READ
B_BEAT2 = B_SWAP + T_SWAP
B_HOLD2 = B_BEAT2 + N_DEN
B_RESET = B_HOLD2 + T_READ
FRAMES = B_RESET + T_RESET
REWIND = (3, 12)                                       # first-draft frames (after a result is hidden) of the knob's rewind
PULSE_PERIOD, PULSE_TRAVEL, PULSE_LIFE = fr(8), fr(10), fr(8)    # a pulse starts, takes, and an adapter's ring lives


def smooth(k):
    k = min(max(k, 0.0), 1.0)
    return k * k * (3 - 2 * k)


def ease_out(k):
    return 1 - (1 - min(max(k, 0.0), 1.0)) ** 3


def ease_in(k):
    return min(max(k, 0.0), 1.0) ** 3


def mix(c0, c1, k):
    """Colour between two '#rrggbb' colours."""
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * k):02x}' for x, y in zip(a, b))


def pale(color, fade):
    """A colour faded towards white (text and lines cannot take an opacity in figlib)."""
    return mix(color, '#ffffff', 1 - min(max(fade, 0.0), 1.0))


def timeline(f):
    """What is on screen at frame f. The last frame equals the first, so the loop is seamless.

    The explain phase (f < E_END) is carried by the lanes' own clocks (lane_t0, t_land); everything else is set here.
    Then: the logP adapter docks (run 1), the noise is denoised and read back, the result goes while the concept adapter
    docks and takes its request (swap), run 2 does the same, and a reset clears the stage for the first frame."""
    s = SimpleNamespace(run=None, p=0.0, running=False, fade_noise=1.0, fade_result=0.0, caption=0.0, glow=0.0,
                        prop=0.0, conc=0.0,                       # adapters: 0 away, 1 docked
                        explain=1.0,                              # opacity of the upper left part
                        rho=0.0, expl_out=0.0, clusters=1.0,      # reset: first molecules back, explanations gone, clusters gone
                        rb=0.0, rb_alpha=0.0, rb_letter=None, lit=0.0,    # read back: front, opacity, concept, glow
                        ask=0.0, ask_ring=0.0, ask_label=0.0)     # the concept adapter's request
    if f < E_END:
        return s
    if f < B_BEAT1:                                        # the logP adapter docks, the explanations dim
        s.explain = 1 - 0.5 * smooth((f - B_DOCK1) / T_DIM)
        s.prop = ease_out((f - B_DOCK1) / T_DOCK)
        return s
    s.explain, s.prop = 0.5, 1.0
    if f < B_HOLD1:                                        # run 1, denoising
        s.run, s.p, s.running = 'property', (f - B_BEAT1) / (N_DEN - 1), True
    elif f < B_SWAP:                                       # run 1, result, read back
        read_back(s, 'property', f - B_HOLD1)
    elif f < B_BEAT2:                                      # swap: result out, noise in, the concept adapter takes its request
        k = f - B_SWAP
        s.run, s.p = 'property', 1.0
        s.fade_result, s.fade_noise = 1 - smooth(k / 6), smooth((k - 4) / 5)
        s.caption = s.glow = 1 - smooth(k / 6)
        s.rb_letter, s.rb, s.rb_alpha, s.lit = 'B', 1.0, 1 - smooth(k / 6), 1 - smooth(k / 6)
        s.ask, s.ask_ring = smooth((k - 3) / 8), smooth((k - 3) / 6)
        s.conc = ease_out((k - 8) / 12)
        s.ask_label = smooth((k - 17) / 5)
    elif f < B_HOLD2:                                      # run 2, denoising
        s.run, s.p, s.running = 'joint', (f - B_BEAT2) / (N_DEN - 1), True
        s.conc, s.ask, s.ask_ring, s.ask_label = 1.0, 1.0, 1.0, 1.0
    elif f < B_RESET:                                      # run 2, result, read back
        read_back(s, 'joint', f - B_HOLD2)
        s.conc, s.ask, s.ask_ring, s.ask_label = 1.0, 1.0, 1.0, 1.0
    else:                                                  # reset: the stage is cleared for the first frame
        k = f - B_RESET
        s.run, s.p = 'joint', 1.0
        s.fade_result, s.fade_noise = 1 - smooth(k / 5), smooth((k - 3) / 5)
        s.caption = s.glow = 1 - smooth(k / 5)
        s.rb_letter, s.rb, s.rb_alpha, s.lit = 'A', 1.0, 1 - smooth(k / 5), 1 - smooth(k / 5)
        s.prop = s.conc = 1 - ease_in(k / 12)
        s.ask = s.ask_ring = 1 - smooth(k / 8)
        s.ask_label = 1 - smooth(k / 5)
        s.clusters = 1 - smooth((k - 2) / 10)
        s.expl_out = smooth(k / 8)
        s.rho = smooth((k - 6) / 10)
        s.explain = 0.5 + 0.5 * smooth((k - 8) / 10)
    if s.running:
        s.fade_noise, s.fade_result = 0.0, 1.0
    return s


def read_back(s, run, k):
    """The result of a run, k frames after the denoising ended: the molecule and its caption fade in, then MEGAN reads
    it back (a line with a dot runs from the caption into the card), and the concept the explanation lands in lights."""
    s.run, s.p = run, 1.0
    s.fade_noise, s.fade_result = 0.0, 1.0
    s.caption, s.glow = smooth(k / 6), smooth(k / 8)
    s.rb_letter, s.rb_alpha = LETTER[run], smooth((k - 7) / 4)
    s.rb = smooth((k - 8) / 14)
    s.lit = smooth((k - 22) / 6)


# ------------------------------------------------------------------------------ data
def step_clock(run, steps):
    """Which recorded step a run shows at fraction p of its frames. Frames are spent where the model is changing
    things: the pace follows the number of flips per step (smoothed), with a floor so that quiet stretches are
    fast-forwarded rather than frozen. The step counter shows the true time of the step on screen."""
    X, B = run['X'], run['bonds']
    act = np.zeros(steps)
    for s in range(steps):
        a, b = {(i, j): c for i, j, c in B[s]}, {(i, j): c for i, j, c in B[s + 1]}
        act[s] = sum(x != y for x, y in zip(X[s], X[s + 1])) + sum(a.get(q, 0) != b.get(q, 0) for q in set(a) | set(b))
    smooth_act = np.convolve(np.pad(act, 6, mode='edge'), np.ones(13) / 13, mode='valid')
    w = 0.3 + smooth_act / max(smooth_act.max(), 1e-9)
    cum = np.concatenate([[0.0], np.cumsum(w)])
    cum /= cum[-1]
    return lambda p: int(round(float(np.interp(p, cum, np.arange(steps + 1)))))


def load():
    with open(DATA) as fh:
        d = json.load(fh)
    atom_types = d['meta']['atom_types']
    n = d['meta']['n']
    slots = scatter_layout(n, NOISE_W, NOISE_H, seed=7, min_dist=9.5, cx=X_NOISE, cy=MID)
    finals = {}
    for kind, run in d['runs'].items():
        Xf, bf = run['X'][-1], run['bonds'][-1]
        unit = molecule_layout(Xf, bf, atom_types)
        finals[kind], _ = fit_layout(unit, SAMPLE_W, SAMPLE_H, BOND_MAX, X_SAMPLE, MID)
    shift = X_SAMPLE - X_NOISE
    noise_pos = assign_slots(slots, [[(x - shift, y) for x, y in f] for f in finals.values()])
    d['clock'] = {kind: step_clock(run, d['meta']['steps']) for kind, run in d['runs'].items()}
    return d, atom_types, noise_pos, finals


def counter_time(f, tl, d):
    """The flow time t the counter and the knob show. While a run goes on it is the time of the step on screen; while a
    result is shown it is 1; after a result is hidden the knob rewinds to 0 (so that the noise starts again from 0)."""
    if tl.running:
        return d['meta']['flow_t'][d['clock'][tl.run](tl.p)]
    if f < B_BEAT1:
        return 0.0
    if f < B_SWAP or B_HOLD2 <= f < B_RESET:
        return 1.0
    k = (f - (B_SWAP if f < B_BEAT2 else B_RESET)) / SLOW
    return 1.0 - smooth((k - REWIND[0]) / (REWIND[1] - REWIND[0]))


def pulse_starts():
    """First frames of the pulses of a run: one every PULSE_PERIOD, none so late that the end of the run would cut
    it off while it is still going."""
    return range(0, N_DEN - PULSE_TRAVEL, PULSE_PERIOD)


def flow_x(tau):
    """x of a pulse at time fraction tau (0..1) of its run. It lingers where it can be seen (the line into the model,
    the arrow) and zips through the stretch behind the model, whose body hides it."""
    return float(np.interp(tau, FLOW_TAU, (FLOW_X0, X_MODEL, X_MODEL + MW, FLOW_X1)))


def flow_pulses(i):
    """Time fraction (0..1 of its run) of each pulse in flight at frame i of a run."""
    return [(i - s) / PULSE_TRAVEL for s in pulse_starts() if 0 <= i - s <= PULSE_TRAVEL]


def ring_ages(i, dock_x):
    """Ages (frames) of the rings of the adapter docked at dock_x, at frame i of a run: a ring fires when a pulse
    passes under the dock."""
    tau_dock = float(np.interp(dock_x, (FLOW_X0, X_MODEL, X_MODEL + MW, FLOW_X1), FLOW_TAU))
    ages = [i - (s + tau_dock * PULSE_TRAVEL) for s in pulse_starts()]
    return [a for a in ages if 0 <= a < PULSE_LIFE]


# ------------------------------------------------------------------------------ small parts
class Poly:
    """A polyline with its arc length, to move a dot along a path and to draw the part of the path it has passed."""

    def __init__(self, pts):
        self.pts = [(float(x), float(y)) for x, y in pts]
        self.cum = [0.0]
        for a, b in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.length = self.cum[-1]

    def at(self, s):
        s = min(max(s, 0.0), self.length)
        j = min(bisect.bisect_right(self.cum, s) - 1, len(self.pts) - 2)
        seg = (self.cum[j + 1] - self.cum[j]) or 1.0
        k = (s - self.cum[j]) / seg
        (x0, y0), (x1, y1) = self.pts[j], self.pts[j + 1]
        return x0 + k * (x1 - x0), y0 + k * (y1 - y0)

    def upto(self, s):
        """The corner points up to arc length s, and the point at s."""
        s = min(max(s, 0.0), self.length)
        j = bisect.bisect_right(self.cum, s) - 1
        return self.pts[:j + 1] + [self.at(s)]


def comet(fig, x, y, ux, uy, color, vis=1.0, name='pulse'):
    """A dot with a short fading trail behind it, moving along the unit vector (ux, uy)."""
    for dist, r, op in ((0, 2.2, 100), (4.2, 1.7, 60), (8.4, 1.2, 32), (12.6, 0.8, 16)):
        fig.ellipse(x - dist * ux, y - dist * uy, r, fill=color, stroke='none', opacity=pct(op * vis), name=name)


def token(fig, cx, cy, r, letter, color, k=1.0, name='concept'):
    """A concept token: a pale disc in the concept's colour with a ring and the letter in black. k fades it in."""
    if k < 0.03:
        return
    fig.ellipse(cx, cy, r, fill=pale(tint(color, 0.50), k), stroke=pale(color, k), sw=0.9,
                effects=[shadow(0.35, opacity=25 * k)], name=f'{name} token')
    size = 1.5 * r
    fig.text(cx, cy + 0.36 * size, letter, size=size, font='rcm', color=tint(BLACK, 1 - k), justify='center')


def snowflake(fig, cx, cy, r, color, sw, name='snowflake'):
    """A six-armed snowflake with a V branch on each arm, drawn as round-capped lines."""
    for k in range(6):
        a = math.radians(90 + 60 * k)
        ux, uy = math.cos(a), -math.sin(a)
        fig.line(cx, cy, cx + r * ux, cy + r * uy, stroke=color, sw=sw, cap='round', name=name)
        bx, by = cx + 0.55 * r * ux, cy + 0.55 * r * uy
        for s in (-1, 1):
            b = a + s * math.radians(52)
            fig.line(bx, by, bx + 0.36 * r * math.cos(b), by - 0.36 * r * math.sin(b), stroke=color, sw=sw,
                     cap='round', name=name)


def frozen_model(fig, x, y, w, h, label, taper=TAPER, size=7.5):
    """Frozen model: white trapezoid (narrow side right) with broad light blue diagonal stripes, and a badge with a
    snowflake on its lower right corner (the pale stripes alone would hide a pale snowflake)."""
    t = h * taper / 2
    pts = [(x, y), (x + w, y + t), (x + w, y + h - t), (x, y + h)]
    fig.path(pts, closed=True, fill=WHITE, stroke='none', effects=[shadow()], name='frozen body')
    poly = ' '.join(f'{px:.2f},{py:.2f}' for px, py in pts)
    lines = ''.join(f'<line x1="{x + k:.2f}" y1="{y + h:.2f}" x2="{x + k + h:.2f}" y2="{y:.2f}" '
                    f'stroke="{FROST}" stroke-width="{FROST_SW}"/>'
                    for k in [i * FROST_GAP - h for i in range(int((w + h) / FROST_GAP) + 2)])
    content = f'<defs><clipPath id="fc"><polygon points="{poly}"/></clipPath></defs><g clip-path="url(#fc)">{lines}</g>'
    fig.place(_svg_file('frost', x, y, w, h, content), x, y, w, h, name='frozen stripes')
    fig.path(pts, closed=True, fill='none', stroke=BLACK, sw=STYLE['outline'], name='frozen outline')
    fig.text(x + w / 2 - 2, y + h / 2 + size * 0.36, label, size=size, font='rcm', justify='center')
    bx, by = x + w - 6, y + h - t - 1                # the lower right corner: both adapters dock along the top edge
    fig.ellipse(bx, by, 8, fill=WHITE, stroke=ICE_D, sw=0.9, effects=[shadow(0.7, opacity=30)], name='frozen badge')
    snowflake(fig, bx, by, 5.4, ICE_D, 1.05)


def model_top_y(x):
    """y of the frozen model's slanted top edge at x."""
    t = MH * TAPER / 2
    return (MID - MH / 2) + t * (x - X_MODEL) / MW


def adapter_top():
    """y of the top of the docked adapters: one level for both. The model's top edge slopes down to the right, so the
    higher docking point (the logP adapter's) sets the level and the adapter on the right gets the longer stem."""
    return min(model_top_y(AP_X), model_top_y(AC_X)) - DOCK_GAP - ADAPTER_H


def adapter(fig, x, a, label):
    """An adapter docking on the frozen model: it slides down from the card above, with a stem into the base."""
    if a < 0.02:
        return
    y = adapter_top() - 15 * (1 - a)
    fig.model(x - ADAPTER_W / 2, y, ADAPTER_W, ADAPTER_H, label, taper=0.26, direction='down', fill=ADAPTER_FILL,
              size=6, font='rc', name='adapter')
    fig.line(x, y + ADAPTER_H, x, model_top_y(x) + 1.5 * (1 - a), sw=1.1, name='adapter into base')


def label_property(fig, k):
    """'drop logP* = 2.5' to the left of the logP adapter, in the colour the fade k gives it."""
    parts = [('logP', 'mr'), ('∗', 'mm', 'sup'), SP, ('=', 'mm'), SP, (LEVEL_TEXT, 'mr')]
    _, w = fig._measure(parts, 7.5)
    top = adapter_top() + ADAPTER_H / 2
    x_end = AP_X - ADAPTER_W / 2 - 4                      # right edge of the label
    fig.rich(x_end - w, top + 2.7, parts, size=7.5, color=tint(BLACK, 1 - k))
    if k > 0.5:
        fig.emoji('💧', x_end - w - 8, top - 0.2, size=8.5)


def pulses(fig, dock_x, ages, color):
    """Ripples at an adapter's docking point: the condition acts on each pulse that passes under the adapter. A
    soft flash first, then a ring that widens and fades."""
    cx, cy = dock_x, model_top_y(dock_x)
    for age in ages:
        k = age / PULSE_LIFE
        flash = 1 - age / (3 * SLOW)
        if flash > 0.02:
            fig.ellipse(cx, cy + 1, 6.5, fill=color, stroke='none', opacity=pct(70 * flash), effects=[blur(3.2)],
                        name='pulse flash')
        fig.ellipse(cx, cy, 3.0 + 12.0 * k, fill='none', stroke=color, sw=2.0 - 1.1 * k,
                    opacity=pct(100 * (1 - k) ** 1.2), name='pulse ring')


def pulse_color(x):
    """A pulse is dark until it passes the logP adapter, orange (the adapters' colour) after it."""
    return mix(PULSE, ORANGE, smooth((x - (AP_X - 2)) / 12))


def draw_flow_pulses(fig, i, x_lo, x_hi):
    """Small pulses that run from the noise behind the model and out along the arrow to the sample: a dot with a short
    fading trail. Only the dots between x_lo and x_hi are drawn: the ones left of the model's right edge BEHIND the
    model (its body hides them while they are inside), the rest in front of the arrow. A pulse takes the colour of an
    adapter as it passes under it."""
    for u in flow_pulses(i):
        head = flow_x(u)
        vis = smooth(u / 0.1) * (1 - smooth((u - 0.88) / 0.12))       # fades in at the start, out on arrival
        if vis < 0.03:
            continue
        for dist, r, op in ((0, 2.3, 100), (4.5, 1.8, 62), (9, 1.3, 36), (13.5, 0.9, 18)):
            x = head - dist
            if x < FLOW_X0 or not (x_lo <= x < x_hi):
                continue
            col = pulse_color(x)
            if dist == 0:
                fig.ellipse(x, MID, 5.0, fill=col, stroke='none', opacity=pct(22 * vis), effects=[blur(2.6)],
                            name='pulse halo')
            fig.ellipse(x, MID, r, fill=col, stroke='none', opacity=pct(op * vis), name='pulse')


def time_axis(fig, t):
    """The flow time as a faint scrubber under the stations: from the noise (t = 0) to the sample (t = 1). The knob
    carries the counter; the part the knob has passed is a shade darker."""
    x0, x1, y = X_NOISE, X_SAMPLE, AXIS_Y
    xt = x0 + (x1 - x0) * t
    fig.line(x0, y, x1, y, stroke=AX_TRACK, sw=1.1, cap='round', name='time track')
    if t > 0.004:
        fig.line(x0, y, xt, y, stroke=AX_PASSED, sw=1.1, cap='round', name='time passed')
    for k in range(1, TICKS):
        xk = x0 + (x1 - x0) * k / TICKS
        fig.line(xk, y - 1.4, xk, y + 1.4, stroke=AX_PASSED if t >= k / TICKS - 1e-9 else AX_TICK_OFF, sw=0.5,
                 name='tick')
    fig.ellipse(x0, y, 1.6, fill=AX_PASSED, stroke='none', name='t = 0')
    fig.ellipse(x1, y, 1.6, fill=AX_PASSED if t >= 0.999 else AX_TICK_OFF, stroke='none', name='t = 1')
    fig.ellipse(xt, y, 2.7, fill=WHITE, stroke=AX_KNOB, sw=0.9, effects=[shadow(0.5, opacity=22)], name='knob')
    fig.rich(xt, T_BASE, [('t', 'mi'), SP, ('=', 'mm'), SP, (f'{t:.2f}', 'mr')], size=7, color=AX_TEXT,
             justify='center')


# ------------------------------------------------------------------------------ the upper part
def smiles_state(smiles):
    """A designed example: the discrete state (element per node, bonds with an order) of a small molecule."""
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    X = [ATOMS.index(a.GetSymbol()) for a in mol.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(round(b.GetBondTypeAsDouble()))) for b in mol.GetBonds()]
    return X, bonds


def example(smiles, cx, cy):
    """Positions and state of a designed example centred on (cx, cy). Every bond is drawn single: at this size the
    order of a bond is noise, and the example only has to carry the explanation."""
    X, bonds = smiles_state(smiles)
    pos, _ = fit_layout(molecule_layout(X, bonds, ATOMS), MOL_W, MOL_H, 6.0, cx, cy)
    return pos, X, [(i, j, 1) for i, j, _ in bonds]


def wave_alphas(w, i, f):
    """Opacity of the input molecule and of the explained molecule of wave w in lane i at frame f: a molecule goes
    into MEGAN, comes out with its explanation, and the explanation leaves for the concepts. The first wave is there
    from the start (the first frame shows the molecules), the last stays."""
    u = f - lane_t0(w, i)
    last = w == WAVES - 1
    a_in = 1.0 if w == 0 else smooth(u / U_IN)
    a_ex = smooth((u - U_EXPL) / 6)
    if not last:
        a_in *= 1 - smooth((u - U_OUT) / 5)
        a_ex *= 1 - smooth((u - (U_DOT + 6)) / 5)
    return a_in, a_ex


def draw_lanes(fig, f, s):
    """MEGAN with its three lanes: molecules go in, explanations (the glow) come out and leave for the concepts."""
    k = s.explain
    yc = ROW[1]
    x_in0 = X_IN + MOL_W / 2 + 4
    for i, y in enumerate(ROW):
        fig.line(x_in0, y, X_MEGAN, y, stroke=pale(BLACK, k), sw=0.9, name='into MEGAN')
    for i in range(len(LANES)):                                            # a pulse on each line, behind MEGAN
        y = ROW[i]
        for w in range(WAVES):
            u = f - lane_t0(w, i)
            if U_PULSE <= u <= U_PULSE + N_PULSE:
                x = x_in0 + (X_MEGAN + 6 - x_in0) * (u - U_PULSE) / N_PULSE
                comet(fig, x, y, 1.0, 0.0, PULSE, vis=k, name='pulse into MEGAN')
    fig.model(X_MEGAN, yc - MEGAN_H / 2, MEGAN_W, MEGAN_H, 'MEGAN', taper=0.2, size=7, font='rcm', color=tint(BLACK, 1 - k),
              stroke=pale(BLACK, k), name='MEGAN')
    for y in ROW:
        fig.arrow([(X_MEGAN + MEGAN_W + 3, y), (X_OUT - MOL_W / 2 - 4, y)], sw=0.9, head=2.6, color=pale(BLACK, k),
                  name='explanation')
        fig.arrow([(X_OUT + MOL_W / 2 + 5.0, y), (CARD_X - 2, y)], sw=0.9, head=2.6, color=pale(BLACK, k),
                  name='into the concepts')
    fig.text(X_IN, yc + MEGAN_H / 2 + 9, 'molecules', size=6, color=pale(GREY, k), justify='center')
    fig.text(X_OUT, yc + MEGAN_H / 2 + 9, 'explanations', size=6, color=pale(GREY, k), justify='center')
    for i, letter in enumerate(LANES):
        y = ROW[i]
        for w in range(WAVES):
            a_in, a_ex = wave_alphas(w, i, f)
            if w == 0:
                a_in = max(a_in, s.rho)
            if w == WAVES - 1:
                a_in *= 1 - s.rho
                a_ex *= 1 - s.expl_out
            smiles, glow_atoms = EXAMPLES[letter][w]
            if a_in * k > 0.02:
                pos, X, bonds = example(smiles, X_IN, y)
                draw_state(fig, pos, X, bonds, ATOMS, r=2.5, r_hetero=2.8, bond_sw=1.0, fade=a_in * k, letters=False,
                           name=f'molecule {letter}{w}')
            if a_ex * k > 0.02:
                pos, X, bonds = example(smiles, X_OUT, y)
                draw_state(fig, pos, X, bonds, ATOMS, r=2.5, r_hetero=2.8, bond_sw=1.0, fade=a_ex * k, letters=False,
                           glow=0.9, glow_atoms=glow_atoms, glow_color=CONCEPT[letter], glow_r=1.7,
                           name=f'explained molecule {letter}{w}')


def draw_dots(fig, f):
    """The explanations on their way into the concepts: a dot with a trail from the explained molecule to its cluster."""
    for i, letter in enumerate(LANES):
        for w in range(WAVES):
            u = f - lane_t0(w, i) - U_DOT
            if u < 0:
                continue
            a, b = dot_ends(i, w)
            n = dot_len(i, w)
            if u * DOT_SPEED < n:
                ux, uy = (b[0] - a[0]) / n, (b[1] - a[1]) / n
                run = u * DOT_SPEED
                comet(fig, a[0] + ux * run, a[1] + uy * run, ux, uy, CONCEPT[letter], vis=1.0, name='explanation in flight')


def concepts_card(fig, f, s):
    """The concepts: clusters of explanations, one colour and one letter each. A dot joins its cluster when an
    explanation arrives; the cluster the sample's explanation lands in lights up, and the requested one has a ring."""
    fig.card(CARD_X, CARD_Y, CARD_W, CARD_H, name='concepts')
    fig.pill(CARD_X + CARD_W / 2, CARD_Y, 'Concepts', anchor='center')
    fig.text(CARD_X + CARD_W - 6, CARD_Y + 14, 'clusters of explanations', size=6, color=GREY, justify='right')
    cf = s.clusters
    for i, letter in enumerate(LANES):
        c = CLUSTERS[letter]
        col = CONCEPT[letter]
        pops = [smooth((f - t_land(i, w)) / 3) for w in range(WAVES)]
        grow = sum(pops) / WAVES
        lit = s.lit if s.rb_letter == letter else 0.0
        ask = s.ask_ring if letter == 'A' else 0.0
        if grow * cf > 0.02:
            fig.ellipse(c['x'], c['y'], c['r'] + 2.5 * lit, fill=col, stroke='none',
                        opacity=pct((38 + 32 * lit) * (0.35 + 0.65 * grow) * cf), effects=[blur(0.55 * c['r'])],
                        name=f'cluster {letter}')
        for w in range(WAVES):
            if pops[w] * cf > 0.05:
                dx, dy = c['wave'][w]
                fig.ellipse(c['x'] + dx, c['y'] + dy, 2.1 * pops[w], fill=col, stroke=WHITE, sw=0.5,
                            opacity=None if cf > 0.995 else pct(100 * cf), name=f'explanation {letter}')
                age = f - t_land(i, w)
                if 0 <= age < 6:                                   # a small ripple where a dot lands
                    fig.ellipse(c['x'] + dx, c['y'] + dy, 2.1 + 4.0 * age / 6, fill='none', stroke=col, sw=0.8,
                                opacity=pct(80 * (1 - age / 6) * cf), name=f'landing {letter}')
        if s.rb_letter == letter and s.rb >= 0.999 and s.rb_alpha * cf > 0.05:      # the sample's explanation has landed
            dx, dy = c['new']
            fig.ellipse(c['x'] + dx, c['y'] + dy, 2.1, fill=col, stroke=WHITE, sw=0.5,
                        opacity=None if s.rb_alpha * cf > 0.995 else pct(100 * s.rb_alpha * cf), name=f'sample explanation {letter}')
        ra = max(lit, 0.75 * ask) * cf
        if ra > 0.03:
            fig.ellipse(c['x'], c['y'], c['r'] + RING_GAP, fill='none', stroke=col, sw=1.1, opacity=pct(100 * ra),
                        name=f'{letter} ring')
        sx, sy = c['token']
        tk = smooth((f - (t_land(i, 0) - 1)) / 4) * cf
        token(fig, c['x'] + sx * (c['r'] + 2), c['y'] + sy * (c['r'] - 1), 4.0, letter, col, tk, name=f'concept {letter}')


def request_line(fig, s):
    """The concept adapter's request: a line from the cluster of the requested concept, A, down into the adapter,
    with phi* (the membership vector, the requested concept filled) beside it."""
    if s.ask < 0.02:
        return
    c = CLUSTERS['A']
    y0 = c['y'] + c['r'] + RING_GAP
    y_adapter = adapter_top() - 15 * (1 - s.conc)           # where the adapter's top is now
    y1 = y0 + s.ask * (y_adapter - y0)
    fig.line(AC_X, y0, AC_X, y1, stroke=CONCEPT['A'], sw=1.6, name='phi* into adapter')
    k = s.ask_label
    if k > 0.03:
        fig.rich(AC_X + 5, CARD_Y + CARD_H + 9.0, [('\U0001D711', 'mm'), ('∗', 'mm', 'sup'), SP, ('=', 'mm')], size=7.5,
                 color=tint(BLACK, 1 - k))
        fig.cells(AC_X + 24, CARD_Y + CARD_H + 3.2, [tint(ORANGE, 0.35), WHITE, WHITE], 5.4, 6.4, opacity=pct(100 * k),
                  name='phi*')


# ------------------------------------------------------------------------------ the sample and its read-back
def caption_geometry(fig, kind, finals, d):
    """Where the caption 'logP = 2.4 (B)' sits just above the finished molecule: its text, baseline, left edge and
    width, and the centre of its concept token."""
    run = d['runs'][kind]
    top = min(y for _, y in finals[kind]) - 4.1
    base = max(12.0, top - 7.5)
    value = f'{run["logp"]:.1f}'.replace('-', '−')
    parts = [('logP', 'mr'), SP, ('=', 'mm'), SP, (value, 'mr')]
    _, w = fig._measure(parts, 7.5)
    x0 = X_SAMPLE - (w + 4 + 7.4) / 2
    return parts, base, x0, (x0 + w + 4 + 3.7, base - 2.5)


def result_caption(fig, kind, finals, d, k):
    parts, base, x0, (tx, ty) = caption_geometry(fig, kind, finals, d)
    fig.rich(x0, base, parts, size=7.5, color=tint(DGREY, 1 - k))
    letter = LETTER[kind]
    token(fig, tx, ty, 3.7, letter, CONCEPT[letter], k, name='sample concept')


def readback_path(kind, finals, d, fig):
    """The route of MEGAN's read-back: up from the sample's concept token, around to the right edge of the card and
    along to the cluster it lands in. The last point is the dot's place in the cluster, the one before it where the
    arrow's head stops (outside the lit ring)."""
    _, _, _, (tx, ty) = caption_geometry(fig, kind, finals, d)
    c = CLUSTERS[LETTER[kind]]
    p0, ctrl, p1 = (tx, ty - 5.5), (tx, float(c['y'])), (CARD_X + CARD_W + 4.0, float(c['y']))
    pts = []
    for j in range(25):
        q = j / 24
        pts.append(((1 - q) ** 2 * p0[0] + 2 * q * (1 - q) * ctrl[0] + q * q * p1[0],
                    (1 - q) ** 2 * p0[1] + 2 * q * (1 - q) * ctrl[1] + q * q * p1[1]))
    tip = (c['x'] + c['r'] + RING_GAP + 2.0, float(c['y']))
    pts.append(tip)
    poly = Poly(pts)
    dot = (c['x'] + c['new'][0], c['y'] + c['new'][1])
    full = Poly(pts + [dot])
    return poly, full


def draw_readback(fig, s, finals, d):
    """MEGAN reads the finished molecule back: a line from the caption's token into the card, a dot at its front."""
    if s.rb_letter is None or s.rb_alpha < 0.03 or s.rb <= 0.0:
        return
    kind = 'property' if s.rb_letter == 'B' else 'joint'
    col = CONCEPT[s.rb_letter]
    poly, full = readback_path(kind, finals, d, fig)
    a = s.rb_alpha
    line_col = pale(col, a)
    front = s.rb * full.length
    if front < poly.length:
        pts = poly.upto(front)
        if len(pts) >= 2:
            fig.path(points=pts, stroke=line_col, sw=1.2, cap='round', name='explanation read back')
    else:
        fig.arrow(poly.pts, head=3.0, color=line_col, sw=1.2, name='explanation read back')
    if s.rb < 0.999:
        x, y = full.at(front)
        fig.ellipse(x, y, 2.2, fill=col, stroke=WHITE, sw=0.5, name='explanation on its way back')
    parts, base, x0, (tx, ty) = caption_geometry(fig, kind, finals, d)
    fig.text(tx - 15, CARD_Y + 66, 'MEGAN reads it back', size=6, color=pale(GREY, a), justify='right')


def sample_slot(fig, tl, d, atom_types, noise_pos, finals):
    """The evolving sample: noise (copied from the left), the half-denoised state, or the finished molecule with the
    explanation of its concept glowing."""
    dx = X_SAMPLE - X_NOISE
    sample_noise = [(x + dx, y) for x, y in noise_pos]
    X0, b0 = d['noise']['X'], d['noise']['bonds']
    run = d['runs'][tl.run] if tl.run else None
    if tl.running:
        s = d['clock'][tl.run](tl.p)
        q = min(tl.p / 0.92, 1.0)
        morph = 0.5 * q + 0.5 * smooth(q)
        pos = keep_apart(lerp_points(sample_noise, finals[tl.run], morph), 7.8)
        draw_state(fig, pos, run['X'][s], run['bonds'][s], atom_types, name='sample')
        return
    if tl.fade_noise > 0.01:
        draw_state(fig, sample_noise, X0, b0, atom_types, fade=tl.fade_noise, name='sample noise')
    if tl.fade_result > 0.01 and run is not None:
        draw_state(fig, finals[tl.run], run['X'][-1], run['bonds'][-1], atom_types, fade=tl.fade_result,
                   glow=0.8 * tl.glow, glow_atoms=run['glow'], glow_color=CONCEPT[LETTER[tl.run]], name='sample result')


# ------------------------------------------------------------------------------ the frame
LEVEL_TEXT = '2.5'


def build(fig):
    global LEVEL_TEXT
    d, atom_types, noise_pos, finals = load()
    LEVEL_TEXT = f"{d['meta']['level']:.1f}"
    tl = timeline(FRAME)
    t = counter_time(FRAME, tl, d)
    i = FRAME - (B_BEAT1 if tl.run == 'property' else B_BEAT2)           # frame within the run that is going

    # the upper part: molecules, MEGAN, the explanations on their way, the concepts
    draw_lanes(fig, FRAME, tl)
    concepts_card(fig, FRAME, tl)
    draw_dots(fig, FRAME)

    # the lower part, left to right: the noise, the frozen generator with its adapters, the sample
    draw_state(fig, noise_pos, d['noise']['X'], d['noise']['bonds'], atom_types, name='noise')
    fig.text(X_NOISE, LABEL, 'Noise', size=7, justify='center')
    time_axis(fig, t)
    fig.line(FLOW_X0, MID, X_MODEL, MID, sw=STYLE['arrow_sw'], name='noise into model')
    if tl.running:
        draw_flow_pulses(fig, i, -1e9, X_MODEL + MW)                      # on the line, and BEHIND the model
    request_line(fig, tl)                                                 # the concept adapter's request, from cluster A
    frozen_model(fig, X_MODEL, MID - MH / 2, MW, MH, 'DeFoG')
    fig.text(X_MODEL + MW / 2, LABEL, 'frozen generator', size=7, justify='center')
    if tl.running:
        pulses(fig, AP_X, ring_ages(i, AP_X), ORANGE)
        if tl.run == 'joint':
            pulses(fig, AC_X, ring_ages(i, AC_X), ORANGE)
    adapter(fig, AP_X, tl.prop, 'property\nadapter')
    adapter(fig, AC_X, tl.conc, 'concept\nadapter')
    if tl.prop > 0.85:
        label_property(fig, smooth((tl.prop - 0.85) / 0.15))
    fig.arrow([(X_MODEL + MW + 3, MID), (X_ARROW, MID)], name='model to sample')
    if tl.running:
        draw_flow_pulses(fig, i, X_MODEL + MW, 1e9)                       # out of the model, on the arrow

    # the evolving sample, its caption and MEGAN's read-back
    sample_slot(fig, tl, d, atom_types, noise_pos, finals)
    fig.text(X_SAMPLE, LABEL, 'Sample', size=7, justify='center')
    if tl.caption > 0.02 and tl.run:
        result_caption(fig, tl.run, finals, d, tl.caption)
    draw_readback(fig, tl, finals, d)
