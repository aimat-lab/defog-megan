#!/usr/bin/env python
"""Record REAL denoising trajectories for the README banner.

The banner shows one frozen DeFoG base, steered by a logP adapter alone, denoising noise into a molecule, and then
the same base steered by the logP adapter AND a concept adapter denoising the SAME noise. Both runs are recorded
here from the actual sampler, so every atom and bond flip in the animation is a flip the model made. The glow on
a finished molecule is the explainer's own explanation (the node importance of MEGAN's second channel), in the
colour of the concept the molecule belongs to.

This is not part of the package's tests. It needs the model zoo (the ``zinc-logp`` group, see the README), the
explainer stack and RDKit, and it writes the data that ``banner.py`` draws. A GPU makes the sampling fast; the
explainer runs on the CPU.

    # 1. many same-noise pairs, ranked for legibility -> a candidate list to choose from
    python record_trajectories.py record --work WORK --concept ch1_cl34 --level 2.5

    # 2. (optional) re-rank the stored runs after changing the scoring
    python record_trajectories.py rank --work WORK

    # 3. full trajectories of the chosen pair -> trajectory.json (what the figure reads)
    python record_trajectories.py export --work WORK --rank 0 --out trajectory.json

The model zoo is found through ``$DEFOG_MEGAN_ZOO`` (or ``--zoo``).

Settings of the recorded runs: a fixed number of heavy atoms, 200 steps, polydec time distortion, eta 0 (calm: an
atom or bond that is right stays right), probability-space product composition. Run 1 uses the logP adapter at
weight 2 (the weight of the paper's property arm); run 2 uses the concept adapter and the logP adapter at weights
1 and 1 (the paper's joint arm). Eta 0, a fixed size and 200 steps are chosen so that the animation is readable;
they are not the paper's configuration (500 steps, eta 500, sizes from the base's own distribution).

REPRODUCING a pair: sampling is deterministic given ``(seed, batch, index)`` and the same library, checkpoints and
hardware class, and those numbers are stored in ``trajectory.json``. The pair was chosen from many (ranked for
legibility), not at random: the banner illustrates the process with real samples, it is not a measurement.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

# the zoo location must be set before defog_megan is imported
for _i, _arg in enumerate(sys.argv):
    if _arg == "--zoo" and _i + 1 < len(sys.argv):
        os.environ["DEFOG_MEGAN_ZOO"] = sys.argv[_i + 1]

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen

RDLogger.DisableLog("rdApp.*")

BOND_CLASSES = ["none", "single", "double", "triple"]
#: elements that read well in the animation (no phosphorus, no iodine, no charges)
LEGIBLE_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br"}
#: a node counts as explained when MEGAN's importance of the channel is above this
GLOW_THRESHOLD = 0.5
#: the channel of the explainer the concepts of the banner live in (the positive one: it raises logP)
CHANNEL = 1
W_PROPERTY_SOLO = 2.0           # the paper's property arm
W_JOINT = (1.0, 1.0)            # the paper's joint arm: (concept, property)


# ============================================================================ sampling with recording
def _recorder(sampler_cls):
    """Subclass of ``sampler_cls`` that keeps the discrete state (class index per node, per node pair) before
    the first step and after every step, for the whole batch."""

    class Recorded(sampler_cls):
        def _run_loop(self, X, E, y, node_mask, use_cfg, show_progress, on_step=None):
            self.X_states = [X.argmax(-1).cpu().numpy().astype(np.int8)]
            self.E_states = [E.argmax(-1).cpu().numpy().astype(np.int8)]
            return super()._run_loop(X, E, y, node_mask, use_cfg, show_progress, on_step=on_step)

        def _advance(self, t_int, X, E, y, node_mask, use_cfg):
            out = super()._advance(t_int, X, E, y, node_mask, use_cfg)
            self.X_states.append(out[0].argmax(-1).cpu().numpy().astype(np.int8))
            self.E_states.append(out[1].argmax(-1).cpu().numpy().astype(np.int8))
            return out

    return Recorded


def load_models(device):
    from defog_adapt.core import CrossAttentionAdapter, DeFoGModel
    from defog_megan.data import load_targets
    from defog_megan.zoo import zoo_path

    base = DeFoGModel.load(zoo_path("zinc-logp", "base.ckpt"), device="cpu").to(device).eval()
    prop = CrossAttentionAdapter.load(zoo_path("zinc-logp", "property_adapter.ckpt"), device="cpu").to(device).eval()
    conc = CrossAttentionAdapter.load(zoo_path("zinc-logp", "concept_adapter.ckpt"), device="cpu").to(device).eval()
    for adapter in (prop, conc):
        adapter.check_compatible(base)
    matrix, ids = load_targets(zoo_path("zinc-logp", "data/concept_targets.npz"), mode="empirical")
    return base, prop, conc, matrix, list(ids)


def sample_pair(models, seed, batch, n, steps, level, concept, device):
    """Two recorded runs of ``batch`` graphs each, from IDENTICAL initial noise (same seed, same size).

    ``property``: the logP adapter alone. ``joint``: the concept adapter and the logP adapter together."""
    from defog_adapt.core import AdaptedSampler, AdapterComposition, ConditionBranch

    base, prop, conc, matrix, ids = models
    ctmc = dict(sample_steps=steps, eta=0.0, omega=0.0, time_distortion="polydec")
    pvec = torch.tensor([[float(level)]], dtype=torch.float32, device=device).expand(batch, -1)
    cvec = torch.tensor(matrix[ids.index(concept)], dtype=torch.float32, device=device).expand(batch, -1)
    branches = {
        "property": [ConditionBranch(prop, pvec, W_PROPERTY_SOLO)],
        "joint": [ConditionBranch(conc, cvec, W_JOINT[0]), ConditionBranch(prop, pvec, W_JOINT[1])],
    }
    runs = {}
    for kind, br in branches.items():
        torch.manual_seed(seed)
        comp = AdapterComposition(br, base=base, mode="product")
        sampler = _recorder(AdaptedSampler)(base, comp, **ctmc)
        graphs = sampler.sample(batch, num_nodes=n, device=torch.device(device), show_progress=False)
        runs[kind] = dict(X=np.stack(sampler.X_states), E=np.stack(sampler.E_states), graphs=graphs)
    assert np.array_equal(runs["property"]["X"][0], runs["joint"]["X"][0]) \
        and np.array_equal(runs["property"]["E"][0], runs["joint"]["E"][0]), "the two runs must start from the same noise"
    return runs


# ============================================================================ legibility of a final molecule
def mol_from_state(X, E, atom_types):
    """RDKit molecule from a final discrete state (class per node, class per pair); None if it does not sanitise.
    The atom index of the molecule is the node index of the state."""
    rw = Chem.RWMol()
    for c in X:
        rw.AddAtom(Chem.Atom(atom_types[int(c)]))
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    n = len(X)
    for i, j in itertools.combinations(range(n), 2):
        c = int(E[i, j])
        if c:
            rw.AddBond(i, j, order[c])
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:                                          # noqa: BLE001
        return None
    return m


def _segments_cross(p, q, r, s):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return orient(p, q, r) * orient(p, q, s) < 0 and orient(r, s, p) * orient(r, s, q) < 0


def layout_ok(mol):
    """A 2D depiction with no overlapping atoms and no crossing bonds (so it can be drawn as ball and stick)."""
    AllChem.Compute2DCoords(mol)
    pts = [tuple(mol.GetConformer().GetAtomPosition(i))[:2] for i in range(mol.GetNumAtoms())]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    mean = float(np.mean([math.dist(pts[i], pts[j]) for i, j in bonds]))
    bonded = {frozenset(b) for b in bonds}
    for i, j in itertools.combinations(range(len(pts)), 2):
        if frozenset((i, j)) not in bonded and math.dist(pts[i], pts[j]) < 0.8 * mean:
            return False
    for (a, b), (c, d) in itertools.combinations(bonds, 2):
        if len({a, b, c, d}) == 4 and _segments_cross(pts[a], pts[b], pts[c], pts[d]):
            return False
    return True


def legible(mol):
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return False
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return False
    if {a.GetSymbol() for a in mol.GetAtoms()} - LEGIBLE_ELEMENTS:
        return False
    sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    if not sizes or min(sizes) < 5 or max(sizes) > 7 or len(sizes) > 3:
        return False
    return layout_ok(mol)


def final_layout(mol):
    """2D coordinates of a final molecule in units of its mean bond length."""
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    mean = np.mean([np.linalg.norm(pts[b.GetBeginAtomIdx()] - pts[b.GetEndAtomIdx()]) for b in mol.GetBonds()])
    return pts / mean


def trajectory_metrics(X, E, mol):
    """How watchable a recorded run is when drawn on the final molecule's layout.

    ``late_chord``: over the last half of the steps, the mean total length (beyond 1.6 bond lengths) of bonds that
    are present in the state but NOT in the final molecule. A bond that the model only removes at the very end is
    a real transient, but drawn across a finished-looking molecule it reads as a drawing error.
    ``last_flip``: the last step at which anything changed (how early the run settles).
    """
    pts = final_layout(mol)
    D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    steps = len(X) - 1
    final_bond = E[-1] != 0
    chord = np.zeros(steps + 1)
    for k in range(steps + 1):
        spurious = np.triu((E[k] != 0) & ~final_bond, 1)
        chord[k] = np.maximum(0.0, D[spurious] - 1.6).sum()
    changed = np.where((X[1:] != X[:-1]).any(1) | (E[1:] != E[:-1]).reshape(steps, -1).any(1))[0]
    return dict(late_chord=float(chord[steps // 2:].mean()), last_flip=int(changed.max()) if len(changed) else 0)


# ============================================================================ the explainer's view of a molecule
def explain(space, mols):
    """For each molecule: its SMILES, its concept memberships phi, and the nodes MEGAN's explanation marks in
    ``CHANNEL`` (as node indices of the molecule, i.e. of the sampler's state). Molecules the explainer cannot
    read come back as None."""
    smiles, orders = [], []
    for m in mols:
        smiles.append(Chem.MolToSmiles(m))
        orders.append([int(i) for i in m.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]])
    graphs, ok = space.graphs_from_smiles(smiles)
    out = [None] * len(mols)
    if not len(graphs):
        return out
    infos = space.model.forward_graphs(graphs, batch_size=64)
    phi = space.phi_from_graphs(graphs)
    k = 0
    for idx in range(len(mols)):
        if not ok[idx]:
            continue
        importance = np.asarray(infos[k]["node_importance"])[:, CHANNEL]
        order = orders[idx]
        if len(order) != len(importance):
            k += 1
            continue
        atoms = sorted(order[i] for i in range(len(order)) if importance[i] >= GLOW_THRESHOLD)
        out[idx] = dict(smiles=smiles[idx], phi=phi[k], importance={order[i]: float(importance[i]) for i in range(len(order))},
                        atoms=atoms)
        k += 1
    return out


def realized_concept(space, phi):
    """The concept of the channel a molecule belongs to (lowest phi, and below 0.5), or None."""
    ids = [c for c in space.concept_ids if c.startswith(f"ch{CHANNEL}_")]
    best = min(ids, key=lambda c: phi[space.index_of(c)])
    return best if phi[space.index_of(best)] < 0.5 else None


def connected(mol, atoms):
    """Whether the atoms form one connected piece of the molecule."""
    if not atoms:
        return False
    atoms, seen, todo = set(atoms), set(), [min(atoms)]
    while todo:
        a = todo.pop()
        if a in seen:
            continue
        seen.add(a)
        todo += [nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in atoms]
    return seen == atoms


# ============================================================================ ranking
def pair_score(info, mols, metrics, noise_hetero, level, steps, concept):
    """Higher is better. The pair is only here when both runs are legible, run 1 does not belong to the requested
    concept and run 2 does. What is scored: both land on the requested logP, both explanations are a few connected
    atoms, the pair has colour, both runs are clean to watch, and the shared noise has some colour too."""
    score = -2.0 * abs(info["joint"]["logp"] - level) - 1.0 * abs(info["property"]["logp"] - level)
    for kind in ("property", "joint"):
        size = len(info[kind]["atoms"])
        score -= 0.25 * abs(size - 6)                                    # a glow of about six atoms reads best
        score -= 0.6 * (0 if connected(mols[kind], info[kind]["atoms"]) else 1)
    elements = {a.GetSymbol() for m in mols.values() for a in m.GetAtoms()}
    score += 0.15 * len(elements)
    score -= 1.0 * sum(m["late_chord"] for m in metrics.values())
    score -= 0.5 * sum(max(0.0, 0.7 - m["last_flip"] / steps) for m in metrics.values())   # settled before 70 %
    score -= 0.25 * max(0, 3 - noise_hetero)
    return score


def rank_candidates(work, tol_joint=0.35, tol_prop=0.6):
    """Rank every usable pair among the recorded runs (from ``states.npz``) and write ``candidates.json``."""
    from defog_megan.bundle import load_space

    with open(os.path.join(work, "meta.json")) as fh:
        meta = json.load(fh)
    st = np.load(os.path.join(work, "states.npz"))
    atom_types, steps, level, concept = meta["atom_types"], meta["steps"], meta["level"], meta["concept"]
    total = st["property_X"].shape[0]

    # 1. cheap filters: both final molecules legible, both near the requested logP
    kept = []
    for run in range(total):
        mols = {kind: mol_from_state(st[f"{kind}_X"][run][-1], st[f"{kind}_E"][run][-1], atom_types)
                for kind in ("property", "joint")}
        if not all(legible(m) for m in mols.values()):
            continue
        logp = {kind: float(Crippen.MolLogP(m)) for kind, m in mols.items()}
        if abs(logp["joint"] - level) > tol_joint or abs(logp["property"] - level) > tol_prop:
            continue
        if Chem.MolToSmiles(mols["joint"]) == Chem.MolToSmiles(mols["property"]):
            continue
        kept.append((run, mols, logp))
    print(f"{len(kept)} of {total} pairs have two legible molecules near logP {level}")

    # 2. the explainer on the survivors
    space = load_space("zinc-logp")
    req_idx = space.index_of(concept)
    candidates = []
    for start in range(0, len(kept), 32):
        chunk = kept[start:start + 32]
        flat = [m for _, mols, _ in chunk for m in (mols["property"], mols["joint"])]
        views = explain(space, flat)
        for c, (run, mols, logp) in enumerate(chunk):
            vp, vj = views[2 * c], views[2 * c + 1]
            if vp is None or vj is None:
                continue
            rp, rj = realized_concept(space, vp["phi"]), realized_concept(space, vj["phi"])
            if rj != concept or rp is None or rp == concept or vp["phi"][req_idx] < 0.8:
                continue
            info = {"property": dict(logp=logp["property"], concept=rp, atoms=vp["atoms"], smiles=vp["smiles"],
                                     phi_requested=float(vp["phi"][req_idx]), phi_realized=float(vp["phi"][space.index_of(rp)])),
                    "joint": dict(logp=logp["joint"], concept=rj, atoms=vj["atoms"], smiles=vj["smiles"],
                                  phi_requested=float(vj["phi"][req_idx]), phi_realized=float(vj["phi"][space.index_of(rj)]))}
            met = {kind: trajectory_metrics(st[f"{kind}_X"][run], st[f"{kind}_E"][run], mols[kind]) for kind in mols}
            hetero = int(sum(atom_types[int(c_)] != "C" for c_ in st["property_X"][run][0]))
            candidates.append(dict(seed=run // meta["batch"], batch=meta["batch"], index=run % meta["batch"], run=run,
                                   score=pair_score(info, mols, met, hetero, level, steps, concept), noise_hetero=hetero,
                                   property=dict(info["property"], **met["property"]),
                                   joint=dict(info["joint"], **met["joint"])))
    candidates.sort(key=lambda c: -c["score"])
    with open(os.path.join(work, "candidates.json"), "w") as fh:
        json.dump(dict(meta=meta, candidates=candidates), fh, indent=1)
    print(f"{len(candidates)} usable pairs (run 1 explained as another concept, run 2 as {concept}); the best:")
    for rank, c in enumerate(candidates[:12]):
        p, j = c["property"], c["joint"]
        print(f"  #{rank:2d} score {c['score']:5.2f}  logP {p['logp']:4.2f} / {j['logp']:4.2f}  "
              f"{p['concept']} ({len(p['atoms'])} atoms) -> {j['concept']} ({len(j['atoms'])} atoms)   "
              f"{p['smiles']}   ->   {j['smiles']}")


# ============================================================================ commands
def cmd_record(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    from defog_megan.molecules import NEUTRAL_KEK_ATOMS

    atom_types = list(NEUTRAL_KEK_ATOMS)
    os.makedirs(args.work, exist_ok=True)
    states = {f"{k}_{a}": [] for k in ("property", "joint") for a in ("X", "E")}
    for seed in range(args.seeds):
        runs = sample_pair(models, seed, args.batch, args.n, args.steps, args.level, args.concept, device)
        for a in ("X", "E"):
            for kind in ("property", "joint"):
                states[f"{kind}_{a}"].append(np.moveaxis(runs[kind][a], 1, 0))        # -> (batch, steps+1, ...)
        print(f"seed {seed}: {(seed + 1) * args.batch} runs recorded", flush=True)
    np.savez_compressed(os.path.join(args.work, "states.npz"), **{k: np.concatenate(v) for k, v in states.items()})
    meta = dict(n=args.n, steps=args.steps, level=args.level, concept=args.concept, atom_types=atom_types,
                batch=args.batch, seeds=args.seeds, w_property_solo=W_PROPERTY_SOLO, w_joint=list(W_JOINT))
    with open(os.path.join(args.work, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    rank_candidates(args.work)


def cmd_rank(args):
    rank_candidates(args.work)


def cmd_export(args):
    with open(os.path.join(args.work, "candidates.json")) as fh:
        data = json.load(fh)
    meta, cand = data["meta"], data["candidates"][args.rank]
    st = np.load(os.path.join(args.work, "states.npz"))
    run, steps = cand["run"], meta["steps"]
    s = np.arange(steps + 1) / steps
    flow_t = (2 * s - s * s).round(5).tolist()                       # polydec: the t the model actually sees

    def trajectory(kind):
        X, E = st[f"{kind}_X"][run], st[f"{kind}_E"][run]
        bonds = [[[int(i), int(j), int(E[k, i, j])] for i, j in itertools.combinations(range(meta["n"]), 2)
                  if E[k, i, j]] for k in range(steps + 1)]
        c = cand[kind]
        return dict(smiles=c["smiles"], logp=round(c["logp"], 2), concept=c["concept"], glow=c["atoms"],
                    phi_requested=round(c["phi_requested"], 3), phi_realized=round(c["phi_realized"], 3),
                    X=X.astype(int).tolist(), bonds=bonds)

    out = dict(
        meta=dict(meta, seed=cand["seed"], batch_index=cand["index"], eta=0.0, omega=0.0, time_distortion="polydec",
                  bond_classes=BOND_CLASSES, flow_t=flow_t, channel=CHANNEL, glow_threshold=GLOW_THRESHOLD,
                  model="frozen ZINC250k DeFoG base (structural RL), neutral kekulized vocabulary",
                  adapters="logP adapter and concept adapter (cross-attention), probability-space product composition",
                  note="chosen from ranked candidates for legibility; real samples, not a measurement"),
        runs={kind: trajectory(kind) for kind in ("property", "joint")})
    out["noise"] = dict(X=out["runs"]["property"]["X"][0], bonds=out["runs"]["property"]["bonds"][0])
    assert out["noise"] == dict(X=out["runs"]["joint"]["X"][0], bonds=out["runs"]["joint"]["bonds"][0])
    with open(args.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    p, j = out["runs"]["property"], out["runs"]["joint"]
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB): property only logP {p['logp']} {p['concept']} "
          f"{p['smiles']}  ->  property + {meta['concept']} logP {j['logp']} {j['concept']} {j['smiles']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zoo", help="the model zoo (default: $DEFOG_MEGAN_ZOO)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record many same-noise pairs and rank them")
    r.add_argument("--work", required=True, help="folder for candidates.json and states.npz")
    r.add_argument("--concept", default="ch1_cl34", help="the requested concept (a channel 1 concept of the zoo)")
    r.add_argument("--level", type=float, default=2.5, help="the requested logP")
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--batch", type=int, default=24)
    r.add_argument("--n", type=int, default=18, help="heavy atoms")
    r.add_argument("--steps", type=int, default=200)
    r.set_defaults(fn=cmd_record)
    k = sub.add_parser("rank", help="re-rank the stored runs (after changing the scoring)")
    k.add_argument("--work", required=True)
    k.set_defaults(fn=cmd_rank)
    e = sub.add_parser("export", help="write the full trajectories of one ranked candidate as JSON")
    e.add_argument("--work", required=True)
    e.add_argument("--rank", type=int, default=0)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
