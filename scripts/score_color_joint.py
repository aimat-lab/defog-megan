"""Score Experiment 2 (joint property x concept steering on colour graphs).

Reads the shard dumps of ``experiments/sample_color_joint.py`` (collected as
``<root>/<stage>/shard*/``, one folder per shard run) and grades every arm against every
cell it applies to, with one code path, so the rows are exactly comparable.

Per graph, all three readouts are exact or MEGAN itself, never a surrogate:

    valid            connected (the colour-graph definition, as in Experiment 1)
    property-match   |exact_target(g) - y*| <= 1   (and the exact rate, tau = 0)
    concept-match    phi_c < 0.5                   (MEGAN, the core-member threshold)
    joint            valid AND property-match AND concept-match

Single-condition arms are scored against every cell they can be read against: the
property arm at level L against all six concepts, the concept arm for c against all five
levels, the unconditional base against every cell. That is what makes them the nulls the
joint arms are read against -- a joint cell that merely matches one of them has composed
nothing.

ARM B (property + rejection filter) is derived, not sampled: under property-only sampling
the acceptance rate is that arm's joint rate, so the cost of the filter is
``samples per accepted hit = 1 / acceptance``. The same is reported for the unconditional
base, the floor of the comparison.

CELL CLASSES are read off the POOL, so the grouping is a property of the data, not of the
choices in the sampler: for concept c at level L, ``pool_exact`` is the number of real
core members of c whose motif sum is exactly L. 0 -> ``unsupported``; a share of at least
15% of c's core members -> ``native``; anything in between -> ``off-native``.

MECHANISMS. For every (cell, arm) the motif-count tuples of the graphs that land exactly on
the requested level are recorded -- the "which reason did you get" distribution the
paper's Figure 1 is built from.

Usage:
    python scripts/score_color_joint.py --root results/color_joint --stages probe fk --out scores/color_joint

The membership vectors of the generated graphs are cached next to each dump (``*.phi.npy``), so a
second scoring of the same dumps does not run the explainer again.
"""
import argparse
import collections
import csv
import glob
import json
import os

import numpy as np
import networkx as nx

from defog_megan.bundle import bundle_paths, load_space
from defog_megan.colorgraphs import load_pool_npz, pyg_to_vgd, wl_hash
from defog_megan.rb_motifs import MOTIF_NAMES, MOTIF_VALUES, count_motifs_pyg, exact_target

CORE = 0.5
TOL = 1
NATIVE_SHARE = 0.15


def load_units(stage_dirs):
    units = []
    for stage_dir in stage_dirs:
        manifests = sorted(glob.glob(os.path.join(stage_dir, "shard*", "manifest.json")))
        if not manifests:
            raise SystemExit(f"no shard manifests under {stage_dir}")
        for path in manifests:
            with open(path) as f:
                man = json.load(f)
            for u in man["units"]:
                u["path"] = os.path.join(os.path.dirname(path), u["file"])
                u["stage"] = man["stage"]
                units.append(u)
    return units


def is_connected(g) -> bool:
    n = int(g.x.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    ei = g.edge_index
    if ei.numel():
        a = ei.cpu().numpy()
        graph.add_edges_from((int(u), int(v)) for u, v in zip(a[0], a[1]) if u < v)
    return n > 0 and nx.is_connected(graph)


def read_unit(u, space):
    """Per-graph readouts for one unit, with phi cached beside the dump."""
    graphs, _ = load_pool_npz(u["path"])
    cache = u["path"].replace(".npz", ".phi.npy")
    if os.path.exists(cache):
        phi = np.load(cache)
    else:
        phi = space.phi_from_graphs([pyg_to_vgd(g) for g in graphs])
        np.save(cache, phi)
    counts = [count_motifs_pyg(g) for g in graphs]
    return {
        "valid": np.array([is_connected(g) for g in graphs]),
        "y": np.array([exact_target(c) for c in counts], dtype=float),
        "motifs": [tuple(int(c[m]) for m in MOTIF_NAMES) for c in counts],
        "phi": phi,
        "keys": [wl_hash(g) for g in graphs],
    }


def grade(r, concept_index, level):
    valid = r["valid"]
    prop = np.abs(r["y"] - level) <= TOL
    exact = r["y"] == level
    conc = r["phi"][:, concept_index] < CORE
    joint = valid & prop & conc
    joint_exact = valid & exact & conc
    mech = collections.Counter(m for m, hit in zip(r["motifs"], exact & valid) if hit)
    acc = float(joint.mean())
    return {
        "n": int(len(valid)),
        "validity": float(valid.mean()),
        "uniqueness": len(set(r["keys"])) / max(1, len(r["keys"])),
        "property_match": float(prop.mean()),
        "property_exact": float(exact.mean()),
        "concept_match": float(conc.mean()),
        "joint": acc,
        "joint_exact": float(joint_exact.mean()),
        "n_joint": int(joint.sum()),
        # each distinct hit counted once (FK resamples particles, so hits can repeat)
        "joint_unique": len({k for k, h in zip(r["keys"], joint) if h}) / len(valid),
        # Arm-B cost when this arm is the proposal: inf when nothing was accepted, so a
        # cell where filtering never succeeds cannot average in as cheap.
        "samples_per_hit": (1.0 / acc) if acc > 0 else float("inf"),
        "samples_per_unique_hit": (len(valid) / max(1, len({k for k, h in zip(r["keys"], joint) if h}))
                                   if joint.any() else float("inf")),
        "mechanisms": {",".join(map(str, k)): v for k, v in mech.most_common()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/color_joint", help="folder holding <stage>/shard*/ run folders")
    ap.add_argument("--stages", nargs="+", default=["probe"])
    ap.add_argument("--out", default="scores/color_joint")
    ap.add_argument("--group", default="colorgraphs", help="the explainer bundle of the zoo that grades")
    args = ap.parse_args()

    units = load_units([os.path.join(args.root, s) for s in args.stages])
    print(f"{len(units)} units from stages {args.stages}")

    space = load_space(args.group)
    bundle = bundle_paths(args.group)

    # --- the pool: which cells are real, and what mechanisms they hold ----------------
    pool = np.load(bundle["pool"], allow_pickle=True)
    pool_m = np.asarray(pool["motifs"], dtype=int)
    pool_y = pool_m @ np.array([MOTIF_VALUES[k] for k in MOTIF_NAMES])
    cond = np.load(bundle["reference"], allow_pickle=True)
    pool_phi = np.asarray(cond["phi"])
    ids = [str(c) for c in cond["concept_ids"]]
    assert ids == list(space.concept_ids), "pool phi and MEGAN disagree on concept order"

    concepts = sorted({u["concept"] for u in units if "concept" in u}, key=ids.index)
    levels = sorted({u["level"] for u in units if "level" in u})
    cells = {}
    for c in concepts:
        core = pool_phi[:, ids.index(c)] < CORE
        for lv in levels:
            n_exact = int((core & (pool_y == lv)).sum())
            share = n_exact / max(1, int(core.sum()))
            cls = ("unsupported" if n_exact == 0 else
                   "native" if share >= NATIVE_SHARE else "off-native")
            mech = collections.Counter(
                tuple(r) for r in pool_m[core & (pool_y == lv)])
            cells[(c, lv)] = {
                "concept": c, "level": lv, "class": cls, "pool_exact": n_exact,
                "pool_share": share,
                "pool_joint_rate": float((core & (np.abs(pool_y - lv) <= TOL)).mean()),
                "pool_mechanisms": {",".join(map(str, k)): v
                                    for k, v in mech.most_common()},
            }

    # --- grade every arm against every cell it applies to ----------------------------
    readouts = {}
    for u in units:
        readouts[u["key"]] = read_unit(u, space)
        print(f"  scored {u['key']:<40} n={u['n']}")

    rows = []
    for (c, lv), cell in cells.items():
        ci = ids.index(c)
        for u in units:
            arm = u["arm"]
            if arm == "property" and u["level"] != lv:
                continue
            if arm == "concept" and u["concept"] != c:
                continue
            # "native" is the end-to-end jointly conditioned model of Appendix C.2
            # (experiments/sample_color_native.py): one unit per cell, like the joint arms.
            if (arm in ("joint", "joint+fk", "fk_only", "native")
                    and (u["concept"], u["level"]) != (c, lv)):
                continue
            # Table 2's operating points keep their bare labels (property w=2, concept w=1,
            # FK lam=1); the Pareto sweep's other weights and lambdas are tagged.
            label = arm
            wc, wp = u.get("w") or (None, None)
            if arm in ("joint", "joint+fk"):
                label = f"{arm} wc{wc:g}/wp{wp:g}"
            elif arm == "property" and wp is not None and wp != 2.0:
                label = f"property wp{wp:g}"
            elif arm == "concept" and wc is not None and wc != 1.0:
                label = f"concept wc{wc:g}"
            if u.get("lam", 1.0) != 1.0:
                label += f" lam{u['lam']:g}"
            g = grade(readouts[u["key"]], ci, lv)
            g.update(arm=label, concept=c, level=lv, cell_class=cell["class"],
                     pool_exact=cell["pool_exact"],
                     seconds_per_sample=u.get("seconds_per_sample"))
            rows.append(g)

    # --- summaries -------------------------------------------------------------------
    def mean_over(rs, key):
        vals = [r[key] for r in rs]
        return float(np.mean(vals)) if vals else float("nan")

    arms = sorted({r["arm"] for r in rows})
    classes = ["native", "off-native", "unsupported"]
    summary = {"tolerance": TOL, "core": CORE, "concepts": concepts, "levels": levels,
               "cells": {f"{c}|{lv:+d}": v for (c, lv), v in cells.items()},
               "arms": {}}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        entry = {"all": {k: mean_over(rs, k) for k in
                         ("validity", "uniqueness", "property_match", "property_exact",
                          "concept_match", "joint", "joint_unique", "joint_exact")}}
        for cls in classes:
            sub = [r for r in rs if r["cell_class"] == cls]
            entry[cls] = {"n_cells": len(sub), "joint": mean_over(sub, "joint"),
                          "joint_exact": mean_over(sub, "joint_exact"),
                          "property_match": mean_over(sub, "property_match"),
                          "concept_match": mean_over(sub, "concept_match")}
        spu = [r["samples_per_unique_hit"] for r in rs if r["cell_class"] != "unsupported"]
        entry["median_samples_per_unique_hit_supported"] = float(np.median(spu)) if spu else float("nan")
        sph = [r["samples_per_hit"] for r in rs if r["cell_class"] != "unsupported"]
        entry["median_samples_per_hit_supported"] = (float(np.median(sph)) if sph
                                                     else float("nan"))
        entry["cells_with_zero_hits_supported"] = int(sum(np.isinf(sph)))
        entry["seconds_per_sample"] = mean_over(
            [r for r in rs if r["seconds_per_sample"] is not None], "seconds_per_sample")
        summary["arms"][arm] = entry

    joint_arms = [a for a in arms if a.startswith("joint ")]
    if joint_arms:
        def supported_joint(a):
            s = summary["arms"][a]
            n = s["native"]["n_cells"] + s["off-native"]["n_cells"]
            return (s["native"]["joint"] * s["native"]["n_cells"]
                    + s["off-native"]["joint"] * s["off-native"]["n_cells"]) / max(1, n)
        best = max(joint_arms, key=lambda a: (supported_joint(a),
                                               summary["arms"][a]["all"]["validity"]))
        summary["recommended_fk_weights"] = best.split(" ", 1)[1]
        summary["recommended_by"] = "mean joint success over native + off-native cells"

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    fields = ["concept", "level", "cell_class", "pool_exact", "arm", "n", "validity",
              "uniqueness", "property_match", "property_exact", "concept_match", "joint",
              "joint_exact", "n_joint", "samples_per_hit", "seconds_per_sample"]
    with open(os.path.join(args.out, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["concept"], r["level"], r["arm"])):
            w.writerow(r)
    with open(os.path.join(args.out, "mechanisms.json"), "w") as f:
        json.dump([{k: r[k] for k in ("concept", "level", "arm", "mechanisms")}
                   for r in rows], f)

    hdr = (f"{'arm':<26} {'valid':>6} {'uniq':>6} {'prop':>6} {'conc':>6} {'JOINT':>6} | "
           f"{'native':>7} {'off-nat':>7} {'unsupp':>7} | {'smp/hit':>8} {'s/graph':>8}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for arm in arms:
        s = summary["arms"][arm]
        a = s["all"]
        print(f"{arm:<26} {a['validity']:>6.3f} {a['uniqueness']:>6.3f} "
              f"{a['property_match']:>6.3f} {a['concept_match']:>6.3f} {a['joint']:>6.3f} | "
              f"{s['native']['joint']:>7.3f} {s['off-native']['joint']:>7.3f} "
              f"{s['unsupported']['joint']:>7.3f} | "
              f"{s['median_samples_per_hit_supported']:>8.1f} {s['seconds_per_sample']:>8.3f}")
    n_cls = collections.Counter(v["class"] for v in cells.values())
    print(f"\ncells: {dict(n_cls)}")
    if "recommended_fk_weights" in summary:
        print(f"recommended FK operating point: {summary['recommended_fk_weights']} "
              f"({summary['recommended_by']})")
    print(f"wrote {args.out}/summary.json, cells.csv, mechanisms.json")


if __name__ == "__main__":
    main()
