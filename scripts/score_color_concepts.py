"""Experiment 1, concept arm: score the concept adapter's samples, one row per concept.

Reads the shard dumps of ``experiments/sample_color_concept_adapter.py`` (``train_color_concept_adapter.py``
writes the same layout for its own evaluation subset) and scores them with the explainer of the zoo:

    concept match   phi_c < 0.5 for the requested concept (MEGAN ConceptSpace, the bundle's scoring settings)
    unc             the same rate on the run's own unconditional baseline (``gen_baseline.npz`` of shard 0)
    lift            match - unc
    property excess mean exact property of the generated graphs minus the mean over the pool graphs that
                    are core members of the concept; the shift is the mean absolute excess over the
                    concepts, also as a share of the property range of 6

Expected layout, one folder per shard, each the run folder of one sampling job:

    <root>/shard0/{manifest.json, gen_baseline.npz, gen_<concept>.npz ...}
    <root>/shard1/ ...

Writes ``per_concept.csv`` and ``summary.json`` to ``--out``.

Usage:
    python scripts/score_color_concepts.py results/color_concepts --out scores/color_concepts
"""
import argparse
import csv
import json
import os

import numpy as np

from defog_megan.bundle import bundle_paths, load_space
from defog_megan.colorgraphs import load_pool_npz, pyg_to_vgd
from defog_megan.rb_motifs import MOTIF_NAMES, MOTIF_VALUES, count_motifs_pyg, exact_target

CORE = 0.5
PROPERTY_RANGE = 6.0


def load(path):
    """(graphs, exact property); the property is recomputed from structure, as the dumps carry NaN targets."""
    graphs, _ = load_pool_npz(path)
    y = np.array([exact_target(count_motifs_pyg(g)) for g in graphs], dtype=float)
    return graphs, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder holding the shard*/ run folders")
    ap.add_argument("--out", default="scores/color_concepts")
    ap.add_argument("--group", default="colorgraphs", help="the explainer bundle of the zoo that grades")
    ap.add_argument("--allow-partial", action="store_true",
                    help="score the concepts that were sampled even if the vocabulary is not covered")
    args = ap.parse_args()

    run_root = args.root
    shards = sorted(d for d in os.listdir(run_root) if d.startswith("shard") and "smoke" not in d)
    if not shards:
        raise SystemExit(f"no shard*/ folders under {run_root}")

    bundle = bundle_paths(args.group)
    pool = np.load(bundle["pool"], allow_pickle=True)
    w = np.array([MOTIF_VALUES[k] for k in MOTIF_NAMES])
    pool_y = np.asarray(pool["motifs"], dtype=float) @ w
    phi_pool = np.load(bundle["reference"], allow_pickle=True)
    phi_all = np.asarray(phi_pool["phi"], dtype=float)
    cids = [str(c) for c in phi_pool["concept_ids"]]

    space = load_space(args.group)

    def phi(graphs):
        return space.phi_from_graphs([pyg_to_vgd(g) for g in graphs])

    base_g, _ = load(os.path.join(run_root, shards[0], "gen_baseline.npz"))
    base_phi = phi(base_g)
    print(f"baseline: n={len(base_g)}")

    files, manifests = {}, {}
    for s in shards:
        with open(os.path.join(run_root, s, "manifest.json")) as fh:
            man = json.load(fh)
        manifests[s] = man
        for c in man["eval_concepts"]:
            assert c not in files, f"{c} sampled twice"
            files[c] = os.path.join(run_root, s, man["per_concept"][c]["file"])
    missing = [c for c in cids if c not in files]
    if missing and not args.allow_partial:
        raise SystemExit(f"the run does not cover the vocabulary, missing {missing} (--allow-partial scores the rest)")
    concepts = [c for c in cids if c in files]

    rows = []
    for c in concepts:
        j = cids.index(c)
        g, y = load(files[c])
        rate = float((phi(g)[:, j] < CORE).mean())
        unc = float((base_phi[:, j] < CORE).mean())
        members = pool_y[phi_all[:, j] < CORE]
        excess = float(y.mean() - members.mean())
        rows.append({
            "concept": c, "channel": int(c[2]), "n": len(g),
            "match": rate, "match_se": float(np.sqrt(rate * (1 - rate) / len(g))),
            "unc": unc, "lift": rate - unc,
            "property_mean": float(y.mean()), "pool_member_mean": float(members.mean()),
            "property_excess": excess,
        })
        print(f"{c:>9} n={len(g):4d}  match {rate:.3f}  unc {unc:.3f}  lift {rate - unc:+.3f}  excess {excess:+.3f}")

    def stats(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}

    abs_ex = np.abs([r["property_excess"] for r in rows])
    summary = {
        "n_per_concept": sorted({r["n"] for r in rows}),
        "n_concepts": len(rows),
        "n_baseline": len(base_g),
        "core_threshold": CORE,
        "match": stats("match"), "unc": stats("unc"), "lift": stats("lift"),
        "property_shift_mean_abs": float(abs_ex.mean()),
        "property_shift_pct_of_range": float(100 * abs_ex.mean() / PROPERTY_RANGE),
        "spread_note": "std is the sample std (ddof=1) over the concepts, not run-to-run noise",
        "sources": {"run_root": run_root,
                    "shards": {s: manifests[s].get("eval_concepts") for s in shards}},
        "sampling": manifests[shards[0]]["sampling"],
        "guidance_weight": manifests[shards[0]]["guidance_weight"],
        "source_adapter": manifests[shards[0]]["source_adapter"],
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_concept.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in ("match", "unc", "lift", "property_shift_mean_abs",
                                               "property_shift_pct_of_range")}, indent=1))


if __name__ == "__main__":
    main()
