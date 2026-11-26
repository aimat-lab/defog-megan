"""Score the colour-graph arms into the paper's main Experiment-1 table (Table 1).

One code path for every arm, so the rows are exactly comparable to each other
whatever the absolute definitions are. Columns, and the definitions they use:

* ``property MAE``  mean ``|exact_target(g) - y*|`` over requested levels. The
  property is recomputed from structure, never predicted. The unconditional row
  has no requested value, so its cell is empty rather than zero.
* ``validity``      the connected rate. Colour graphs have no chemistry to
  violate, so by the paper's definition a graph is valid iff it is
  connected. This makes a separate connectivity column redundant.
* ``uniqueness``    distinct canonical forms / generated, and
* ``novelty``       not-in-pool / distinct  -- the same two definitions
  the molecular scorers use (novelty is measured over the UNIQUE set, the
  MOSES/GuacaMol convention), so the rungs can be read against each other. The
  canonical form is a Weisfeiler-Lehman hash over the graph with node colour as
  the node label (``defog_megan.colorgraphs.wl_hash``), standing in for a SMILES
  string.
* ``KL``            ``kl_colors + kl_degrees + kl_sizes`` from
  :func:`structural_report` -- the same total ``distribution_score`` already
  exponentiates. The three terms are reported separately for the appendix.
* ``time``          seconds per generated graph, read from the timing block the
  eval experiments record. One process owns one GPU in every arm, so this is
  already per-GPU and the FK arms are not flattered by parallelism.

A CAVEAT THAT BELONGS IN THE CAPTION, not in a footnote nobody reads: the
conditional arms are sampled at equal N on all seven levels, while the pool's own
level distribution is flattened but not uniform. So part of every conditional
arm's KL is the deliberate uniform level coverage rather than a distributional
defect, and the unconditional row has an inherent advantage on that column. The
KL column compares arms to the pool, not arms to each other's difficulty.

Usage::

    python scripts/score_color_arms.py \
        --arm "Base=<eval_target_adapter run folder>:baseline" \
        --arm "Adapter (w=2)=<eval_target_adapter run folder>:w2.0" \
        --arm "Adapter + FK=<eval_color_fk run folder>" \
        --out scores/color_arms

Each ``--arm`` names a run folder (``gen_*.npz`` plus the timing json the experiment wrote); the
optional ``:FILTER`` keeps the files whose name contains it (a guidance weight, an FK setting).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from defog_megan.colorgraphs import RB_PALETTE, load_pool_npz, structural_report, wl_hash
from defog_megan.rb_motifs import count_motifs_pyg, exact_target
from defog_megan.zoo import zoo_path

POOL = zoo_path("colorgraphs", "data/pool.npz")
#: KL reference size, matching eval_target_adapter's REFERENCE_SIZE.
REFERENCE_SIZE = 5000
SEED = 42
_LEVEL_RE = re.compile(r"_lv([+-]\d+)")


def score_arm(files: dict, reference, pool_keys: set, timing: dict) -> dict:
    """``files`` maps requested level (or None) -> npz path."""
    all_graphs, maes = [], []
    for level, path in sorted(files.items(), key=lambda kv: (kv[0] is None, kv[0])):
        graphs, _ = load_pool_npz(path)
        all_graphs += list(graphs)
        if level is not None:
            achieved = np.array([exact_target(count_motifs_pyg(g)) for g in graphs],
                                dtype=float)
            maes.append(float(np.abs(achieved - level).mean()))

    keys = [wl_hash(g) for g in all_graphs]
    distinct = set(keys)
    novel = {k for k in distinct if k not in pool_keys}
    report = structural_report(all_graphs, reference)

    return {
        "n": len(all_graphs),
        "n_levels": sum(1 for lv in files if lv is not None),
        "property_mae": float(np.mean(maes)) if maes else None,
        "validity": report["connected_rate"],
        "uniqueness": len(distinct) / len(keys) if keys else 0.0,
        "novelty": len(novel) / len(distinct) if distinct else 0.0,
        "kl_total": report["kl_colors"] + report["kl_degrees"] + report["kl_sizes"],
        "kl_colors": report["kl_colors"],
        "kl_degrees": report["kl_degrees"],
        "kl_sizes": report["kl_sizes"],
        "mean_nodes": report["mean_nodes"],
        "mean_degree": report["mean_degree"],
        "seconds_per_sample": timing.get("seconds_per_sample"),
        "device": timing.get("device"),
    }


def collect(spec: str) -> tuple:
    """``DIR`` or ``DIR:filter`` -> (files, timing). ``filter`` selects a weight
    or FK setting when one directory holds several arms."""
    path, _, filt = spec.partition(":")
    names = sorted(glob.glob(os.path.join(path, "gen_*.npz")))
    if filt:
        names = [n for n in names if filt in os.path.basename(n)]
    if not names:
        raise SystemExit(f"no gen_*.npz under {path!r}" + (f" matching {filt!r}" if filt else ""))

    files = {}
    for n in names:
        m = _LEVEL_RE.search(os.path.basename(n))
        files[int(m.group(1)) if m else None] = n

    timing = {}
    for meta, key in ((os.path.join(path, "calibration.json"), "timing"),
                      (os.path.join(path, "conditional_samples.json"), "timing")):
        if os.path.exists(meta):
            block = json.load(open(meta)).get(key, {}) or {}
            # The unconditional row's cost is the baseline timer; every other arm's
            # is the mean over its sampled cells.
            if set(files) == {None}:
                timing = dict(block.get("baseline", {}), device=block.get("device"))
            else:
                timing = {"seconds_per_sample": block.get("mean_seconds_per_sample"),
                          "device": block.get("device")}
            break
    return files, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR[:FILTER]",
                    help="repeatable; NAME is the table's row label")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out", default=None, help="directory for summary.json")
    args = ap.parse_args()
    if not args.arm:
        raise SystemExit("need at least one --arm")

    pool_graphs, _ = load_pool_npz(args.pool)
    pool_graphs = list(pool_graphs)
    print(f"pool: {len(pool_graphs)} graphs")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pool_graphs), size=min(REFERENCE_SIZE, len(pool_graphs)),
                     replace=False)
    reference = [pool_graphs[i] for i in idx]
    print(f"KL reference: {len(reference)} graphs (seed {SEED})")
    pool_keys = {wl_hash(g) for g in pool_graphs}
    print(f"novelty reference: {len(pool_keys)} distinct canonical forms")

    rows = {}
    for spec in args.arm:
        # rpartition, not partition: row labels legitimately contain "=" ("w=1.0")
        name, _, where = spec.rpartition("=")
        files, timing = collect(where)
        rows[name] = score_arm(files, reference, pool_keys, timing)
        print(f"scored {name!r}: n={rows[name]['n']} from {len(files)} file(s)")

    hdr = (f"{'arm':<26} {'MAE':>7} {'novel':>7} {'uniq':>7} {'valid':>7} "
           f"{'KL':>7} {'s/graph':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        mae = "  --  " if r["property_mae"] is None else f"{r['property_mae']:>7.3f}"
        sec = "    --  " if r["seconds_per_sample"] is None else f"{r['seconds_per_sample']:>8.3f}"
        print(f"{name:<26} {mae} {r['novelty']:>7.3f} {r['uniqueness']:>7.3f} "
              f"{r['validity']:>7.3f} {r['kl_total']:>7.3f} {sec}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        dest = os.path.join(args.out, "summary.json")
        with open(dest, "w") as fh:
            json.dump({"rows": rows, "pool": args.pool,
                       "reference_size": len(reference), "seed": SEED}, fh, indent=2)
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
