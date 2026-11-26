"""Score Experiments 3 and 4 (logP + concept steering on molecules).

    python scripts/score_mol_joint.py --root results/mol_joint --stages single probe fk --out scores/mol_joint \
        --train-csv <the ZINC molecules the adapters trained on>

Reads ``<root>/<stage>/shard*/manifest.json`` (written by ``experiments/sample_mol_joint.py``) and
grades every generated molecule:

    valid           decodes, parses in RDKit, and is a single fragment
    property match  RDKit Crippen logP of the decoded molecule within TOL of the target
                    (primary 0.5, also 1.0). EXACT -- nothing is predicted.
    concept match   MEGAN's phi for the requested concept below CORE (0.5)
    joint           valid AND property match AND concept match, over ALL graphs generated
                    (invalid ones count as misses)

CELLS are (concept, logP level) pairs classed by the pool the adapters were trained on: a cell
is *native* when at least 15% of the concept's core members lie in the +-0.5 band around the
level, *unsupported* when fewer than 10 do, *off-native* otherwise. The pool is the table of
memberships and logP values of the training molecules (``--conditions`` and ``--logp``, built by
``experiments/prepare_concept_dataset.py``); the molecules themselves are only needed for the
novelty column (``--train-csv``). None of these tables is part of the zoo's download.

ARMS are graded against every cell they apply to (the base and the property arm against all
concepts at their level, the concept arm against all levels of its concept, the joint arms
against their own cell), as in the colored-graph experiment (scripts/score_color_joint.py).

EXPERIMENT 3 additionally gets (``exp3.json``): the property adapter's MAE and match rates per
level and weight; the concept adapter's lift and enrichment over the base for every concept
that the base reaches often enough to measure; and the CROSS-AXIS DISPLACEMENT, referenced to
what the pool implies for the axis not requested rather than to unconditional generation
(the property is a function of the structure, so conditioning on either axis necessarily
moves the other and measuring against the base would report the data as interference).
"""
import argparse
import collections
import csv
import glob
import json
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

from defog_megan.bundle import load_space
from defog_megan.molecules import warn_if_rdkit_differs
from defog_megan.zoo import zoo_path

RDLogger.DisableLog("rdApp.*")
TRAIN = frozenset()
CORE = 0.5
TOLS = (0.5, 1.0)
BAND = 0.5                     # half-width of the pool band that defines a cell's support
NATIVE_SHARE = 0.15
MIN_SUPPORT = 10
MIN_BASELINE_HITS = 3          # a concept the base never reaches has no measurable lift


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
                u["_man"] = {k: man[k] for k in ("concepts", "levels", "concept_ids")}
                units.append(u)
    return units


def read_unit(u, space):
    """Per-graph readouts for one unit, with phi cached beside the dump."""
    with open(u["path"]) as f:
        smiles = json.load(f)["smiles"]
    n = len(smiles)
    valid = np.array([s is not None and "." not in s and Chem.MolFromSmiles(s) is not None
                      for s in smiles])
    logp = np.full(n, np.nan)
    for i in np.where(valid)[0]:
        logp[i] = Crippen.MolLogP(Chem.MolFromSmiles(smiles[i]))
    cache = u["path"].replace(".json", ".phi.npy")
    if os.path.exists(cache):
        phi = np.load(cache)
    else:
        phi = np.ones((n, space.num_concepts))
        if valid.any():
            phi[valid] = space.phi([smiles[i] for i in np.where(valid)[0]])
        np.save(cache, phi)
    return {"valid": valid, "logp": logp, "phi": phi, "smiles": smiles}


def grade(r, ci, level, train=frozenset()):
    """Metrics of one readout against one cell (concept column ``ci``, logP ``level``)."""
    n, valid = len(r["valid"]), r["valid"]
    miss = np.abs(r["logp"] - level)
    conc = valid & (r["phi"][:, ci] < CORE)
    out = {"n": n, "validity": float(valid.mean()),
           "uniqueness": len({s for s, v in zip(r["smiles"], valid) if v}) / max(1, int(valid.sum())),
           "concept_match": float(conc.mean()),
           "mae": float(np.nanmean(miss)) if valid.any() else float("nan")}
    for tol in TOLS:
        prop = valid & (miss <= tol)
        j = valid & prop & conc
        out[f"property_match_{tol:g}"] = float(prop.mean())
        out[f"joint_{tol:g}"] = float(j.mean())
        out[f"n_joint_{tol:g}"] = int(j.sum())
        out[f"samples_per_hit_{tol:g}"] = (1.0 / j.mean()) if j.any() else float("inf")
        # Duplicates inflate a rate when a sampler collapses (FK resamples particles): count
        # each hit once, and once more only if the molecule is not in the training set.
        hits = {r["smiles"][i] for i in np.where(j)[0]}
        out[f"joint_unique_{tol:g}"] = len(hits) / n
        out[f"joint_novel_{tol:g}"] = len(hits - train) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/mol_joint", help="folder holding <stage>/shard*/ run folders")
    ap.add_argument("--stages", nargs="+", default=["single"])
    ap.add_argument("--out", default="scores/mol_joint")
    ap.add_argument("--group", default="zinc-logp", help="the explainer bundle of the zoo that grades")
    ap.add_argument("--conditions", default=zoo_path("zinc-logp", "data/concept_conditions.npz"),
                    help="the pool: memberships of the training molecules (experiments/prepare_concept_dataset.py)")
    ap.add_argument("--logp", default=zoo_path("zinc-logp", "data/logp.npy"), help="the Crippen logP of every pool molecule")
    ap.add_argument("--train-csv", required=True, help="the molecules the adapters trained on (a SMILES column), for the novelty column")
    args = ap.parse_args()
    warn_if_rdkit_differs("the property of the logP scorer")
    os.makedirs(args.out, exist_ok=True)

    units = load_units([os.path.join(args.root, s) for s in args.stages])
    print(f"{len(units)} units from stages {args.stages}")
    space = load_space(args.group)
    ids = [str(c) for c in space.concept_ids]
    man = units[0]["_man"]
    assert ids == list(man["concept_ids"]), "scorer and sampler disagree on the concept order"
    cells_concepts, levels = list(man["concepts"]), [float(x) for x in man["levels"]]

    global TRAIN
    TRAIN = frozenset(Chem.MolToSmiles(m) for m in
                      (Chem.MolFromSmiles(x) for x in
                       __import__("pandas").read_csv(args.train_csv)["smiles"]) if m is not None)
    print(f"training set: {len(TRAIN)} canonical molecules")

    # --- the pool: what each cell's support is ---------------------------------------
    pool = np.load(args.conditions, allow_pickle=True)
    assert [str(c) for c in pool["concept_ids"]] == ids, "pool and MEGAN disagree on concepts"
    pool_phi, pool_logp = np.asarray(pool["phi"]), np.load(args.logp)
    pool_core = pool_phi < CORE
    cells = {}
    for c in cells_concepts:
        core = pool_core[:, ids.index(c)]
        for lv in levels:
            band = core & (np.abs(pool_logp - lv) <= BAND)
            share = band.sum() / max(1, core.sum())
            cls = ("unsupported" if band.sum() < MIN_SUPPORT
                   else "native" if share >= NATIVE_SHARE else "off-native")
            cells[(c, lv)] = {"concept": c, "level": lv, "class": cls,
                              "pool_in_band": int(band.sum()), "pool_share": float(share)}
    print("cells:", dict(collections.Counter(v["class"] for v in cells.values())))

    # --- read and grade every unit ---------------------------------------------------
    readouts = {}
    for u in units:
        readouts[u["key"]] = read_unit(u, space)
        print(f"  read {u['key']:<46} n={u['n']}")

    def label(u):
        wc, wp = u.get("w") or (None, None)
        if u["arm"] in ("joint", "joint+fk"):
            return f"{u['arm']} wc{wc:g}/wp{wp:g}"
        if u["arm"] == "property":
            return f"property w{wp:g}"
        if u["arm"] == "concept":
            return f"concept w{wc:g}"
        return u["arm"]

    rows = []
    for (c, lv), cell in cells.items():
        ci = ids.index(c)
        for u in units:
            a = u["arm"]
            if a == "property" and float(u["level"]) != lv:
                continue
            if a == "concept" and u["concept"] != c:
                continue
            # "native" is the end-to-end jointly conditioned model of Appendix C.2
            # (experiments/sample_mol_native.py): one unit per cell, like the joint arms
            if a in ("joint", "joint+fk", "fk_only", "native") and (u["concept"], float(u["level"])) != (c, lv):
                continue
            g = grade(readouts[u["key"]], ci, lv, TRAIN)
            g.update(arm=label(u), concept=c, level=lv, cell_class=cell["class"],
                     pool_in_band=cell["pool_in_band"],
                     seconds_per_sample=u.get("seconds_per_sample"))
            rows.append(g)

    def mean_over(rs, key):
        v = [r[key] for r in rs if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(v)) if v else float("nan")

    arms = sorted({r["arm"] for r in rows})
    summary = {"tolerances": list(TOLS), "core": CORE, "band": BAND, "concepts": cells_concepts,
               "levels": levels, "cells": {f"{c}|{lv:+.1f}": v for (c, lv), v in cells.items()},
               "arms": {}}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        keys = ("validity", "uniqueness", "concept_match", "mae", "property_match_0.5",
                "property_match_1", "joint_0.5", "joint_1", "joint_unique_0.5",
                "joint_novel_0.5")
        entry = {"all": {k: mean_over(rs, k) for k in keys}}
        for cls in ("native", "off-native", "unsupported"):
            sub = [r for r in rs if r["cell_class"] == cls]
            entry[cls] = {"n_cells": len(sub), **{k: mean_over(sub, k) for k in keys}}
        sph = [r["samples_per_hit_0.5"] for r in rs if r["cell_class"] != "unsupported"]
        entry["median_samples_per_hit_supported"] = float(np.median(sph)) if sph else float("nan")
        entry["cells_with_zero_hits_supported"] = int(sum(np.isinf(sph)))
        entry["seconds_per_sample"] = mean_over(
            [r for r in rs if r["seconds_per_sample"] is not None], "seconds_per_sample")
        summary["arms"][arm] = entry

    joint_arms = [a for a in arms if a.startswith("joint ")]
    if joint_arms:
        def supported_joint(a):
            s = summary["arms"][a]
            n = s["native"]["n_cells"] + s["off-native"]["n_cells"]
            return (s["native"]["joint_0.5"] * s["native"]["n_cells"]
                    + s["off-native"]["joint_0.5"] * s["off-native"]["n_cells"]) / max(1, n)
        best = max(joint_arms, key=supported_joint)
        summary["recommended_fk_weights"] = best.split(" ", 1)[1]
        summary["recommended_by"] = "mean joint match (tol 0.5) over native + off-native cells"

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    fields = ["concept", "level", "cell_class", "pool_in_band", "arm", "n", "validity",
              "uniqueness", "mae", "property_match_0.5", "property_match_1", "concept_match",
              "joint_0.5", "joint_1", "joint_unique_0.5", "joint_novel_0.5", "n_joint_0.5",
              "samples_per_hit_0.5", "seconds_per_sample"]
    with open(os.path.join(args.out, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["arm"], r["concept"], r["level"])))

    # --- Experiment 3: each adapter alone ------------------------------------------
    has_single = any(st == "single" or st.startswith("single_") for st in args.stages)   # e.g. single_eta500
    if has_single:
        exp3 = {}
        base_u = next(u for u in units if u["arm"] == "uncond")
        base_r = readouts[base_u["key"]]
        base_core = (base_r["phi"] < CORE) & base_r["valid"][:, None]
        # property adapter: per weight and level
        prop = collections.defaultdict(dict)
        for u in units:
            if u["arm"] != "property":
                continue
            r = readouts[u["key"]]
            miss = np.abs(r["logp"] - float(u["level"]))
            prop[f"w{u['w'][1]:g}"][f"{float(u['level']):+.1f}"] = {
                "n": u["n"], "validity": float(r["valid"].mean()),
                "mae": float(np.nanmean(miss)),
                **{f"match_{t:g}": float((r["valid"] & (miss <= t)).mean()) for t in TOLS}}
        exp3["property"] = prop
        # base reference on the same levels (what no steering gives)
        bm = {f"{lv:+.1f}": {f"match_{t:g}": float((base_r["valid"] & (np.abs(base_r["logp"] - lv) <= t)).mean())
                              for t in TOLS} | {"mae": float(np.nanmean(np.abs(base_r["logp"] - lv)))}
              for lv in levels}
        exp3["base_by_level"] = bm
        # concept adapter: lift and enrichment for every concept
        conc = collections.defaultdict(dict)
        for u in units:
            if u["arm"] != "concept":
                continue
            ci = ids.index(u["concept"])
            r = readouts[u["key"]]
            rate = float(((r["phi"][:, ci] < CORE) & r["valid"]).mean())
            base_rate = float(base_core[:, ci].mean())
            conc[f"w{u['w'][0]:g}"][u["concept"]] = {
                "core_rate": rate, "base_rate": base_rate, "lift": rate - base_rate,
                "enrichment": (rate / base_rate) if base_rate > 0 else float("inf"),
                "measurable": bool(base_rate * len(base_r["valid"]) >= MIN_BASELINE_HITS),
                "validity": float(r["valid"].mean())}
        exp3["concept"] = conc
        # cross-axis displacement, referenced to the POOL (not to unconditional generation)
        disp = {}
        for w, per in conc.items():
            shifts, own = [], 0
            for cid in per:
                ci = ids.index(cid)
                u = next(x for x in units if x["arm"] == "concept" and x["concept"] == cid
                         and f"w{x['w'][0]:g}" == w)
                r = readouts[u["key"]]
                gen = np.nanmean(r["logp"][r["valid"] & (r["phi"][:, ci] < CORE)]) \
                    if (r["valid"] & (r["phi"][:, ci] < CORE)).any() else np.nan
                ref = pool_logp[pool_core[:, ci]].mean()
                shifts.append((cid, float(gen - ref)))
            shifts = [(c, s) for c, s in shifts if not np.isnan(s)]
            ch = lambda cid: int(cid[2])
            own = sum(1 for c, s in shifts if (s < 0) == (ch(c) == 0))
            disp[w] = {"mean_abs_property_shift": float(np.mean([abs(s) for _, s in shifts])),
                       "own_channel_direction": own, "n_concepts": len(shifts),
                       "per_concept": {c: s for c, s in shifts}}
        exp3["concept_property_displacement"] = disp
        pdisp = {}
        for w, per_level in prop.items():
            diffs = []
            for lv in levels:
                u = next(x for x in units if x["arm"] == "property" and float(x["level"]) == lv
                         and f"w{x['w'][1]:g}" == w)
                r = readouts[u["key"]]
                gen_rate = ((r["phi"] < CORE) & r["valid"][:, None]).mean(0)
                band = np.abs(pool_logp - lv) <= BAND
                pool_rate = pool_core[band].mean(0)
                diffs.append(np.abs(gen_rate - pool_rate).mean())
            pdisp[w] = {"mean_abs_concept_shift": float(np.mean(diffs))}
        exp3["property_concept_displacement"] = pdisp
        with open(os.path.join(args.out, "exp3.json"), "w") as f:
            json.dump(exp3, f, indent=1, default=float)
        print("\n=== Experiment 3 ===")
        for w, per in sorted(prop.items()):
            print(f"property adapter {w}: " + "  ".join(
                f"{lv}: MAE {v['mae']:.2f} m{TOLS[0]:g} {v['match_0.5']:.2f}" for lv, v in per.items()))
        for w, d in disp.items():
            print(f"concept adapter {w}: mean |logP shift vs pool| {d['mean_abs_property_shift']:.3f}, "
                  f"own-channel direction {d['own_channel_direction']}/{d['n_concepts']}")
        for w, d in pdisp.items():
            print(f"property adapter {w}: mean |concept-rate shift vs pool| {d['mean_abs_concept_shift']:.4f}")
        for w, per in conc.items():
            m = [v for v in per.values() if v["measurable"]]
            print(f"concept adapter {w}: {len(m)}/{len(per)} measurable, median enrichment "
                  f"{np.median([v['enrichment'] for v in m]):.2f}x, mean lift "
                  f"{np.mean([v['lift'] for v in m]):+.3f}")

    # --- Experiment 4 table ---------------------------------------------------------
    print("\narm                          valid   uniq   prop  conc  JOINT | native off-nat  unsupp |"
          "  JOINT@1 | smp/hit  s/graph")
    print("-" * 118)
    for arm in arms:
        s = summary["arms"][arm]
        a = s["all"]
        print(f"{arm:28s} {a['validity']:.3f}  {a['uniqueness']:.3f}  {a['property_match_0.5']:.3f} "
              f"{a['concept_match']:.3f} {a['joint_0.5']:.3f} |  {s['native']['joint_0.5']:.3f}  "
              f"{s['off-native']['joint_0.5']:.3f}  {s['unsupported']['joint_0.5']:.3f} | "
              f"  {a['joint_1']:.3f}  | {s['median_samples_per_hit_supported']:7.1f} "
              f"{s['seconds_per_sample']:.3f}")
    print("\njoint match (tol 0.5) counting each hit once / once and only if absent from the training set")
    for arm in arms:
        s = summary["arms"][arm]
        print(f"{arm:28s} all {s['all']['joint_0.5']:.3f} -> {s['all']['joint_unique_0.5']:.3f} -> "
              f"{s['all']['joint_novel_0.5']:.3f} | native {s['native']['joint_0.5']:.3f} -> "
              f"{s['native']['joint_unique_0.5']:.3f} -> {s['native']['joint_novel_0.5']:.3f} | off-native "
              f"{s['off-native']['joint_0.5']:.3f} -> {s['off-native']['joint_unique_0.5']:.3f} -> "
              f"{s['off-native']['joint_novel_0.5']:.3f}")
    if "recommended_fk_weights" in summary:
        print(f"\nrecommended FK operating point: {summary['recommended_fk_weights']} "
              f"({summary['recommended_by']})")
    print(f"wrote {args.out}/summary.json, cells.csv" + (", exp3.json" if has_single else ""))


if __name__ == "__main__":
    main()
