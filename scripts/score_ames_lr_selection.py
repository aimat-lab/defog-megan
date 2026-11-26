"""Choose the concept adapter's learning rate (model-graded).

    python scripts/score_ames_lr_selection.py --roots lr0.0001=results/ames_joint/single_lrsel_0.0001 ... \
        --base results/ames_joint/single_lrsel_base

Each root holds the concept arm of ``experiments/sample_ames_joint.py`` (stage ``single``) for ONE adapter (one learning
rate): N graphs per concept at w = 1, sizes from the Ames set. A concept match is MEGAN's phi for the requested concept below 0.5
on a valid molecule, counted over all graphs generated; the base pool (same sizes) gives each concept's chance rate, and the
adapter's lift is the difference. The rate is chosen on the concepts that are NOT cells of the rung (``--exclude``), so the final
cells are not selected on; all of them are printed as well.
"""
import argparse
import glob
import json
import os

import numpy as np
from rdkit import Chem, RDLogger

from defog_megan.bundle import load_space
from defog_megan.zoo import zoo_path

RDLogger.DisableLog("rdApp.*")
CORE = 0.5


def units(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "shard*", "manifest.json"))):
        man = json.load(open(path))
        for u in man["units"]:
            u["path"] = os.path.join(os.path.dirname(path), u["file"])
            out.append(u)
    return out


def phi_of(u, space):
    smiles = json.load(open(u["path"]))["smiles"]
    valid = np.array([s is not None and "." not in s and Chem.MolFromSmiles(s) is not None for s in smiles])
    cache = u["path"].replace(".json", ".phi.npy")
    if os.path.exists(cache):
        return valid, np.load(cache)
    phi = np.ones((len(smiles), space.num_concepts))
    if valid.any():
        phi[valid] = space.phi([smiles[i] for i in np.where(valid)[0]])
    np.save(cache, phi)
    return valid, phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="name=dir pairs, one per learning rate")
    ap.add_argument("--base", required=True)
    ap.add_argument("--group", default="ames", help="the explainer bundle of the zoo that grades")
    ap.add_argument("--cells", default=zoo_path("ames", "data/cells.json"), help="the rung's cells file; its concepts are excluded from the selection")
    ap.add_argument("--exclude", nargs="*", default=None, help="overrides the concepts of --cells")
    args = ap.parse_args()
    if args.exclude is None:
        args.exclude = json.load(open(args.cells))["concepts"]
    space = load_space(args.group)
    ids = [str(c) for c in space.concept_ids]
    base_units = [u for u in units(args.base) if u["arm"] == "uncond"]
    valid_b, phi_b = zip(*[phi_of(u, space) for u in base_units])
    valid_b, phi_b = np.concatenate(valid_b), np.concatenate(phi_b)
    chance = ((phi_b < CORE) & valid_b[:, None]).mean(0)
    print(f"base pool: {len(valid_b)} graphs, validity {valid_b.mean():.3f}; chance rate of a concept match: mean {chance.mean():.3f}")
    keep = [i for i, c in enumerate(ids) if c not in set(args.exclude)]
    print(f"selection concepts: {len(keep)} of {len(ids)} (cells excluded: {args.exclude})\n")
    sel = f"match({len(keep)})"
    print(f"{'adapter':>10} {'validity':>9} {sel:>10} {'lift(sel)':>9} {'match(all)':>11} {'lift(all)':>10} {'concepts reached (lift>0.05)':>30}")
    for spec in args.roots:
        name, root = spec.split("=")
        rate, val = np.full(len(ids), np.nan), []
        for u in units(root):
            if u["arm"] != "concept":
                continue
            valid, phi = phi_of(u, space)
            i = ids.index(u["concept"])
            rate[i] = ((phi[:, i] < CORE) & valid).mean()
            val.append(valid.mean())
        lift = rate - chance
        print(f"{name:>10} {np.mean(val):9.3f} {np.nanmean(rate[keep]):10.3f} {np.nanmean(lift[keep]):9.3f} {np.nanmean(rate):11.3f} {np.nanmean(lift):10.3f} "
              f"{int(np.nansum(lift > 0.05)):>14d} of {int(np.sum(~np.isnan(rate)))}")


if __name__ == "__main__":
    main()
