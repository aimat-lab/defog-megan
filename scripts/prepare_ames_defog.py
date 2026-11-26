"""Prepare the Ames mutagenicity set for the real-world rung (Experiment 4) under the CHARGE-AWARE base.

A molecule is kept if the charge-aware vocabulary (``charged_kek_v1``: the nine ZINC elements plus N+, O-, N-)
can represent it and the base can generate it:
  * encoding and decoding it gives the same molecule back (stereo is ignored, as the base ignores it),
  * it has at most 38 heavy atoms (the base's max_nodes).
The SMILES written is the canonical decoded form. Where several source molecules collapse onto one graph
(stereoisomers, duplicates) their labels must agree to keep the graph; graphs with conflicting labels are dropped
and counted. ``--vocab kekulized_v2`` runs the same test for the neutral 12-element vocabulary, which gives the coverage
that the charged classes add.

The source is the Ames set of the visual graph datasets package (downloaded on first use unless ``--dataset`` names a
local copy). The tables written below hold molecules, so they are not part of the zoo.

Writes into ``--out``
  ames_defog.csv          smiles, label                 the file MEGAN trains on (label 1 = mutagenic)
  ames_defog_meta.csv     smiles, label, n_src, n_heavy, name, src_smiles
  ames_defog_sizes.json   heavy-atom histogram of the kept graphs (the sampler's size distribution)
  ames_defog_prep_<vocab>.json  the counts quoted in the paper

    python scripts/prepare_ames_defog.py --out tables/ames
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from defog_megan.molecules import (
    CHARGED_KEK_V1_ATOMS, KEKULE_BONDS, build_encoders, pyg_data_to_mol, smiles_to_pyg_data,
)

RDLogger.DisableLog("rdApp.*")
#: the vocabularies whose coverage of the set can be tested: the charge-aware one of the paper's base, and the neutral
#: twelve-element one it is compared with
VOCABULARIES = {
    "charged_kek_v1": list(CHARGED_KEK_V1_ATOMS),
    "kekulized_v2": ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"],
}
MAX_NODES = 38


def canon(m):
    return Chem.MolToSmiles(m, isomericSmiles=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="a local copy of the Ames set (the visual graph dataset \"mutag\"); "
                                                    "downloaded if omitted")
    ap.add_argument("--out", default="tables/ames")
    ap.add_argument("--vocab", default="charged_kek_v1", choices=sorted(VOCABULARIES),
                    help="representation to test coverage against")
    args = ap.parse_args()
    if args.dataset is None:
        from visual_graph_datasets.web import ensure_dataset
        args.dataset = ensure_dataset("mutag")
    atom_types = VOCABULARIES[args.vocab]
    ae, ad, be, bd = build_encoders(atom_types, KEKULE_BONDS)
    elements = {a for a in atom_types if not a.endswith(("+", "-"))}

    rows = []
    for f in sorted(glob.glob(os.path.join(args.dataset, "**", "*.json"), recursive=True)):
        d = json.load(open(f))
        if "smiles" in d and "target" in d:
            rows.append((d.get("name", ""), d["smiles"], int(np.argmax(d["target"]))))
    reasons = collections.Counter()
    kept = []
    for name, smi, label in rows:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            reasons["unparsable"] += 1
        elif len(Chem.GetMolFrags(m)) > 1:
            reasons["several fragments"] += 1
        elif any(a.GetSymbol() not in elements for a in m.GetAtoms()):
            reasons["element outside the vocabulary"] += 1
        elif m.GetNumHeavyAtoms() > MAX_NODES:
            reasons[f"more than {MAX_NODES} heavy atoms"] += 1
        else:
            try:
                d = smiles_to_pyg_data(smi, ae, be, kekulize=True)
                back = pyg_data_to_mol(d, ad, bd) if d is not None else None
                ok = back is not None and canon(Chem.MolFromSmiles(Chem.MolToSmiles(back))) == canon(m)
            except Exception:
                ok = False
            if not ok:
                reasons["changed by the encoding"] += 1
            else:
                reasons["kept"] += 1
                kept.append((canon(m), label, name, smi, m.GetNumHeavyAtoms()))
    print(f"{len(rows)} source molecules, vocabulary {args.vocab}")
    for r, c in reasons.most_common():
        print(f"  {r:40s} {c:6d}  {c / len(rows):6.1%}")
    n_pos = sum(l for _, l, *_ in kept)
    print(f"  kept class balance: {n_pos} mutagenic of {len(kept)}")

    df = pd.DataFrame(kept, columns=["smiles", "label", "name", "src_smiles", "n_heavy"])
    g = df.groupby("smiles", sort=False)
    meta = g.agg(label=("label", "max"), label_min=("label", "min"), n_src=("label", "size"),
                 n_heavy=("n_heavy", "first"), name=("name", "first"), src_smiles=("src_smiles", "first")).reset_index()
    conflict = meta[meta.label != meta.label_min]
    meta = meta[meta.label == meta.label_min].drop(columns="label_min")
    dup = meta[meta.n_src > 1]
    print(f"{len(df)} kept molecules -> {len(meta) + len(conflict)} distinct graphs "
          f"({int((meta.n_src > 1).sum() + len(conflict))} from several sources); "
          f"{len(conflict)} graphs with conflicting labels dropped -> {len(meta)} graphs, "
          f"{int(meta.label.sum())} mutagenic ({meta.label.mean():.1%})")

    os.makedirs(args.out, exist_ok=True)
    hist = collections.Counter(int(n) for n in meta.n_heavy)
    if args.vocab == "charged_kek_v1":
        # the tables of the paper's base; the comparison vocabulary only writes its coverage report
        meta[["smiles", "label"]].to_csv(os.path.join(args.out, "ames_defog.csv"), index=False)
        meta.to_csv(os.path.join(args.out, "ames_defog_meta.csv"), index=False)
        json.dump({"counts": {str(k): hist[k] for k in sorted(hist)}, "n": int(len(meta)),
                   "mean": float(meta.n_heavy.mean()), "max_nodes": MAX_NODES},
                  open(os.path.join(args.out, "ames_defog_sizes.json"), "w"), indent=1)

    n = meta.n_heavy.to_numpy()
    q = lambda a: [float(x) for x in np.percentile(a, [5, 25, 50, 75, 95])]
    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    has_nitro = np.array([Chem.MolFromSmiles(s).HasSubstructMatch(nitro) for s in meta.smiles])
    result = {"vocabulary": args.vocab, "source": len(rows), "reasons": dict(reasons),
              "distinct_graphs": int(len(meta) + len(conflict)), "conflicting_label_graphs": int(len(conflict)),
              "kept_graphs": int(len(meta)), "mutagenic": int(meta.label.sum()),
              "nitro_share": float(has_nitro.mean()), "nitro_share_mutagenic": float(has_nitro[meta.label.to_numpy() == 1].mean()),
              "heavy_atoms": {"mean": float(n.mean()), "quantiles_5_25_50_75_95": q(n),
                              "share_up_to_12": float((n <= 12).mean())}}
    json.dump(result, open(os.path.join(args.out, f"ames_defog_prep_{args.vocab}.json"), "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
