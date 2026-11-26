"""Build the class-label conditioning table and split for the mutagenicity rung (Experiment 4).

The scalar adapter conditions on the MEASURED Ames label (0 non-mutagenic, 1 mutagenic) of the graphs of
``ames_defog_meta.csv`` (scripts/prepare_ames_defog.py). The tables are in the format of
experiments/train_property_adapter.py:

  ames_defog_label_conditions.npz   smiles, smiles_original, value (= label)
  ames_defog_label_split.json       train indices; dev_targets / report_targets are held-out molecules

(The research code called the array ``logs``; ``--VALUE_KEY '"logs"'`` of the trainer reads such a table.)

    python scripts/prepare_ames_defog_label.py --meta tables/ames/ames_defog_meta.csv --out tables/ames
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEED = 20261104
HELDOUT_FRACTION = 0.05
N_DEV_TARGETS, N_REPORT_TARGETS = 30, 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default="tables/ames/ames_defog_meta.csv")
    ap.add_argument("--out", default="tables/ames")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    OUT_TABLE = os.path.join(args.out, "ames_defog_label_conditions.npz")
    OUT_SPLIT = os.path.join(args.out, "ames_defog_label_split.json")
    meta = pd.read_csv(args.meta)
    smiles = meta.smiles.tolist()
    values = meta.label.to_numpy(dtype=np.float32)
    print(f"{len(smiles)} graphs; mutagenic share {values.mean():.3f}")
    np.savez_compressed(OUT_TABLE, smiles=np.array(smiles, dtype=object),
                        smiles_original=np.array(meta.src_smiles.tolist(), dtype=object), value=values)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(smiles))
    n_held = int(round(HELDOUT_FRACTION * len(smiles)))
    held, train = order[:n_held], order[n_held:]
    report_idx = held[np.random.default_rng(SEED + 1).choice(len(held), size=N_REPORT_TARGETS, replace=False)]
    remaining = np.setdiff1d(held, report_idx)
    dev_idx = remaining[np.random.default_rng(SEED + 2).choice(len(remaining), size=N_DEV_TARGETS, replace=False)]
    split = {
        "seed": SEED, "created": "2026-11-03", "source_table": os.path.basename(OUT_TABLE),
        "rationale": "molecule-level split of the 6,301 representable Ames graphs; train = adapter training rows, the "
                     "held-out 5% never enter training (the rung scores the fixed levels 0 and 1, not these molecules)",
        "heldout_fraction": HELDOUT_FRACTION, "n_train": int(len(train)), "n_heldout": int(n_held),
        "train": sorted(int(i) for i in train),
        "dev_targets": [{"index": int(i), "smiles": smiles[i], "value": float(values[i])} for i in dev_idx],
        "report_targets": [{"index": int(i), "smiles": smiles[i], "value": float(values[i])} for i in report_idx],
    }
    json.dump(split, open(OUT_SPLIT, "w"), indent=2)
    print(f"wrote {OUT_TABLE} and {OUT_SPLIT}: train {len(train)}, held out {n_held}")


if __name__ == "__main__":
    main()
