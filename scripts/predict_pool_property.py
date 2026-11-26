"""MEGAN's predicted property for every molecule of the concept pool.

The real-world rung grades every property by MEGAN's own prediction, so the pool that classes the cells
(native / off-native / unsupported) must carry the same prediction, not the measured value: a cell is native when
enough of a concept's core members lie near the level AS MEGAN SEES THEM. Writes an array aligned with the rows of the
concept pool (``concept_conditions.npz`` of ``experiments/prepare_concept_dataset.py``).

    python scripts/predict_pool_property.py --conditions tables/ames/concept_conditions.npz --out tables/ames/prob_pred.npy
"""
import argparse

import numpy as np

from defog_megan.bundle import load_space
from defog_megan.evaluate import property_readout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="ames", help="the explainer bundle of the zoo")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--kind", default="class_probability", choices=["regression", "class_probability"],
                    help="class_probability: predicted probability of the mutagenic class; regression: predicted value")
    args = ap.parse_args()
    space = load_space(args.group)
    pool = np.load(args.conditions, allow_pickle=True)
    smiles = [str(s) for s in pool["smiles"]]
    pred = property_readout(args.kind)(space, smiles, args.batch)
    if len(pred) != len(smiles):
        raise SystemExit(f"MEGAN encoded {len(pred)} of {len(smiles)} pool molecules")
    np.save(args.out, pred.astype(np.float32))
    print(f"wrote {args.out}: {len(pred)} predictions, mean {pred.mean():.2f} sd {pred.std():.2f}, "
          f"range [{pred.min():.2f}, {pred.max():.2f}]")


if __name__ == "__main__":
    main()
