"""The Crippen logP of every molecule of a concept pool, aligned with its rows.

The cells of the logP rung are classed by the pool the adapters were trained on (a cell is native when enough of a concept's
core members lie near the requested logP), so the scorer needs the logP of each pool molecule. The property is exact (RDKit's
Crippen method of the decoded molecule), not predicted.

    python scripts/pool_logp.py --conditions tables/zinc/concept_conditions.npz --out tables/zinc/logp.npy
"""
import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", required=True, help="concept_conditions.npz of experiments/prepare_concept_dataset.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen

    from defog_megan.molecules import warn_if_rdkit_differs

    warn_if_rdkit_differs("the pool's logP")

    RDLogger.DisableLog("rdApp.*")
    smiles = [str(s) for s in np.load(args.conditions, allow_pickle=True)["smiles"]]
    logp = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)
    np.save(args.out, logp)
    print(f"wrote {args.out}: {len(logp)} values, mean {logp.mean():.3f} sd {logp.std():.3f}")


if __name__ == "__main__":
    main()
