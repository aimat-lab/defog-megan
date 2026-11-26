"""Build the public reference table that checks an explainer bundle of the molecular groups of the zoo.

The explainer must reproduce the membership vectors that the adapters were trained against, or a condition no longer means
what the clustering says it means, and nothing downstream shows it (``defog_megan.bundle.reference_check``). For the
colored graphs the reference is the table of the whole pool. For the molecular groups the tables of the paper hold SMILES of
the datasets, so they are not redistributed; this table holds well-known molecules instead (a few dozen textbook compounds and
structural-alert examples, public facts) with the memberships, and for the Ames group the predicted class probability, that the
bundle's own explainer computes for them.

    python scripts/build_reference_check.py --group ames --out ames_reference_check.npz

It is run once by the maintainer of a zoo, on the release's own explainer environment, and the result is part of the group.
"""
import argparse

import numpy as np

#: (name, SMILES): neutral compounds of both groups
NEUTRAL = [
    ("benzene", "c1ccccc1"), ("toluene", "Cc1ccccc1"), ("phenol", "Oc1ccccc1"), ("aniline", "Nc1ccccc1"),
    ("chlorobenzene", "Clc1ccccc1"), ("naphthalene", "c1ccc2ccccc2c1"), ("pyridine", "c1ccncc1"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"), ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"), ("paracetamol", "CC(=O)Nc1ccc(O)cc1"), ("ethanol", "CCO"),
    ("acetic acid", "CC(=O)O"), ("acetone", "CC(C)=O"), ("urea", "NC(N)=O"), ("glycine", "NCC(=O)O"),
    ("sulfanilamide", "Nc1ccc(cc1)S(N)(=O)=O"), ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc12"),
    ("nicotine", "CN1CCCC1c1cccnc1"), ("benzoic acid", "OC(=O)c1ccccc1"), ("cyclohexane", "C1CCCCC1"),
    ("hexane", "CCCCCC"), ("dimethyl sulfoxide", "CS(C)=O"), ("dimethyl sulfone", "CS(C)(=O)=O"),
    ("methanesulfonamide", "CS(N)(=O)=O"), ("1,4-dioxane", "C1COCCO1"), ("tetrahydrofuran", "C1CCOC1"),
    ("morpholine", "C1COCCN1"), ("piperidine", "C1CCNCC1"), ("imidazole", "c1c[nH]cn1"), ("indole", "c1ccc2[nH]ccc2c1"),
    ("quinoline", "c1ccc2ncccc2c1"), ("thiophene", "c1ccsc1"), ("furan", "c1ccoc1"), ("acetanilide", "CC(=O)Nc1ccccc1"),
    ("benzaldehyde", "O=Cc1ccccc1"), ("acetophenone", "CC(=O)c1ccccc1"), ("styrene", "C=Cc1ccccc1"),
    ("biphenyl", "c1ccc(cc1)-c1ccccc1"), ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("ethylene oxide", "C1CO1"), ("styrene oxide", "C1OC1c1ccccc1"), ("N-nitrosodimethylamine", "CN(C)N=O"),
    ("azobenzene", "c1ccc(cc1)N=Nc1ccccc1"), ("4-aminoazobenzene", "Nc1ccc(cc1)N=Nc1ccccc1"),
    ("2-naphthylamine", "Nc1ccc2ccccc2c1"), ("4-aminobiphenyl", "Nc1ccc(cc1)-c1ccccc1"),
    ("benzo[a]pyrene", "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"), ("acrylamide", "C=CC(N)=O"), ("formaldehyde", "C=O"),
]
#: compounds with charged atoms, which only the charge-aware vocabulary of the Ames group can write
CHARGED = [
    ("nitrobenzene", "[O-][N+](=O)c1ccccc1"), ("2,4-dinitrotoluene", "Cc1ccc(cc1[N+]([O-])=O)[N+]([O-])=O"),
    ("4-nitroaniline", "Nc1ccc(cc1)[N+]([O-])=O"), ("2-nitrofuran", "[O-][N+](=O)c1ccco1"),
    ("metronidazole", "Cc1ncc([N+]([O-])=O)n1CCO"), ("azidobenzene", "[N-]=[N+]=Nc1ccccc1"),
    ("pyridine N-oxide", "[O-][n+]1ccccc1"), ("tetramethylammonium", "C[N+](C)(C)C"),
    ("trimethylamine N-oxide", "C[N+](C)(C)[O-]"), ("benzoate", "O=C([O-])c1ccccc1"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, choices=["zinc-logp", "ames"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from rdkit import Chem

    from defog_megan.bundle import load_space
    from defog_megan.evaluate import predict_class_probability

    molecules = NEUTRAL + (CHARGED if args.group == "ames" else [])
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for _, s in molecules]
    assert len(set(smiles)) == len(smiles), "a duplicate molecule in the list"
    space = load_space(args.group)
    phi = space.phi(smiles, batch_size=64)
    if len(phi) != len(smiles):
        raise SystemExit(f"the explainer encoded {len(phi)} of {len(smiles)} reference molecules")
    arrays = {"concept_ids": np.asarray(space.concept_ids, dtype=np.str_), "smiles": np.asarray(smiles, dtype=np.str_),
              "names": np.asarray([n for n, _ in molecules], dtype=np.str_), "phi": phi}
    if args.group == "ames":
        arrays["pred"] = np.asarray(predict_class_probability(space, smiles, 64), dtype=np.float64)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}: {len(smiles)} molecules x {phi.shape[1]} concepts" + (", with predictions" if "pred" in arrays else ""))


if __name__ == "__main__":
    main()
