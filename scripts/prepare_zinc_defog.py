"""Prepare the ZINC molecules of the logP rung (Experiment 3): the molecules the ZINC base represents, decoded.

The paper's ZINC molecules are not the strings of the public ``zinc64k`` dataset but what the base makes of them. Each
molecule is encoded into the base's graph vocabulary (nine neutral elements, kekulized bonds, 6 to 38 heavy atoms) and
decoded again, which drops formal charges and stereochemistry, and a decoded molecule that an earlier one already produced
is dropped, keeping the first. What is left is the corpus the adapters train on, the one the novelty column of the scorer
compares against and the one ``experiments/prepare_concept_dataset.py`` computes the memberships of. It is a fixed point of
the codec, so that experiment changes none of its molecules.

Each molecule is given with the exact Crippen logP of the decoded molecule (the property of the rung is exact, not
predicted). The logP depends on the RDKit release, see ``defog_megan.molecules.PAPER_RDKIT``.

The source is the ``zinc64k`` set of the visual graph datasets package (downloaded on first use unless ``--dataset`` names a
local copy). The tables written below hold molecules, so they are not part of the zoo.

Writes into ``--out``
  molecules.csv          smiles, logp                 the corpus, in the dataset's order (the file is sorted by element file name)
  molecules_prep.json    the counts: source molecules, what was dropped and why, the molecules kept

    python scripts/prepare_zinc_defog.py --out tables/zinc
"""
import argparse
import collections
import glob
import json
import os
import typing as t

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

from defog_megan.molecules import (
    KEKULE_BONDS, NEUTRAL_KEK_ATOMS, build_encoders, mol_to_smiles, pyg_data_to_mol, smiles_to_pyg_data, warn_if_rdkit_differs,
)

RDLogger.DisableLog("rdApp.*")

#: the frozen ZINC base: its vocabulary, and the numbers of heavy atoms it supports (the settings of
#: ``experiments/prepare_concept_dataset.py``, which reads this corpus next)
ATOM_TYPES = list(NEUTRAL_KEK_ATOMS)
BOND_TYPES = list(KEKULE_BONDS)
MIN_NODES = 6
MAX_NODES = 38


def read_source(dataset: str) -> t.List[str]:
    """The SMILES of the elements of a visual graph dataset folder (every ``*.json`` with a ``smiles``), in sorted file order."""
    smiles = []
    for path in sorted(glob.glob(os.path.join(dataset, "**", "*.json"), recursive=True)):
        with open(path) as handle:
            element = json.load(handle)
        if "smiles" in element:
            smiles.append(element["smiles"])
    return smiles


def decode_corpus(source: t.Iterable[str]) -> t.Tuple[t.List[str], collections.Counter]:
    """Round-trip every molecule through the base's vocabulary and keep each decoded molecule once, in order of first occurrence.

    :returns: the decoded canonical SMILES, and the counts of what happened to the source molecules (``kept`` and the reasons
        for dropping the others)
    """
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(ATOM_TYPES, BOND_TYPES)
    counts: collections.Counter = collections.Counter()
    seen = set()
    corpus = []
    for smiles in source:
        counts["source"] += 1
        reference = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if reference is not None:
            n_atoms = reference.GetNumAtoms()
            if n_atoms > MAX_NODES:
                counts["more than 38 heavy atoms"] += 1
                continue
            if n_atoms < MIN_NODES:
                counts["fewer than 6 heavy atoms"] += 1
                continue
        try:
            data = smiles_to_pyg_data(smiles, atom_encoder, bond_encoder, kekulize=True)
        except Exception:
            data = None
        if data is None:
            counts["not encodable"] += 1
            continue
        mol = pyg_data_to_mol(data, atom_decoder, bond_decoder)
        decoded = mol_to_smiles(mol) if mol is not None else None
        if decoded is None or Chem.MolFromSmiles(decoded) is None:
            counts["not decodable"] += 1
            continue
        if decoded in seen:
            counts["repeats an earlier decoded molecule"] += 1
            continue
        seen.add(decoded)
        corpus.append(decoded)
    counts["kept"] = len(corpus)
    return corpus, counts


def crippen_logp(smiles: t.Iterable[str]) -> t.List[float]:
    return [float(Crippen.MolLogP(Chem.MolFromSmiles(s))) for s in smiles]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="zinc64k", help="name of the dataset in the visual graph datasets repository")
    ap.add_argument("--dataset", default=None, help="a local copy of the dataset folder (skips the download)")
    ap.add_argument("--out", required=True, help="the folder to write molecules.csv and molecules_prep.json into")
    args = ap.parse_args()
    if args.dataset is None:
        from visual_graph_datasets.web import ensure_dataset
        args.dataset = ensure_dataset(args.name)

    warn_if_rdkit_differs("the corpus logP")
    source = read_source(args.dataset)
    corpus, counts = decode_corpus(source)
    logp = crippen_logp(corpus)

    import rdkit

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "molecules.csv"), "w") as handle:
        handle.write("smiles,logp\n")
        for smiles, value in zip(corpus, logp):
            handle.write(f"{smiles},{value!r}\n")
    report = {"dataset": args.name, "counts": dict(counts), "atom_types": ATOM_TYPES, "min_nodes": MIN_NODES, "max_nodes": MAX_NODES,
              "rdkit": rdkit.__version__}
    with open(os.path.join(args.out, "molecules_prep.json"), "w") as handle:
        json.dump(report, handle, indent=1)
    dropped = ", ".join(f"{count} {reason}" for reason, count in counts.items() if reason not in ("source", "kept"))
    print(f"wrote {os.path.join(args.out, 'molecules.csv')}: {len(corpus)} molecules from {counts['source']} source molecules"
          + (f" (dropped: {dropped})" if dropped else ""))


if __name__ == "__main__":
    main()
