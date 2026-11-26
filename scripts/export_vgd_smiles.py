"""Write the SMILES of a visual graph dataset (for example ``zinc64k``) to a CSV, in the dataset's own order.

The molecular tables of the paper are built from the public datasets (``experiments/prepare_concept_dataset.py`` takes a
table with a SMILES column), and the zoo does not redistribute them. This reads every ``*.json`` element of a dataset
folder, in sorted order, and writes ``smiles`` (and the name and the first target value when the elements have them).
The dataset is downloaded on first use unless ``--dataset`` names a local copy.

    python scripts/export_vgd_smiles.py --name zinc64k --out tables/zinc/molecules.csv
"""
import argparse
import csv
import glob
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="zinc64k", help="name of the dataset in the visual graph datasets repository")
    ap.add_argument("--dataset", default=None, help="a local copy of the dataset folder (skips the download)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.dataset is None:
        from visual_graph_datasets.web import ensure_dataset
        args.dataset = ensure_dataset(args.name)

    files = sorted(glob.glob(os.path.join(args.dataset, "**", "*.json"), recursive=True))
    rows = []
    for path in files:
        element = json.load(open(path))
        if "smiles" not in element:
            continue
        target = element.get("target")
        rows.append((element["smiles"], element.get("name", ""), (target[0] if target else "")))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "name", "target"])
        writer.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} molecules from {len(files)} files")


if __name__ == "__main__":
    main()
