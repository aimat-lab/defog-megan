"""The ZINC corpus step of the table chain (``scripts/prepare_zinc_defog.py``).

The paper's ZINC molecules are the decoded molecules of the public dataset, each kept once. The expected strings below are
written by hand from the chemistry (a stereocentre and a charge are not represented, the order is the order of first
occurrence), not computed with the code under test.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import Crippen

ROOT = Path(__file__).resolve().parents[1]

#: (file, element) of a small dataset folder, in the order the script reads them (sorted by path)
ELEMENTS = [
    ("a/00.json", {"smiles": "CC(=O)Nc1ccccc1"}),                 # kept as it is
    ("a/01.json", {"smiles": "C[C@H](N)c1ccccc1"}),               # kept, its stereocentre is not represented
    ("a/02.json", {"smiles": "C[C@@H](N)c1ccccc1"}),              # the enantiomer: the same decoded molecule again
    ("b/00.json", {"smiles": "C[NH2+]Cc1ccccc1"}),                # kept, the formal charge is dropped
    ("b/01.json", {"smiles": "CNCc1ccccc1"}),                     # the neutral amine: the same decoded molecule again
    ("b/02.json", {"smiles": "CC"}),                              # below the six heavy atoms of the base
    ("b/03.json", {"smiles": "OB(O)c1ccccc1"}),                   # boron is not in the base's vocabulary
    ("b/04.json", {"smiles": "not a molecule"}),                  # cannot be parsed
    ("c/00.json", {"name": "an element without a molecule"}),     # not a molecule at all, not counted
    ("c/01.json", {"smiles": "C" * 39}),                          # above the 38 heavy atoms of the base
    ("c/02.json", {"smiles": "CCc1ccc[nH]1"}),                    # kept, the N-H of the pyrrole survives the kekulized codec
]
EXPECTED = ["CC(=O)Nc1ccccc1", "CC(N)c1ccccc1", "CNCc1ccccc1", "CCc1ccc[nH]1"]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("prepare_zinc_defog_for_test", ROOT / "scripts" / "prepare_zinc_defog.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def dataset(tmp_path):
    folder = tmp_path / "dataset"
    for name, element in ELEMENTS:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(element))
    return folder


def test_source_is_read_in_sorted_file_order_and_only_molecules(script, dataset):
    assert script.read_source(str(dataset)) == [e["smiles"] for _, e in ELEMENTS if "smiles" in e]


def test_corpus_is_the_decoded_molecules_each_kept_once_in_order(script, dataset):
    corpus, counts = script.decode_corpus(script.read_source(str(dataset)))
    assert corpus == EXPECTED
    assert dict(counts) == {
        "source": 10, "kept": 4, "repeats an earlier decoded molecule": 2, "fewer than 6 heavy atoms": 1,
        "more than 38 heavy atoms": 1, "not encodable": 2,
    }


def test_corpus_is_a_fixed_point_of_the_codec(script):
    corpus, counts = script.decode_corpus(EXPECTED)
    assert corpus == EXPECTED
    assert dict(counts) == {"source": 4, "kept": 4}


def test_first_occurrence_wins_whatever_the_order(script):
    corpus, _ = script.decode_corpus(["CNCc1ccccc1", "C[NH2+]Cc1ccccc1", "CC(=O)Nc1ccccc1"])
    assert corpus == ["CNCc1ccccc1", "CC(=O)Nc1ccccc1"]


def test_written_table_has_the_exact_crippen_logp(script, dataset, tmp_path, monkeypatch):
    out = tmp_path / "tables"
    monkeypatch.setattr(sys, "argv", ["prepare_zinc_defog.py", "--dataset", str(dataset), "--out", str(out)])
    script.main()
    lines = (out / "molecules.csv").read_text().splitlines()
    assert lines[0] == "smiles,logp"
    rows = list(csv.DictReader(lines))
    assert [row["smiles"] for row in rows] == EXPECTED
    for row in rows:
        assert float(row["logp"]) == Crippen.MolLogP(Chem.MolFromSmiles(row["smiles"]))
        assert row["logp"] == repr(float(row["logp"]))                      # the full double, as the paper's table has it
    report = json.loads((out / "molecules_prep.json").read_text())
    assert report["counts"]["kept"] == len(EXPECTED) and report["counts"]["source"] == 10
    assert report["min_nodes"] == 6 and report["max_nodes"] == 38
