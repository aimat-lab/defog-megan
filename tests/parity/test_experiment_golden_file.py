"""The experiment golden file is a published artifact: check its structure and the digests it relies on.

These tests need neither torch nor the zoo, so they run in the ordinary suite. They cannot show that an experiment reproduces
the digests (that is ``test_experiment_golden.py``, opt-in); they show that the file is whole, that it covers every case, and
that the digest function notices what it must.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from cases_experiments import CASES
from digests import digest_file, digest_run

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden" / "experiments.json"
EXPERIMENTS = HERE.parents[1] / "experiments"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def test_the_file_records_the_environment_it_is_bit_exact_in(golden):
    env = golden["environment"]
    assert {"python", "torch", "rdkit", "threads", "device"} <= set(env) and env["threads"] == 1 and env["device"] == "cpu"


def test_every_case_is_in_the_file_and_nothing_else(golden):
    assert sorted(golden["cases"]) == sorted(CASES), "a case was added or changed without regenerating the golden file"


def test_each_case_names_an_experiment_and_its_arguments_as_the_gate_runs_them(golden):
    for name, case in golden["cases"].items():
        assert (EXPERIMENTS / f"{case['experiment']}.py").is_file(), name
        assert [case["experiment"], case["arguments"]] == [CASES[name][0], CASES[name][1]], name
        assert "--__TESTING__" in case["arguments"], f"{name}: the gate runs the tiny testing mode"


def test_every_digest_is_a_sha256_and_every_case_has_files(golden):
    for name, case in golden["cases"].items():
        assert case["files"], name
        assert all(SHA256.match(d) and re.match(r"^gen_.*\.(npz|json)$", f) for f, d in case["files"].items()), name


def test_the_steered_cases_record_how_often_the_explainer_was_called(golden):
    for name in ("zinc-logp/fk", "ames/fk"):
        assert golden["cases"][name]["energy_calls"], f"{name}: the number of explainer calls is part of what is reproduced"


def test_the_digest_of_a_graph_file_notices_one_changed_value_and_the_order_of_nothing(tmp_path):
    arrays = {"node_classes": np.arange(12, dtype=np.int64), "targets": np.array([1.0, np.nan, 3.0])}
    np.savez(tmp_path / "gen_a.npz", **arrays)
    np.savez(tmp_path / "gen_same.npz", **dict(reversed(list(arrays.items()))))
    arrays["node_classes"] = arrays["node_classes"].copy()
    arrays["node_classes"][5] += 1
    np.savez(tmp_path / "gen_changed.npz", **arrays)
    assert digest_file(tmp_path / "gen_a.npz") == digest_file(tmp_path / "gen_same.npz")      # NaN included, order of arrays not
    assert digest_file(tmp_path / "gen_a.npz") != digest_file(tmp_path / "gen_changed.npz")


def test_the_digest_of_a_smiles_file_notices_one_changed_molecule_and_ignores_the_rest(tmp_path):
    (tmp_path / "gen_x.json").write_text(json.dumps({"key": "x", "smiles": ["CCO", None, "c1ccccc1"]}))
    (tmp_path / "gen_y.json").write_text(json.dumps({"key": "other key", "smiles": ["CCO", None, "c1ccccc1"]}))
    (tmp_path / "gen_z.json").write_text(json.dumps({"key": "x", "smiles": ["CCO", None, "c1ccccc2"]}))
    assert digest_file(tmp_path / "gen_x.json") == digest_file(tmp_path / "gen_y.json")
    assert digest_file(tmp_path / "gen_x.json") != digest_file(tmp_path / "gen_z.json")
    (tmp_path / "gen_x.phi.npy").write_bytes(b"not a digest input")
    assert "gen_x.phi.npy" not in digest_run(tmp_path)                                        # caches are not outputs
