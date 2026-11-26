"""Parity gate of the experiments: run each experiment in testing mode and reproduce the research code's outputs, bit for bit.

Opt-in (marker ``parity``): it needs the model zoo, the explainer stack and the torch build of the golden file, and takes
about half an hour on one CPU core. The cases are tiny (a few graphs, 20 steps) but run every code path of the
experiment: the adapters, the composition, the codec, Feynman-Kac steering with the explainer as its energy.

    DEFOG_MEGAN_ZOO=<zoo> OMP_NUM_THREADS=1 python -m pytest -m parity tests/parity/test_experiment_golden.py

The digests are bit-exact only within one torch build on one CPU architecture; another build skips the tests, and
``DEFOG_MEGAN_PARITY_ANY_BUILD=1`` runs them anyway (a mismatch is then not necessarily a defect).
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from digests import digest_run
from experiment_parity import REPO, run

pytestmark = pytest.mark.parity

GOLDEN = Path(__file__).resolve().parent / "golden" / "experiments.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def zoo(golden):
    path = os.environ.get("DEFOG_MEGAN_ZOO")
    if not path or not Path(path).is_dir():
        pytest.skip("DEFOG_MEGAN_ZOO does not name the model zoo")
    torch = pytest.importorskip("torch")
    for module in ("graph_attention_student", "megan_global_explanations", "visual_graph_datasets", "pycomex", "rdkit"):
        pytest.importorskip(module)
    recorded = golden["environment"]["torch"]
    if torch.__version__ != recorded and os.environ.get("DEFOG_MEGAN_PARITY_ANY_BUILD") != "1":
        pytest.skip(f"the golden digests are bit-exact only on torch {recorded}, this is {torch.__version__}")
    return str(path)


def _cases():
    return sorted(json.loads(GOLDEN.read_text())["cases"])


@pytest.mark.parametrize("name", _cases())
def test_the_experiment_reproduces_the_research_output(golden, zoo, name):
    case = golden["cases"][name]
    scratch = Path(tempfile.mkdtemp(prefix="parity-"))
    try:
        folder = run(REPO / "experiments" / f"{case['experiment']}.py", case["arguments"], scratch, [str(REPO / "src")], zoo)
        got = digest_run(folder)
        assert sorted(got) == sorted(case["files"]), "a different set of generated files"
        assert got == case["files"], "the generated outputs differ from the research code's: " + ", ".join(
            k for k in case["files"] if got.get(k) != case["files"][k])
        manifest = json.loads((folder / "manifest.json").read_text())
        calls = {u["key"]: u.get("energy_calls") for u in manifest.get("units", []) if u.get("energy_calls")}
        assert calls == case["energy_calls"], "a different number of explainer calls"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
