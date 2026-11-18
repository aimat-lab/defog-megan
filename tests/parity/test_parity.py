"""Parity gate for the colored-graph generator-side cases: the candidate must reproduce the golden digests bit for bit.

Opt-in (marker ``parity``): it needs the model zoo and the research environment's torch build.

    DEFOG_MEGAN_ZOO=<zoo> PYTHONPATH=<defog-adapt>/src:<this repo>/src <research python> -m pytest -m parity
    DEFOG_MEGAN_PARITY_TIER=paper ...                  # the paper's 500 steps and 16 particles; much longer
    DEFOG_MEGAN_PARITY_PACKAGE=<other package> ...     # check another candidate than defog_adapt

The digests are bit-exact only within one torch build on one CPU architecture. A different build skips the tests with
that reason; set ``DEFOG_MEGAN_PARITY_ANY_BUILD=1`` to run anyway (a mismatch is then not necessarily a defect).
"""
import json
import os
from pathlib import Path

import pytest

from parity_constants import CASES

pytestmark = pytest.mark.parity

GOLDEN = Path(__file__).resolve().parent / "golden" / "colorgraphs.json"
TIER = os.environ.get("DEFOG_MEGAN_PARITY_TIER", "quick")
PACKAGE = os.environ.get("DEFOG_MEGAN_PARITY_PACKAGE", "defog_adapt")


@pytest.fixture(scope="module")
def golden():
    data = json.loads(GOLDEN.read_text())
    if TIER not in data["tiers"]:
        pytest.skip(f"the {TIER} tier is not in the golden file")
    return data


@pytest.fixture(scope="module")
def zoo():
    path = os.environ.get("DEFOG_MEGAN_ZOO")
    if not path or not Path(path).is_dir():
        pytest.skip("DEFOG_MEGAN_ZOO does not name the model zoo")
    return Path(path)


@pytest.fixture(scope="module")
def runner(golden, zoo):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip(PACKAGE)
    recorded = golden["environment"]["torch"]
    if torch.__version__ != recorded and os.environ.get("DEFOG_MEGAN_PARITY_ANY_BUILD") != "1":
        pytest.skip(f"the golden digests are bit-exact only on torch {recorded}, this is {torch.__version__}")
    import cases
    assert cases.input_hashes(zoo) == golden["inputs"], "the zoo's files are not the ones the golden file was made with"
    return cases.Runner(PACKAGE, zoo)


@pytest.mark.parametrize("name", list(CASES))
def test_the_candidate_reproduces_the_golden_digest(runner, golden, name):
    expected = golden["tiers"][TIER]["cases"][name]
    got = runner.run(name, TIER)
    assert got["sizes"] == expected["sizes"], f"{name}: different graph sizes ({CASES[name][1]})"
    assert got["digest"] == expected["digest"], f"{name}: the generated graphs differ ({CASES[name][1]})"
