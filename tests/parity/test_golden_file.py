"""The golden file is a published artifact: check its structure, and that it agrees with the manifest and the definitions.

These tests need neither torch nor the checkpoints, so they run in the ordinary suite. They cannot show that a port
reproduces the golden digests (that is ``test_parity.py``, which is opt-in); they show that the golden file is whole,
that it describes the files the zoo manifest describes, and that nobody changed a case or a setting without regenerating.
"""
import json
import re
from pathlib import Path

import pytest

from defog_megan.zoo import Manifest
from parity_constants import CASES, FILES, GROUP, SETTINGS, TIERS

GOLDEN = Path(__file__).resolve().parent / "golden" / "colorgraphs.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def test_the_file_describes_the_colored_graphs_and_names_its_reference(golden):
    assert golden["format"] == 2 and golden["group"] == GROUP
    reference = golden["reference"]
    assert reference["package"] == "defog" and re.match(r"^[0-9a-f]{7,40}$", reference["revision"])


def test_the_environment_is_recorded_because_bit_exactness_depends_on_it(golden):
    env = golden["environment"]
    assert {"python", "torch", "numpy", "threads", "machine", "system"} <= set(env)
    assert env["threads"] == 1


def test_the_quick_tier_is_always_present(golden):
    assert "quick" in golden["tiers"]


@pytest.mark.parametrize("tier", ["quick", "paper"])
def test_a_tier_that_is_present_is_complete_and_consistent(golden, tier):
    if tier not in golden["tiers"]:
        pytest.skip(f"the {tier} tier has not been generated yet")
    section = golden["tiers"][tier]
    assert section["config"] == TIERS[tier], "a tier's steps or batch sizes changed without regenerating it"
    assert set(section["cases"]) == set(CASES)
    for name, entry in section["cases"].items():
        assert SHA256.match(entry["digest"]), name
        assert len(entry["sizes"]) == TIERS[tier]["graphs"][name], f"{name}: one size per generated graph"
        assert all(isinstance(n, int) and n > 0 for n in entry["sizes"]), name
        assert entry["pins"] == CASES[name][1], f"{name}: the description changed without regenerating"


@pytest.mark.parametrize("tier", ["quick", "paper"])
def test_the_cases_of_a_tier_are_distinct_runs(golden, tier):
    """A golden file in which two cases collapsed to one digest would pin nothing about one of them."""
    if tier not in golden["tiers"]:
        pytest.skip(f"the {tier} tier has not been generated yet")
    digests = [entry["digest"] for entry in golden["tiers"][tier]["cases"].values()]
    assert len(set(digests)) == len(digests)


def test_the_settings_in_the_file_are_the_definitions_in_the_code(golden):
    assert golden["settings"] == SETTINGS


def test_the_golden_inputs_are_the_files_of_the_manifest(golden):
    """The checkpoints the golden digests were produced with are, by hash, the ones the zoo manifest ships as core."""
    manifest = Manifest.load()
    by_path = {f.path: f for f in manifest.files(GROUP)}
    assert set(golden["inputs"]) == set(FILES.values())
    for path, digest in golden["inputs"].items():
        assert path in by_path, f"{path} is not in the manifest"
        assert by_path[path].sha256 == digest, f"{path}: golden and manifest disagree about the file"
        assert by_path[path].ship == "core"


def test_the_pinned_settings_are_the_papers_explicit_choices():
    """Guards the three settings whose defaults differ between packages or were launcher overrides in the research."""
    assert SETTINGS["compose_mode"] == "product"
    assert SETTINGS["blend_space"] == "prob"
    assert SETTINGS["fk"]["resample_scheme"] == "multinomial"
    assert SETTINGS["eta"] == 0.0 and SETTINGS["omega"] == 0.0 and SETTINGS["time_distortion"] == "polydec"
    assert TIERS["paper"]["steps"]["joint_product"] == 500 and TIERS["paper"]["graphs"]["fk_joint"] == 16
