"""The manifest that ships with the package: what it must contain and what it must never contain."""
import json
import re
from importlib import resources

import pytest

from defog_megan.zoo import Manifest
from test_no_private_references import PATTERNS

GROUPS = ("ames", "colorgraphs", "zinc-logp")


@pytest.fixture(scope="module")
def manifest():
    return Manifest.load()


def test_it_has_the_three_groups_of_the_paper(manifest):
    assert sorted(manifest.groups) == sorted(GROUPS)


def test_the_core_download_is_the_size_the_inventory_measured(manifest):
    """About 563 MB across the three groups (254.9, 162.2 and 145.9 MB). A band, not an exact number: a file may
    legitimately move between core and optional, but a group should not silently shrink to nothing."""
    total = manifest.total_bytes(ship=("core",)) / 1e6
    assert 540 < total < 590
    for group, low, high in (("colorgraphs", 240, 270), ("zinc-logp", 150, 175), ("ames", 135, 160)):
        assert low < manifest.total_bytes(group, ("core",)) / 1e6 < high, group


@pytest.mark.parametrize("group", GROUPS)
def test_every_group_has_the_explainer_bundle_in_full(manifest, group):
    """The explainer is useless without its clustering and processing module, and unusable without knowing how good
    it is: the four travel together."""
    paths = {f.path for f in manifest.files(group, ("core",))}
    for needed in ("megan/model.ckpt", "megan/clustering.clu", "megan/process.py"):
        assert needed in paths, f"{group} lacks {needed}"
    assert any(p.startswith("megan/report") for p in paths), f"{group} has no quality report"


@pytest.mark.parametrize("group", GROUPS)
def test_every_adapter_comes_with_a_base_and_the_contracts_that_travel_with_it(manifest, group):
    files = manifest.files(group, ("core",))
    kinds = {f.kind for f in files}
    assert "base" in kinds and "adapter" in kinds
    assert "concept_targets" in kinds and "concept_table" in kinds, \
        f"{group}: the request vectors and the scoring contract must ship with the concept adapter"


def test_the_molecule_groups_ship_the_vocabulary_that_no_checkpoint_stores(manifest):
    zinc = {f.path for f in manifest.files("zinc-logp", ("core",))}
    ames = {f.path for f in manifest.files("ames", ("core",))}
    assert "base.schema.json" in zinc
    assert "base.vocabulary.json" in ames      # the charged vocabulary (N+, O-, N-) is in no checkpoint


def test_nothing_with_molecule_tables_is_core(manifest):
    """The data behind ZINC and Ames models is not ours to redistribute. Everything that holds SMILES is withheld."""
    for f in manifest.files(ship=("core", "optional")):
        assert f.kind not in {"labels", "concept_conditions"} or f.group == "colorgraphs", \
            f"{f.group}/{f.path} holds molecule tables and must be withheld"
    withheld = {(f.group, f.path) for f in manifest.files(ship=("withheld",))}
    assert ("ames", "data/concept_conditions.npz") in withheld
    assert ("zinc-logp", "data/labels.csv") in withheld


def test_the_synthetic_group_ships_its_dataset_and_reference_memberships(manifest):
    """Synthetic data has no third-party terms: the pool and the reference membership table ship, and the latter is
    what a membership computation is checked against."""
    paths = {f.path for f in manifest.files("colorgraphs", ("core",))}
    assert {"data/pool.npz", "data/concept_conditions.npz"} <= paths


def test_the_files_that_are_not_in_hand_are_known_and_few(manifest):
    missing = {(f.group, f.id) for f in manifest.files() if not f.in_hand}
    assert missing == {("colorgraphs", "C3"), ("ames", "A2")}


def test_the_manifest_holds_no_machine_path_account_or_job_number():
    """The manifest is published data. It names files by bundle path only."""
    text = resources.files("defog_megan.zoo").joinpath("manifest.json").read_text(encoding="utf-8")
    values = []

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            values.append(node)

    walk(json.loads(text))
    assert len(values) > 100                      # the walk really reached the entries
    for value in values:
        for label, pattern in PATTERNS.items():   # the same generic patterns the repository scan uses
            assert not re.search(pattern, value, flags=re.IGNORECASE), f"{label} in the manifest: {value!r}"


def test_the_notes_say_what_a_user_must_know_about_the_two_surprising_models(manifest):
    zinc_base = manifest.get("zinc-logp", "L1").note.lower()
    assert "reinforcement" in zinc_base, "the molecular base is RL-fine-tuned; its note must say so"
    assert "no training log" in manifest.get("zinc-logp", "L3").note.lower()
