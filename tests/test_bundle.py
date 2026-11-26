"""Reading a bundle's scoring contract, and the reference check against the zoo's own memberships."""
import json
from pathlib import Path

import numpy as np
import pytest

from defog_megan.bundle import FILES, SCORING_KEYS, bundle_paths, read_scoring_spec
from defog_megan.zoo import ENV_VAR, Manifest, zoo_root

GROUPS = ("colorgraphs", "zinc-logp", "ames")


def write_table(path, **spec):
    path.write_text(json.dumps({"spec": spec, "table": []}))
    return path


GOOD = dict(fidelity_threshold=0.75, dataset_type="classification", score_method="knn", score_k=5, sharpen="sparsemax",
            sharpen_temperature=0.5, num_concepts=29)


def test_the_scoring_settings_are_read_from_the_table_and_nothing_else(tmp_path):
    table = write_table(tmp_path / "t.json", **GOOD)
    spec = read_scoring_spec(table)
    assert set(spec) == set(SCORING_KEYS)                    # the bookkeeping field num_concepts is not a setting
    assert spec["fidelity_threshold"] == 0.75 and spec["dataset_type"] == "classification"


@pytest.mark.parametrize("missing", SCORING_KEYS)
def test_a_table_that_does_not_record_a_setting_is_refused_not_guessed(tmp_path, missing):
    spec = {k: v for k, v in GOOD.items() if k != missing}
    with pytest.raises(KeyError, match=missing):
        read_scoring_spec(write_table(tmp_path / "t.json", **spec))


def test_the_parts_of_a_bundle_are_the_files_the_manifest_lists(monkeypatch, tmp_path):
    """The bundle's layout and the manifest agree for every group, so a verified zoo is a loadable zoo."""
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    manifest = Manifest.load()
    for group in GROUPS:
        listed = {f.path for f in manifest.files(group)}
        for part in ("model", "clustering", "processing", "table"):
            assert FILES[part] in listed, f"{group}: {FILES[part]} is not in the manifest"
        assert Path(bundle_paths(group)["model"]) == tmp_path / group / "megan" / "model.ckpt"


# -- against the real zoo (needs the explainer stack and the files; skipped otherwise) --------------------------------

def _zoo_has(group, *parts):
    return all((zoo_root() / group / FILES[p]).is_file() for p in parts)


@pytest.fixture(scope="module")
def explainer_stack():
    pytest.importorskip("graph_attention_student")
    pytest.importorskip("megan_global_explanations")
    pytest.importorskip("visual_graph_datasets")


@pytest.mark.parametrize("group", GROUPS)
def test_the_explainer_reproduces_the_reference_memberships(explainer_stack, group):
    """The load-bearing test: recomputed memberships equal the table the adapters were trained against. If this ever
    fails, a condition no longer means what the clustering says it means, and nothing downstream would show it."""
    if not _zoo_has(group, "model", "clustering", "processing", "table", "reference"):
        pytest.skip(f"the zoo (or its reference table) is not available for {group}")
    if group == "colorgraphs" and not _zoo_has(group, "pool"):
        pytest.skip("the colored-graph pool is not available")
    from defog_megan.bundle import load_space, reference_check
    space = load_space(group)
    deviation = reference_check(space, group, n=32)
    assert deviation <= 1e-5
    assert space.num_concepts == {"colorgraphs": 19, "zinc-logp": 42, "ames": 29}[group]
