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


# -- the gate must not move the run it guards ---------------------------------------------------------------------------

def test_the_reference_check_leaves_the_random_streams_untouched(tmp_path):
    """The explainer draws from the global torch generator (it builds a data loader per call). A check that runs before a
    sampling run must not shift that run's random numbers: found when the FK arms of a ported experiment stopped being
    bit-identical to the research code's, because the new gate called the explainer once before sampling."""
    import random

    import torch

    from defog_megan.bundle import preserved_rng, reference_check
    from defog_megan.colorgraphs import RB_PALETTE, save_pool_npz, vgd_to_pyg

    graphs = [vgd_to_pyg({"node_attributes": [list(RB_PALETTE[i % 7]) for i in range(3 + k)],
                          "edge_indices": [[i, i + 1] for i in range(2 + k)] + [[i + 1, i] for i in range(2 + k)],
                          "edge_attributes": [[1.0]] * (4 + 2 * k)}) for k in range(3)]
    pool = tmp_path / "pool.npz"
    save_pool_npz(str(pool), graphs, [{} for _ in graphs])
    phi = np.array([[0.1, 0.9], [0.5, 0.5], [1.0, 0.0]])
    reference = tmp_path / "reference.npz"
    np.savez(reference, concept_ids=np.array(["a", "b"]), phi=phi)

    class Space:                                     # an explainer that, like the real one, consumes the global generator
        concept_ids = ["a", "b"]

        def phi_from_graphs(self, graphs, batch_size=512):
            torch.rand(1)
            np.random.rand()
            random.random()
            return phi[: len(graphs)]

    torch.manual_seed(1), np.random.seed(1), random.seed(1)
    before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    deviation = reference_check(Space(), "colorgraphs", n=3, reference=str(reference), pool=str(pool))
    after = (torch.get_rng_state(), np.random.get_state()[1], random.getstate())
    assert deviation == 0.0
    assert torch.equal(before[0], after[0]) and np.array_equal(before[1], after[1]) and before[2] == after[2]

    with preserved_rng():                            # and the helper itself, on a stream that is moved inside it
        torch.rand(5)
    assert torch.equal(torch.get_rng_state(), after[0])


def test_the_reference_check_also_gates_the_predicted_property(tmp_path, monkeypatch):
    """The Ames energy asks the explainer for the property, so the gate checks the prediction as well as the memberships."""
    from defog_megan import evaluate
    from defog_megan.bundle import reference_check

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    phi = np.array([[0.1, 0.9], [0.5, 0.5], [1.0, 0.0], [0.3, 0.7]])
    pred = np.array([0.1, 0.8, 0.3, 0.6])
    reference = tmp_path / "reference.npz"
    np.savez(reference, concept_ids=np.array(["a", "b"]), smiles=np.array(smiles), phi=phi, pred=pred)

    class Space:
        concept_ids = ["a", "b"]

        def phi(self, batch, batch_size=512):
            return np.array([phi[smiles.index(s)] for s in batch])

    served = {"offset": 0.0}
    monkeypatch.setattr(evaluate, "property_readout",
                        lambda kind: (lambda space, batch, batch_size: np.array([pred[smiles.index(s)] + served["offset"] for s in batch])))
    assert reference_check(Space(), "ames", n=4, reference=str(reference), property_kind="class_probability") == 0.0
    # without the property kind the prediction is not looked at
    served["offset"] = 0.5
    assert reference_check(Space(), "ames", n=4, reference=str(reference)) == 0.0
    with pytest.raises(AssertionError, match="predicted property"):
        reference_check(Space(), "ames", n=4, reference=str(reference), property_kind="class_probability")
