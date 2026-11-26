"""An explainer bundle of the model zoo, loaded with the scoring settings that its own concept table records.

A membership vector is only defined together with the settings it was computed with (fidelity threshold, how a molecule
is scored against the clustering, the sharpening). The research code re-specified those settings at every place that built
a concept space; the concept table written next to each bundle recorded them and nothing read it. This module reads them
from there, so a bundle always means what it was trained and scored as.

Besides the loader there is a reference check: a bundle's explainer is run on a few of the graphs of the bundle's own
reference table and must reproduce the stored memberships. A different explainer release, a different processing module
or a corrupted file shows up there, where nothing downstream would show it.
"""
from __future__ import annotations

import contextlib
import json
import random
import typing as t
from pathlib import Path

import numpy as np
import torch

from defog_megan.zoo import zoo_path

#: the scoring settings a concept table records, in the order ``ConceptSpace`` takes them
SCORING_KEYS = ("fidelity_threshold", "dataset_type", "score_method", "score_k", "sharpen", "sharpen_temperature")

#: where each part of an explainer bundle sits inside its group of the zoo
FILES = {
    "model": "megan/model.ckpt",
    "clustering": "megan/clustering.clu",
    "processing": "megan/process.py",
    "table": "data/concept_table.json",
    "reference": "data/concept_conditions.npz",
    "pool": "data/pool.npz",
}


def bundle_paths(group: str) -> t.Dict[str, str]:
    """The zoo paths of the parts of a group's explainer bundle."""
    return {name: zoo_path(group, relative) for name, relative in FILES.items()}


def read_scoring_spec(table_path: t.Union[str, Path]) -> dict:
    """The scoring settings recorded in a concept table (``spec`` section), as keyword arguments for ``ConceptSpace``."""
    spec = json.loads(Path(table_path).read_text())["spec"]
    missing = [key for key in SCORING_KEYS if key not in spec]
    if missing:
        raise KeyError(f"the concept table {table_path} does not record {missing}")
    return {key: spec[key] for key in SCORING_KEYS}


def load_space(group: str, **overrides):
    """The ``ConceptSpace`` of a group's explainer bundle, with the settings its concept table records.

    ``overrides`` replace single settings (for an experiment that deliberately scores differently); nothing else is
    guessed. Needs the explainer stack (the ``megan`` extra).
    """
    from defog_megan.torch_compat import load_concept_space

    paths = bundle_paths(group)
    settings = {**read_scoring_spec(paths["table"]), **overrides}
    return load_concept_space(paths["model"], paths["clustering"], paths["processing"], **settings)


@contextlib.contextmanager
def preserved_rng():
    """Leave the global random streams exactly as they were found.

    The explainer builds a data loader on every call, and a data loader draws its base seed from the global torch
    generator, so merely *asking* the explainer moves the stream of whatever samples afterwards. A check that guards a
    run must not change that run.
    """
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def reference_check(space, group: str, n: int = 64, tolerance: float = 1e-5,
                    reference: t.Optional[str] = None, pool: t.Optional[str] = None,
                    property_kind: t.Optional[str] = None, property_tolerance: float = 1e-4) -> float:
    """Recompute the memberships of ``n`` evenly spread rows of the group's reference table, return the largest deviation.

    Raises ``AssertionError`` if the concept order differs from the table's or the deviation exceeds ``tolerance``
    (the bound the research gated on before every steered run). ``reference`` and ``pool`` name other files than the
    zoo's, for an experiment that was pointed at a bundle of its own. The global random streams are left as they were.

    With ``property_kind`` (``"class_probability"`` or ``"regression"``, see ``evaluate.property_readout``) and a ``pred``
    array in the reference, the explainer's predicted property must reproduce it within ``property_tolerance`` as well:
    an energy that asks the explainer for the property needs that to be right, too. The return value is the largest of
    the two deviations.
    """
    with preserved_rng():
        return _reference_check(space, group, n, tolerance, reference, pool, property_kind, property_tolerance)


def _reference_check(space, group, n, tolerance, reference, pool, property_kind, property_tolerance) -> float:
    paths = bundle_paths(group)
    reference = np.load(reference or paths["reference"], allow_pickle=False)
    assert list(space.concept_ids) == [str(c) for c in reference["concept_ids"]], "the concept order differs"
    total = reference["phi"].shape[0]
    rows = sorted({int(i) for i in np.linspace(0, total - 1, num=min(n, total))})
    if "smiles" in reference.files:                       # the molecular groups
        computed = space.phi([str(reference["smiles"][i]) for i in rows])
    else:                                                 # colored graphs: rows of the pool
        from defog_megan.colorgraphs import load_pool_npz, pyg_to_vgd
        graphs, _ = load_pool_npz(pool or paths["pool"])
        computed = space.phi_from_graphs([pyg_to_vgd(graphs[i]) for i in rows])
    deviation = float(np.abs(computed - reference["phi"][rows]).max())
    assert deviation <= tolerance, f"{group}: the explainer deviates from the reference table by {deviation:.3e}"
    if property_kind is not None and "pred" in reference.files and "smiles" in reference.files:
        from defog_megan.evaluate import property_readout
        predicted = np.asarray(property_readout(property_kind)(space, [str(reference["smiles"][i]) for i in rows], 256))
        gap = float(np.abs(predicted - reference["pred"][rows]).max())
        assert gap <= property_tolerance, f"{group}: the predicted property deviates from the reference by {gap:.3e}"
        deviation = max(deviation, gap)
    return deviation
