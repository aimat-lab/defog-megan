"""
Tests for the ``φ`` pipeline.

The pure functions are tested here. The load-bearing test, that recomputing ``φ`` with a bundle's explainer
reproduces the bundle's own reference table, runs against the model zoo in ``test_bundle.py``.
"""

import numpy as np
import pytest

from defog_megan.concepts import (
    ConceptSpace,
    compute_fidelity,
    parse_concept_id,
    sort_concept_ids,
)

# -- pure functions (no artifacts needed) ------------------------------------

def test_parse_concept_id():
    assert parse_concept_id("ch0_cl7") == (0, 7)
    assert parse_concept_id("ch1_cl41") == (1, 41)


def test_sort_concept_ids_is_channel_then_index():
    # Deliberately shuffled, including the lexicographic trap cl10 < cl9.
    ids = ["ch1_cl41", "ch0_cl10", "ch0_cl9", "ch1_cl31", "ch0_cl0"]
    assert sort_concept_ids(ids) == ["ch0_cl0", "ch0_cl9", "ch0_cl10", "ch1_cl31", "ch1_cl41"]


def test_compute_fidelity_regression_negates_channel_zero():
    deviation = np.array([[-2.0, 3.0]])          # (O=1, K=2)
    assert np.allclose(compute_fidelity(deviation, "regression"), [2.0, 3.0])


def test_compute_fidelity_rejects_unknown_dataset_type():
    with pytest.raises(ValueError):
        compute_fidelity(np.zeros((1, 2)), "clustering")
