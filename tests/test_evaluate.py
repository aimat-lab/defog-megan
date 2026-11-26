"""
Tests for the round-trip evaluation metrics.

These use a stub concept space rather than the real MEGAN model: the point is to
pin down the *arithmetic* of hit rate, lift and enrichment (including the empty
and zero-baseline edge cases), which is where a silently wrong number would be
easiest to miss and most damaging to believe.
"""

import numpy as np
import pytest

from defog_megan.evaluate import ASSIGNED, CORE, concept_hit_rate, evaluate_concept


class StubSpace:
    """A ConceptSpace stand-in that returns canned φ rows keyed by SMILES."""

    def __init__(self, scores_by_smiles, concept_ids=("ch0_cl0", "ch0_cl1")):
        self.concept_ids = list(concept_ids)
        self._scores = scores_by_smiles

    def index_of(self, concept_id):
        return self.concept_ids.index(concept_id)

    def phi(self, smiles, batch_size=256):
        return np.array([self._scores[s] for s in smiles], dtype=float)


def test_hit_rate_counts_anything_below_unassigned():
    # 0.2 and 0.8 are assigned; 1.0 is not. Only 0.2 clears the CORE bar.
    space = StubSpace({"a": [0.2, 1.0], "b": [0.8, 1.0], "c": [1.0, 1.0]})
    hit = concept_hit_rate(space, ["a", "b", "c"], "ch0_cl0")
    assert hit["n"] == 3
    assert hit["hit_rate"] == pytest.approx(2 / 3)
    assert hit["core_rate"] == pytest.approx(1 / 3)
    assert hit["mean_score"] == pytest.approx(2.0 / 3)


def test_hit_rate_on_empty_input_is_nan_not_zero():
    # Zero would read as "steering failed"; NaN correctly reads as "no data".
    hit = concept_hit_rate(StubSpace({}), [], "ch0_cl0")
    assert hit["n"] == 0
    assert np.isnan(hit["hit_rate"])


def test_evaluate_concept_reports_lift_over_baseline():
    space = StubSpace({
        "g1": [0.1, 1.0], "g2": [0.3, 1.0], "g3": [1.0, 1.0], "g4": [0.4, 1.0],
        "b1": [1.0, 1.0], "b2": [1.0, 1.0], "b3": [0.5, 1.0], "b4": [1.0, 1.0],
    })
    out = evaluate_concept(space, ["g1", "g2", "g3", "g4"],
                           ["b1", "b2", "b3", "b4"], "ch0_cl0")
    assert out["conditioned"]["hit_rate"] == pytest.approx(0.75)
    assert out["baseline"]["hit_rate"] == pytest.approx(0.25)
    assert out["lift"] == pytest.approx(0.5)
    assert out["enrichment"] == pytest.approx(3.0)
    # Conditioned molecules sit closer to the concept, so the delta is negative.
    assert out["score_delta"] < 0


def test_enrichment_is_infinite_when_baseline_never_reaches_the_concept():
    space = StubSpace({"g": [0.1, 1.0], "b": [1.0, 1.0]})
    out = evaluate_concept(space, ["g"], ["b"], "ch0_cl0")
    assert out["baseline"]["hit_rate"] == 0.0
    assert np.isinf(out["enrichment"])
    assert out["lift"] == pytest.approx(1.0)


def test_no_steering_gives_zero_lift():
    space = StubSpace({"x": [0.2, 1.0], "y": [0.2, 1.0]})
    out = evaluate_concept(space, ["x"], ["y"], "ch0_cl0")
    assert out["lift"] == pytest.approx(0.0)
    assert out["enrichment"] == pytest.approx(1.0)


def test_thresholds_are_ordered():
    assert CORE < ASSIGNED < 1.0


def test_rare_concept_is_flagged_unmeasurable():
    # The baseline never reaches this concept, so enrichment is infinite. That is
    # an artifact of a too-small baseline, not a spectacular result, and it must
    # not be averaged into a summary as if it were a finding.
    space = StubSpace({"g": [0.1, 1.0], "b1": [1.0, 1.0], "b2": [1.0, 1.0]})
    out = evaluate_concept(space, ["g"], ["b1", "b2"], "ch0_cl0")
    assert out["baseline_hits"] == 0.0
    assert out["measurable"] is False
    assert np.isinf(out["enrichment"])


def test_well_supported_concept_is_measurable():
    baseline = {f"b{i}": [0.2 if i < 5 else 1.0, 1.0] for i in range(20)}
    space = StubSpace({"g": [0.1, 1.0], **baseline})
    out = evaluate_concept(space, ["g"], list(baseline), "ch0_cl0")
    assert out["baseline_hits"] == pytest.approx(5.0)
    assert out["measurable"] is True


def test_measurability_threshold_is_configurable():
    baseline = {f"b{i}": [0.2 if i < 4 else 1.0, 1.0] for i in range(20)}
    space = StubSpace({"g": [0.1, 1.0], **baseline})
    assert evaluate_concept(space, ["g"], list(baseline), "ch0_cl0",
                            min_baseline_hits=3)["measurable"] is True
    assert evaluate_concept(space, ["g"], list(baseline), "ch0_cl0",
                            min_baseline_hits=10)["measurable"] is False


def test_precomputed_scores_match_recomputed():
    # The caching path must be an optimization, not a behavior change.
    space = StubSpace({"g1": [0.1, 1.0], "g2": [1.0, 1.0],
                       "b1": [0.4, 1.0], "b2": [1.0, 1.0], "b3": [0.3, 1.0]})
    gen, base = ["g1", "g2"], ["b1", "b2", "b3"]
    slow = evaluate_concept(space, gen, base, "ch0_cl0")
    fast = evaluate_concept(space, gen, base, "ch0_cl0",
                            generated_phi=space.phi(gen), baseline_phi=space.phi(base))
    for key in ("lift", "enrichment", "score_delta", "baseline_hits", "measurable"):
        assert slow[key] == fast[key], key
    assert slow["conditioned"] == fast["conditioned"]
    assert slow["baseline"] == fast["baseline"]
