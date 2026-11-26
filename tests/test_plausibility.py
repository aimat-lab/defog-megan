"""
Tests for the macrocycle plausibility penalty.

The penalty exists because concept metrics live in MEGAN's embedding space, where
a molecule can score highly while being chemically implausible -- `ch0_cl7` hit
164x enrichment while emitting strained polyaza macrocycles. These tests pin the
ring-counting and the shape of the penalty, not the chemistry judgement itself.
"""

import pytest
from rdkit import Chem

from defog_megan.plausibility import (
    MACRO_CAP,
    MACRO_RING_SIZE,
    count_macrocycles,
    count_macrocycles_smiles,
    macrocycle_fraction,
    macrocycle_penalty,
)


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1", 0),                 # benzene, 6
    ("C1CCCCCC1", 0),                # 7-ring, at the boundary but not over
    ("C1CCCCCCC1", 0),               # 8-ring, exactly the threshold
    ("C1CCCCCCCC1", 1),              # 9-ring, over
    ("C1CCCCCCCCCC1", 1),            # 11-ring
    ("CCO", 0),                      # acyclic
    ("C1CCC2CCCCC2C1", 0),           # fused 6+6
])
def test_ring_counting(smiles, expected):
    assert count_macrocycles_smiles(smiles) == expected


def test_threshold_is_strictly_greater_than():
    # A ring of exactly MACRO_RING_SIZE must NOT count, or 8-membered rings --
    # which do occur in real chemistry -- would be penalized.
    ring = "C1" + "C" * (MACRO_RING_SIZE - 1) + "1"
    assert count_macrocycles_smiles(ring) == 0


def test_none_molecule_is_zero_not_an_error():
    assert count_macrocycles(None) == 0


def test_penalty_is_graded_then_capped():
    # Graded so the search keeps a gradient between one macrocycle and three;
    # capped so ring geometry cannot swamp the concept signal.
    class FakeRingInfo:
        def __init__(self, n): self._n = n
        def AtomRings(self): return [tuple(range(12))] * self._n

    class FakeMol:
        def __init__(self, n): self._ri = FakeRingInfo(n)
        def GetRingInfo(self): return self._ri

    assert macrocycle_penalty(FakeMol(0), 0.5) == 0.0
    assert macrocycle_penalty(FakeMol(1), 0.5) == pytest.approx(0.5)
    assert macrocycle_penalty(FakeMol(2), 0.5) == pytest.approx(1.0)
    # capped
    assert macrocycle_penalty(FakeMol(5), 0.5) == pytest.approx(0.5 * MACRO_CAP)


def test_penalty_is_never_negative():
    assert macrocycle_penalty(Chem.MolFromSmiles("CCO"), 0.5) >= 0.0


def test_zero_weight_disables_the_penalty():
    assert macrocycle_penalty(Chem.MolFromSmiles("C1CCCCCCCCCC1"), 0.0) == 0.0


def test_fraction_ignores_unparseable_molecules():
    frac = macrocycle_fraction(["C1CCCCCCCCCC1", "CCO", "not-a-molecule"])
    assert frac == pytest.approx(0.5)          # 1 of 2 parseable


def test_fraction_of_empty_set_is_zero():
    assert macrocycle_fraction([]) == 0.0
