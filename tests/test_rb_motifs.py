"""
Tests for the external motif grader.

This module is the reason the synthetic experiment is worth running: it grades a
generated graph against a definition MEGAN never saw, which is the circularity
defence the molecular side has to argue for. So it has to be *exactly* the
generator's definition, not a plausible approximation.

The definitive check is not here but in ``validate_against_pool``, which
re-derives counts for real pool graphs and diffs them against the ``motifs``
metadata the generator itself wrote (25,000 graphs, 0 mismatches at the time of
writing). What is pinned here is the behaviour that check cannot see: the
induced-match rules, and the disjoint packing that stops one motif site from
counting as twenty.
"""

import numpy as np
import pytest

from defog_megan.colorgraphs import BLUE, GRAY, GREEN, RED, YELLOW
from defog_megan.rb_motifs import (
    MOTIF_NAMES,
    MOTIF_VALUES,
    count_motifs,
    exact_target,
    find_stars,
    pack,
    _Graph,
)


# -- the four definitions ----------------------------------------------------

def test_a_red_ring_is_a_green_plus_two_adjacent_reds():
    counts = count_motifs([GREEN, RED, RED], [(0, 1), (0, 2), (1, 2)])
    assert counts["red_ring"] == 1
    assert counts["blue_ring"] == 0


def test_a_blue_ring_is_the_same_shape_in_blue():
    counts = count_motifs([GREEN, BLUE, BLUE], [(0, 1), (0, 2), (1, 2)])
    assert counts["blue_ring"] == 1
    assert counts["red_ring"] == 0


def test_a_triangle_without_the_green_start_is_not_a_ring():
    counts = count_motifs([RED, RED, RED], [(0, 1), (0, 2), (1, 2)])
    assert counts["red_ring"] == 0


def test_an_open_path_is_not_a_ring():
    # The closing edge between the two red nodes is what makes it a ring.
    counts = count_motifs([GREEN, RED, RED], [(0, 1), (0, 2)])
    assert counts["red_ring"] == 0


def test_a_red_star_is_a_yellow_hub_with_three_reds():
    counts = count_motifs([YELLOW, RED, RED, RED], [(0, 1), (0, 2), (0, 3)])
    assert counts["red_star"] == 1
    assert counts["blue_star"] == 0


def test_a_star_needs_three_leaves_not_two():
    counts = count_motifs([YELLOW, RED, RED], [(0, 1), (0, 2)])
    assert counts["red_star"] == 0


def test_star_leaves_must_be_mutually_non_adjacent():
    # The match is node-induced: an edge between two leaves disqualifies it.
    base = [YELLOW, RED, RED, RED]
    assert count_motifs(base, [(0, 1), (0, 2), (0, 3)])["red_star"] == 1
    assert count_motifs(base, [(0, 1), (0, 2), (0, 3), (1, 2)])["red_star"] == 0


def test_wrong_hub_colour_is_not_a_star():
    counts = count_motifs([GRAY, RED, RED, RED], [(0, 1), (0, 2), (0, 3)])
    assert counts["red_star"] == 0


# -- disjoint packing --------------------------------------------------------

def test_one_hub_with_six_leaves_is_one_site_not_twenty():
    # The upstream audit's motivating case: C(6,3) = 20 raw induced stars around
    # a single hub. Raw counting drove target values past +/-100.
    classes = [YELLOW] + [RED] * 6
    edges = [(0, i) for i in range(1, 7)]

    graph = _Graph(classes, edges)
    assert len(find_stars(graph, YELLOW, RED, k=3)) == 20      # raw
    assert count_motifs(classes, edges)["red_star"] == 1       # packed


def test_two_separate_sites_count_twice():
    classes = [YELLOW, RED, RED, RED, YELLOW, RED, RED, RED]
    edges = [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7)]
    assert count_motifs(classes, edges)["red_star"] == 2


def test_pack_returns_pairwise_disjoint_sets():
    occurrences = [frozenset({0, 1, 2}), frozenset({2, 3, 4}), frozenset({5, 6, 7})]
    packed = pack(occurrences)
    for a, b in zip(packed, packed[1:]):
        assert not (a & b)
    assert len(packed) == 2      # the middle one conflicts with the first


def test_pack_is_a_noop_on_short_input():
    assert pack([]) == []
    single = [frozenset({1, 2, 3})]
    assert pack(single) == single


# -- the target ---------------------------------------------------------------

def test_motif_values_match_the_generator():
    assert MOTIF_VALUES == {"blue_star": -2.0, "blue_ring": -1.0,
                            "red_ring": +1.0, "red_star": +2.0}
    assert MOTIF_NAMES == ["blue_star", "blue_ring", "red_ring", "red_star"]


def test_exact_target_is_the_signed_sum():
    assert exact_target({"red_star": 1}) == pytest.approx(2.0)
    assert exact_target({"blue_star": 1}) == pytest.approx(-2.0)
    # Values apply per occurrence -- two red rings are worth +2, which is the v1
    # defect the release exists to fix.
    assert exact_target({"red_ring": 2}) == pytest.approx(2.0)
    assert exact_target({"red_star": 1, "blue_ring": 1}) == pytest.approx(1.0)


def test_empty_graph_has_no_motifs_and_a_zero_target():
    counts = count_motifs([], [])
    assert all(counts[name] == 0 for name in MOTIF_NAMES)
    assert exact_target(counts) == 0.0


def test_both_edge_directions_are_tolerated():
    # Sampled graphs carry both directions; pool json does too.
    one_way = count_motifs([GREEN, RED, RED], [(0, 1), (0, 2), (1, 2)])
    two_way = count_motifs([GREEN, RED, RED],
                           [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)])
    assert one_way == two_way


# -- the Feynman-Kac energy --------------------------------------------------

def test_color_energy_tiers_are_strictly_ordered():
    """A disconnected graph must cost more than the worst connected one, or FK can prefer fragments."""
    from defog_megan.color_energy import ColorTargetEnergy
    e = ColorTargetEnergy(target=0.0)
    assert e.disconnect_energy > e.clamp


def test_color_energy_refuses_an_inverted_tiering():
    from defog_megan.color_energy import ColorTargetEnergy
    with pytest.raises(ValueError, match="must exceed the clamp"):
        ColorTargetEnergy(target=0.0, clamp=5.0)


def _path_graph_batch(n=4, batch=2):
    import torch
    from defog_megan.colorgraphs import RB_PALETTE
    C = len(RB_PALETTE)
    X1 = torch.zeros(batch, n, C); X1[..., 0] = 1.0
    E1 = torch.zeros(batch, n, n, 2); E1[..., 0] = 1.0
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        for u, v in ((a, b), (b, a)):
            E1[:, u, v, 0] = 0.0; E1[:, u, v, 1] = 1.0
    return X1, E1, torch.ones(batch, n, dtype=torch.bool)


def test_color_energy_scores_by_distance_to_the_target():
    """A connected graph whose motif sum equals the request has the lowest energy; a farther request costs more."""
    from defog_megan.color_energy import ColorTargetEnergy
    X1, E1, node_mask = _path_graph_batch()
    at_target = ColorTargetEnergy(target=0.0)(X1, E1, node_mask)       # the all-gray path has motif sum 0
    far = ColorTargetEnergy(target=6.0)(X1, E1, node_mask)
    assert float(at_target.max()) == 0.0
    assert float(far.min()) > float(at_target.max())


def test_color_energy_is_capped_below_the_disconnected_tier():
    from defog_megan.color_energy import ColorTargetEnergy
    X1, E1, node_mask = _path_graph_batch()
    energy = ColorTargetEnergy(target=1e6)                              # absurdly far: saturates at the clamp
    out = energy(X1, E1, node_mask)
    assert float(out.max()) == energy.clamp < energy.disconnect_energy


def test_color_energy_puts_a_split_graph_in_its_own_tier():
    import torch
    from defog_megan.color_energy import ColorTargetEnergy
    X1, E1, node_mask = _path_graph_batch()
    E1 = E1.clone()
    for u, v in ((1, 2), (2, 1)):                                       # sever the middle edge: two components
        E1[:, u, v, 0] = 1.0; E1[:, u, v, 1] = 0.0
    energy = ColorTargetEnergy(target=0.0)
    out = energy(X1, E1, node_mask)
    assert torch.all(out == energy.disconnect_energy)
    assert energy.last_stats["connected_rate"] == 0.0
