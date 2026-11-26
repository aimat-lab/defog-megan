"""
Tests for the colour-graph seam.

The load-bearing property is that the VGD -> PyG -> VGD round trip is *exact*.
On the molecular side the equivalent round trip loses 75% of zinc64k to the
atom-symbol encoding, and phi therefore has to be recomputed on the decoded
molecule. Here it must lose nothing, because that is the whole reason the
synthetic dataset is a clean test of the method. If these tests ever fail, the
synthetic experiment inherits the molecular side's confound.

The second thing pinned here is the palette contract: node class ``k`` means
``RB_PALETTE[k]``, an unknown colour is an error rather than a fallback class,
and the ordering is not frequency-derived.
"""

import numpy as np
import pytest
import torch

from defog_megan.colorgraphs import (
    BLUE,
    GRAY,
    GREEN,
    RED,
    YELLOW,
    distribution_score,
    RB_PALETTE,
    RB_PALETTE_NAMES,
    ColorGraphDomain,
    assert_palette,
    color_to_class,
    load_pool_npz,
    pyg_to_vgd,
    save_pool_npz,
    structural_report,
    vgd_to_pyg,
    wl_hash,
)


def make_graph(classes, edges):
    """A minimal VGD-shaped graph dict from palette indices and undirected edges."""
    node_attributes = [list(RB_PALETTE[c]) for c in classes]
    edge_indices = []
    for u, v in edges:
        edge_indices += [[u, v], [v, u]]
    return {
        "node_attributes": node_attributes,
        "edge_indices": edge_indices,
        "edge_attributes": [[1.0]] * len(edge_indices),
    }


# -- the palette contract ----------------------------------------------------

def test_palette_and_names_line_up():
    assert len(RB_PALETTE) == len(RB_PALETTE_NAMES) == 7


def test_palette_matches_the_generators_canonical_order():
    # The contract: DeFoG's node class index IS the dataset's node type index.
    # These values are copied from
    # visual_graph_datasets.generation.synth.colors.COLOR_RGB / COLOR_NAMES.
    # (The installed vgd is 0.15.7, which predates that module, so they are
    # pinned here rather than imported.)
    assert RB_PALETTE_NAMES == ["gray", "red", "green", "blue",
                                "yellow", "cyan", "magenta"]
    assert RB_PALETTE[GRAY] == (0.8, 0.8, 0.8)
    assert RB_PALETTE[RED] == (1.0, 0.0, 0.0)
    assert RB_PALETTE[GREEN] == (0.0, 1.0, 0.0)
    assert RB_PALETTE[BLUE] == (0.0, 0.0, 1.0)
    assert RB_PALETTE[YELLOW] == (1.0, 1.0, 0.0)


def test_color_to_class_round_trips_every_entry():
    for k, rgb in enumerate(RB_PALETTE):
        assert color_to_class(rgb) == k


def test_unknown_colour_raises_rather_than_bucketing():
    with pytest.raises(ValueError):
        color_to_class((0.5, 0.25, 0.125))


def test_assert_palette_rejects_a_pool_missing_a_colour():
    graphs = [make_graph([0, 1], [(0, 1)])]
    with pytest.raises(ValueError, match="never observed"):
        assert_palette(graphs)


def test_assert_palette_accepts_a_pool_using_all_of_it():
    graphs = [make_graph(list(range(len(RB_PALETTE))),
                         [(i, i + 1) for i in range(len(RB_PALETTE) - 1)])]
    assert_palette(graphs)          # must not raise


# -- encoding ----------------------------------------------------------------

def test_encoding_shapes_and_edge_class_convention():
    graph = make_graph([4, 0, 3], [(0, 1), (1, 2)])
    data = vgd_to_pyg(graph)

    assert data.x.shape == (3, len(RB_PALETTE))
    # Edge class 0 is reserved for "no edge"; every real edge is class 1.
    assert data.edge_attr.shape == (4, 2)
    assert torch.equal(data.edge_attr.argmax(dim=-1), torch.ones(4, dtype=torch.long))
    assert data.x.argmax(dim=-1).tolist() == [4, 0, 3]


def test_edgeless_graph_is_dropped():
    graph = {"node_attributes": [list(RB_PALETTE[0])],
             "edge_indices": [], "edge_attributes": []}
    assert vgd_to_pyg(graph) is None


def test_condition_vector_is_attached_with_a_batch_dimension():
    graph = make_graph([0, 1], [(0, 1)])
    data = vgd_to_pyg(graph, y=[0.25, 1.0, 1.0])
    assert data.y.shape == (1, 3)


# -- the round trip ----------------------------------------------------------

@pytest.mark.parametrize("classes,edges", [
    ([4, 0, 3], [(0, 1), (1, 2)]),
    ([0, 1, 2, 3, 4, 5, 6], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)]),
    ([3, 3, 3], [(0, 1), (0, 2), (1, 2)]),                     # triangle
    ([4, 4, 4, 4, 4], [(0, 1), (0, 2), (0, 3), (0, 4)]),       # star
])
def test_round_trip_is_exact(classes, edges):
    graph = make_graph(classes, edges)
    decoded = pyg_to_vgd(vgd_to_pyg(graph))

    assert np.allclose(np.asarray(decoded["node_attributes"]),
                       np.asarray(graph["node_attributes"]))

    def undirected(pairs):
        return {tuple(sorted(p)) for p in np.asarray(pairs).reshape(-1, 2).tolist()}

    assert undirected(decoded["edge_indices"]) == undirected(graph["edge_indices"])
    # Both directions are emitted, matching the pool's own convention.
    assert len(decoded["edge_indices"]) == 2 * len(undirected(graph["edge_indices"]))


def test_decoded_dict_has_the_keys_megan_reads():
    # graph_attention_student.torch.data reads exactly these three.
    decoded = pyg_to_vgd(vgd_to_pyg(make_graph([0, 4], [(0, 1)])))
    for key in ("node_attributes", "edge_indices", "edge_attributes"):
        assert key in decoded
    assert np.asarray(decoded["edge_attributes"]).shape == (2, 1)


def test_decode_never_fails_on_a_degenerate_sample():
    # Unlike the molecular decode, there is nothing to sanitize: a sampled graph
    # with no edges at all still decodes to a well-formed colour graph.
    data = vgd_to_pyg(make_graph([0, 1], [(0, 1)]))
    data.edge_index = torch.zeros((2, 0), dtype=torch.long)
    decoded = pyg_to_vgd(data)
    assert np.asarray(decoded["edge_indices"]).shape == (0, 2)
    assert len(decoded["node_attributes"]) == 2


def test_self_loops_are_dropped_on_decode():
    data = vgd_to_pyg(make_graph([0, 1], [(0, 1)]))
    data.edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long)
    decoded = pyg_to_vgd(data)
    assert np.asarray(decoded["edge_indices"]).tolist() == [[0, 1], [1, 0]]


# -- the domain --------------------------------------------------------------

def test_domain_reports_everything_valid():
    # Deliberate: there is no chemistry to violate, so realism is distributional.
    domain = ColorGraphDomain()
    assert domain.is_valid(vgd_to_pyg(make_graph([0, 1], [(0, 1)]))) is True


def test_domain_identity_separates_colourings_of_the_same_shape():
    domain = ColorGraphDomain()
    a = vgd_to_pyg(make_graph([0, 0, 0], [(0, 1), (1, 2)]))
    b = vgd_to_pyg(make_graph([4, 4, 4], [(0, 1), (1, 2)]))
    same = vgd_to_pyg(make_graph([0, 0, 0], [(0, 1), (1, 2)]))
    assert domain.identity(a) != domain.identity(b)
    assert domain.identity(a) == domain.identity(same)


def test_domain_renders_without_error():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domain = ColorGraphDomain()
    fig, ax = plt.subplots()
    domain.render(ax, vgd_to_pyg(make_graph([0, 4, 3], [(0, 1), (1, 2)])))
    plt.close(fig)


def test_domain_caption_flags_disconnection():
    domain = ColorGraphDomain()
    two_components = make_graph([0, 1, 2, 3], [(0, 1), (2, 3)])
    assert "disconn" in domain.caption(vgd_to_pyg(two_components))
    connected = make_graph([0, 1], [(0, 1)])
    assert "disconn" not in domain.caption(vgd_to_pyg(connected))


# -- the distributional report -----------------------------------------------

def test_identical_sets_have_zero_divergence():
    graphs = [vgd_to_pyg(make_graph([0, 4, 3], [(0, 1), (1, 2)])),
              vgd_to_pyg(make_graph([4, 4, 0], [(0, 1), (1, 2), (0, 2)]))]
    report = structural_report(graphs, graphs)
    assert report["kl_colors"] == pytest.approx(0.0, abs=1e-9)
    assert report["kl_degrees"] == pytest.approx(0.0, abs=1e-9)
    assert report["kl_sizes"] == pytest.approx(0.0, abs=1e-9)


def test_divergence_grows_when_the_colour_mix_is_wrong():
    reference = [vgd_to_pyg(make_graph([0, 0, 0], [(0, 1), (1, 2)]))]
    same = [vgd_to_pyg(make_graph([0, 0, 0], [(0, 1), (1, 2)]))]
    different = [vgd_to_pyg(make_graph([4, 4, 4], [(0, 1), (1, 2)]))]
    assert (structural_report(different, reference)["kl_colors"]
            > structural_report(same, reference)["kl_colors"])


def test_distribution_score_is_one_for_a_perfect_match():
    graphs = [vgd_to_pyg(make_graph([0, 4, 3], [(0, 1), (1, 2)]))]
    assert distribution_score(structural_report(graphs, graphs)) == pytest.approx(1.0, abs=1e-6)


def test_distribution_score_punishes_over_density():
    # The bug this exists to prevent: an over-dense model is trivially connected,
    # so connectivity alone would rank it top. The score must rank it below a
    # correctly-sparse model.
    reference = [vgd_to_pyg(make_graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)]))]
    sparse = [vgd_to_pyg(make_graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)]))]
    dense = [vgd_to_pyg(make_graph(
        [0, 1, 2, 3],
        [(a, b) for a in range(4) for b in range(a + 1, 4)]))]     # complete graph

    dense_report = structural_report(dense, reference)
    assert dense_report["connected_rate"] == 1.0        # trivially connected
    assert (distribution_score(dense_report)
            < distribution_score(structural_report(sparse, reference)))


def test_connectivity_is_reported_not_enforced():
    disconnected = [vgd_to_pyg(make_graph([0, 1, 2, 3], [(0, 1), (2, 3)]))]
    connected = [vgd_to_pyg(make_graph([0, 1], [(0, 1)]))]
    report = structural_report(disconnected, connected)
    assert report["connected_rate"] == 0.0
    assert report["connected_rate_ref"] == 1.0


def test_save_pool_npz_accepts_cuda_resident_graphs():
    """A sampler's own output is CUDA-resident; every prior caller built graphs
    from JSON (always CPU), so this path was never exercised until a training
    experiment dumped sampled graphs directly -- see colorgraphs.save_pool_npz.
    """
    import torch as _torch
    if not _torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA on this machine")
    from torch_geometric.data import Data
    data = Data(
        x=torch.eye(len(RB_PALETTE))[torch.tensor([0, 1, 2])].cuda(),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda(),
        edge_attr=torch.zeros(2, 2).cuda(),
    )
    path_dir = "/tmp/claude-1000/-media-ssd2-Programming-defog-megan/123aec8d-c984-4390-bada-5885af795ac4/scratchpad"
    path = f"{path_dir}/cuda_pool_test.npz"
    save_pool_npz(path, [data], [{}])          # must not raise
    loaded, _ = load_pool_npz(path)
    assert loaded[0].x.shape[0] == 3


# -- the isomorphism key -------------------------------------------------------------


def _pyg(classes, edges):
    return vgd_to_pyg(make_graph(classes, edges))


def test_wl_hash_ignores_node_order():
    # a path 0-1-2 with colours (RED, GRAY, BLUE), and the same graph with the nodes listed in reverse order
    a = _pyg([RED, GRAY, BLUE], [(0, 1), (1, 2)])
    b = _pyg([BLUE, GRAY, RED], [(1, 2), (0, 1)])
    assert wl_hash(a) == wl_hash(b)


def test_wl_hash_separates_colourings_of_the_same_shape():
    a = _pyg([RED, GRAY, BLUE], [(0, 1), (1, 2)])
    b = _pyg([RED, GRAY, RED], [(0, 1), (1, 2)])
    assert wl_hash(a) != wl_hash(b)


def test_wl_hash_separates_shapes_with_the_same_colours():
    path = _pyg([RED, RED, RED, RED], [(0, 1), (1, 2), (2, 3)])
    star = _pyg([RED, RED, RED, RED], [(0, 1), (0, 2), (0, 3)])
    assert wl_hash(path) != wl_hash(star)
