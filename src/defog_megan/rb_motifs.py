"""
The external grader: exact detection of the four planted ``rb_dual_motifs`` motifs.

This is what makes the synthetic experiment worth running. On the molecular side
a generated molecule can only be graded by the *same* MEGAN that produced the
condition, which is the circularity the paper has to defend against. Here the
ground truth is planted by construction, so a generated graph can be graded by
exact subgraph match against a definition MEGAN never saw.

The four motifs, verbatim from
``visual_graph_datasets.generation.synth.colors``::

    BLUE_STAR = StarPattern(YELLOW, BLUE, k=3)     value -2.0
    BLUE_RING = RingPattern(GREEN,  BLUE, k=2)     value -1.0
    RED_RING  = RingPattern(GREEN,  RED,  k=2)     value +1.0
    RED_STAR  = StarPattern(YELLOW, RED,  k=3)     value +2.0

A *star* is a hub of the centre colour with ``k`` leaves of the leaf colour that
are **mutually non-adjacent** (the match is node-induced). A *ring* with ``k=2``
is a triangle of one start-coloured node and two adjacent ring-coloured ones.

**Counting is disjoint, not raw.** A yellow hub with six mutually non-adjacent
red neighbours contains C(6,3) = 20 induced stars but is obviously one motif
site; the dataset's own labeller therefore reduces occurrences to a maximal
node-disjoint packing before counting, and so does this module. Getting that
wrong does not produce a slightly different number, it produces target values
past +/-100 -- the upstream audit hit exactly that.

The installed ``visual_graph_datasets`` is 0.15.7, which predates the ``synth``
module, so the algorithm is reimplemented here rather than imported. It is not
trusted on that basis: :func:`validate_against_pool` re-derives the counts for
real pool graphs and checks them against the ``motifs`` metadata the generator
itself wrote, which is the only evidence that matters.
"""

from __future__ import annotations

import typing as t
from itertools import combinations

import numpy as np

from defog_megan.colorgraphs import BLUE, GREEN, RED, YELLOW, colors_to_classes


#: Motif name -> its contribution to the graph target, per occurrence.
MOTIF_VALUES: t.Dict[str, float] = {
    "blue_star": -2.0,
    "blue_ring": -1.0,
    "red_ring": +1.0,
    "red_star": +2.0,
}

#: Canonical motif order, matching the dataset's ``motifs`` metadata.
MOTIF_NAMES: t.List[str] = ["blue_star", "blue_ring", "red_ring", "red_star"]


class _Graph:
    """Minimal typed graph: per-node colour class plus a neighbour-set adjacency."""

    __slots__ = ("types", "adj", "n")

    def __init__(self, types: t.Sequence[int], edges: t.Iterable[t.Tuple[int, int]]):
        self.types = list(types)
        self.n = len(self.types)
        self.adj: t.List[t.Set[int]] = [set() for _ in range(self.n)]
        for u, v in edges:
            if u == v or not (0 <= u < self.n and 0 <= v < self.n):
                continue
            self.adj[u].add(v)
            self.adj[v].add(u)

    def nodes_of_type(self, type_: int) -> t.List[int]:
        return [i for i in range(self.n) if self.types[i] == type_]

    def neighbors(self, node: int) -> t.List[int]:
        return sorted(self.adj[node])

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adj[u]


def find_stars(graph: _Graph, center_type: int, leaf_type: int,
               k: int = 3) -> t.List[t.FrozenSet[int]]:
    """Every node-induced star: a hub with ``k`` mutually non-adjacent leaves."""
    results: t.List[t.FrozenSet[int]] = []
    for hub in graph.nodes_of_type(center_type):
        leaves = [n for n in graph.neighbors(hub) if graph.types[n] == leaf_type]
        if len(leaves) < k:
            continue
        for subset in combinations(leaves, k):
            if all(not graph.has_edge(a, b) for a, b in combinations(subset, 2)):
                results.append(frozenset((hub,) + subset))
    return results


def find_rings(graph: _Graph, start_type: int,
               ring_type: int) -> t.List[t.FrozenSet[int]]:
    """Every triangle of one start-coloured node and two adjacent ring-coloured ones."""
    results: t.List[t.FrozenSet[int]] = []
    for start in graph.nodes_of_type(start_type):
        members = [n for n in graph.neighbors(start) if graph.types[n] == ring_type]
        for a, b in combinations(members, 2):
            if graph.has_edge(a, b):
                results.append(frozenset((start, a, b)))
    return results


def pack(occurrences: t.List[t.FrozenSet[int]]) -> t.List[t.FrozenSet[int]]:
    """Greedily reduce occurrences to a maximal pairwise node-disjoint subset.

    Repeatedly takes the occurrence conflicting with the fewest remaining
    candidates. Exact maximum set packing is NP-hard, but occurrences here are
    three or four nodes and few per graph, so greedy is fast and in practice
    optimal. Being greedy it can only under-count, so any count is a lower bound.
    """
    remaining = list(occurrences)
    if len(remaining) <= 1:
        return remaining
    chosen: t.List[t.FrozenSet[int]] = []
    while remaining:
        conflicts = [
            sum(1 for j, other in enumerate(remaining) if j != i and other & candidate)
            for i, candidate in enumerate(remaining)
        ]
        pick = remaining[min(range(len(remaining)), key=conflicts.__getitem__)]
        chosen.append(pick)
        remaining = [other for other in remaining if not (other & pick)]
    return chosen


def count_motifs(node_classes: t.Sequence[int],
                 edges: t.Iterable[t.Tuple[int, int]]) -> t.Dict[str, int]:
    """Disjoint occurrence count of each of the four motifs.

    :param node_classes: per-node palette index (i.e. the dataset's node type).
    :param edges: undirected edges as ``(u, v)`` pairs; both directions tolerated.
    """
    graph = _Graph(node_classes, edges)
    return {
        "blue_star": len(pack(find_stars(graph, YELLOW, BLUE, k=3))),
        "blue_ring": len(pack(find_rings(graph, GREEN, BLUE))),
        "red_ring": len(pack(find_rings(graph, GREEN, RED))),
        "red_star": len(pack(find_stars(graph, YELLOW, RED, k=3))),
    }


def count_motifs_vgd(graph: dict) -> t.Dict[str, int]:
    """Motif counts for a VGD-style graph dict (``node_attributes``/``edge_indices``)."""
    classes = colors_to_classes(graph["node_attributes"]).tolist()
    edges = np.asarray(graph["edge_indices"], dtype=np.int64).reshape(-1, 2)
    return count_motifs(classes, [(int(u), int(v)) for u, v in edges])


def count_motifs_pyg(data) -> t.Dict[str, int]:
    """Motif counts for a sampled PyG ``Data`` (one-hot ``x`` over the palette)."""
    classes = data.x.argmax(dim=-1).tolist()
    edge_index = getattr(data, "edge_index", None)
    edges: t.List[t.Tuple[int, int]] = []
    if edge_index is not None and edge_index.numel() > 0:
        edges = [(int(edge_index[0, k]), int(edge_index[1, k]))
                 for k in range(edge_index.shape[1])]
    return count_motifs(classes, edges)


def exact_target(counts: t.Mapping[str, int]) -> float:
    """The noise-free target implied by a motif count vector.

    The dataset's label is this sum plus uniform measurement noise (half-width
    0.5 in the shipped release), so this is the signal the generator would have
    to reproduce for its label distribution to match.
    """
    return float(sum(MOTIF_VALUES[name] * counts.get(name, 0) for name in MOTIF_NAMES))


def motif_report(samples: t.Sequence[t.Any],
                 reference: t.Sequence[t.Any]) -> dict:
    """Compare generated motif content against a reference set of graphs.

    Both arguments are sequences of PyG ``Data``. Reports, per motif, the
    fraction of graphs containing it and the mean disjoint count, for generated
    and reference alike -- plus the distribution of the exact target the counts
    imply.

    :returns: a dict of plain floats and lists, safe to store in an experiment.
    """
    def _tally(graphs):
        present = {name: 0 for name in MOTIF_NAMES}
        totals = {name: 0 for name in MOTIF_NAMES}
        targets = []
        per_graph = []
        for graph in graphs:
            counts = count_motifs_pyg(graph)
            per_graph.append(counts)
            for name in MOTIF_NAMES:
                if counts[name]:
                    present[name] += 1
                totals[name] += counts[name]
            targets.append(exact_target(counts))
        n = max(len(graphs), 1)
        return {
            "rate": {name: present[name] / n for name in MOTIF_NAMES},
            "mean_count": {name: totals[name] / n for name in MOTIF_NAMES},
            "targets": np.asarray(targets, dtype=np.float64),
            "per_graph": per_graph,
        }

    gen = _tally(samples)
    ref = _tally(reference)

    # Target histogram over the integer motif-sum, on a shared support.
    lo = int(min(gen["targets"].min(initial=0), ref["targets"].min(initial=0)))
    hi = int(max(gen["targets"].max(initial=0), ref["targets"].max(initial=0)))
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    gen_hist, _ = np.histogram(gen["targets"], bins=edges)
    ref_hist, _ = np.histogram(ref["targets"], bins=edges)

    from defog_megan.colorgraphs import _discrete_kl

    return {
        "motif_rate": gen["rate"],
        "motif_rate_ref": ref["rate"],
        "motif_mean_count": gen["mean_count"],
        "motif_mean_count_ref": ref["mean_count"],
        "kl_motif_targets": _discrete_kl(gen_hist, ref_hist),
        "target_support": [int(v) for v in np.arange(lo, hi + 1)],
        "target_hist": gen_hist.tolist(),
        "target_hist_ref": ref_hist.tolist(),
        "mean_exact_target": float(gen["targets"].mean()) if len(gen["targets"]) else 0.0,
        "mean_exact_target_ref": float(ref["targets"].mean()) if len(ref["targets"]) else 0.0,
        "std_exact_target": float(gen["targets"].std()) if len(gen["targets"]) else 0.0,
        "std_exact_target_ref": float(ref["targets"].std()) if len(ref["targets"]) else 0.0,
    }


def validate_against_pool(root: str, limit: int = 2000) -> dict:
    """Re-derive motif counts for real pool graphs and diff against the metadata.

    The only real evidence that this reimplementation matches the generator. Any
    mismatch means the grader is wrong and every motif number downstream is
    meaningless, so callers should treat a non-zero ``mismatches`` as fatal.
    """
    from defog_megan.colorgraphs import load_vgd_dataset

    records = load_vgd_dataset(root, limit=limit)
    mismatches, checked = [], 0
    for record in records:
        stored = record.get("motifs")
        if stored is None:
            continue
        derived = count_motifs_vgd(record["graph"])
        checked += 1
        if any(int(stored.get(name, 0)) != derived[name] for name in MOTIF_NAMES):
            mismatches.append({"index": record.get("index"),
                               "stored": stored, "derived": derived})
    return {"checked": checked, "mismatches": len(mismatches),
            "examples": mismatches[:5]}
