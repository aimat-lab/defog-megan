"""
The colour-graph seam: ``rb_dual_motifs`` <-> DeFoG <-> MEGAN.

The synthetic side of the project needs the same round trip the molecular side
has -- encode a dataset for DeFoG, sample, decode, re-score with MEGAN -- but for
graphs whose nodes carry a *colour* instead of an element symbol. DeFoG's core is
already agnostic here: it only ever sees categorical node/edge classes, and
``molecular_features`` is opt-in. So this module is a data adapter, not a model
change.

Three conversions, and one thing that is genuinely easier than it is for
molecules::

    VGD json  --vgd_to_pyg-->  PyG Data  --DeFoG-->  PyG Data  --pyg_to_vgd-->  graph dict
                                                                                    |
                                                          ConceptSpace.phi_from_graphs

**The decode is lossless.** There is no valence model, no sanitization and no
aromatic/kekule codec, so a sampled graph always decodes to a well-formed colour
graph. The 24.8% round-trip loss and the ``[nH]`` codec failure that dominate the
molecular results simply have no analogue here -- which is exactly what makes this
dataset a clean test of the *method* rather than of the encoding.

Two contracts that will bite if broken:

* **The palette order is frozen** (:data:`RB_PALETTE`). Node class ``k`` means
  ``RB_PALETTE[k]`` in a trained checkpoint forever after. It is deliberately
  *not* derived by frequency the way ``derive_atom_types`` builds an atom
  vocabulary: red and blue occur 25849 vs 25815 times in a 3000-graph sample, so
  a frequency-sorted palette can swap the two most semantically loaded colours
  between one subsample and the next. The order here is lexicographic by RGB,
  which is stable under any subsample. :func:`assert_palette` enforces that a
  pool actually uses this palette.
* **Edge class 0 is "no edge"**, matching DeFoG's convention throughout. The pool
  has a single real edge type, so ``num_edge_classes == 2``.

Validity is deliberately *not* defined as a gate. Any colour graph is a
well-formed member of the domain -- there is no chemistry to violate -- so
:class:`ColorGraphDomain` reports every sample valid and structural realism is
measured distributionally instead, by :func:`structural_report`.
"""

from __future__ import annotations

import glob
import json
import os
import typing as t

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from defog_adapt.core.domain import GenericGraphDomain


def _num_nodes(data) -> int:
    """The number of nodes of a PyG graph (0 without node features)."""
    x = getattr(data, "x", None)
    return int(x.shape[0]) if x is not None else 0


# -- the palette contract -----------------------------------------------------

#: The ``rb_dual_motifs`` node palette. This is the generator's own canonical
#: order -- ``visual_graph_datasets.generation.synth.colors.COLOR_RGB``, a frozen
#: upstream constant -- so DeFoG's node class index *is* the dataset's node type
#: index. That equality is what lets the motif detector read node classes
#: directly and keeps every number comparable to the dataset's own audit.
#:
#: Deriving this by frequency instead would be a trap: red and blue occur 25849
#: vs 25815 times in a 3000-graph sample, so a frequency-sorted palette can swap
#: the two most semantically loaded colours between one subsample and the next.
RB_PALETTE: t.List[t.Tuple[float, float, float]] = [
    (0.8, 0.8, 0.8),   # 0 gray  (the "background" colour)
    (1.0, 0.0, 0.0),   # 1 red
    (0.0, 1.0, 0.0),   # 2 green
    (0.0, 0.0, 1.0),   # 3 blue
    (1.0, 1.0, 0.0),   # 4 yellow
    (0.0, 1.0, 1.0),   # 5 cyan
    (1.0, 0.0, 1.0),   # 6 magenta
]

#: Human-readable names in the same order, for captions and reports.
RB_PALETTE_NAMES: t.List[str] = [
    "gray", "red", "green", "blue", "yellow", "cyan", "magenta",
]

# Named indices, mirroring the generator's own module-level constants. The motif
# definitions are stated in terms of these.
GRAY, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA = range(7)

#: The four planted motifs the generator plants, i.e. the exact ground truth this
#: dataset exists to provide. Present in the per-graph ``motifs`` metadata.
RB_MOTIFS: t.List[str] = ["blue_star", "blue_ring", "red_ring", "red_star"]

#: Tolerance when matching a stored RGB triple to a palette entry. The values are
#: written to json at full precision, so this only has to absorb float formatting.
_COLOR_TOL = 1e-6


#: Decimal places the palette lookup rounds to. The colours are exact small
#: decimals written to json at full precision, so this only absorbs formatting;
#: it is what lets the lookup be a dict rather than a scan of ``np.allclose``.
_COLOR_ROUND = 6


def _lookup(palette: t.Sequence[t.Sequence[float]]) -> t.Dict[t.Tuple[float, ...], int]:
    return {tuple(np.round(np.asarray(c, dtype=np.float64), _COLOR_ROUND)): k
            for k, c in enumerate(palette)}


def colors_to_classes(node_attributes,
                      palette: t.Sequence[t.Sequence[float]] = RB_PALETTE) -> np.ndarray:
    """Map an ``(N, 3)`` array of RGB rows onto palette indices.

    A dict lookup on rounded tuples rather than a scan of ``np.allclose`` per
    colour per node: the scan costs ``N x len(palette)`` numpy calls, which on
    the 3.2M nodes of the full pool is minutes rather than seconds.

    :raises ValueError: on any colour outside the palette. Hard error by design
        -- an unrecognised colour means the pool does not match the frozen
        palette, and bucketing it silently would corrupt node-class semantics
        for every graph downstream.
    """
    arr = np.round(np.asarray(node_attributes, dtype=np.float64), _COLOR_ROUND)
    if arr.ndim != 2:
        arr = arr.reshape(-1, len(palette[0]))
    table = _lookup(palette)
    out = np.empty(len(arr), dtype=np.int64)
    for i, row in enumerate(map(tuple, arr)):
        try:
            out[i] = table[row]
        except KeyError:
            raise ValueError(f"colour {row} is not in the frozen palette") from None
    return out


def color_to_class(rgb: t.Sequence[float],
                   palette: t.Sequence[t.Sequence[float]] = RB_PALETTE) -> int:
    """Index of a single ``rgb`` in ``palette`` (see :func:`colors_to_classes`)."""
    return int(colors_to_classes(np.asarray(rgb, dtype=np.float64)[None, :], palette)[0])


def assert_palette(graphs: t.Iterable[dict],
                   palette: t.Sequence[t.Sequence[float]] = RB_PALETTE) -> None:
    """Check that ``graphs`` use exactly ``palette`` -- no extras, none missing.

    Cheap insurance against pointing the pipeline at a differently-coloured pool
    (``rgb_dual_motifs_10k``, a regenerated variant) and training a base whose
    node classes mean something other than what the checkpoint claims.
    """
    seen: t.Set[int] = set()
    for graph in graphs:
        seen.update(colors_to_classes(graph["node_attributes"], palette).tolist())
    missing = sorted(set(range(len(palette))) - seen)
    if missing:
        names = ", ".join(RB_PALETTE_NAMES[k] if k < len(RB_PALETTE_NAMES) else str(k)
                          for k in missing)
        raise ValueError(f"palette colours never observed in this pool: {names}")


# -- VGD <-> PyG --------------------------------------------------------------

def vgd_to_pyg(graph: dict,
               palette: t.Sequence[t.Sequence[float]] = RB_PALETTE,
               y: t.Optional[t.Sequence[float]] = None) -> t.Optional[Data]:
    """Encode one visual-graph-dataset graph dict as a DeFoG-ready ``Data``.

    Mirrors ``defog.domains.molecule.smiles_to_pyg_data``: one-hot ``x`` over the
    palette, one-hot ``edge_attr`` with class 0 reserved for "no edge", edges
    stored in both directions and sorted by ``(row * N + col)``.

    :param graph: a VGD graph dict (``node_attributes``, ``edge_indices``, ...).
    :param y: optional condition vector to attach as ``data.y`` (shape ``(1, C)``).
    :returns: the encoded graph, or ``None`` if it has no edges (DeFoG cannot
        represent an edgeless graph any more than it can an unbonded molecule).
    """
    node_attrs = np.asarray(graph["node_attributes"], dtype=np.float64)
    n = int(node_attrs.shape[0])
    type_idx = colors_to_classes(node_attrs, palette)

    edge_indices = np.asarray(graph["edge_indices"], dtype=np.int64)
    if edge_indices.size == 0:
        return None

    # The pool stores both directions already; keep whatever is there and let the
    # undirected de-duplication happen at the consumer, exactly as the molecular
    # path does with RDKit bonds expanded into two rows.
    row = edge_indices[:, 0].tolist()
    col = edge_indices[:, 1].tolist()

    edge_index = torch.tensor([row, col], dtype=torch.long)
    # Single real edge type -> class 1; class 0 stays reserved for "no edge".
    edge_type = torch.ones(edge_index.shape[1], dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=2).float()

    perm = (edge_index[0] * n + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                  num_classes=len(palette)).float()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(1, -1)
    return data


def pyg_to_vgd(data,
               palette: t.Sequence[t.Sequence[float]] = RB_PALETTE) -> dict:
    """Decode a sampled ``Data`` back into a MEGAN-consumable graph dict.

    The inverse of :func:`vgd_to_pyg`, and unlike the molecular decode it cannot
    fail: there is nothing to sanitize. The emitted dict carries exactly the keys
    ``graph_attention_student.torch.data`` reads -- ``node_attributes``,
    ``edge_indices``, ``edge_attributes`` -- plus ``node_indices`` for VGD
    compatibility, so it can be handed straight to
    :meth:`ConceptSpace.phi_from_graphs`.

    Edges are emitted in both directions, matching the pool's own convention.
    """
    x = data.x
    classes = x.argmax(dim=-1).tolist() if x.dim() == 2 and x.shape[1] > 1 else [0] * _num_nodes(data)
    n = len(classes)

    node_attributes = np.asarray([palette[c] for c in classes], dtype=np.float64)

    edge_index = getattr(data, "edge_index", None)
    pairs: t.List[t.Tuple[int, int]] = []
    if edge_index is not None and edge_index.numel() > 0:
        seen = set()
        for k in range(edge_index.shape[1]):
            u, v = int(edge_index[0, k]), int(edge_index[1, k])
            if u == v:
                continue          # DeFoG never emits self-loops; defensive
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            pairs += [(key[0], key[1]), (key[1], key[0])]

    edge_indices = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    edge_attributes = np.ones((edge_indices.shape[0], 1), dtype=np.float64)

    return {
        "node_indices": np.arange(n, dtype=np.int64),
        "node_attributes": node_attributes,
        "edge_indices": edge_indices,
        "edge_attributes": edge_attributes,
    }


# -- loading the pool ---------------------------------------------------------

def load_vgd_dataset(root: str,
                     limit: t.Optional[int] = None,
                     with_metadata: bool = True) -> t.List[dict]:
    """Read a visual graph dataset's json files.

    :param root: the dataset directory (one ``NNNNNNN.json`` per graph).
    :param limit: read only the first ``limit`` graphs, in filename order.
    :param with_metadata: keep the per-graph ``target`` / ``motifs`` metadata
        alongside the graph. The motif counts are the exact planted ground truth,
        so they are what an external grader scores generated graphs against.
    :returns: a list of ``{"graph": ..., "target": ..., "motifs": ...}`` dicts
        (metadata keys omitted when ``with_metadata`` is False).
    """
    paths = sorted(glob.glob(os.path.join(root, "*.json")))
    if limit is not None:
        paths = paths[:limit]

    out = []
    for path in paths:
        with open(path) as handle:
            record = json.load(handle)
        if "graph" not in record:
            continue           # process.py and other non-graph json in the folder
        entry = {"graph": record["graph"]}
        if with_metadata:
            entry["target"] = record.get("target")
            entry["motifs"] = record.get("motifs")
            entry["index"] = record.get("index")
        out.append(entry)
    return out


def save_pool_npz(path: str, graphs: t.Sequence[Data], metadata: t.Sequence[dict],
                  palette: t.Sequence[t.Sequence[float]] = RB_PALETTE) -> None:
    """Serialize an encoded pool as flat arrays.

    Deliberately **not** ``torch.save`` of the ``Data`` objects. A pool is built on
    one machine and read on another, with another torch and PyG; pickled PyG objects
    are a class-definition contract across that gap, and a long job is a bad place to
    discover it broke. Integers and offsets survive any version.

    Node classes ride as ``int8`` and edges as ``int32`` undirected pairs, which
    also makes the file ~7x smaller than the one-hot form it rebuilds into.

    Every caller before this one built ``graphs`` straight from JSON (always CPU),
    so the ``.cpu()`` below was never exercised until a training experiment tried
    to dump a sampler's own output here directly -- ``model.sample(..., device=
    "cuda", ...)`` returns CUDA-resident ``Data`` objects, and ``.numpy()`` on one
    raises rather than silently doing the wrong thing. Cheap and a no-op on data
    that was already on CPU, so it is applied unconditionally rather than only
    where it was observed to matter.
    """
    node_classes, node_ptr = [], [0]
    edges, edge_ptr = [], [0]
    for data in graphs:
        classes = data.x.argmax(dim=-1).to(torch.int8).cpu().numpy()
        node_classes.append(classes)
        node_ptr.append(node_ptr[-1] + len(classes))

        ei = data.edge_index.cpu().numpy()
        undirected = np.unique(np.sort(ei.T, axis=1), axis=0) if ei.size else np.zeros((0, 2))
        edges.append(undirected.astype(np.int32))
        edge_ptr.append(edge_ptr[-1] + len(undirected))

    targets = np.array(
        [(m.get("target") or [np.nan])[0] if m.get("target") is not None else np.nan
         for m in metadata], dtype=np.float64)
    motif_names = ["blue_star", "blue_ring", "red_ring", "red_star"]
    motifs = np.array(
        [[int((m.get("motifs") or {}).get(name, -1)) for name in motif_names]
         for m in metadata], dtype=np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        node_classes=np.concatenate(node_classes) if node_classes else np.zeros(0, np.int8),
        node_ptr=np.asarray(node_ptr, dtype=np.int64),
        edges=np.concatenate(edges) if edges else np.zeros((0, 2), np.int32),
        edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
        targets=targets,
        motifs=motifs,
        motif_names=np.asarray(motif_names),
        palette=np.asarray([list(c) for c in palette], dtype=np.float64),
    )


def load_pool_npz(path: str,
                  palette: t.Sequence[t.Sequence[float]] = RB_PALETTE,
                  ) -> t.Tuple[t.List[Data], t.List[dict]]:
    """Rebuild an encoded pool saved by :func:`save_pool_npz`.

    Verifies the stored palette against ``palette`` first: node class indices are
    meaningless without it, and a silent mismatch would train a model whose
    colours mean something other than the checkpoint claims.
    """
    blob = np.load(path, allow_pickle=False)
    stored = [tuple(row) for row in blob["palette"]]
    if stored != [tuple(c) for c in palette]:
        raise ValueError(f"pool {path} was built with a different palette: {stored}")

    node_classes, node_ptr = blob["node_classes"], blob["node_ptr"]
    edges, edge_ptr = blob["edges"], blob["edge_ptr"]
    targets, motifs = blob["targets"], blob["motifs"]
    motif_names = [str(s) for s in blob["motif_names"]]

    graphs, metadata = [], []
    for i in range(len(node_ptr) - 1):
        classes = torch.tensor(node_classes[node_ptr[i]:node_ptr[i + 1]], dtype=torch.long)
        pairs = edges[edge_ptr[i]:edge_ptr[i + 1]]
        n = len(classes)

        row = np.concatenate([pairs[:, 0], pairs[:, 1]]) if len(pairs) else np.zeros(0, np.int64)
        col = np.concatenate([pairs[:, 1], pairs[:, 0]]) if len(pairs) else np.zeros(0, np.int64)
        edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
        edge_attr = F.one_hot(torch.ones(edge_index.shape[1], dtype=torch.long),
                              num_classes=2).float()
        perm = (edge_index[0] * n + edge_index[1]).argsort()

        graphs.append(Data(
            x=F.one_hot(classes, num_classes=len(palette)).float(),
            edge_index=edge_index[:, perm],
            edge_attr=edge_attr[perm],
        ))
        metadata.append({
            "target": [float(targets[i])],
            "motifs": {name: int(v) for name, v in zip(motif_names, motifs[i])},
            "index": i,
        })
    return graphs, metadata


def save_concept_conditions(path: str, phi: np.ndarray, concept_ids: t.Sequence[str],
                            pool_path: str) -> str:
    """Write the ``(graph, φ)`` conditioning table for a colour-graph concept run.

    Unlike the molecular table (:func:`defog_megan.data.save_conditions`), there is
    no per-row identity to store alongside ``φ`` -- colour graphs have no SMILES,
    and unlike molecules there is no encode/decode round-trip loss to reconcile
    (:mod:`colorgraphs`' own module docstring: "the decode is lossless"). So ``φ``
    is aligned **by row position** against ``pool_path`` (:data:`RB_DUAL_MOTIFS_CACHE`
    normally) rather than carried with a copy of the graph itself: row ``i`` here
    describes row ``i`` of that pool. ``pool_path`` is stored so a consumer can
    verify the two files still agree on which pool they are talking about instead
    of silently pairing mismatched rows.

    :param phi: ``(N, n_concepts)`` membership matrix, aligned to the pool.
    :param concept_ids: column layout of ``phi``, so a mismatch fails loudly
        rather than training against a permuted vector.
    :param pool_path: the ``.npz`` these rows are aligned against.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        phi=np.asarray(phi, dtype=np.float32),
        concept_ids=np.asarray(list(concept_ids), dtype=np.str_),
        pool_path=np.asarray(str(pool_path)),
    )
    return path


def load_concept_conditions(path: str, pool_path: t.Optional[str] = None
                            ) -> t.Tuple[np.ndarray, t.List[str]]:
    """Read a table written by :func:`save_concept_conditions`.

    :param pool_path: if given, verified against the path recorded at save time --
        the whole point of storing it. Pass the path you are about to align
        ``phi`` against; a mismatch means row ``i`` here does not describe row
        ``i`` of the pool you are loading, which would train on scrambled
        conditions with no error anywhere else in the pipeline.

        Compared by **basename**, not full absolute path. A table is built on one
        machine and read on another, with different mount roots for what is the
        same logical pool file, so an ``os.path.abspath`` comparison fails on every
        cross-machine load, a false positive that basename comparison does not
        have. What this still catches is the thing worth catching -- ``pool.npz`` vs some other
        pool's name entirely.
    :returns: ``(phi, concept_ids)``.
    """
    bundle = np.load(path, allow_pickle=False)
    stored_pool = str(bundle["pool_path"])
    if pool_path is not None and os.path.basename(stored_pool) != os.path.basename(pool_path):
        raise ValueError(
            f"{path} was built against pool {stored_pool!r}, not {pool_path!r} -- "
            f"row alignment would be silently wrong")
    return (np.asarray(bundle["phi"], dtype=np.float32),
            [str(c) for c in bundle["concept_ids"]])


def build_pyg_dataset(root: str,
                      limit: t.Optional[int] = None,
                      palette: t.Sequence[t.Sequence[float]] = RB_PALETTE,
                      cache_path: t.Optional[str] = None,
                      verify_palette: bool = True) -> t.Tuple[t.List[Data], t.List[dict]]:
    """Load a VGD pool and encode it for DeFoG, with an optional on-disk cache.

    Reading 100k json files takes minutes, which is a poor thing to repeat once
    per experiment; ``cache_path`` (an ``.npz``, see :func:`save_pool_npz`) makes
    the second run a single load, and is also what gets shipped to the cluster
    where the raw pool does not exist.

    :returns: ``(pyg_graphs, metadata)`` aligned elementwise. Graphs that cannot
        be encoded (no edges) are dropped from *both*.
    """
    if cache_path and os.path.exists(cache_path):
        return load_pool_npz(cache_path, palette)

    records = load_vgd_dataset(root, limit=limit)
    if verify_palette:
        assert_palette((r["graph"] for r in records), palette)

    graphs, metadata = [], []
    for record in records:
        data = vgd_to_pyg(record["graph"], palette)
        if data is None:
            continue
        graphs.append(data)
        metadata.append({k: v for k, v in record.items() if k != "graph"})

    if cache_path:
        save_pool_npz(cache_path, graphs, metadata, palette)
    return graphs, metadata


# -- the domain ---------------------------------------------------------------

class ColorGraphDomain(GenericGraphDomain):
    """DeFoG domain adapter for palette-coloured graphs.

    Renders in the pool's *actual* colours rather than an arbitrary colormap, so
    a generated grid is directly comparable to the dataset's own ``.png``
    depictions. Identity is the Weisfeiler-Lehman hash over colour-labelled
    nodes, inherited from :class:`GenericGraphDomain`, which gives uniqueness and
    novelty for free.

    ``is_valid`` is left at the base class's unconditional ``True``: every colour
    graph is a well-formed member of this domain. Structural realism is a
    distributional question here, not a per-sample gate -- see
    :func:`structural_report`.
    """

    def __init__(self,
                 palette: t.Sequence[t.Sequence[float]] = RB_PALETTE,
                 names: t.Sequence[str] = RB_PALETTE_NAMES,
                 layout_seed: int = 42,
                 node_size: int = 90):
        super().__init__(layout_seed=layout_seed, node_size=node_size)
        self.palette = [tuple(c) for c in palette]
        self.names = list(names)

    def render(self, ax, data) -> None:
        import networkx as nx

        graph, n = self._to_nx(data)
        if n == 0:
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            return

        classes = data.x.argmax(dim=-1).tolist() if data.x.dim() == 2 else [0] * n
        colors = [self.palette[c % len(self.palette)] for c in classes]
        pos = nx.spring_layout(graph, seed=self.layout_seed)
        nx.draw_networkx(
            graph, pos, ax=ax,
            node_color=colors, node_size=self.node_size,
            edgecolors="#333333", linewidths=0.6,
            width=0.9, edge_color="#888888", with_labels=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def caption(self, data) -> t.Optional[str]:
        import networkx as nx

        graph, n = self._to_nx(data)
        if n == 0:
            return None
        connected = nx.is_connected(graph) if n > 0 and graph.number_of_nodes() else False
        return f"n={n} e={graph.number_of_edges()}{'' if connected else ' (disconn.)'}"

    def summarize(self, samples: t.Sequence[t.Any]) -> str:
        base = super().summarize(samples)
        if not samples:
            return base
        import networkx as nx

        n_conn = 0
        for sample in samples:
            graph, n = self._to_nx(sample)
            if n and nx.is_connected(graph):
                n_conn += 1
        return f"{base} | connected: {n_conn / len(samples):.0%}"


# -- distributional realism ---------------------------------------------------

def _discrete_kl(observed: np.ndarray, reference: np.ndarray,
                 smoothing: float = 1e-3) -> float:
    """KL(observed || reference) over a shared discrete support, with smoothing.

    Both inputs are count vectors on the same support. The smoothing keeps the
    divergence finite when the generator visits a bin the reference never did,
    which is the interesting case rather than an error.
    """
    p = np.asarray(observed, dtype=np.float64) + smoothing
    q = np.asarray(reference, dtype=np.float64) + smoothing
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _histogram(values: t.Iterable[int], size: int) -> np.ndarray:
    counts = np.zeros(size, dtype=np.float64)
    for v in values:
        if 0 <= v < size:
            counts[v] += 1
    return counts


def _graph_stats(samples: t.Sequence[t.Any], num_classes: int,
                 max_degree: int, max_nodes: int) -> dict:
    """Colour / degree / size histograms plus connectivity for a batch."""
    import networkx as nx

    colors = np.zeros(num_classes, dtype=np.float64)
    degrees = np.zeros(max_degree + 1, dtype=np.float64)
    sizes = np.zeros(max_nodes + 1, dtype=np.float64)
    n_connected = 0
    n_isolated = 0
    n_nodes_total = 0

    for sample in samples:
        x = sample.x
        n = int(x.shape[0])
        classes = x.argmax(dim=-1).tolist() if x.dim() == 2 else [0] * n
        colors += _histogram(classes, num_classes)
        sizes += _histogram([n], max_nodes + 1)

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        ei = getattr(sample, "edge_index", None)
        if ei is not None and ei.numel() > 0:
            for k in range(ei.shape[1]):
                u, v = int(ei[0, k]), int(ei[1, k])
                if u < v:
                    graph.add_edge(u, v)
        deg = [d for _, d in graph.degree()]
        degrees += _histogram(deg, max_degree + 1)
        n_isolated += sum(1 for d in deg if d == 0)
        n_nodes_total += n
        if n and nx.is_connected(graph):
            n_connected += 1

    return {
        "colors": colors, "degrees": degrees, "sizes": sizes,
        "connected_rate": n_connected / len(samples) if samples else 0.0,
        "isolated_rate": n_isolated / n_nodes_total if n_nodes_total else 0.0,
    }


def structural_report(samples: t.Sequence[t.Any],
                      reference: t.Sequence[t.Any],
                      num_classes: int = len(RB_PALETTE),
                      max_degree: int = 16,
                      max_nodes: int = 64) -> dict:
    """How closely generated colour graphs match the pool, distributionally.

    This stands in for the molecular ``validity`` column. There is no chemistry
    to violate, so instead of gating each sample we ask whether the *population*
    of generated graphs looks like the training pool along the three axes the
    generator could plausibly get wrong -- which colours it uses, how densely it
    wires them, and how big the graphs are -- and report connectivity rather than
    enforcing it.

    :returns: KL divergences (generated || reference, nats; lower is better),
        the two connectivity rates for both sets, and the raw means.
    """
    gen = _graph_stats(samples, num_classes, max_degree, max_nodes)
    ref = _graph_stats(reference, num_classes, max_degree, max_nodes)

    def _mean_size(stats):
        total = stats["sizes"].sum()
        return float((np.arange(len(stats["sizes"])) * stats["sizes"]).sum() / total) if total else 0.0

    def _mean_degree(stats):
        total = stats["degrees"].sum()
        return float((np.arange(len(stats["degrees"])) * stats["degrees"]).sum() / total) if total else 0.0

    return {
        "kl_colors": _discrete_kl(gen["colors"], ref["colors"]),
        "kl_degrees": _discrete_kl(gen["degrees"], ref["degrees"]),
        "kl_sizes": _discrete_kl(gen["sizes"], ref["sizes"]),
        "connected_rate": gen["connected_rate"],
        "connected_rate_ref": ref["connected_rate"],
        "isolated_rate": gen["isolated_rate"],
        "isolated_rate_ref": ref["isolated_rate"],
        "mean_nodes": _mean_size(gen),
        "mean_nodes_ref": _mean_size(ref),
        "mean_degree": _mean_degree(gen),
        "mean_degree_ref": _mean_degree(ref),
        "color_marginals": (gen["colors"] / max(gen["colors"].sum(), 1.0)).tolist(),
        "color_marginals_ref": (ref["colors"] / max(ref["colors"].sum(), 1.0)).tolist(),
    }


def distribution_score(report: dict) -> float:
    """Collapse a :func:`structural_report` into one higher-is-better scalar.

    Needed because DeFoG's training monitor selects the best checkpoint on a
    single ``validity`` number, and this domain has no validity. The obvious
    substitute -- connectivity -- is actively wrong on its own: an *over-dense*
    model wires nearly everything together and scores 1.0 for free. A 2-epoch
    smoke run produced exactly that, at mean degree 10.3 against the pool's 2.7.

    So connectivity is kept only as a multiplier, and the divergence terms do the
    real work: ``exp(-sum_kl)`` is 1.0 for a perfect distributional match and
    decays as the generator drifts in colour mix, wiring density or size.

    :returns: a score in ``[0, 1]``, 1.0 being an exact distributional match.
    """
    total_kl = report["kl_colors"] + report["kl_degrees"] + report["kl_sizes"]
    return float(np.exp(-total_kl) * report["connected_rate"])


def _to_nx(data):
    """PyG -> networkx with the colour class attached as the node label."""
    import networkx as nx

    g = nx.Graph()
    classes = np.asarray(data.x).argmax(axis=1) if getattr(data, "x", None) is not None \
        else np.zeros(data.num_nodes, dtype=int)
    for i in range(int(data.num_nodes)):
        g.add_node(i, color=int(classes[i]))
    ei = np.asarray(data.edge_index)
    if ei.size:
        for a, b in np.unique(np.sort(ei.T, axis=1), axis=0):
            if a != b:
                g.add_edge(int(a), int(b))
    return g


def wl_hash(data) -> str:
    """Isomorphism-invariant, colour-aware key of one graph.

    A Weisfeiler-Lehman hash (three iterations) over the graph with the node colour as the node
    label. It stands in for a SMILES string wherever graphs are counted as distinct (uniqueness,
    novelty, the distinct joint hits of the steering experiments). Edge direction and node order
    do not matter.
    """
    import networkx as nx

    return nx.weisfeiler_lehman_graph_hash(_to_nx(data), node_attr="color", iterations=3)
