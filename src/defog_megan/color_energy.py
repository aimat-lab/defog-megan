"""
The Feynman-Kac energy for the colored graphs: exact motif counting, one target motif sum.

``ColorTargetEnergy`` is the property-only energy of the paper's colored-graph rung (Table 1, the Feynman-Kac row);
the joint energy with the concept term (``color_joint_energy``) reuses its normalisation unchanged.

Colored graphs have no invalid samples (any colored graph is a well-formed member of the domain), but they can be
**disconnected**: the pool is fully connected and the base reaches 99.4 %, so a disconnected graph is off-distribution.
It gets its own energy tier, above the clamp, so that without it the cheapest way to hit a motif count cannot be to plant
motifs on fragments that no longer form one graph. The target is a motif sum; the energy scores against the noise-free
exact sum recomputed from the generated graph.
"""

from __future__ import annotations

import torch

from defog_megan.rb_motifs import count_motifs_pyg, exact_target

#: Motif-sum error mapping to one unit of energy. The pool's own target sd is ~2.0 over the seven integer levels, so 3.0
#: keeps the whole in-support range inside the linear region while saturating well before the disconnected tier.
ERROR_SCALE: float = 3.0

#: Energy for a disconnected graph. Above the clamp so FK culls them first.
DISCONNECT_ENERGY: float = 4.0
#: Ceiling on the squared error term.
ENERGY_CLAMP: float = 2.0


class ColorTargetEnergy:
    """FK potential source: squared normalised distance from ONE target motif sum.

    ``(X1, E1, node_mask) -> (K,)``, lower = better, single fixed target -- the FK
    contract, which carries no per-row condition.

    Normalised by ``error_scale`` so that ``beta`` reads as a log-weight ratio: the energy gap between a good and a
    bad particle is about 1.
    """

    def __init__(self, target: float, error_scale: float = ERROR_SCALE,
                 clamp: float = ENERGY_CLAMP,
                 disconnect_energy: float = DISCONNECT_ENERGY):
        if disconnect_energy <= clamp:
            raise ValueError(
                f"disconnect_energy ({disconnect_energy}) must exceed the clamp "
                f"({clamp}), or a split graph can outrank a connected one")
        self.target = float(target)
        self.error_scale = float(error_scale)
        self.clamp = float(clamp)
        self.disconnect_energy = float(disconnect_energy)
        self.last_stats: dict = {}

    def __call__(self, X1, E1, node_mask) -> torch.Tensor:
        import networkx as nx
        from defog_adapt.core.data import dense_to_pyg

        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.disconnect_energy)
        n_ok = 0
        for i, data in enumerate(datas):
            size = int(data.x.shape[0])
            graph = nx.Graph()
            graph.add_nodes_from(range(size))
            ei = data.edge_index
            if ei.numel():
                arr = ei.cpu().numpy()
                graph.add_edges_from(
                    (int(a), int(b)) for a, b in zip(arr[0], arr[1]) if a < b)
            if size == 0 or not nx.is_connected(graph):
                continue
            err = (float(exact_target(count_motifs_pyg(data))) - self.target) / self.error_scale
            out[i] = min(err * err, self.clamp)
            n_ok += 1
        self.last_stats = {"n": len(datas), "connected_rate": n_ok / max(len(datas), 1)}
        return out
