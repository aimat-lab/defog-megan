"""Feynman-Kac energy over (concept membership, exact motif sum) on colour graphs.

The colour-graph counterpart of :class:`~defog_megan.joint_energy.MeganJointEnergy`, and
simpler than it in one important way: the property term is EXACT. The motif sum is
recomputed from structure by :mod:`defog_megan.rb_motifs`, so only the concept term needs
MEGAN, and nothing about the property half of the energy is a learned approximation.

**The two terms are already on comparable scales.** The concept term is MEGAN's sharpened
phi distance, bounded in [0, 1] with the core-member threshold at 0.5. The property term
reuses :class:`~defog_megan.color_energy.ColorTargetEnergy`'s normalisation unchanged --
``min((err / 3)^2, 2)``, so a one-ring miss costs 0.11 and a three-motif miss costs 1.0 --
which is what the FK rows of Table 1 were steered with. Keeping it identical means the
joint+FK arm differs from those rows only by the concept term it adds. ``lam`` weights the
property term against the concept term.

**sum vs max.** ``mode="sum"`` is ``phi + lam * prop``, the same form the molecular joint
energy uses. ``mode="max"`` is the bottleneck that
:class:`~defog_megan.color_multi_concept_fk.MultiConceptEnergy` argues for on a
conjunction. Over two terms with different scales the bottleneck is dominated by whichever
term is larger, so ``sum`` is the default and ``max`` is kept for contrast.

Disconnected graphs are tiered above anything a connected graph can score, as in every
other energy in this project, so a fragment can never outrank a real graph.
"""

import typing as t

import numpy as np
import torch

from defog_megan.color_energy import ENERGY_CLAMP, ERROR_SCALE
from defog_megan.rb_motifs import count_motifs_pyg, exact_target


class ColorJointEnergy:
    """``energy_fn(X1, E1, node_mask) -> (K,)``; lower is better.

    :param space: a loaded :class:`~defog_megan.concepts.ConceptSpace` for the colour
        vocabulary.
    :param concept_index: the phi column to drive down.
    :param target: the requested exact motif sum.
    :param lam: weight on the property term relative to the concept term.
    :param mode: ``"sum"`` or ``"max"``, see the module docstring.
    """

    def __init__(self, space, concept_index: int, target: float, lam: float = 1.0,
                 mode: str = "sum", error_scale: float = ERROR_SCALE,
                 clamp: float = ENERGY_CLAMP, disconnect_energy: t.Optional[float] = None):
        assert mode in ("sum", "max"), f"unknown mode {mode!r}"
        # The worst a connected graph can score: phi at its ceiling plus a clamped miss.
        worst = 1.0 + lam * clamp if mode == "sum" else max(1.0, lam * clamp)
        if disconnect_energy is None:
            disconnect_energy = worst + 1.0
        if disconnect_energy <= worst:
            raise ValueError(
                f"disconnect_energy ({disconnect_energy}) must exceed the worst connected "
                f"score ({worst}), or a disconnected graph can outrank a real one")
        self.space = space
        self.concept_index = int(concept_index)
        self.target = float(target)
        self.lam = float(lam)
        self.mode = mode
        self.error_scale = float(error_scale)
        self.clamp = float(clamp)
        self.disconnect_energy = float(disconnect_energy)
        self.n_calls = 0
        self.last_stats: dict = {}

    @torch.no_grad()
    def __call__(self, X1, E1, node_mask) -> torch.Tensor:
        import networkx as nx
        from defog_adapt.core.data import dense_to_pyg
        from defog_megan.colorgraphs import pyg_to_vgd

        self.n_calls += 1
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.disconnect_energy)

        connected_idx, connected_vgd, prop_terms = [], [], []
        for i, data in enumerate(datas):
            size = int(data.x.shape[0])
            graph = nx.Graph()
            graph.add_nodes_from(range(size))
            ei = data.edge_index
            if ei.numel():
                arr = ei.cpu().numpy()
                graph.add_edges_from(
                    (int(a), int(b)) for a, b in zip(arr[0], arr[1]) if a < b)
            if size and nx.is_connected(graph):
                err = (float(exact_target(count_motifs_pyg(data))) - self.target) \
                    / self.error_scale
                connected_idx.append(i)
                connected_vgd.append(pyg_to_vgd(data))
                prop_terms.append(min(err * err, self.clamp))

        phi = None
        if connected_vgd:
            phi = np.clip(self.space.phi_from_graphs(connected_vgd)[:, self.concept_index],
                          0.0, 1.0)
            prop = np.asarray(prop_terms)
            combined = (phi + self.lam * prop if self.mode == "sum"
                        else np.maximum(phi, self.lam * prop))
            for idx, value in zip(connected_idx, combined):
                out[idx] = float(value)

        self.last_stats = {
            "n": len(datas),
            "connected_rate": len(connected_idx) / max(len(datas), 1),
            "mean_phi": float(phi.mean()) if phi is not None else float("nan"),
            "mean_prop": float(np.mean(prop_terms)) if prop_terms else float("nan"),
        }
        return out
