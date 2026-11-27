"""Generate colored graphs for a requested property AND a requested concept, and check them.

The frozen unconditional DeFoG base ignores both requests. With a property adapter and a concept adapter on it, composed
by summing their log-ratios (the paper's joint arm, both at weight 1), most generated graphs have the requested property
and are a member of the requested concept. Both are read back exactly, the property from the motif counts of the graph
(the generator's own ground truth) and the concept from the explainer, so the table at the end is a real check and not a
self-report.

    python -m defog_megan.zoo fetch colorgraphs          # once, about 300 MB
    python examples/quickstart.py                        # the request below, 64 graphs per arm
    python examples/quickstart.py --concept ch1_cl12 --level 3 --n 128

The settings are those of the paper's runs (500 steps, eta 0, polydec time distortion, the adapters at weight 1 each),
except for the number of graphs. It needs the model zoo (``$DEFOG_MEGAN_ZOO``,
default ``~/.cache/defog-megan/zoo``). Use a GPU: the graphs are sampled in chunks (``--chunk``) so that a small card is
enough (the defaults took 13 minutes on a shared 6 GB RTX 2060, and the paper's runs suggest a couple of minutes on a recent
card), but on a CPU they take hours (try ``--n 8 --steps 50`` there, which shows the effect at lower quality).

Requests that the models serve well (concept, level): ``ch0_cl3`` at -3, ``ch1_cl12`` at 3, ``ch0_cl4`` at -3,
``ch0_cl7`` at 0. The property levels of the paper are -3, -1, 0, 1 and 3; the concepts are listed in
``<zoo>/colorgraphs/data/concept_table.json``.
"""
import argparse
import time

import networkx as nx
import numpy as np
import torch
from defog_adapt.core import (
    AdaptedSampler, AdapterComposition, ConditionBranch, CrossAttentionAdapter, DeFoGModel, Sampler,
)

from defog_megan.bundle import load_space
from defog_megan.colorgraphs import pyg_to_vgd
from defog_megan.data import load_targets
from defog_megan.rb_motifs import count_motifs_pyg, exact_target
from defog_megan.zoo import zoo_path

TOLERANCE = 1.0          # a graph has the property when its exact value is within this of the request
CORE = 0.5               # a graph is a member of a concept when its membership value is below this
CTMC = dict(eta=0.0, omega=0.0, time_distortion="polydec")


def connected(graph) -> bool:
    """A generated graph counts as valid when it is one connected piece."""
    g = nx.Graph()
    g.add_nodes_from(range(int(graph.x.shape[0])))
    if graph.edge_index.numel():
        edges = graph.edge_index.cpu().numpy()
        g.add_edges_from((int(a), int(b)) for a, b in zip(edges[0], edges[1]) if a < b)
    return g.number_of_nodes() > 0 and nx.is_connected(g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--concept", default="ch0_cl3", help="the requested concept")
    ap.add_argument("--level", type=float, default=-3.0, help="the requested property (-3, -1, 0, 1 or 3)")
    ap.add_argument("--n", type=int, default=64, help="graphs per arm")
    ap.add_argument("--steps", type=int, default=500, help="sampling steps (the paper's runs use 500)")
    ap.add_argument("--chunk", type=int, default=16, help="graphs sampled at once (lower it if the GPU runs out of memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when there is one)")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # the frozen base, the two adapters, and the request vectors of the concepts
    base = DeFoGModel.load(zoo_path("colorgraphs", "base.ckpt"), device="cpu").to(device).eval()
    prop = CrossAttentionAdapter.load(zoo_path("colorgraphs", "property_adapter.ckpt"), device="cpu").to(device).eval()
    conc = CrossAttentionAdapter.load(zoo_path("colorgraphs", "concept_adapter.ckpt"), device="cpu").to(device).eval()
    matrix, ids = load_targets(zoo_path("colorgraphs", "data/concept_targets.npz"), mode="empirical")
    ids = list(ids)
    if args.concept not in ids:
        raise SystemExit(f"unknown concept {args.concept!r}; the zoo has {ids}")
    space = load_space("colorgraphs")                      # the explainer that reads the concept of a graph back
    concept_index = space.index_of(args.concept)

    def sampler_for(arm: str, batch: int):
        """The sampler of an arm for a batch of ``batch`` graphs. The requests are the property as a number and the
        concept as its membership vector, one row per graph."""
        if arm == "frozen base":
            return Sampler(base, sample_steps=args.steps, **CTMC)
        property_request = torch.tensor([[args.level]], dtype=torch.float32, device=device).expand(batch, -1)
        concept_request = torch.tensor(matrix[ids.index(args.concept)], dtype=torch.float32, device=device).expand(batch, -1)
        both = AdapterComposition([ConditionBranch(conc, concept_request, 1.0), ConditionBranch(prop, property_request, 1.0)],
                                  base=base, mode="product")
        return AdaptedSampler(base, both, sample_steps=args.steps, **CTMC)

    print(f"request: concept {args.concept}, property {args.level:g} (within {TOLERANCE:g}); "
          f"{args.n} graphs per arm, {args.steps} steps on {device}")
    print(f"{'arm':24s} {'valid':>6s} {'property':>9s} {'concept':>8s} {'joint':>6s} {'seconds':>8s}")
    for name in ("frozen base", "base + both adapters"):
        torch.manual_seed(args.seed)
        started = time.time()
        graphs = []
        while len(graphs) < args.n:
            batch = min(args.chunk, args.n - len(graphs))
            graphs += sampler_for(name, batch).sample(num_samples=batch, device=device, show_progress=False)
        seconds = time.time() - started
        valid = np.array([connected(g) for g in graphs])
        value = np.array([exact_target(count_motifs_pyg(g)) for g in graphs], dtype=float)
        member = space.phi_from_graphs([pyg_to_vgd(g) for g in graphs])[:, concept_index] < CORE
        has_property = np.abs(value - args.level) <= TOLERANCE
        joint = valid & has_property & member
        print(f"{name:24s} {valid.mean():6.2f} {has_property.mean():9.2f} {member.mean():8.2f} {joint.mean():6.2f} {seconds:8.0f}")
    print("joint = a connected graph with the requested property that is a member of the requested concept")


if __name__ == "__main__":
    main()
