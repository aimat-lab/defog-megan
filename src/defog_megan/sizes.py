"""Graph-size distributions for the molecular rungs.

Generation draws each graph's node count before denoising begins. By default that comes from the *base model's* training
marginal, which for the ZINC base means about 23 heavy atoms. A dataset the adapter is trained on can live elsewhere (the
Ames molecules are smaller), and then conditioned generation is asked to place a small motif inside a large graph: the
surplus atoms have to go somewhere, and the plausibility failures observed (macrocycles, disconnected fragments) are what
padding a small motif out to a large graph looks like. Drawing the size from the dataset's own marginal removes that
confound at its source.

Three helpers, all independent of the condition: :class:`DatasetSizeDistribution` draws every graph's size from one
dataset histogram, :func:`dataset_size_histogram` builds that histogram from SMILES, and :func:`base_size_histogram` reads
the frozen base's own marginal. The paper's sampling runs read the histogram of the dataset from ``sizes.json`` of the zoo
instead (``defog_adapt``'s ``EmpiricalSizeDistribution``). The concept-conditional size distribution of the prototype
appendix belongs to the appendix code.
"""
import typing as t

import numpy as np
import torch

from defog_adapt.core.size_distribution import SizeDistribution


class DatasetSizeDistribution(SizeDistribution):
    """Draw every graph's size from ONE dataset histogram, ignoring the condition.

    For a property that correlates with size the choice is a claim about what is being measured, not a plumbing detail.
    Drawing the size conditional on the requested value would let the sampler satisfy much of an extreme target by making
    the graph big, and the resulting error would credit the adapter for it. Drawing from the dataset marginal instead fixes
    only the base/dataset mismatch and leaves every bit of the steering to the adapter.

    :param histogram: ``(sizes, counts)`` over the dataset's heavy-atom counts.
    :param max_nodes: hard ceiling from the base model; larger sizes are dropped because the model cannot represent them.
    :param min_nodes: the base's floor, for the same reason.
    """

    def __init__(self, histogram, max_nodes: int = 38, min_nodes: int = 1):
        self.max_nodes = int(max_nodes)
        self.min_nodes = int(min_nodes)
        sizes, counts = histogram
        sizes = np.asarray(sizes)
        counts = np.asarray(counts, dtype=np.float64)
        keep = (sizes >= self.min_nodes) & (sizes <= self.max_nodes)
        assert keep.any(), (
            f"no dataset sizes inside the base's [{self.min_nodes}, {self.max_nodes}] "
            f"support -- the histogram and the base do not belong together")
        self.histogram = (sizes[keep], counts[keep])

    @property
    def max_size(self) -> int:
        return int(self.histogram[0].max())

    def sample(self, num_samples, condition=None, device=None, generator=None):
        # `condition` is accepted and deliberately ignored; see the class docstring.
        if generator is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        sizes, counts = self.histogram
        drawn = rng.choice(sizes, size=num_samples, p=counts / counts.sum())
        return torch.as_tensor(drawn, dtype=torch.long,
                               device=device if device is not None else "cpu")


def dataset_size_histogram(smiles: t.Sequence[str]) -> t.Tuple[np.ndarray, np.ndarray]:
    """``(sizes, counts)`` of heavy-atom counts over a set of molecules."""
    from rdkit import Chem

    counts: t.Dict[int, int] = {}
    for entry in smiles:
        mol = Chem.MolFromSmiles(entry)
        if mol is None:
            continue
        n = mol.GetNumAtoms()
        if n >= 1:
            counts[n] = counts.get(n, 0) + 1
    assert counts, "no molecules parsed -- cannot build a size histogram"
    keys = np.array(sorted(counts), dtype=np.int64)
    return keys, np.array([float(counts[int(k)]) for k in keys])


def base_size_histogram(model, max_nodes: t.Optional[int] = None):
    """``(sizes, counts)`` of the frozen base's own training size marginal, or ``None`` if it has none.

    The pmf lives at ``model.node_dist.prob`` (a ``DistributionNodes``), NOT at a plain ``node_counts`` attribute.
    Getting this wrong does not raise: a fallback would be synthesised quietly, so an *unconditional* baseline would end
    up drawn at other sizes than the base's own, which is precisely the comparison this histogram exists to preserve.
    """
    distribution = getattr(model, "node_dist", None)
    counts = getattr(distribution, "prob", None) if distribution is not None else None
    if counts is None:
        return None
    counts = torch.as_tensor(counts).reshape(-1).detach().cpu().numpy()
    sizes = np.arange(len(counts), dtype=np.int64)
    keep = counts > 0
    sizes, counts = sizes[keep], counts[keep].astype(np.float64)
    if max_nodes is not None:
        keep = sizes <= max_nodes
        sizes, counts = sizes[keep], counts[keep]
    return (sizes, counts) if len(sizes) else None
