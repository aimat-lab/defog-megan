"""
Concept round-trip evaluation.

The question this project has to answer is not "did the loss go down" but *did the
molecules actually come out containing the requested motif*. Because the condition
is produced by a model we still have on hand, that is directly checkable: generate
under a target concept, decode, push the results back through MEGAN and the same
clustering, and see where they land.

:func:`concept_hit_rate` is that measurement. It is deliberately reported against
an unconditional baseline, because some concepts are simply common -- a hit rate
of 40% means nothing until you know the base rate was 35%.
"""

import typing as t

import numpy as np
from rdkit import Chem


#: A generated molecule counts as *reaching* a concept when its sharpened distance
#: to that concept is below this. 1.0 means "not assigned at all", so anything
#: below it is a genuine (if partial) assignment by the scorer.
ASSIGNED = 0.999

#: Stricter bar: sparsemax has committed most of its mass to this concept.
CORE = 0.5


def decode_samples(samples, atom_decoder, bond_decoder) -> t.Tuple[t.List[str], dict]:
    """Decode generated PyG graphs into canonical SMILES.

    A vocabulary with charged atom classes (the Ames base) is read strictly, any other the relaxed way, as in
    :mod:`defog_megan.molecules`.

    :returns: ``(smiles, stats)`` where ``smiles`` holds only the valid molecules
        and ``stats`` carries validity/uniqueness over the full sample list.
    """
    from defog_megan.molecules import mol_to_smiles, pyg_data_to_mol

    valid = []
    for sample in samples:
        mol = pyg_data_to_mol(sample, atom_decoder, bond_decoder)
        smiles = mol_to_smiles(mol) if mol is not None else None
        if smiles is not None and Chem.MolFromSmiles(smiles) is not None:
            valid.append(smiles)
    total = len(samples)
    unique = set(valid)
    return valid, {
        "n_sampled": total,
        "n_valid": len(valid),
        "validity": len(valid) / total if total else 0.0,
        "n_unique": len(unique),
        "uniqueness": len(unique) / len(valid) if valid else 0.0,
    }


def concept_hit_rate(
    space,
    smiles: t.Sequence[str],
    concept_id: str,
    threshold: float = ASSIGNED,
    batch_size: int = 256,
    scores: t.Optional[np.ndarray] = None,
) -> dict:
    """How often a set of molecules actually reaches ``concept_id``.

    :param space: The :class:`~defog_megan.concepts.ConceptSpace` that defines φ.
    :param smiles: Decoded molecules to score (typically the valid generations).
    :param concept_id: The concept that was asked for.
    :param threshold: Score below which the concept counts as reached.
    :param scores: Optional precomputed ``(N,)`` column of φ for this concept. The
        same molecule set is typically scored against many concepts (a baseline is
        reused by every one), and φ is a full ``(N, n_concepts)`` matrix anyway --
        so computing it once and slicing is strictly cheaper than re-running MEGAN
        per concept.
    :returns: hit rate, mean/median score, and the fraction reaching :data:`CORE`.
    """
    if not len(smiles):
        return {"n": 0, "hit_rate": float("nan"), "core_rate": float("nan"),
                "mean_score": float("nan"), "median_score": float("nan")}

    if scores is None:
        index = space.index_of(concept_id)
        scores = space.phi(list(smiles), batch_size=batch_size)[:, index]
    scores = np.asarray(scores)
    return {
        "n": int(len(scores)),
        "hit_rate": float((scores < threshold).mean()),
        "core_rate": float((scores < CORE).mean()),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
    }


#: A concept needs at least this many baseline hits before its lift means
#: anything. Below it, the baseline rate is dominated by sampling noise and
#: enrichment degenerates to inf -- which reads as a spectacular result when it
#: is really "we could not measure this".
MIN_BASELINE_HITS = 3


def evaluate_concept(
    space,
    generated: t.Sequence[str],
    baseline: t.Sequence[str],
    concept_id: str,
    threshold: float = ASSIGNED,
    batch_size: int = 256,
    min_baseline_hits: int = MIN_BASELINE_HITS,
    generated_phi: t.Optional[np.ndarray] = None,
    baseline_phi: t.Optional[np.ndarray] = None,
) -> dict:
    """Steering result for one concept, against an unconditional baseline.

    ``lift`` is the headline number: how much more often the requested concept is
    reached than it would have been by chance. ``enrichment`` expresses the same
    thing as a ratio, which is the more meaningful framing for rare concepts.

    Pass ``generated_phi`` / ``baseline_phi`` (full ``(N, n_concepts)`` matrices) to
    skip re-scoring. A baseline is shared by every concept, so recomputing it per
    concept dominates the runtime of a full sweep.

    ``measurable`` says whether the baseline saw this concept often enough for
    either number to mean anything. Rare concepts routinely produce zero baseline
    hits, and the resulting infinite enrichment is an artifact, not a finding --
    callers should summarize over measurable concepts and report the rest
    separately.
    """
    index = space.index_of(concept_id)
    conditioned = concept_hit_rate(
        space, generated, concept_id, threshold, batch_size,
        scores=None if generated_phi is None else generated_phi[:, index])
    reference = concept_hit_rate(
        space, baseline, concept_id, threshold, batch_size,
        scores=None if baseline_phi is None else baseline_phi[:, index])

    lift = conditioned["hit_rate"] - reference["hit_rate"]
    base_rate = reference["hit_rate"]
    baseline_hits = base_rate * reference["n"] if reference["n"] else 0.0
    enrichment = (conditioned["hit_rate"] / base_rate) if base_rate > 0 else float("inf")
    return {
        "concept_id": concept_id,
        "conditioned": conditioned,
        "baseline": reference,
        "lift": float(lift),
        "core_lift": float(conditioned["core_rate"] - reference["core_rate"]),
        "enrichment": float(enrichment),
        "baseline_hits": float(baseline_hits),
        "measurable": bool(baseline_hits >= min_baseline_hits),
        # Distance shrinking is the smoother signal: it moves even when the
        # binary hit rate has not yet crossed the threshold, and it stays
        # meaningful for concepts the baseline never reaches.
        "score_delta": float(conditioned["mean_score"] - reference["mean_score"]),
    }


def predict_property(space, smiles: t.Sequence[str], batch_size: int = 256) -> np.ndarray:
    """MEGAN's predicted property for each molecule (empty array if none encode)."""
    graphs, _ = space.graphs_from_smiles(list(smiles))
    if not graphs:
        return np.array([])
    infos = space.model.forward_graphs(graphs, batch_size=batch_size)
    return np.array([float(np.asarray(i["graph_output"]).ravel()[0]) for i in infos])


def predict_class_probability(space, smiles: t.Sequence[str], batch_size: int = 256,
                              class_index: int = 1) -> np.ndarray:
    """MEGAN's predicted probability of class ``class_index`` (softmax of its logits) for each
    molecule; empty array if none encode. For a classification MEGAN, whose ``graph_output`` holds one
    logit per class."""
    graphs, _ = space.graphs_from_smiles(list(smiles))
    if not graphs:
        return np.array([])
    infos = space.model.forward_graphs(graphs, batch_size=batch_size)
    out = []
    for i in infos:
        z = np.asarray(i["graph_output"], dtype=np.float64).ravel()
        z = np.exp(z - z.max())
        out.append(float(z[class_index] / z.sum()))
    return np.array(out)


def property_readout(kind: str):
    """The per-molecule property function of a rung: ``"regression"`` (MEGAN's predicted value, logS) or
    ``"class_probability"`` (MEGAN's probability of the positive class, mutagenicity)."""
    if kind == "regression":
        return predict_property
    if kind == "class_probability":
        return predict_class_probability
    raise ValueError(f"unknown property kind {kind!r}")


def property_shift(space, generated: t.Sequence[str], baseline: t.Sequence[str],
                   batch_size: int = 256,
                   baseline_values: t.Optional[np.ndarray] = None) -> dict:
    """Mean shift in MEGAN's predicted property, conditioned vs baseline.

    A secondary sanity signal: a concept carries a measured contribution to the
    prediction, so steering into it should move the predicted property in that
    direction. Independent of the cluster-distance metric, so it fails differently.
    """
    gen = predict_property(space, generated, batch_size)
    base = (np.asarray(baseline_values) if baseline_values is not None
            else predict_property(space, baseline, batch_size))
    return {
        "generated_mean": float(gen.mean()) if gen.size else float("nan"),
        "baseline_mean": float(base.mean()) if base.size else float("nan"),
        "shift": float(gen.mean() - base.mean()) if gen.size and base.size else float("nan"),
    }
