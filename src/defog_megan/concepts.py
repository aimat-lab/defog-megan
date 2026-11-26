"""
The ``φ`` pipeline: molecule -> MEGAN concept-membership vector.

A :class:`ConceptSpace` bundles the three artifacts that jointly define a concept
vocabulary -- a trained MEGAN model, the HDBSCAN concept clustering extracted from
its explanation latent space (a ``.clu`` archive), and the dataset's own
``process.py`` feature encoder -- and exposes the single function this project is
built around::

    space = ConceptSpace.from_paths(model_path, clustering_path, processing_path)
    phi = space.phi(["CCO", "c1ccccc1"])        # (N, n_concepts)

``phi`` follows the *sharpened-distance* convention of
``megan_global_explanations``: **lower means better fit**, ``0.0`` is a core member
of that concept and ``1.0`` means "not assigned". A molecule whose per-channel
fidelity falls below ``fidelity_threshold`` gets all-``1.0`` in that channel, i.e.
"this channel explains nothing here" -- which is also the natural *no-concept*
condition (:meth:`ConceptSpace.null_vector`).

The vector layout is the canonical ordering of :attr:`ConceptSpace.concept_ids`
(sorted by channel, then by cluster index). This ordering is what an adapter gets
trained against, so it must be persisted alongside any trained adapter -- see
:meth:`ConceptSpace.spec`.
"""

import os
import typing as t

import numpy as np


# Concept ids produced by megan_global_explanations look like ``ch0_cl7``.
def parse_concept_id(concept_id: str) -> t.Tuple[int, int]:
    """Split a ``ch{channel}_cl{index}`` id into its ``(channel, index)`` pair."""
    channel_part, cluster_part = concept_id.split("_cl")
    return int(channel_part[2:]), int(cluster_part)


def sort_concept_ids(concept_ids: t.Iterable[str]) -> t.List[str]:
    """Canonical ordering: by channel, then by cluster index.

    This reproduces the column order of the ``embeddings_scores.csv`` emitted by
    the ``vgd_concept_extraction`` experiments, so a vector computed live is
    directly comparable to the archived scores.
    """
    return sorted(concept_ids, key=parse_concept_id)


def compute_fidelity(deviation: np.ndarray, dataset_type: str) -> np.ndarray:
    """Per-channel fidelity from a MEGAN leave-one-out deviation matrix.

    Mirrors the formula used inside ``vgd_concept_extraction`` (and
    ``predict_and_score.py``) so that ``fidelity_threshold`` keeps the meaning it
    was calibrated with when the clustering was produced.

    :param deviation: ``(O, K)`` leave-one-out deviation for a single graph.
    :param dataset_type: ``'regression'`` or ``'classification'``.
    :returns: ``(K,)`` fidelity vector.
    """
    dev = np.asarray(deviation)
    if dataset_type == "regression":
        # 2-channel regressor: channel 0 collects negative, channel 1 positive
        # contributions, so channel 0's deviation is negated to make "more
        # explanatory" mean "larger" on both channels.
        return np.array([-dev[0, 0], dev[0, 1]])
    if dataset_type == "classification":
        return np.diag(np.asarray(dev))
    raise ValueError(f"unknown dataset_type: {dataset_type!r}")


class ConceptSpace:
    """A MEGAN model plus its concept clustering, exposed as ``molecule -> φ``.

    :param model: A loaded ``Megan`` instance.
    :param clustering: A loaded ``Clustering`` instance (from a ``.clu`` archive).
    :param processing: The dataset's ``ProcessingBase`` instance -- **must** be the
        one the model was trained with, or the node/edge feature widths will not
        line up with the model's input layer.
    :param fidelity_threshold: Channels below this fidelity are reported as
        entirely unassigned (all ``1.0``). Match the value used when the
        clustering was extracted.
    :param dataset_type: ``'regression'`` or ``'classification'``.
    :param score_method: ``'knn'`` (mean of the ``score_k`` nearest member
        distances) or ``'distance'`` (distance to the cluster representative).
    :param score_k: Neighbour count for ``'knn'``.
    :param sharpen: ``'sparsemax'``, ``'softmax'`` or ``None`` for raw distances.
    :param sharpen_temperature: Lower = more decisive assignments.
    """

    def __init__(
        self,
        model,
        clustering,
        processing,
        fidelity_threshold: float = 0.75,
        dataset_type: str = "regression",
        score_method: str = "knn",
        score_k: int = 5,
        sharpen: t.Optional[str] = "sparsemax",
        sharpen_temperature: float = 0.5,
    ):
        self.model = model
        self.clustering = clustering
        self.processing = processing
        self.fidelity_threshold = float(fidelity_threshold)
        self.dataset_type = dataset_type
        self.score_method = score_method
        self.score_k = int(score_k)
        self.sharpen = sharpen
        self.sharpen_temperature = float(sharpen_temperature)

        self.channels: t.List[int] = sorted(clustering.channels.keys())
        ids: t.List[str] = []
        for channel in self.channels:
            for cluster in clustering.channels[channel]["clusters"]:
                ids.append(f"ch{channel}_cl{cluster['index']}")
        self.concept_ids: t.List[str] = sort_concept_ids(ids)
        self._index_of = {cid: i for i, cid in enumerate(self.concept_ids)}

    # -- construction --------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        model_path: str,
        clustering_path: str,
        processing_path: str,
        **kwargs,
    ) -> "ConceptSpace":
        """Load all three artifacts from disk.

        :param model_path: ``.ckpt`` of the trained MEGAN model.
        :param clustering_path: ``.clu`` archive from a concept-extraction run.
        :param processing_path: ``process.py`` shipped with the visual graph
            dataset the model was trained on.
        """
        from graph_attention_student.torch.megan import Megan
        from megan_global_explanations.data import Clustering
        from visual_graph_datasets.util import dynamic_import

        for path, what in ((model_path, "model"), (clustering_path, "clustering"),
                           (processing_path, "processing")):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{what} not found: {path}")

        model = Megan.load(model_path)
        model.eval()
        processing = dynamic_import(processing_path).processing
        return cls(model, Clustering.load(clustering_path), processing, **kwargs)

    # -- properties ----------------------------------------------------------

    @property
    def num_concepts(self) -> int:
        """Dimensionality of ``φ`` -- i.e. the adapter's ``cond_dim``."""
        return len(self.concept_ids)

    def index_of(self, concept_id: str) -> int:
        """Position of a concept in the ``φ`` vector."""
        return self._index_of[concept_id]

    # -- the phi pipeline ----------------------------------------------------

    def graphs_from_smiles(
        self, smiles: t.Sequence[str]
    ) -> t.Tuple[t.List[dict], np.ndarray]:
        """Encode SMILES into MEGAN graph dicts.

        Molecules the processing instance cannot encode are dropped; the returned
        boolean mask says which inputs survived, so callers can align results back
        onto their original list.

        :returns: ``(graphs, ok_mask)`` with ``ok_mask`` of shape ``(len(smiles),)``.
        """
        graphs, ok = [], np.zeros(len(smiles), dtype=bool)
        for i, smi in enumerate(smiles):
            try:
                graphs.append(self.processing.process(smi))
                ok[i] = True
            except Exception:
                continue
        return graphs, ok

    def phi_from_graphs(self, graphs: t.Sequence[dict], batch_size: int = 512) -> np.ndarray:
        """Concept-membership vectors for already-encoded graph dicts.

        :returns: ``(len(graphs), num_concepts)`` array of sharpened distances.
        """
        if not len(graphs):
            return np.ones((0, self.num_concepts))

        graphs = list(graphs)
        infos = self.model.forward_graphs(graphs, batch_size=batch_size)
        deviations = self.model.leave_one_out_deviations(graphs, batch_size=batch_size)

        out = np.ones((len(graphs), self.num_concepts), dtype=np.float64)
        for row, (info, dev) in enumerate(zip(infos, deviations)):
            embedding = np.asarray(info["graph_embedding"])   # (D, K)
            fidelity = compute_fidelity(dev, self.dataset_type)
            for channel in self.channels:
                scores = self.clustering.score(
                    embedding[:, channel],
                    channel=channel,
                    method=self.score_method,
                    k=self.score_k,
                    sharpen=self.sharpen,
                    sharpen_temperature=self.sharpen_temperature,
                    fidelity=float(fidelity[channel]),
                    fidelity_threshold=self.fidelity_threshold,
                )
                for concept_id, score in scores.items():
                    out[row, self._index_of[concept_id]] = score
        return out

    def phi(
        self, smiles: t.Sequence[str], batch_size: int = 512, return_mask: bool = False
    ):
        """Concept-membership vectors for a list of SMILES.

        Molecules that cannot be encoded are represented by the null (all-``1.0``)
        vector, so the output always has one row per input.

        :param return_mask: Also return the boolean mask of encodable molecules.
        :returns: ``(N, num_concepts)`` array, or ``(array, mask)``.
        """
        graphs, ok = self.graphs_from_smiles(smiles)
        out = np.ones((len(smiles), self.num_concepts), dtype=np.float64)
        if len(graphs):
            out[ok] = self.phi_from_graphs(graphs, batch_size=batch_size)
        return (out, ok) if return_mask else out

    # -- condition vectors for generation ------------------------------------

    def null_vector(self) -> np.ndarray:
        """The *no concept* condition: unassigned everywhere."""
        return np.ones(self.num_concepts, dtype=np.float64)

    def target_vector(self, concept_id: str, mode: str = "empirical") -> np.ndarray:
        """The condition vector that asks for one specific concept.

        Two conventions, because "give me concept X" does not have a single
        obviously-correct encoding:

        * ``'ideal'`` -- ``0.0`` at ``concept_id``, ``1.0`` everywhere else. Clean
          semantics and exactly what a dropdown implies, but slightly out of
          distribution: real sparsemax outputs often split mass across a few
          neighbouring concepts rather than committing fully to one.
        * ``'empirical'`` (default) -- the mean ``φ`` of the concept's own cluster
          members, i.e. "what a molecule of this concept actually looks like to
          the scorer". In-distribution with respect to adapter training.

        :param concept_id: e.g. ``'ch0_cl7'``.
        :param mode: ``'empirical'`` or ``'ideal'``.
        """
        if concept_id not in self._index_of:
            raise KeyError(f"unknown concept {concept_id!r}")

        if mode == "ideal":
            vector = self.null_vector()
            vector[self._index_of[concept_id]] = 0.0
            return vector

        if mode == "empirical":
            channel, _ = parse_concept_id(concept_id)
            members = np.asarray(self.clustering[concept_id]["members"])
            if not len(members):
                return self.target_vector(concept_id, mode="ideal")
            rows = np.ones((len(members), self.num_concepts), dtype=np.float64)
            for i, member in enumerate(members):
                # Members are, by construction, above the fidelity threshold in
                # their own channel; the gate is therefore not applied here.
                scores = self.clustering.score(
                    member, channel=channel, method=self.score_method, k=self.score_k,
                    sharpen=self.sharpen, sharpen_temperature=self.sharpen_temperature,
                )
                for cid, score in scores.items():
                    rows[i, self._index_of[cid]] = score
            return rows.mean(axis=0)

        raise ValueError(f"unknown mode: {mode!r}")

    # -- metadata ------------------------------------------------------------

    def concept_table(self) -> t.List[dict]:
        """One row per concept -- the data a selection UI (dropdown) needs.

        Includes the concept's channel, its member count and, when the extraction
        recorded it, the mean contribution of the concept to the model's
        prediction (positive = pushes the predicted property up).
        """
        table = []
        for concept_id in self.concept_ids:
            channel, index = parse_concept_id(concept_id)
            cluster = self.clustering[concept_id]
            row = {
                "concept_id": concept_id,
                "position": self._index_of[concept_id],
                "channel": channel,
                "cluster_index": index,
                "channel_name": self.clustering.channels[channel].get("name", ""),
                "num_members": int(len(np.asarray(cluster["members"]))),
            }
            metadata = cluster["metadata"] if "metadata" in cluster else {}
            row["name"] = metadata.get("name", "")
            row["color"] = metadata.get("color", "")
            table.append(row)
        return table

    def spec(self) -> dict:
        """The scoring contract, for persisting next to a trained adapter.

        An adapter is only meaningful together with the exact ``φ`` that produced
        its training conditions, so this records both the vector layout and every
        scoring knob that would silently change that vector.
        """
        return {
            "concept_ids": list(self.concept_ids),
            "num_concepts": self.num_concepts,
            "channels": list(self.channels),
            "fidelity_threshold": self.fidelity_threshold,
            "dataset_type": self.dataset_type,
            "score_method": self.score_method,
            "score_k": self.score_k,
            "sharpen": self.sharpen,
            "sharpen_temperature": self.sharpen_temperature,
        }
