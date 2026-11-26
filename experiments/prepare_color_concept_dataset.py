"""
Build the ``(graph, φ)`` training table for a colour-graph concept adapter.

Mirrors ``prepare_concept_dataset.py``, but simpler in one respect and stricter in
another.

SIMPLER: no round-trip reconciliation. The molecular pipeline has to recompute φ
on the *decoded* molecule because DeFoG's atom-by-symbol encoding loses charge and
stereochemistry (only 24.8% of zinc64k survives unchanged). Colour graphs decode
losslessly -- no valence model, no sanitisation -- so φ is computed once, on the
same graphs the training pool was already built from.

STRICTER: that "same graphs" claim is not assumed here, it is checked.
``colorgraphs.build_pyg_dataset`` (and therefore the pool) drops
any graph with no edges -- ``vgd_to_pyg`` returns ``None`` for them -- so the
cached training pool is a strict subset of the raw VGD folder (99,995 of ~100,000).
Assuming row *i* of a fresh ``load_vgd_dataset`` call is row *i* of the pool would
be exactly the kind of silent misalignment that trains an adapter on scrambled
conditions with no error anywhere downstream. So this experiment re-applies the
same edgeless-graph filter and then verifies, row by row, that the result agrees
with the actual pool cache on node count -- not just on the filter's logic.

The clustering used by default is the one of the zoo (19 concepts, leaf selection): checked
against the exact planted motif counts, it has both better-populated concepts and cleaner
single-motif correlation than the variants that were tried.

Two sources. ``SOURCE = "pool"`` (the default) reads the graphs from the pool of the zoo, which
is the compact copy of the dataset, so the memberships can be recomputed (and compared with the
table of the zoo) without the raw dataset. ``"vgd"`` re-derives the graphs from the raw visual
graph dataset folder ``VGD_ROOT`` and verifies the alignment against the pool.

Usage:
    python experiments/prepare_color_concept_dataset.py --__TESTING__ True
    python experiments/prepare_color_concept_dataset.py --CLUSTERING_PATH '"...clustering.clu"'
"""
import json
import os
import time

import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_megan.colorgraphs import (
    load_pool_npz, load_vgd_dataset, pyg_to_vgd, save_concept_conditions, vgd_to_pyg,
)
from defog_megan.concepts import ConceptSpace
from defog_megan.zoo import zoo_path

# ============================================================================
# Parameters
# ============================================================================

# :param SOURCE: "pool" (read the graphs from POOL_PATH) or "vgd" (re-derive them from the raw
#       dataset folder VGD_ROOT, which the zoo does not ship).
SOURCE: str = "pool"
VGD_ROOT: str = ""
POOL_PATH: str = zoo_path("colorgraphs", "data/pool.npz")

# :param MEGAN_CKPT / CLUSTERING_PATH / PROCESSING_PATH: the three artifacts that
#       jointly define the concept vocabulary. PROCESSING_PATH MUST be the one the
#       MEGAN model and the clustering were built against (this pool's own, 7
#       colour classes / 1 edge type -- not a generic MoleculeProcessing).
MEGAN_CKPT: str = zoo_path("colorgraphs", "megan/model.ckpt")
CLUSTERING_PATH: str = zoo_path("colorgraphs", "megan/clustering.clu")
PROCESSING_PATH: str = zoo_path("colorgraphs", "megan/process.py")

# :param scoring knobs: must match the concept-extraction run's own parameters,
#       otherwise φ means something other than what the concept report shows.
#       These are the recorded values of the concept extraction (and of the zoo's concept table).
FIDELITY_THRESHOLD: float = 0.5
DATASET_TYPE: str = "regression"
SCORE_METHOD: str = "knn"
SCORE_K: int = 5
SHARPEN: str = "sparsemax"
SHARPEN_TEMPERATURE: float = 0.5

BATCH_SIZE: int = 512
LIMIT: int = 0            # 0 = the whole pool

# :param OUT_PATH: where to write the table. Empty = ``concept_conditions.npz`` in the run folder (the
#       default never overwrites the paper's table in the zoo).
OUT_PATH: str = ""

__DEBUG__: bool = False
__TESTING__: bool = False
# no desktop notification when a run ends: on a machine without a desktop session it fails with a (harmless) traceback
__NOTIFY__: bool = False


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    # MEGAN inference stays on CPU: upstream ``forward_graphs`` builds its own CPU
    # DataLoader and ignores the model's device.
    e.log("building the colour-graph concept-conditioning table (MEGAN on CPU)")

    space = ConceptSpace.from_paths(
        e.MEGAN_CKPT, e.CLUSTERING_PATH, e.PROCESSING_PATH,
        fidelity_threshold=e.FIDELITY_THRESHOLD, dataset_type=e.DATASET_TYPE,
        score_method=e.SCORE_METHOD, score_k=e.SCORE_K,
        sharpen=e.SHARPEN, sharpen_temperature=e.SHARPEN_TEMPERATURE,
    )
    e.log(f"concept space: {space.num_concepts} concepts over channels {space.channels}")
    e["concepts/spec"] = space.spec()
    e["concepts/table"] = space.concept_table()

    # -- reproduce build_pyg_dataset's own edgeless-graph filter, so row i here IS
    # -- row i of the pool the adapter trains on.
    if e.SOURCE == "pool":
        # The pool is the compact copy of the same graphs in the same rows, so row i here IS row i of the
        # pool by construction: there is nothing to align.
        pool_graphs, _ = load_pool_npz(e.POOL_PATH)
        if e.LIMIT:
            pool_graphs = pool_graphs[:e.LIMIT]
        kept_graphs = [pyg_to_vgd(g) for g in pool_graphs]
        e.log(f"{len(kept_graphs)} graphs read from the pool {os.path.basename(e.POOL_PATH)}")
        e["data/n_kept"] = int(len(kept_graphs))
    elif e.SOURCE == "vgd":
        records = load_vgd_dataset(e.VGD_ROOT, limit=(e.LIMIT or None))
        kept_graphs = []
        n_edgeless = 0
        for record in records:
            if vgd_to_pyg(record["graph"]) is None:
                n_edgeless += 1
                continue
            kept_graphs.append(record["graph"])
        e.log(f"{len(records)} raw graphs -> {len(kept_graphs)} kept "
              f"({n_edgeless} edgeless, dropped -- the same filter build_pyg_dataset applies)")
        e["data/n_raw"] = int(len(records))
        e["data/n_kept"] = int(len(kept_graphs))
        e["data/n_edgeless"] = int(n_edgeless)

        # -- verify alignment against the ACTUAL training pool, not just the filter
        # -- logic. A prefix check: with LIMIT unset this must be an exact match; with
        # -- LIMIT set (testing) it only has to agree on the rows actually built.
        if not os.path.exists(e.POOL_PATH):
            raise FileNotFoundError(
                f"{e.POOL_PATH} does not exist -- build it first "
                f"(train_colorgraph_base.py / colorgraphs.build_pyg_dataset), phi has "
                f"nothing to align against otherwise")
        pool_graphs, _ = load_pool_npz(e.POOL_PATH)
        n_check = len(kept_graphs)
        if len(pool_graphs) < n_check:
            raise ValueError(
                f"pool has only {len(pool_graphs)} graphs but this run kept {n_check} "
                f"-- the pool is smaller than the table being built against it")
        if not e.LIMIT and len(pool_graphs) != n_check:
            raise ValueError(
                f"row-count mismatch with no LIMIT set: pool has {len(pool_graphs)} "
                f"graphs, this run's filter kept {n_check} -- phi would misalign "
                f"row-for-row against every graph in the pool")
        mismatches = [
            i for i in range(n_check)
            if int(pool_graphs[i].x.shape[0]) != int(np.asarray(kept_graphs[i]["node_attributes"]).shape[0])
        ]
        if mismatches:
            raise ValueError(
                f"{len(mismatches)}/{n_check} rows disagree on node count between "
                f"{e.POOL_PATH} and this re-derivation (first mismatch at row "
                f"{mismatches[0]}) -- ordering has drifted, phi would train against "
                f"the wrong graphs")
        e.log(f"alignment verified: {n_check} rows agree with "
              f"{os.path.basename(e.POOL_PATH)}"
              + (" (prefix, LIMIT set)" if e.LIMIT else " (the full pool)"))
    else:
        raise ValueError(f"SOURCE must be 'pool' or 'vgd', got {e.SOURCE!r}")

    # -- score -----------------------------------------------------------------
    e.log(f"scoring {len(kept_graphs)} graphs...")
    phi = np.ones((len(kept_graphs), space.num_concepts))
    started = time.time()
    for start in range(0, len(kept_graphs), e.BATCH_SIZE):
        chunk = kept_graphs[start:start + e.BATCH_SIZE]
        phi[start:start + len(chunk)] = space.phi_from_graphs(chunk, batch_size=e.BATCH_SIZE)
        done = min(start + len(chunk), len(kept_graphs))
        if (start // e.BATCH_SIZE) % 20 == 0:
            rate = done / max(time.time() - started, 1e-9)
            eta = (len(kept_graphs) - done) / max(rate, 1e-9)
            e.log(f"  scored {done}/{len(kept_graphs)}  "
                  f"({rate:.0f} graphs/s, eta {eta / 60:.1f} min)")
    e["phi/scoring_seconds"] = float(time.time() - started)
    e.log(f"scoring took {time.time() - started:.1f}s "
          f"({len(kept_graphs) / max(time.time() - started, 1e-9):.0f} graphs/s)")

    # -- report how informative the conditions actually are -------------------
    assigned = phi < 0.999
    per_graph = assigned.sum(axis=1)
    e["phi/mean_assigned_per_graph"] = float(per_graph.mean())
    e["phi/frac_no_concept"] = float((per_graph == 0).mean())
    e["phi/per_concept_support"] = {
        cid: int(assigned[:, i].sum()) for i, cid in enumerate(space.concept_ids)
    }
    e.log(f"φ: mean {per_graph.mean():.2f} concepts assigned per graph; "
          f"{(per_graph == 0).mean():.1%} carry the no-concept condition")
    empty = [cid for i, cid in enumerate(space.concept_ids) if assigned[:, i].sum() == 0]
    if empty:
        e.log(f"WARNING {len(empty)} concepts have no support in this table: {empty}")

    # -- persist -------------------------------------------------------------
    out_path = save_concept_conditions(
        e.OUT_PATH or os.path.join(e.path, "concept_conditions.npz"), phi=phi,
        concept_ids=space.concept_ids, pool_path=os.path.basename(e.POOL_PATH))
    e.log(f"wrote {out_path} ({len(kept_graphs)} rows x {space.num_concepts} "
          f"concepts, aligned by row to {e.POOL_PATH})")

    with open(os.path.join(e.path, "concept_spec.json"), "w") as file:
        json.dump({"spec": space.spec(), "table": space.concept_table(),
                   "megan_ckpt": os.path.basename(e.MEGAN_CKPT), "clustering": os.path.basename(e.CLUSTERING_PATH),
                   "processing": os.path.basename(e.PROCESSING_PATH), "pool_path": os.path.basename(e.POOL_PATH)},
                  file, indent=2)
    e.log("Done.")


@experiment.testing
def testing(e: Experiment) -> None:
    e.LIMIT = 400
    e.BATCH_SIZE = 64


experiment.run_if_main()
