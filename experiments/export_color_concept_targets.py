"""
Export the per-concept target vectors for colour-graph concept steering.

Mirrors ``export_concept_targets.py``. Training and sampling never need MEGAN itself -- ``φ`` is
already baked into the conditioning table built by ``prepare_color_concept_dataset.py``, and the
only other thing they need from MEGAN is the *target vector* for each concept: a fixed
``(n_concepts, n_concepts)`` matrix, computed once here and shipped as data.

The tables go into the run folder; ``--OUTPUT_PATH`` writes them elsewhere as well (the default
never overwrites the paper's tables in the zoo).

Usage:
    python experiments/export_color_concept_targets.py
"""
import json
import os

import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_megan.concepts import ConceptSpace
from defog_megan.zoo import zoo_path

# ============================================================================
# Parameters
# ============================================================================

MEGAN_CKPT: str = zoo_path("colorgraphs", "megan/model.ckpt")
CLUSTERING_PATH: str = zoo_path("colorgraphs", "megan/clustering.clu")
PROCESSING_PATH: str = zoo_path("colorgraphs", "megan/process.py")

# Must match prepare_color_concept_dataset.py -- these are the recorded extraction
# parameters of the concept vocabulary, not free knobs.
FIDELITY_THRESHOLD: float = 0.5
DATASET_TYPE: str = "regression"
SCORE_METHOD: str = "knn"
SCORE_K: int = 5
SHARPEN: str = "sparsemax"
SHARPEN_TEMPERATURE: float = 0.5

TARGET_MODES: list = ["empirical", "ideal"]

# :param OUTPUT_PATH: a second place to write ``concept_targets.npz`` to (empty = the run folder only).
OUTPUT_PATH: str = ""

__DEBUG__: bool = False
__TESTING__: bool = False
# no desktop notification when a run ends: on a machine without a desktop session it fails with a (harmless) traceback
__NOTIFY__: bool = False


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    space = ConceptSpace.from_paths(
        e.MEGAN_CKPT, e.CLUSTERING_PATH, e.PROCESSING_PATH,
        fidelity_threshold=e.FIDELITY_THRESHOLD, dataset_type=e.DATASET_TYPE,
        score_method=e.SCORE_METHOD, score_k=e.SCORE_K,
        sharpen=e.SHARPEN, sharpen_temperature=e.SHARPEN_TEMPERATURE,
    )
    e.log(f"concept space: {space.num_concepts} concepts over channels {space.channels}")

    arrays = {"concept_ids": np.asarray(space.concept_ids, dtype=np.str_)}
    for mode in e.TARGET_MODES:
        e.log(f"building '{mode}' target vectors...")
        matrix = np.stack([space.target_vector(cid, mode=mode)
                           for cid in space.concept_ids])
        arrays[f"targets_{mode}"] = matrix
        # A target vector is only useful if its own concept is the closest entry
        # -- otherwise steering to it would be aiming somewhere else.
        self_best = int(sum(matrix[i].argmin() == i for i in range(len(matrix))))
        e.log(f"  {mode}: {self_best}/{len(matrix)} target vectors have their own "
              f"concept as the closest entry")
        e[f"targets/{mode}/self_best"] = self_best
    arrays["null"] = space.null_vector()

    targets_paths = [os.path.join(e.path, "concept_targets.npz")]
    table_paths = [os.path.join(e.path, "concept_table.json")]
    if e.OUTPUT_PATH:
        targets_paths.append(e.OUTPUT_PATH)
        table_paths.append(os.path.join(os.path.dirname(e.OUTPUT_PATH) or ".", "concept_table.json"))
    for path in targets_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **arrays)
        e.log(f"wrote {path}")
    for table_path in table_paths:
        with open(table_path, "w") as file:
            json.dump({"spec": space.spec(), "table": space.concept_table()}, file, indent=2)
        e.log(f"wrote {table_path}")


experiment.run_if_main()
