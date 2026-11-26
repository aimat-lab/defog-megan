"""
Sample an already-trained colour-graph concept adapter, across the WHOLE vocabulary.

``train_color_concept_adapter.py`` trains and then dumps only its own eval subset -- four
of nineteen concepts, chosen by support. Experiment 1's concept arm needs all of them, and
there is no reason to retrain to get them: the adapter checkpoint is on disk, so this is a
sampling-only pass.

Writes the same layout as ``train_color_concept_adapter.py`` -- one ``gen_<concept>.npz`` per
concept plus ``gen_baseline.npz`` and a ``manifest.json`` -- so grading is one step
(``scripts/score_color_concept_1024.py``) and the concepts that training already dumped act as a
consistency check.

Deliberately does NOT import MEGAN: phi comes from the precomputed conditions table and the
target vectors from the exported ``.npz``, so this runs where MEGAN is not installed.

Usage:
    python experiments/sample_color_concept_adapter.py --__TESTING__ True
    python experiments/sample_color_concept_adapter.py \
        --ADAPTER_CKPT '"..."' --N_PER_CONCEPT 256
"""
import json
import os

import numpy as np
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_adapt.core import (
    DeFoGModel, CrossAttentionAdapter, AdapterComposition, ConditionBranch, AdaptedSampler,
    Sampler,
)
from defog_megan.zoo import zoo_path
from defog_megan.colorgraphs import (
    RB_PALETTE, load_pool_npz, save_pool_npz, structural_report, distribution_score,
    load_concept_conditions,
)
from defog_megan.data import load_targets

# ============================================================================
#: The trained adapter (copy it out of a `debug` run folder first: PyComex wipes those at the start
#: of any debug-mode run).
ADAPTER_CKPT: str = zoo_path("colorgraphs", "concept_adapter.ckpt")
BASE_CKPT: str = zoo_path("colorgraphs", "base.ckpt")
CONDITIONS_PATH: str = zoo_path("colorgraphs", "data/concept_conditions.npz")
TARGETS_PATH: str = zoo_path("colorgraphs", "data/concept_targets.npz")
POOL_PATH: str = zoo_path("colorgraphs", "data/pool.npz")

# :param CONCEPTS: empty = the ENTIRE vocabulary, which is the point of this module.
CONCEPTS: list = []
TARGET_MODE: str = "empirical"

# These must match the training run or the samples are not comparable with the concepts it
# already dumped.
GUIDANCE_WEIGHT: float = 1.0
COMPOSE_MODE: str = "product"     # single branch: product == mean
EVAL_STEPS: int = 500
ETA: float = 0.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"

N_PER_CONCEPT: int = 128
N_BASELINE: int = 500
EVAL_CHUNK: int = 32
REFERENCE_SIZE: int = 5000
TRAIN_SPLIT: float = 0.9

# :param SHARD / NUM_SHARDS: the concepts are dealt out to the shards in turn (shard k takes every NUM_SHARDS-th concept
#       from k), so that the vocabulary can be sampled on several GPUs. Only shard 0 draws the full unconditional baseline the
#       per-concept rates are read against; the others draw a token baseline of 8 graphs. Each shard seeds with SEED + SHARD.
SHARD: int = 0
NUM_SHARDS: int = 1
SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    if e.__TESTING__:
        e.N_PER_CONCEPT = 8
        e.N_BASELINE = 8
        e.EVAL_CHUNK = 8
        e.EVAL_STEPS = 20
        e.REFERENCE_SIZE = 200

    torch.manual_seed(e.SEED + e.SHARD)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e.log(f"device: {device} | shard {e.SHARD}/{e.NUM_SHARDS}")

    phi, concept_ids = load_concept_conditions(e.CONDITIONS_PATH)
    concept_ids = list(concept_ids)
    target_matrix, target_ids = load_targets(e.TARGETS_PATH, mode=e.TARGET_MODE)
    if list(target_ids) != concept_ids:
        raise ValueError("targets and conditions disagree on the concept ordering -- "
                         "the adapter would be steered with a permuted vector")
    support = (np.asarray(phi) < 0.999).sum(axis=0)

    concepts = (list(e.CONCEPTS) or concept_ids)[e.SHARD::e.NUM_SHARDS]
    if e.__TESTING__:
        concepts = concepts[:2]
    e.log(f"sampling {len(concepts)} of {len(concept_ids)} concepts, "
          f"{e.N_PER_CONCEPT} each")

    graphs, _ = load_pool_npz(e.POOL_PATH)
    n_train = int(len(graphs) * e.TRAIN_SPLIT)
    generator = torch.Generator().manual_seed(e.SEED)
    perm = torch.randperm(len(graphs), generator=generator).tolist()
    reference = [graphs[i] for i in perm[n_train:]][: e.REFERENCE_SIZE]
    e.log(f"reference: {len(reference)} held-out graphs")

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    if base.num_node_classes != len(RB_PALETTE):
        raise ValueError(f"base has {base.num_node_classes} node classes, palette has "
                         f"{len(RB_PALETTE)} -- wrong base")
    adapter = CrossAttentionAdapter.load(e.ADAPTER_CKPT, device="cpu").to(device).eval()
    # A FiLM-only reload would silently sweep a different model than the one graded.
    if getattr(adapter, "xattn_tokens", 0) == 0:
        raise ValueError("adapter reloaded WITHOUT cross-attention")
    adapter.check_compatible(base)
    if adapter.cond_dim != len(concept_ids):
        raise ValueError(f"adapter cond_dim={adapter.cond_dim} but the vocabulary has "
                         f"{len(concept_ids)} concepts")
    e.log(f"base {sum(p.numel() for p in base.parameters()):,} (frozen) | "
          f"adapter {sum(p.numel() for p in adapter.parameters()):,} "
          f"xattn={adapter.xattn_tokens}")

    def sample_with(sampler, n):
        out, remaining = [], n
        while remaining > 0:
            take = min(e.EVAL_CHUNK, remaining)
            out += sampler.sample(take, device=device, show_progress=False)
            remaining -= take
        return out

    baseline_sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA,
                               sample_steps=e.EVAL_STEPS,
                               time_distortion=e.TIME_DISTORTION)
    baseline = sample_with(baseline_sampler, e.N_BASELINE if e.SHARD == 0 else min(8, e.N_BASELINE))
    baseline_report = structural_report(baseline, reference)
    baseline_score = distribution_score(baseline_report)
    save_pool_npz(os.path.join(e.path, "gen_baseline.npz"), baseline, [{}] * len(baseline))
    e.log(f"baseline: {len(baseline)} samples  distribution_score={baseline_score:.4f} "
          f"connected={baseline_report['connected_rate']:.3f}")

    manifest = {
        "stage": "concepts", "shard": e.SHARD, "num_shards": e.NUM_SHARDS,
        "source_adapter": e.ADAPTER_CKPT,
        "target_mode": e.TARGET_MODE,
        "guidance_weight": e.GUIDANCE_WEIGHT,
        "blend_space": "prob",
        "sampling": {"steps": e.EVAL_STEPS, "eta": e.ETA, "omega": e.OMEGA,
                     "time_distortion": e.TIME_DISTORTION},
        "concept_ids": concept_ids,
        "eval_concepts": concepts,
        "baseline_file": "gen_baseline.npz",
        "baseline_structural": {"distribution_score": baseline_score,
                                "connected_rate": baseline_report["connected_rate"]},
        "per_concept": {},
    }

    e.log(f"{'concept':>10} {'support':>8} {'n':>5} {'dist.score':>11} {'connected':>10}")
    for concept_id in concepts:
        cond = torch.tensor(target_matrix[concept_ids.index(concept_id)],
                            dtype=torch.float32)
        composition = AdapterComposition(
            [ConditionBranch(adapter, cond, e.GUIDANCE_WEIGHT)],
            base=base, mode=e.COMPOSE_MODE)
        sampler = AdaptedSampler(base, composition, eta=e.ETA, omega=e.OMEGA,
                                 sample_steps=e.EVAL_STEPS,
                                 time_distortion=e.TIME_DISTORTION)
        samples = sample_with(sampler, e.N_PER_CONCEPT)
        report = structural_report(samples, reference)
        score = distribution_score(report)
        fname = f"gen_{concept_id}.npz"
        save_pool_npz(os.path.join(e.path, fname), samples, [{}] * len(samples))
        manifest["per_concept"][concept_id] = {
            "file": fname, "n": len(samples), "distribution_score": score,
            "connected_rate": report["connected_rate"],
            "support": int(support[concept_ids.index(concept_id)]),
        }
        e.log(f"{concept_id:>10} {support[concept_ids.index(concept_id)]:>8} "
              f"{len(samples):>5} {score:>11.4f} {report['connected_rate']:>10.3f}")

    with open(os.path.join(e.path, "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)
    e.log(f"wrote manifest.json and {len(concepts)} concept dumps")
    e.log("Done.")


experiment.run_if_main()
