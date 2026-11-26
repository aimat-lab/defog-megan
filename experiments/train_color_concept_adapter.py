"""
Train the MEGAN-concept adapter on the frozen colour-graph base.

One adapter takes the whole φ vector (19 dims, the ``redesign`` clustering) as its
condition, trained with the base's own denoising CE loss. At inference a single
concept is selected and its target vector steers generation. Mirrors
``experiments/train_concept_adapter.py`` (the molecular concept adapter), adapted
for a domain with no valence, no SMILES and -- unlike molecules -- planted ground
truth for a cheap, MEGAN-free structural sanity check during training.

This module deliberately never imports MEGAN. Everything MEGAN-derived was
precomputed upstream and travels as data:

* ``data/rb_dual_motifs_v2_concept_conditions.npz`` -- φ per training graph
  (``prepare_color_concept_dataset.py``)
* ``data/rb_dual_motifs_v2_concept_targets.npz``     -- per-concept target vectors
  (``export_color_concept_targets.py``)

``cond_dim = 19`` is above the ``<= 8`` bound ``CrossAttentionAdapter.for_base`` requires for
Fourier condition bands, so those are off automatically -- this adapter gets
cross-attention only, same as the molecular concept adapters (42-dim and 29-dim) and unlike
the scalar property adapter. The architecture is asked for with ``None`` sentinels and
logged as *built*, not assumed, so this experiment follows the library's ``for_base``.

ON THE GUIDANCE WEIGHT: do not assume w=2. That was the measured optimum for the scalar
property adapter, but the best weight depends on the condition: for a multi-dimensional
concept condition a larger weight can cost many concepts their core membership, abruptly
rather than gradually. So the in-training probe stays at w=1 and the real sweep -- which must
check structural health at every weight, not just report a mean -- is a separate step.

Grading against MEGAN is a separate step (``scripts/score_color_concept_1024.py``): this job
dumps decoded graphs per concept in the ``.npz`` pool format that
``colorgraphs.save_pool_npz`` writes.

Usage:
    python experiments/train_color_concept_adapter.py --__TESTING__ True
    python experiments/train_color_concept_adapter.py --LEARNING_RATE 3e-4
"""
import json
import os
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_adapt.core import (
    DeFoGModel, CrossAttentionAdapter, AdapterModule, AdapterComposition, ConditionBranch,
    AdaptedSampler, Sampler, EMACallback,
)
from defog_megan.zoo import zoo_path
from defog_megan.colorgraphs import (
    RB_PALETTE, distribution_score, load_concept_conditions, load_pool_npz,
    save_pool_npz, structural_report,
)
from defog_megan.data import load_targets
from defog_megan.gpu import largest_fitting_batch
from defog_megan.rb_motifs import MOTIF_NAMES, count_motifs_pyg

# ============================================================================
# Parameters
# ============================================================================

CONDITIONS_PATH: str = zoo_path("colorgraphs", "data/concept_conditions.npz")
TARGETS_PATH: str = zoo_path("colorgraphs", "data/concept_targets.npz")
POOL_PATH: str = zoo_path("colorgraphs", "data/pool.npz")

# :param BASE_CKPT: the frozen unconditional colour base (the best checkpoint of the base
#       training, EMA @ epoch 80, distribution_score 0.9862).
BASE_CKPT: str = zoo_path("colorgraphs", "base.ckpt")

# --- Adapter architecture ----------------------------------------------------
H_HIDDEN: int = 256
TIME_CONDITIONED: bool = True
STREAMS: list = ["X", "E", "y"]

# ``None`` = take the library's current for_base default, so this experiment
# tracks the library rather than a pinned copy of its numbers. cond_fourier ends
# up 0 regardless of what is asked for -- for_base refuses Fourier bands above
# cond_dim=8 -- so leaving it None just means "let the library decide", same as
# the xattn knobs.
XATTN_TOKENS: int = None
XATTN_DIM: int = None
XATTN_HEADS: int = None
COND_FOURIER: int = None

# --- Training -----------------------------------------------------------
EPOCHS: int = 80
BATCH_SIZE: int = 128          # upper bound; probed down with headroom
MIN_BATCH_SIZE: int = 8
MEMORY_HEADROOM: float = 0.80
LEARNING_RATE: float = 4e-4
# No condition dropout: the adapter's gate head is zero-initialised, so the unconditional
# branch IS the frozen base. Nothing to drop out to.
MAX_TIME_HOURS: float = 12.0
EMA_DECAY: float = 0.999
CHECKPOINT_EVERY_K: int = 5

# :param TARGET_MODE: 'empirical' (mean φ of the concept's own cluster members)
#       or 'ideal' (0.0 at the concept, 1.0 elsewhere). Both are in the npz.
TARGET_MODE: str = "empirical"

TRAIN_SPLIT: float = 0.9
REFERENCE_SIZE: int = 5000
# :param LIMIT: cap on how many pool graphs to load (0 = the whole 99,995-graph
#       pool). Smoke-test only -- omitting this cost the first testing run ~10
#       minutes of CPU time training 2 tiny epochs over the FULL pool, which
#       defeats the point of a quick smoke gate.
LIMIT: int = 0

# --- Sampling / in-training probe --------------------------------------------
EVAL_STEPS: int = 500
ETA: float = 0.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
# :param GUIDANCE_WEIGHT: the ONE weight this job checks -- see the module
#       docstring for why w=2 is not assumed here. The real sweep, which must
#       check structural health at every weight rather than report a mean, is a
#       separate job.
GUIDANCE_WEIGHT: float = 1.0
COMPOSE_MODE: str = "product"

# :param EVAL_CONCEPTS: empty = the N best-supported concepts in the training
#       table (steering to a concept with a handful of examples measures the
#       data, not the method).
EVAL_CONCEPTS: list = []
N_EVAL_CONCEPTS: int = 4
N_PER_CONCEPT: int = 128
N_BASELINE: int = 256
EVAL_CHUNK: int = 32

# --- mid-training probe (MEGAN-free: structural realism + exact motif counts) --
PROBE_EVERY_K: int = 5
PROBE_N: int = 32
PROBE_STEPS: int = 100

SEED: int = 42
# False: every run gets its own time-stamped folder (True writes to one `debug/` folder, which the next run wipes).
__DEBUG__: bool = False
__TESTING__: bool = False


class GenerationProbe(pl.Callback):
    """Every K epochs: sample under the top eval concept and report health.

    Deliberately MEGAN-free -- it cannot tell you whether the requested CONCEPT
    was hit, only whether the adapter is still producing realistic colour graphs
    (via :func:`structural_report`), plus their exact planted-motif composition
    since that is free in this domain and unlike molecules needs no solver or
    sanitiser. The concept question -- did steering actually move φ -- is answered
    by ``scripts/score_color_concept_1024.py`` afterwards.
    """

    def __init__(self, e, target, reference):
        super().__init__()
        self.e, self.target, self.reference = e, target, reference

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        loss = trainer.callback_metrics.get(
            "adapter/loss_epoch", trainer.callback_metrics.get("adapter/loss"))
        self.e.log(f"[epoch {epoch}] adapter/loss={float(loss):.4f}"
                   if loss is not None else f"[epoch {epoch}] done")
        if not self.e.PROBE_EVERY_K or (epoch + 1) % self.e.PROBE_EVERY_K != 0:
            return
        try:
            self._probe(pl_module, epoch)
        except Exception as exc:
            self.e.log(f"[epoch {epoch}] PROBE failed (non-fatal): {exc}")

    @torch.no_grad()
    def _probe(self, pl_module, epoch):
        composition = AdapterComposition(
            [ConditionBranch(pl_module.adapter,
                             torch.tensor(self.target, dtype=torch.float32),
                             self.e.GUIDANCE_WEIGHT)],
            base=pl_module.base, mode=self.e.COMPOSE_MODE)
        sampler = AdaptedSampler(pl_module.base, composition, eta=self.e.ETA,
                                 omega=self.e.OMEGA, sample_steps=self.e.PROBE_STEPS,
                                 time_distortion=self.e.TIME_DISTORTION)
        samples, remaining = [], self.e.PROBE_N
        while remaining > 0:
            take = min(self.e.EVAL_CHUNK, remaining)
            samples += sampler.sample(take, device=pl_module.device, show_progress=False)
            remaining -= take
        report = structural_report(samples, self.reference)
        score = distribution_score(report)
        motif_means = np.mean(
            [[count_motifs_pyg(s)[m] for m in MOTIF_NAMES] for s in samples], axis=0)
        self.e.log(
            f"[epoch {epoch}] PROBE(w={self.e.GUIDANCE_WEIGHT}): dist={score:.4f} "
            f"connected={report['connected_rate']:.3f}  motifs="
            + " ".join(f"{m}={v:.2f}" for m, v in zip(MOTIF_NAMES, motif_means)))


class PeriodicAdapterCheckpoint(pl.Callback):
    """Rolling adapter snapshot, so a crash costs epochs rather than the run."""

    def __init__(self, adapter, checkpoint_dir: str, every_k_epochs: int = 5):
        super().__init__()
        self.adapter = adapter
        self.checkpoint_dir = checkpoint_dir
        self.every_k_epochs = every_k_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.every_k_epochs:
            return
        try:
            self.adapter.save(os.path.join(self.checkpoint_dir, "adapter_last"))
        except Exception as exc:
            warnings.warn(f"periodic adapter checkpoint failed at epoch {epoch}: {exc}")


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e.log(f"MEGAN-concept adapter on the frozen colour-graph base (device={device})")

    if e.__TESTING__:
        e.LIMIT = 1000
        e.EPOCHS = 2
        e.BATCH_SIZE = 16
        e.MIN_BATCH_SIZE = 4
        e.MAX_TIME_HOURS = 0.2
        e.H_HIDDEN = 32
        e.EVAL_STEPS = 5
        e.PROBE_STEPS = 5
        e.PROBE_EVERY_K = 1
        e.PROBE_N = 6
        e.N_EVAL_CONCEPTS = 2
        e.N_PER_CONCEPT = 8
        e.N_BASELINE = 8
        e.EVAL_CHUNK = 8
        e.REFERENCE_SIZE = 200
        e.CHECKPOINT_EVERY_K = 1
        e.log("TESTING mode")

    # -- the two precomputed MEGAN artifacts ---------------------------------
    graphs, _ = load_pool_npz(e.POOL_PATH)
    phi, concept_ids = load_concept_conditions(e.CONDITIONS_PATH, pool_path=e.POOL_PATH)
    if len(graphs) != phi.shape[0]:
        raise ValueError(f"pool has {len(graphs)} graphs but phi has {phi.shape[0]} rows")
    if e.LIMIT:
        graphs, phi = graphs[: e.LIMIT], phi[: e.LIMIT]   # keeps row alignment
    e.log(f"conditions: {len(graphs)} graphs x {len(concept_ids)} concepts")

    target_matrix, target_ids = load_targets(e.TARGETS_PATH, mode=e.TARGET_MODE)
    if target_ids != concept_ids:
        raise ValueError(
            "target vectors and conditions table disagree on the concept ordering "
            "-- the adapter would be steered with a permuted vector")
    e.log(f"targets: '{e.TARGET_MODE}' {target_matrix.shape}")

    support = (phi < 0.999).sum(axis=0)
    e["data/support"] = {cid: int(s) for cid, s in zip(concept_ids, support)}
    e.log(f"per-concept support: min={support.min()} "
          f"median={int(np.median(support))} max={support.max()}")
    e.log(f"phi std per concept: min={phi.std(axis=0).min():.4f} "
          f"max={phi.std(axis=0).max():.4f}")

    if e.EVAL_CONCEPTS:
        eval_concepts = list(e.EVAL_CONCEPTS)
    else:
        eval_concepts = [concept_ids[i] for i in np.argsort(-support)[:e.N_EVAL_CONCEPTS]]
    e["eval/concepts"] = eval_concepts
    e.log("eval concepts: " + ", ".join(
        f"{c}(n={support[concept_ids.index(c)]})" for c in eval_concepts))
    targets = {c: target_matrix[concept_ids.index(c)] for c in eval_concepts}

    # -- data: attach the RAW phi row as .cond, held out a reference slice ----
    n_train = int(len(graphs) * e.TRAIN_SPLIT)
    generator = torch.Generator().manual_seed(e.SEED)
    perm = torch.randperm(len(graphs), generator=generator).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    for i in train_idx:
        graphs[i].cond = torch.tensor(phi[i], dtype=torch.float).reshape(1, -1)
    train_set = [graphs[i] for i in train_idx]
    reference = [graphs[i] for i in val_idx][: e.REFERENCE_SIZE]
    e.log(f"train {len(train_set)} | held-out reference {len(reference)}")

    # -- frozen base + adapter ------------------------------------------------
    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    if base.cond_dim != 0:
        raise ValueError(f"expected an unconditional base, got cond_dim={base.cond_dim}")
    if base.num_node_classes != len(RB_PALETTE):
        raise ValueError(
            f"base has {base.num_node_classes} node classes, palette has "
            f"{len(RB_PALETTE)} -- wrong base checkpoint")

    arch = {k: v for k, v in (
        ("xattn_tokens", e.XATTN_TOKENS), ("xattn_dim", e.XATTN_DIM),
        ("xattn_heads", e.XATTN_HEADS), ("cond_fourier", e.COND_FOURIER),
    ) if v is not None}
    adapter = CrossAttentionAdapter.for_base(
        base, cond_dim=len(concept_ids), hidden=e.H_HIDDEN,
        time_conditioned=e.TIME_CONDITIONED, streams=tuple(e.STREAMS),
        cond_mean=phi.mean(axis=0), cond_std=phi.std(axis=0),
        name="color_concept_adapter", cond_type="rb_dual_motifs_concepts", **arch,
    ).to(device)
    built = {"xattn_tokens": int(getattr(adapter, "xattn_tokens", 0)),
             "xattn_dim": int(getattr(adapter, "xattn_dim", 0)),
             "xattn_heads": int(getattr(adapter, "xattn_heads", 0)),
             "cond_fourier": int(getattr(adapter, "cond_fourier", 0))}
    e["adapter/architecture"] = built
    e["adapter/num_params"] = int(sum(p.numel() for p in adapter.parameters()))
    e.log(f"adapter: {e['adapter/num_params']:,} params "
          f"(base {sum(p.numel() for p in base.parameters()):,} frozen)")
    e.log(f"architecture as built: {built}")
    if built["cond_fourier"] != 0:
        e.log(f"WARNING: cond_fourier={built['cond_fourier']} despite cond_dim="
              f"{len(concept_ids)}>8 -- for_base's guard may have changed")

    # -- train -----------------------------------------------------------------
    module = AdapterModule(base, adapter, cond_attr="cond", lr=e.LEARNING_RATE)

    batch_size = largest_fitting_batch(
        module, train_set, e.BATCH_SIZE, e.MIN_BATCH_SIZE, e.log, e.MEMORY_HEADROOM)
    e["config/batch_size"] = batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    e.log(f"training: batch {batch_size}, {len(train_set) // batch_size} steps/epoch")

    probe = GenerationProbe(e, targets[eval_concepts[0]], reference)
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS}, accelerator="auto",
        devices=1, enable_progress_bar=False, enable_checkpointing=False, logger=False,
        gradient_clip_val=1.0,
        callbacks=[probe, EMACallback(decay=e.EMA_DECAY),
                   PeriodicAdapterCheckpoint(adapter, e.path, e.CHECKPOINT_EVERY_K)],
    )
    e.log(f"training: epochs<={e.EPOCHS} lr={e.LEARNING_RATE} max_time={e.MAX_TIME_HOURS}h")
    trainer.fit(module, train_dataloaders=train_loader)

    checkpoint = adapter.save(os.path.join(e.path, "concept_adapter"))
    e.log(f"saved adapter -> {checkpoint}")

    # -- sample under each concept, dump graphs for MEGAN grading ------------------
    e.log("=" * 60)
    # Lightning's teardown moves the module (and with it base + adapter) back to
    # CPU when fit() returns -- put both back explicitly, or sampling dies inside
    # CrossAttentionAdapter.normalize with cond_mean/cond_std on CPU and the condition on
    # CUDA.
    base, adapter = base.to(device).eval(), adapter.to(device).eval()

    def sample_with(sampler, n):
        out, remaining = [], n
        while remaining > 0:
            take = min(e.EVAL_CHUNK, remaining)
            out += sampler.sample(take, device=device, show_progress=False)
            remaining -= take
        return out

    e.log(f"baseline: {e.N_BASELINE} unconditional samples")
    baseline_sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA,
                               sample_steps=e.EVAL_STEPS, time_distortion=e.TIME_DISTORTION)
    baseline_samples = sample_with(baseline_sampler, e.N_BASELINE)
    baseline_report = structural_report(baseline_samples, reference)
    baseline_score = distribution_score(baseline_report)
    e.log(f"  distribution_score={baseline_score:.4f} "
          f"connected={baseline_report['connected_rate']:.3f}")
    save_pool_npz(os.path.join(e.path, "gen_baseline.npz"), baseline_samples,
                  [{}] * len(baseline_samples))

    manifest = {
        "target_mode": e.TARGET_MODE,
        "guidance_weight": e.GUIDANCE_WEIGHT,
        "blend_space": "prob",
        "sampling": {"steps": e.EVAL_STEPS, "eta": e.ETA, "omega": e.OMEGA,
                     "time_distortion": e.TIME_DISTORTION},
        "concept_ids": concept_ids,
        "eval_concepts": eval_concepts,
        "baseline_file": "gen_baseline.npz",
        "baseline_structural": {"distribution_score": baseline_score,
                                "connected_rate": baseline_report["connected_rate"]},
        "per_concept": {},
    }
    for concept_id in eval_concepts:
        composition = AdapterComposition(
            [ConditionBranch(adapter, torch.tensor(targets[concept_id], dtype=torch.float32),
                             e.GUIDANCE_WEIGHT)],
            base=base, mode=e.COMPOSE_MODE)
        sampler = AdaptedSampler(base, composition, eta=e.ETA, omega=e.OMEGA,
                                 sample_steps=e.EVAL_STEPS, time_distortion=e.TIME_DISTORTION)
        samples = sample_with(sampler, e.N_PER_CONCEPT)
        report = structural_report(samples, reference)
        score = distribution_score(report)
        fname = f"gen_{concept_id}.npz"
        save_pool_npz(os.path.join(e.path, fname), samples, [{}] * len(samples))
        manifest["per_concept"][concept_id] = {
            "file": fname, "n": len(samples),
            "distribution_score": score, "connected_rate": report["connected_rate"],
            "support": int(support[concept_ids.index(concept_id)]),
        }
        e.log(f"  {concept_id} (support={support[concept_ids.index(concept_id)]}): "
              f"dist={score:.4f} connected={report['connected_rate']:.3f}")

    with open(os.path.join(e.path, "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)
    e.log("wrote manifest.json + gen_*.npz -- grade them with MEGAN (scripts/score_color_concept_1024.py)")
    e.log("Done.")


experiment.run_if_main()
