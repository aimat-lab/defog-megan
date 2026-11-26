"""
Target-conditioned adapter on the frozen colour-graph base.

An AdaLN adapter reads the dataset's scalar regression label ``y`` and steers the
frozen unconditional base toward graphs with that target. It is the first
conditional experiment on this dataset, and it is the cleanest one the project
can run: the requested quantity is an *exact function of the planted motifs*, so
a generated graph can be graded against the request with no model in the loop.

What makes it worth doing beyond "conditioning works":

* **The target is many-to-one in motif content.** ``+2`` is one ``red_star``, or
  two ``red_rings``, or ``red_star + red_ring - blue_ring``. Conditioning fixes
  the sum, not the route. The unconditional base was measured to over-produce
  rings (all 6 ring deltas positive across three evaluations, mean +1.5 points,
  against stars at ~0), so the model has a cheap route available and may take it.
  :func:`composition_report` measures whether it does.
* **The base cannot reach the extremes.** The pool is uniform on the seven
  integer levels ``-3..+3`` by rejection sampling; the base instead emits a bell
  curve, under-producing +/-3 by ~40% relative and putting ~6% of its mass
  *outside* the pool's support entirely. Whether conditioning can pull mass back
  to +/-3 is a direct, exactly-measurable instance of distributional reachability.
* **Extrapolation is meaningful here.** Requesting +/-4 and +/-5 asks for
  something the adapter never saw a label for, in a region the base does visit.

Conditioning uses the **stored noisy label** (``exact motif sum + U(-0.5, 0.5)``,
sd 0.289) rather than the noise-free sum, because that is the observable a real
dataset provides and the one MEGAN was trained on. Grading stays exact either
way -- the detector recomputes the achieved noise-free sum.

No target-conditional size model: the label is uncorrelated with graph size
(r = -0.004) and mean degree (r = +0.004) in this pool, unlike the molecular
setting where per-property size matching was the single largest win.

Imports nothing MEGAN-derived.

Usage::

    python experiments/train_target_adapter.py --__TESTING__ True
    python experiments/train_target_adapter.py
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
    AdaptedSampler, EMACallback,
)
from defog_megan.zoo import zoo_path
from defog_megan.colorgraphs import (
    RB_PALETTE, ColorGraphDomain, distribution_score, load_pool_npz, structural_report,
)
from defog_megan.gpu import largest_fitting_batch
from defog_megan.rb_motifs import (
    MOTIF_NAMES, count_motifs_pyg, exact_target,
)

# ============================================================================
# Parameters
# ============================================================================

POOL_PATH: str = zoo_path("colorgraphs", "data/pool.npz")

# :param BASE_CKPT:
#     The frozen unconditional base: the best checkpoint of the base training
#     (EMA @ epoch 80), measured at distribution_score 0.9862.
BASE_CKPT: str = zoo_path("colorgraphs", "base.ckpt")

TRAIN_SPLIT: float = 0.9
SEED: int = 42

# --- Adapter architecture (the proven concept/fingerprint-adapter recipe) ----
H_HIDDEN: int = 256
TIME_CONDITIONED: bool = True
STREAMS: list = ["X", "E", "y"]

# --- The architecture: cross-attention with Fourier bands (the library default) ---
# ``None`` = take the library's for_base default. What was actually built is logged
# and stored in the archive.
#
# Set XATTN_TOKENS=0 and COND_FOURIER=0 to get the FiLM-only adapter (the control arm of
# Table 1, which measured slope 0.264).
XATTN_TOKENS: int = None
XATTN_DIM: int = None
XATTN_HEADS: int = None
COND_FOURIER: int = None

# --- Training ---------------------------------------------------------------
# :param EPOCHS:
#     80 epochs at LR 4e-4 is the recipe that goes with the cross-attention architecture, not a
#     separate choice: at the older 2e-4 the larger adapter lands at a worse error. The FiLM-only
#     control arm of Table 1 was trained with the same schedule; it differs from the full adapter
#     in architecture only.
EPOCHS: int = 80
BATCH_SIZE: int = 128          # upper bound; probed down with headroom
MIN_BATCH_SIZE: int = 16
MEMORY_HEADROOM: float = 0.80
LEARNING_RATE: float = 4e-4
# The adapter's gate head is zero-initialised, so the unconditional branch IS the frozen base
# exactly; there is nothing to drop out to, and no condition dropout is used.
EMA_DECAY: float = 0.999
MAX_TIME_HOURS: float = 8.0
CHECKPOINT_EVERY_K: int = 5

# --- Evaluation -------------------------------------------------------------
# :param GUIDANCE_WEIGHTS:
#     The weights of the in-training calibration check. The real sweep over the weights
#     (above 1, where this adapter measures best) is eval_target_adapter.py; this job keeps
#     w=1 as a cheap sanity check only.
GUIDANCE_WEIGHTS: list = [1.0]
# :param REQUEST_LEVELS:
#     -3..+3 are the pool's seven levels. +/-4 and +/-5 are OUTSIDE the training
#     support (the pool contains exactly zero such graphs) and test whether
#     conditioning can steer into a region only the base visits.
REQUEST_LEVELS: list = [-3, -2, -1, 0, 1, 2, 3]
N_PER_CELL: int = 200
EVAL_CHUNK: int = 64
EVAL_SAMPLE_STEPS: int = 500
ETA: float = 0.0        # matches every base measurement, so numbers are comparable
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
REFERENCE_SIZE: int = 5000

# False: every run gets its own time-stamped folder (True writes to one `debug/` folder, which the next run wipes).
__DEBUG__: bool = False
__TESTING__: bool = False


def composition_report(samples, level: int) -> dict:
    """How the model *routed* to a requested target, and whether it landed.

    The target is many-to-one in motif content, so hitting it is only half the
    question -- the other half is which motifs were spent getting there.

    :returns: achieved-target statistics plus mean per-motif counts.
    """
    counts = [count_motifs_pyg(s) for s in samples]
    achieved = np.array([exact_target(c) for c in counts], dtype=np.float64)
    return {
        "requested": int(level),
        "n": len(samples),
        "mean_achieved": float(achieved.mean()) if len(achieved) else float("nan"),
        "sd_achieved": float(achieved.std()) if len(achieved) else float("nan"),
        "hit_rate": float((achieved == level).mean()) if len(achieved) else 0.0,
        "mae": float(np.abs(achieved - level).mean()) if len(achieved) else float("nan"),
        "mean_motifs": {name: float(np.mean([c[name] for c in counts]))
                        for name in MOTIF_NAMES} if counts else {},
    }


def pool_composition(reference, level: int) -> dict:
    """The pool's own motif composition among graphs at ``level``.

    The comparison the model's routing is judged against -- without it, a mean
    count is a number with nothing to mean anything relative to.
    """
    counts = [count_motifs_pyg(g) for g in reference]
    keep = [c for c in counts if exact_target(c) == level]
    if not keep:
        return {"n": 0, "mean_motifs": {name: float("nan") for name in MOTIF_NAMES}}
    return {
        "n": len(keep),
        "mean_motifs": {name: float(np.mean([c[name] for c in keep]))
                        for name in MOTIF_NAMES},
    }


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


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    e.log("Target-conditioned adapter on the frozen colour-graph base")

    if e.__TESTING__:
        e.EPOCHS = 2
        e.N_PER_CELL = 8
        e.EVAL_CHUNK = 8
        e.EVAL_SAMPLE_STEPS = 20
        e.GUIDANCE_WEIGHTS = [1.0]
        e.REQUEST_LEVELS = [-3, 0, 3]
        e.BATCH_SIZE = 16
        e.REFERENCE_SIZE = 300
        e.MAX_TIME_HOURS = 0.5
        e.CHECKPOINT_EVERY_K = 1
        e.log("TESTING mode")

    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Data: attach the RAW label; the adapter normalizes internally --------
    graphs, meta = load_pool_npz(e.POOL_PATH)
    labels = np.array([m["target"][0] for m in meta], dtype=np.float64)
    e.log(f"pool: {len(graphs)} graphs, label mean {labels.mean():+.3f} "
          f"sd {labels.std():.3f} range [{labels.min():.2f}, {labels.max():.2f}]")

    n_train = int(len(graphs) * e.TRAIN_SPLIT)
    generator = torch.Generator().manual_seed(e.SEED)
    perm = torch.randperm(len(graphs), generator=generator).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    for i in train_idx:
        graphs[i].cond = torch.tensor([[labels[i]]], dtype=torch.float)
    train_set = [graphs[i] for i in train_idx]
    reference = [graphs[i] for i in val_idx][: e.REFERENCE_SIZE]
    e.log(f"train {len(train_set)} | held-out reference {len(reference)}")

    if e.__TESTING__:
        train_set = train_set[:400]

    cond_mean = float(labels[train_idx].mean())
    cond_std = float(labels[train_idx].std())
    e["config/cond_mean"], e["config/cond_std"] = cond_mean, cond_std

    # -- Frozen base + adapter -----------------------------------------------
    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    if base.cond_dim != 0:
        raise ValueError(f"expected an unconditional base, got cond_dim={base.cond_dim}")
    if base.num_node_classes != len(RB_PALETTE):
        raise ValueError(
            f"base has {base.num_node_classes} node classes, palette has "
            f"{len(RB_PALETTE)} -- wrong base checkpoint"
        )
    e.log(f"frozen base: {sum(p.numel() for p in base.parameters()):,} params")

    arch = {k: v for k, v in (
        ("xattn_tokens", e.XATTN_TOKENS), ("xattn_dim", e.XATTN_DIM),
        ("xattn_heads", e.XATTN_HEADS), ("cond_fourier", e.COND_FOURIER),
    ) if v is not None}
    adapter = CrossAttentionAdapter.for_base(
        base, cond_dim=1, hidden=e.H_HIDDEN, time_conditioned=e.TIME_CONDITIONED,
        streams=tuple(e.STREAMS), cond_mean=np.array([cond_mean]),
        cond_std=np.array([cond_std]), name="target_adapter",
        cond_type="rb_dual_motifs_target", **arch,
    ).to(device)
    e["adapter/num_params"] = int(sum(p.numel() for p in adapter.parameters()))
    built = {"xattn_tokens": int(getattr(adapter, "xattn_tokens", 0)),
             "xattn_dim": int(getattr(adapter, "xattn_dim", 0)),
             "xattn_heads": int(getattr(adapter, "xattn_heads", 0)),
             "cond_fourier": int(getattr(adapter, "cond_fourier", 0))}
    e["adapter/architecture"] = built
    e.log(f"adapter: {e['adapter/num_params']:,} trainable params")
    e.log(f"  architecture as built: {built}")
    if built["xattn_tokens"] == 0:
        e.log("  NOTE: cross-attention is OFF -- this is the FiLM-only control arm")

    module = AdapterModule(base, adapter, cond_attr="cond", lr=e.LEARNING_RATE)

    batch_size = largest_fitting_batch(
        module, train_set, e.BATCH_SIZE, e.MIN_BATCH_SIZE, e.log, e.MEMORY_HEADROOM)
    e["config/batch_size"] = batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    e.log(f"training: batch {batch_size}, {len(train_set) // batch_size} steps/epoch")

    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS},
        accelerator="auto", devices=1, enable_progress_bar=False,
        enable_checkpointing=False, logger=False, gradient_clip_val=1.0,
        callbacks=[EMACallback(decay=e.EMA_DECAY),
                   PeriodicAdapterCheckpoint(adapter, e.path, e.CHECKPOINT_EVERY_K)],
    )
    trainer.fit(module, train_dataloaders=train_loader)

    # Save BEFORE evaluating: the sweep is hours long and must not be able to
    # take the trained adapter down with it.
    ckpt = adapter.save(os.path.join(e.path, "target_adapter"))
    e.log(f"saved adapter to {ckpt}")

    # -- Evaluation: calibration + routing across weights and levels ----------
    # Lightning's teardown moves the LightningModule (and with it the adapter and
    # the frozen base) back to CPU when fit() returns. Sampling then dies inside
    # CrossAttentionAdapter.normalize with the cond_mean/cond_std buffers on CPU and the
    # condition on CUDA. Put both back explicitly.
    base = base.to(device).eval()
    adapter = adapter.to(device).eval()
    domain = ColorGraphDomain()
    pool_at_level = {lv: pool_composition(reference, lv) for lv in e.REQUEST_LEVELS}

    results = {}
    for weight in e.GUIDANCE_WEIGHTS:
        e.log("=" * 66)
        e.log(f"GUIDANCE WEIGHT w = {weight}")
        e.log(f"  {'req':>4} {'achieved':>9} {'sd':>6} {'hit':>7} {'MAE':>6}   "
              + " ".join(f"{n.split('_')[0][0].upper()}{n.split('_')[1][0]:>3}"
                         for n in MOTIF_NAMES))
        per_level = {}
        for level in e.REQUEST_LEVELS:
            cond = torch.tensor([[float(level)]], dtype=torch.float, device=device)
            samples = []
            while len(samples) < e.N_PER_CELL:
                cur = min(e.EVAL_CHUNK, e.N_PER_CELL - len(samples))
                composition = AdapterComposition(
                    [ConditionBranch(adapter, cond.expand(cur, -1), weight)],
                    base=base)
                # sample_steps / eta / omega belong to the SAMPLER, not to
                # sample(); AdaptedSampler.sample takes no step count.
                sampler = AdaptedSampler(base, composition, eta=e.ETA,
                                         omega=e.OMEGA,
                                         sample_steps=e.EVAL_SAMPLE_STEPS,
                                         time_distortion=e.TIME_DISTORTION)
                samples += sampler.sample(num_samples=cur, device=device,
                                          show_progress=False)
            report = composition_report(samples, level)
            report["pool"] = pool_at_level[level]
            per_level[str(level)] = report
            e.log(f"  {level:>4} {report['mean_achieved']:>9.3f} "
                  f"{report['sd_achieved']:>6.3f} {report['hit_rate']:>7.3f} "
                  f"{report['mae']:>6.3f}   "
                  + " ".join(f"{report['mean_motifs'][n]:>4.2f}" for n in MOTIF_NAMES))
        results[str(weight)] = per_level

    e["eval/results"] = results
    e["eval/pool_at_level"] = {str(k): v for k, v in pool_at_level.items()}
    with open(os.path.join(e.path, "calibration.json"), "w") as handle:
        json.dump({"results": results,
                   "pool_at_level": {str(k): v for k, v in pool_at_level.items()},
                   "levels": list(e.REQUEST_LEVELS),
                   "weights": list(e.GUIDANCE_WEIGHTS)}, handle, indent=2)

    # -- Calibration figure ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    levels = list(e.REQUEST_LEVELS)
    ax.plot(levels, levels, "k--", lw=1, label="perfect", zorder=1)
    ax.axvspan(-3, 3, color="0.92", zorder=0, label="training support")
    for weight in e.GUIDANCE_WEIGHTS:
        means = [results[str(weight)][str(lv)]["mean_achieved"] for lv in levels]
        sds = [results[str(weight)][str(lv)]["sd_achieved"] for lv in levels]
        ax.errorbar(levels, means, yerr=sds, marker="o", capsize=3,
                    label=f"w = {weight}")
    ax.set_xlabel("requested target")
    ax.set_ylabel("achieved exact motif sum")
    ax.set_title("calibration (shaded = levels the pool contains)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for weight in e.GUIDANCE_WEIGHTS:
        hits = [results[str(weight)][str(lv)]["hit_rate"] for lv in levels]
        ax2.plot(levels, hits, marker="s", label=f"w = {weight}")
    ax2.axvspan(-3, 3, color="0.92", zorder=0)
    ax2.set_xlabel("requested target")
    ax2.set_ylabel("exact hit rate")
    ax2.set_title("fraction landing exactly on the requested target")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(e.path, "calibration.png"), dpi=140)
    plt.close(fig)
    e.log(f"wrote calibration figure to {e.path}")


experiment.run_if_main()
