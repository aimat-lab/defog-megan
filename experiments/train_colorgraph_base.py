"""
Unconditional DeFoG base on the synthetic colour-graph pool (``rb_dual_motifs``).

This is the generative half of the planted-motif experiment, and it is
deliberately the *unconditional* half: it depends on no MEGAN checkpoint and no
concept clustering, so it can be trained to completion while the explanation side
is still moving. Whichever clustering eventually wins gets adapters fitted on top
of this frozen base, exactly as the molecular adapters sit on the frozen ZINC base.

What differs from the molecular base runs:

* ``molecular_features=False`` -- there is no valence and no atomic weight. This
  is the DeFoG default; the molecular path is the special case, not this one.
* 7 node classes (the frozen :data:`~defog_megan.colorgraphs.RB_PALETTE`) and 2
  edge classes (class 0 = no edge, class 1 = the pool's single edge type).
* Validity is not a gate. Any colour graph is well-formed, so quality is measured
  distributionally by :func:`~defog_megan.colorgraphs.structural_report` --
  colour / degree / size divergence from the pool, plus connectivity, which the
  training pool satisfies 100% of the time and is therefore a real signal.
* Graphs are bigger than molecules (20-48 nodes vs ~23 heavy atoms), so the dense
  edge tensor is roughly 4x larger per graph. Batch size is set accordingly.

Imports nothing MEGAN-derived, so it runs anywhere DeFoG runs.

Usage::

    python experiments/train_colorgraph_base.py --__TESTING__ True   # smoke
    python experiments/train_colorgraph_base.py
"""
import os
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_adapt.core import (
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback
)
from defog_megan.zoo import zoo_path
from defog_megan.colorgraphs import (
    RB_PALETTE, RB_PALETTE_NAMES, ColorGraphDomain, build_pyg_dataset, load_pool_npz,
    distribution_score, structural_report,
)
from defog_megan.rb_motifs import MOTIF_NAMES, motif_report

# ============================================================================
# Parameters
# ============================================================================

# :param POOL_PATH:
#     The pool to train on: the compact copy of the dataset that the zoo ships.
# :param VGD_ROOT:
#     Alternatively the raw visual graph dataset folder; when it is set it is read instead of the pool.
#     It must use the frozen palette -- build_pyg_dataset asserts this rather than silently remapping colours.
POOL_PATH: str = zoo_path("colorgraphs", "data/pool.npz")
VGD_ROOT: str = ""

# :param LIMIT:
#     Cap on how many graphs to read (None = the whole 100k pool).
LIMIT: int = None

# --- Model architecture -----------------------------------------------------
# Sized between the authors' non-molecular configs and the molecular runs. The
# graphs are larger than molecules but the alphabet is far smaller (7x2 classes
# against 9x4), so depth matters more here than width.
N_LAYERS: int = 8
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
# :param RRWP_STEPS:
#     Random-walk depth. The planted motifs are rings and stars of radius <= 2,
#     but ring detection needs walks long enough to close: 16 covers the largest
#     planted ring with margin.
RRWP_STEPS: int = 16

# --- Training ---------------------------------------------------------------
# :param EPOCHS:
#     150 over 90k graphs is ~13.5M graph presentations. More than that is waste: the best
#     checkpoint is selected on distribution_score anyway, so a long tail costs time without
#     costing quality.
EPOCHS: int = 150
# :param BATCH_SIZE:
#     Upper bound, not a promise -- MAX_BATCH_PROBE below reduces it until a real
#     forward/backward fits the GPU. The dense edge tensor is (bs, n, n, de) with
#     n up to 48, so this is the parameter that decides whether the job OOMs.
BATCH_SIZE: int = 256
# :param MIN_BATCH_SIZE:
#     Floor for the probe. If even this does not fit, fail loudly at startup
#     rather than 20 hours in.
MIN_BATCH_SIZE: int = 16
# :param MEMORY_HEADROOM:
#     Fraction of GPU memory a worst-case training step may occupy. 0.80 leaves
#     room for the allocator fragmentation that killed the previous run at
#     epoch 140 after it had been accepted at 90% occupancy.
MEMORY_HEADROOM: float = 0.80
# :param CHECKPOINT_EVERY_K:
#     Epoch interval for the rolling last_model.ckpt snapshot.
CHECKPOINT_EVERY_K: int = 10
LEARNING_RATE: float = 2e-4
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
# :param LAMBDA_EDGE:
#     Edge loss weight. The whole signal here is topological -- a ring and a star
#     differ only in wiring -- so edges carry more of the task than in molecules.
LAMBDA_EDGE: float = 5.0
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999
TRAIN_SPLIT: float = 0.9

# --- Sampling / evaluation --------------------------------------------------
SAMPLE_STEPS: int = 100
EVAL_SAMPLE_STEPS: int = 500
GEN_SAMPLE_STEPS: int = 200
GEN_ETA: float = 5.0
ETA: float = 0.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"
# :param NUM_EVAL_SAMPLES:
#     5000 puts the standard error on a motif rate near 0.5 at ~0.7 points, which
#     is well inside what the comparison against the pool needs to resolve.
NUM_EVAL_SAMPLES: int = 5000
EVAL_CHUNK: int = 128
SAMPLE_VIS_EVERY_K: int = 25
# :param REFERENCE_SIZE:
#     How many training graphs the distributional report compares against.
REFERENCE_SIZE: int = 5000

SEED: int = 42

__DEBUG__: bool = True
__TESTING__: bool = False


def largest_fitting_batch(model, graphs, upper: int, lower: int, log,
                          headroom: float = MEMORY_HEADROOM) -> int:
    """Halve the batch size until a real forward+backward fits *with headroom*.

    The dense edge tensor is ``(bs, n, n, de)``, so memory scales with the square
    of the graph size and this dataset's graphs are ~4x a typical ZINC graph's.
    A 24-hour job is the wrong place to find that out, and the probe costs
    seconds. Uses the *largest* graphs in the pool so the estimate is worst-case
    rather than average.

    **"Fits once" is not "fits for 100k steps."** An earlier run of this
    experiment accepted batch 128 at a 21.7 GiB peak on a 24 GiB card -- a 10%
    margin -- trained happily for 140 epochs, and then died at the optimizer step
    with 2.9 GiB of the GPU stranded in unusable allocator fragments. So a batch
    is only accepted if its worst-case peak stays under ``headroom`` of total
    device memory; anything above that is treated as not fitting.

    :returns: the largest batch whose worst-case peak left the required headroom.
    """
    import torch as _torch
    from torch_geometric.loader import DataLoader as _DataLoader

    if not _torch.cuda.is_available():
        log("no CUDA -- skipping the batch probe")
        return min(upper, 32)

    biggest = sorted(graphs, key=lambda g: -int(g.x.shape[0]))[:upper]
    batch_size = upper
    while batch_size >= lower:
        try:
            probe = _DataLoader(biggest[:batch_size], batch_size=batch_size)
            batch = next(iter(probe))
            out = model.training_step(batch.to("cuda"), 0)
            loss = out["loss"] if isinstance(out, dict) else out
            loss.backward()
            model.zero_grad(set_to_none=True)
            _torch.cuda.synchronize()
            peak = _torch.cuda.max_memory_allocated() / 2 ** 30
            total = _torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            _torch.cuda.empty_cache()
            _torch.cuda.reset_peak_memory_stats()
            worst_n = max(int(g.x.shape[0]) for g in biggest[:batch_size])
            if peak > headroom * total:
                log(f"batch probe: {batch_size} fits but peaks at {peak:.1f}/"
                    f"{total:.1f} GiB ({peak / total:.0%} > {headroom:.0%} limit) "
                    f"-- too tight for a long run, halving")
                batch_size //= 2
                continue
            log(f"batch probe: {batch_size} accepted (peak {peak:.1f}/{total:.1f} GiB, "
                f"{peak / total:.0%}, on the {worst_n}-node worst case)")
            return batch_size
        except _torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            _torch.cuda.empty_cache()
            _torch.cuda.reset_peak_memory_stats()
            log(f"batch probe: {batch_size} does not fit, halving")
            batch_size //= 2

    raise RuntimeError(
        f"even batch size {lower} does not fit on this GPU -- the model or the "
        f"graph sizes need to come down before this run is worth starting"
    )


class PeriodicCheckpoint(pl.Callback):
    """Write a rolling ``last_model.ckpt`` every ``every_k_epochs``.

    Exists because the previous run of this experiment lost six hours of
    training. Best-checkpoint selection keys off the in-training generation
    probe, that probe plateaued at epoch 80, and when the job then died at epoch
    140 there was nothing on disk between the two -- epochs 81-140 evaporated.

    A rolling snapshot is the cheap insurance: 34 MB per write, and it makes any
    crash cost at most ``every_k_epochs`` epochs instead of everything since the
    last score improvement.

    Note this saves the LIVE weights, not the EMA ones -- :class:`EMACallback`
    only swaps its shadow in during validation and at fit end. That is the right
    trade for a crash-recovery snapshot (it is what an optimizer resume would
    need) but it means ``last_model.ckpt`` is not directly comparable to
    ``best_model.ckpt``, which is EMA. Compare like with like.
    """

    def __init__(self, checkpoint_dir: str, every_k_epochs: int = 10):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.every_k_epochs = every_k_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.every_k_epochs:
            return
        try:
            pl_module.save(os.path.join(self.checkpoint_dir, "last_model"))
        except Exception as exc:      # never let snapshotting kill training
            warnings.warn(f"periodic checkpoint failed at epoch {epoch}: {exc}")


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    e.log("Unconditional colour-graph DeFoG base (rb_dual_motifs)")

    if e.__TESTING__:
        e.LIMIT = 400
        e.EPOCHS = 2
        e.N_LAYERS = 2
        e.HIDDEN_DIM = 64
        e.HIDDEN_MLP_DIM = 128
        e.N_HEADS = 4
        e.RRWP_STEPS = 8
        e.NUM_EVAL_SAMPLES = 16
        e.EVAL_SAMPLE_STEPS = 30
        e.GEN_SAMPLE_STEPS = 30
        e.REFERENCE_SIZE = 200
        e.SAMPLE_VIS_EVERY_K = 1
        e.BATCH_SIZE = 16          # keep the probe off a possibly-busy local GPU
        e.EVAL_CHUNK = 16
        e.log("TESTING mode: tiny model, 400 graphs, 2 epochs")

    pl.seed_everything(e.SEED, workers=True)

    # -- Data ---------------------------------------------------------------
    if e.VGD_ROOT:
        graphs, metadata = build_pyg_dataset(e.VGD_ROOT, limit=e.LIMIT)
        source = e.VGD_ROOT
    else:
        graphs, metadata = load_pool_npz(e.POOL_PATH)
        if e.LIMIT:
            graphs, metadata = graphs[:e.LIMIT], metadata[:e.LIMIT]
        source = e.POOL_PATH
    e.log(f"Loaded {len(graphs)} colour graphs from {source}")
    e["config/palette"] = [list(c) for c in RB_PALETTE]
    e["config/palette_names"] = list(RB_PALETTE_NAMES)

    sizes = np.array([int(g.x.shape[0]) for g in graphs])
    e.log(f"Nodes: min {sizes.min()} max {sizes.max()} mean {sizes.mean():.1f}")
    e["data/num_graphs"] = len(graphs)
    e["data/max_nodes"] = int(sizes.max())

    from torch_geometric.loader import DataLoader
    n_train = int(len(graphs) * e.TRAIN_SPLIT)
    perm = torch.randperm(len(graphs)).tolist()
    train_set = [graphs[i] for i in perm[:n_train]]
    val_set = [graphs[i] for i in perm[n_train:]]
    e.log(f"Train: {len(train_set)}  Val: {len(val_set)}")

    # The distributional comparison is against HELD-OUT graphs, so a model that
    # memorised the training set does not get to look good for it.
    reference = (val_set or train_set)[: e.REFERENCE_SIZE]
    e.log(f"Reference for the distributional report: {len(reference)} held-out graphs")

    # -- Model --------------------------------------------------------------
    # Batch size here is irrelevant -- from_dataloader only walks the data to
    # collect class counts, marginals and the size distribution.
    stats_loader = DataLoader(train_set, batch_size=256)
    model = DeFoGModel.from_dataloader(
        stats_loader,
        n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT, noise_type=e.NOISE_TYPE,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        molecular_features=False,        # no chemistry here -- see module docstring
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY,
        lambda_edge=e.LAMBDA_EDGE, train_time_distortion=e.TRAIN_TIME_DISTORTION,
        lr_scheduler=e.LR_SCHEDULER, lr_min=e.LR_MIN,
        sample_steps=e.SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
    )
    if model.num_node_classes != len(RB_PALETTE) or model.num_edge_classes != 2:
        raise ValueError(
            f"class counts do not match the palette contract: "
            f"{model.num_node_classes} node / {model.num_edge_classes} edge classes, "
            f"expected {len(RB_PALETTE)} / 2"
        )
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"Params: {e['model/num_params']:,}  "
          f"({model.num_node_classes} node classes, {model.num_edge_classes} edge classes, "
          f"cond_dim={model.cond_dim})")

    # -- Batch size: probed, not assumed ------------------------------------
    if torch.cuda.is_available():
        model = model.to("cuda")
    batch_size = largest_fitting_batch(
        model, train_set, e.BATCH_SIZE, e.MIN_BATCH_SIZE, e.log
    )
    e["config/batch_size"] = batch_size
    e.log(f"Training with batch size {batch_size} "
          f"({len(train_set) // batch_size} steps/epoch)")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size) if val_set else None

    domain = ColorGraphDomain()

    # -- Train --------------------------------------------------------------
    def gen_metrics_fn(samples):
        """In-training probe: distributional realism, since validity is free."""
        report = structural_report(samples, reference)
        # The monitor selects the best checkpoint on "validity". There is none
        # here, and connectivity alone would be perverse -- an over-dense model
        # scores 1.0 for free -- so feed it the combined distributional score.
        return {
            "validity": distribution_score(report),
            "uniqueness": domain.metrics(samples)["uniqueness"],
            "connected_rate": report["connected_rate"],
            "kl_colors": report["kl_colors"],
            "kl_degrees": report["kl_degrees"],
            "kl_sizes": report["kl_sizes"],
        }

    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=gen_metrics_fn, gen_every_k=10, gen_num_samples=64,
        gen_sample_steps=e.GEN_SAMPLE_STEPS, gen_eta=e.GEN_ETA, checkpoint_dir=e.path,
    )
    sampler = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=e.SAMPLE_VIS_EVERY_K,
        sample_steps=e.GEN_SAMPLE_STEPS, eta=e.GEN_ETA,
        domain=domain,
        figure_callback=lambda fig: e.track("samples", fig),
    )
    callbacks = [monitor, sampler,
                 PeriodicCheckpoint(e.path, every_k_epochs=e.CHECKPOINT_EVERY_K)]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
        e.log(f"EMA enabled (decay={e.EMA_DECAY})")

    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=False, logger=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model_path = model.save(os.path.join(e.path, "model"))
    e.log(f"Saved final model to {model_path}")

    # -- Evaluation ---------------------------------------------------------
    e.log("=" * 60)
    e.log(f"EVALUATION: {e.NUM_EVAL_SAMPLES} unconditional samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_path = os.path.join(e.path, "best_model")
    if os.path.exists(best_path + ".ckpt"):
        e.log(f"Loading best checkpoint (best connectivity={monitor.best_validity:.3f})")
        model = DeFoGModel.load(best_path)
    model = model.to(device).eval()

    # Sampling has no backward pass so it is lighter than training, but a chunk
    # larger than the batch the GPU accepted is still asking for trouble.
    eval_chunk = min(e.EVAL_CHUNK, batch_size)
    e.log(f"sampling in chunks of {eval_chunk}")

    samples = []
    remaining = e.NUM_EVAL_SAMPLES
    while remaining > 0:
        cur = min(eval_chunk, remaining)
        samples += model.sample(num_samples=cur, sample_steps=e.EVAL_SAMPLE_STEPS,
                                device=device, show_progress=False)
        remaining -= cur

    report = structural_report(samples, reference)
    report["distribution_score"] = distribution_score(report)
    gen_metrics = domain.metrics(samples)
    for key, value in {**report, **gen_metrics}.items():
        if isinstance(value, float):
            e[f"eval/{key}"] = value
            e.log(f"  {key:24s} {value:.4f}")
        else:
            e[f"eval/{key}"] = value

    e.log("-" * 60)
    e.log("colour marginals  gen vs pool:")
    for name, gen_p, ref_p in zip(RB_PALETTE_NAMES,
                                  report["color_marginals"],
                                  report["color_marginals_ref"]):
        e.log(f"  {name:8s} {gen_p:.4f}  vs  {ref_p:.4f}")

    # -- The motif grader ---------------------------------------------------
    # Exact subgraph matching against the planted ground truth. This is the
    # number the synthetic experiment exists to produce: it is graded by the
    # generator's own definition, with no MEGAN anywhere in the loop.
    e.log("=" * 60)
    e.log("MOTIF FIDELITY (exact subgraph match, disjoint counting)")
    motifs = motif_report(samples, reference)
    e["eval/motifs"] = motifs

    e.log(f"  {'motif':12s} {'gen rate':>9s} {'pool rate':>9s} "
          f"{'gen mean':>9s} {'pool mean':>9s}")
    for name in MOTIF_NAMES:
        e.log(f"  {name:12s} {motifs['motif_rate'][name]:9.4f} "
              f"{motifs['motif_rate_ref'][name]:9.4f} "
              f"{motifs['motif_mean_count'][name]:9.4f} "
              f"{motifs['motif_mean_count_ref'][name]:9.4f}")

    e.log(f"  exact target: gen mean {motifs['mean_exact_target']:+.3f} "
          f"sd {motifs['std_exact_target']:.3f}  |  "
          f"pool mean {motifs['mean_exact_target_ref']:+.3f} "
          f"sd {motifs['std_exact_target_ref']:.3f}")
    e.log(f"  KL(gen || pool) over the exact target: {motifs['kl_motif_targets']:.4f}")

    e.log("  exact-target histogram (motif-sum, noise-free):")
    for value, gen_n, ref_n in zip(motifs["target_support"],
                                   motifs["target_hist"],
                                   motifs["target_hist_ref"]):
        n_gen = gen_n / max(sum(motifs["target_hist"]), 1)
        n_ref = ref_n / max(sum(motifs["target_hist_ref"]), 1)
        e.log(f"    {value:+3d}  gen {n_gen:.4f}   pool {n_ref:.4f}")

    # A final grid, drawn in the pool's own colours.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    for ax, sample in zip(axes.flat, samples):
        domain.render(ax, sample)
        ax.set_title(domain.caption(sample), fontsize=7)
    fig.suptitle(f"unconditional colour-graph samples | {domain.summarize(samples)}")
    fig.tight_layout()
    e.commit_fig("final_samples.png", fig)
    plt.close(fig)


experiment.run_if_main()
