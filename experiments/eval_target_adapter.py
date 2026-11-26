"""
Full calibration sweep for a trained target-conditioned adapter.

Separate from ``train_target_adapter.py`` on purpose. The sweep is 3 guidance
weights x 11 request levels x 1000 samples = 33k adapted samples, and an adapted
sampler runs both branches per step, so it costs roughly twice unconditional
sampling -- around ten GPU-hours. Bolting that onto the training job would put a
trained adapter behind a ten-hour failure surface, which is precisely how the
unconditional base run lost six hours of training.

It also means the sweep can be re-run -- different weights, more levels, more
samples -- without retraining anything.

What it measures, per (weight, level) cell:

* **calibration** -- mean and sd of the *achieved* exact motif sum against the
  request, plus the fraction landing exactly on it;
* **routing** -- which motifs were spent getting there, against the pool's own
  composition at that same level. The target is many-to-one in motif content
  (``+2`` = one ``red_star`` = two ``red_rings`` = ...), so hitting the number is
  only half the question;
* **extrapolation** -- levels +/-4 and +/-5 lie outside the training support
  entirely (the pool contains exactly zero such graphs) while the base does emit
  ~6% of its mass there, so this asks whether conditioning can steer into a
  region it never saw a label for.

Usage::

    python experiments/eval_target_adapter.py --__TESTING__ True
    python experiments/eval_target_adapter.py --ADAPTER_CKPT '"/path/to/target_adapter"'
"""
import json
import os
import time

import numpy as np
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_adapt.core import (
    DeFoGModel, CrossAttentionAdapter, AdapterComposition, ConditionBranch, AdaptedSampler,
    Sampler,
)
from defog_megan.zoo import zoo_path
from defog_megan.colorgraphs import RB_PALETTE, load_pool_npz, save_pool_npz
from defog_megan.rb_motifs import MOTIF_NAMES, count_motifs_pyg, exact_target

# ============================================================================
# Parameters
# ============================================================================

POOL_PATH: str = zoo_path("colorgraphs", "data/pool.npz")
BASE_CKPT: str = zoo_path("colorgraphs", "base.ckpt")
# :param ADAPTER_CKPT:
#     The property adapter to sweep (as written by CrossAttentionAdapter.save; the .ckpt suffix is
#     optional). The default is the cross-attention adapter of Table 1; pass the FiLM-only adapter of
#     the zoo (property_adapter_film_only.ckpt) for the control arm.
ADAPTER_CKPT: str = zoo_path("colorgraphs", "property_adapter.ckpt")
# :param EXPECT_XATTN:
#     1 = the reloaded adapter must HAVE cross-attention (the Table 1 adapter), 0 = it must NOT (the FiLM-only control row),
#     -1 = not checked. A FiLM/cross-attention mix-up is a silent wrong answer, not a crash: the run would finish and fill
#     the wrong table row.
EXPECT_XATTN: int = -1

# :param GUIDANCE_WEIGHTS:
#     The two weights the paper's main table reports: w=1.0 and w=2.0 (the weight the
#     Feynman-Kac arms are run at, so that the rows compare). Weights above 1 are meaningful
#     because the guidance is blended in probability space.
GUIDANCE_WEIGHTS: list = [1.0, 2.0]
# :param REQUEST_LEVELS:
#     In-support only. The pool contains exactly these seven levels, and the 0.264
#     and 0.905 numbers this is compared against were fitted over them. The
#     extrapolation levels (+/-4, +/-5) are deliberately dropped: a slope fitted
#     through requests the pool never contained measures reach, not calibration.
REQUEST_LEVELS: list = [-3, -2, -1, 0, 1, 2, 3]
#     1024 rather than 1000: the FK arms need N to be a multiple of their
#     particle count K (16 and 128), and every arm in the main table has to be
#     sampled at the same N or the novelty/uniqueness columns are not comparable.
N_PER_CELL: int = 1024
EVAL_CHUNK: int = 64
EVAL_SAMPLE_STEPS: int = 500
ETA: float = 0.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"

# :param SAVE_GRAPHS: dump every sampled graph as a pool-format .npz, one file per
#       (weight, level) cell, plus an unconditional baseline. The first run of this
#       experiment saved only aggregate statistics, and that made two later questions
#       unanswerable: re-scoring property-match against a TOLERANCE rather than exact
#       equality, and pushing the graphs through MEGAN to fill in the concept-match
#       column of Experiment 1's cross-conditioning table. Both need the graphs, and
#       aggregates cannot be un-aggregated.
SAVE_GRAPHS: bool = True
# :param N_BASELINE: unconditional samples from the frozen base, saved alongside --
#       the first row of that table. Independent of weight and level, so drawn once.
N_BASELINE: int = 1024
# :param BASELINE_ONLY: draw only the unconditional baseline and skip the weight
#       sweep. Added because the KL estimator is biased upward at small n -- the
#       pool against itself scores ~0.018 at n=1024 and ~0.005 at n=7168 -- so a
#       baseline row sampled at 1024 cannot be compared against conditional rows
#       sampled at 7168. This lets the baseline be re-drawn at the conditional
#       arms' sample size so every row carries the same estimator bias.
BASELINE_ONLY: bool = False
TRAIN_SPLIT: float = 0.9
REFERENCE_SIZE: int = 5000
SEED: int = 42

# False: every run gets its own time-stamped folder (True writes to one `debug/` folder, which the next run wipes).
__DEBUG__: bool = False
__TESTING__: bool = False
# no desktop notification when a run ends: on a machine without a desktop session it fails with a (harmless) traceback
__NOTIFY__: bool = False


def _sync_clock(device) -> float:
    """``perf_counter``, but only after CUDA has actually finished. An
    unsynchronised timer around ``sample()`` measures kernel-launch overhead
    rather than sampling, which would make the per-sample cost of the FK arms
    (many particles per call) look identical to the plain adapter's."""
    if torch.cuda.is_available() and getattr(device, "type", str(device)) != "cpu":
        torch.cuda.synchronize()
    return time.perf_counter()


def composition_report(samples, level: int) -> dict:
    """Achieved-target statistics plus the motif route taken to get there."""
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
    """The pool's own motif composition among held-out graphs at ``level``."""
    keep = [c for c in (count_motifs_pyg(g) for g in reference)
            if exact_target(c) == level]
    if not keep:
        return {"n": 0, "mean_motifs": {name: float("nan") for name in MOTIF_NAMES}}
    return {"n": len(keep),
            "mean_motifs": {name: float(np.mean([c[name] for c in keep]))
                            for name in MOTIF_NAMES}}


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    if e.__TESTING__:
        e.N_PER_CELL = 8
        e.EVAL_CHUNK = 8
        e.EVAL_SAMPLE_STEPS = 20
        e.GUIDANCE_WEIGHTS = [1.0]
        e.REQUEST_LEVELS = [-3, 0, 3]
        e.REFERENCE_SIZE = 300
        e.N_BASELINE = 8

    torch.manual_seed(e.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e.log(f"device: {device}")

    # -- Held-out reference, split identically to the training run ------------
    graphs, _ = load_pool_npz(e.POOL_PATH)
    n_train = int(len(graphs) * e.TRAIN_SPLIT)
    generator = torch.Generator().manual_seed(e.SEED)
    perm = torch.randperm(len(graphs), generator=generator).tolist()
    reference = [graphs[i] for i in perm[n_train:]][: e.REFERENCE_SIZE]
    e.log(f"reference: {len(reference)} held-out graphs")

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    if base.num_node_classes != len(RB_PALETTE):
        raise ValueError(f"base has {base.num_node_classes} node classes, "
                         f"palette has {len(RB_PALETTE)} -- wrong base")
    adapter = CrossAttentionAdapter.load(e.ADAPTER_CKPT, device="cpu").to(device).eval()
    if e.EXPECT_XATTN in (0, 1) and int(getattr(adapter, "xattn_tokens", 0) > 0) != e.EXPECT_XATTN:
        raise ValueError(f"EXPECT_XATTN={e.EXPECT_XATTN}, but the adapter has xattn_tokens="
                         f"{getattr(adapter, 'xattn_tokens', 0)} -- wrong adapter file for this table row")
    e.log(f"base {sum(p.numel() for p in base.parameters()):,} params (frozen) | "
          f"adapter {sum(p.numel() for p in adapter.parameters()):,} params")
    if adapter.cond_dim != 1:
        raise ValueError(f"expected a scalar-target adapter, got cond_dim="
                         f"{adapter.cond_dim}")

    pool_at_level = {lv: pool_composition(reference, lv) for lv in e.REQUEST_LEVELS}
    e.log("pool composition per level (held-out):")
    for lv in e.REQUEST_LEVELS:
        p = pool_at_level[lv]
        e.log(f"  {lv:+3d}  n={p['n']:>5}  " + " ".join(
            f"{n}={p['mean_motifs'][n]:.2f}" for n in MOTIF_NAMES))

    # Wall-clock per sample, recorded per cell. One process owns one GPU here, so
    # seconds_per_sample is already per-GPU and stays comparable to the FK arms,
    # which run one combination per process on a GPU each.
    timing = {"device": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu"),
              "n_gpus_visible": torch.cuda.device_count(),
              "sample_steps": int(e.EVAL_SAMPLE_STEPS),
              "cells": {}}

    if e.SAVE_GRAPHS and e.N_BASELINE:
        base_sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA,
                               sample_steps=e.EVAL_SAMPLE_STEPS,
                               time_distortion=e.TIME_DISTORTION)
        baseline = []
        t0 = _sync_clock(device)
        while len(baseline) < e.N_BASELINE:
            cur = min(e.EVAL_CHUNK, e.N_BASELINE - len(baseline))
            baseline += base_sampler.sample(num_samples=cur, device=device,
                                            show_progress=False)
        baseline_seconds = _sync_clock(device) - t0
        save_pool_npz(os.path.join(e.path, "gen_baseline.npz"), baseline,
                      [{}] * len(baseline))
        e.log(f"saved {len(baseline)} unconditional graphs -> gen_baseline.npz")
        e.log(f"unconditional sampling: {baseline_seconds:.1f} s for "
              f"{len(baseline)} graphs = "
              f"{baseline_seconds / max(1, len(baseline)):.3f} s/graph")
        timing["baseline"] = {"seconds": baseline_seconds, "n": len(baseline),
                              "seconds_per_sample":
                                  baseline_seconds / max(1, len(baseline))}

    results = {}
    for weight in ([] if e.BASELINE_ONLY else e.GUIDANCE_WEIGHTS):
        e.log("=" * 74)
        e.log(f"GUIDANCE WEIGHT w = {weight}")
        e.log(f"  {'req':>4} {'achieved':>9} {'sd':>6} {'hit':>7} {'MAE':>6}   "
              + "  ".join(f"{n:>9}" for n in MOTIF_NAMES))
        per_level = {}
        for level in e.REQUEST_LEVELS:
            cond = torch.tensor([[float(level)]], dtype=torch.float, device=device)
            samples = []
            t0 = _sync_clock(device)
            while len(samples) < e.N_PER_CELL:
                cur = min(e.EVAL_CHUNK, e.N_PER_CELL - len(samples))
                composition = AdapterComposition(
                    [ConditionBranch(adapter, cond.expand(cur, -1), weight)], base=base)
                sampler = AdaptedSampler(base, composition, eta=e.ETA, omega=e.OMEGA,
                                         sample_steps=e.EVAL_SAMPLE_STEPS,
                                         time_distortion=e.TIME_DISTORTION)
                samples += sampler.sample(num_samples=cur, device=device,
                                          show_progress=False)
            cell_seconds = _sync_clock(device) - t0
            timing["cells"][f"w{weight}_lv{level:+d}"] = {
                "seconds": cell_seconds, "n": len(samples),
                "seconds_per_sample": cell_seconds / max(1, len(samples))}
            if e.SAVE_GRAPHS:
                fname = f"gen_w{weight}_lv{level:+d}.npz"
                save_pool_npz(os.path.join(e.path, fname), samples,
                              [{}] * len(samples))
            report = composition_report(samples, level)
            report["pool"] = pool_at_level[level]
            if e.SAVE_GRAPHS:
                report["graphs_file"] = fname
            per_level[str(level)] = report
            e.log(f"  {level:>4} {report['mean_achieved']:>9.3f} "
                  f"{report['sd_achieved']:>6.3f} {report['hit_rate']:>7.3f} "
                  f"{report['mae']:>6.3f}   "
                  + "  ".join(f"{report['mean_motifs'][n]:>9.3f}" for n in MOTIF_NAMES))
            e.log(f"       pool composition at this level:          "
                  + "  ".join(f"{pool_at_level[level]['mean_motifs'][n]:>9.3f}"
                              for n in MOTIF_NAMES))
        results[str(weight)] = per_level

    per_sample = [c["seconds_per_sample"] for c in timing["cells"].values()]
    if per_sample:
        timing["mean_seconds_per_sample"] = float(np.mean(per_sample))
        e.log(f"adapted sampling on {timing['device']}: "
              f"{timing['mean_seconds_per_sample']:.3f} s/graph "
              f"averaged over {len(per_sample)} cells")

    e["eval/results"] = results
    e["eval/pool_at_level"] = {str(k): v for k, v in pool_at_level.items()}
    e["eval/timing"] = timing
    with open(os.path.join(e.path, "calibration.json"), "w") as handle:
        json.dump({"results": results,
                   "pool_at_level": {str(k): v for k, v in pool_at_level.items()},
                   "levels": list(e.REQUEST_LEVELS),
                   "weights": list(e.GUIDANCE_WEIGHTS),
                   "timing": timing}, handle, indent=2)

    # -- Figures --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = list(e.REQUEST_LEVELS)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    ax = axes[0]
    ax.axvspan(min(levels), max(levels), color="0.96", zorder=0,
           label="training support (all levels shown)")
    ax.plot(levels, levels, "k--", lw=1, zorder=1, label="perfect")
    for weight in e.GUIDANCE_WEIGHTS:
        means = [results[str(weight)][str(lv)]["mean_achieved"] for lv in levels]
        sds = [results[str(weight)][str(lv)]["sd_achieved"] for lv in levels]
        ax.errorbar(levels, means, yerr=sds, marker="o", capsize=3, label=f"w = {weight}")
    ax.set_xlabel("requested target")
    ax.set_ylabel("achieved exact motif sum")
    ax.set_title("calibration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for weight in e.GUIDANCE_WEIGHTS:
        hits = [results[str(weight)][str(lv)]["hit_rate"] for lv in levels]
        ax.plot(levels, hits, marker="s", label=f"w = {weight}")
    ax.set_xlabel("requested target")
    ax.set_ylabel("exact hit rate")
    ax.set_title("fraction landing exactly on the request")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Routing: rings vs stars used, model against pool, at the strongest weight.
    ax = axes[2]
    strongest = str(max(e.GUIDANCE_WEIGHTS))
    rings_m = [results[strongest][str(lv)]["mean_motifs"]["red_ring"]
               + results[strongest][str(lv)]["mean_motifs"]["blue_ring"] for lv in levels]
    stars_m = [results[strongest][str(lv)]["mean_motifs"]["red_star"]
               + results[strongest][str(lv)]["mean_motifs"]["blue_star"] for lv in levels]
    rings_p = [pool_at_level[lv]["mean_motifs"]["red_ring"]
               + pool_at_level[lv]["mean_motifs"]["blue_ring"] for lv in levels]
    stars_p = [pool_at_level[lv]["mean_motifs"]["red_star"]
               + pool_at_level[lv]["mean_motifs"]["blue_star"] for lv in levels]
    ax.plot(levels, rings_m, marker="o", color="tab:red", label="rings (model)")
    ax.plot(levels, rings_p, marker="o", ls="--", color="tab:red", label="rings (pool)")
    ax.plot(levels, stars_m, marker="s", color="tab:blue", label="stars (model)")
    ax.plot(levels, stars_p, marker="s", ls="--", color="tab:blue", label="stars (pool)")
    ax.set_xlabel("requested target")
    ax.set_ylabel("mean motifs per graph")
    ax.set_title(f"routing at w = {strongest}: which motifs were spent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(e.path, "calibration.png"), dpi=140)
    plt.close(fig)
    e.log(f"wrote calibration.png and calibration.json to {e.path}")


experiment.run_if_main()
