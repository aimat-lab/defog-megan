"""
Feynman-Kac steering stacked on the colour-graph adapter. Eval only.

FK runs a population of ``K`` particles through the adapter-conditioned dynamics,
scores each by its predicted-clean energy, and periodically resamples -- killing
particles heading away from the target and cloning ones heading toward it. The
adapter is the *proposal*; FK concentrates it further. Nothing is trained.

ONE COMBINATION PER PROCESS.  ``BETA`` and ``PARTICLES`` are single values, not
lists, so the four combinations run as four processes on four GPUs rather than
serially inside one. That also means a combination that dies takes only its own
GPU down.

TWO THINGS THAT DECIDE WHETHER THIS MEANS ANYTHING

*K is the batch.* Particles must compete inside one ``sample()`` call, so the
chunk size IS the particle count -- it cannot be set independently. ``N_PER_LEVEL``
is therefore a multiple of the largest K in the sweep, so every K divides it
exactly and no cell is graded on a short final batch that ran a different FK.

*The weight is the deployed one.* FK sits on top of the adapter at the weight
the colour sweep selected (w=2). Stacking it at a different weight would
measure FK plus a weight change and attribute both to FK.

WHAT FK IS EXPECTED TO MOVE, and what it cannot.  The energy is squared distance to
the requested motif sum, so FK attacks per-sample precision. It is a *selection* method -- it can only concentrate the distribution
the adapter already proposes, so it should tighten the conditional spread and
lower MAE, and it has no mechanism to correct a systematic slope error. If the
slope moves at all it will move the way selection toward a fixed target moves it,
which is toward that target -- i.e. FK at extreme requests may pull the mean out,
partially undoing the shrinkage RL introduced.

This grades IN-JOB: the motif detector is self-contained and needs no solver, so the
achieved target is computed here directly.

The defaults are the paper's Table 1 Feynman-Kac row (beta 16, K = 128, w = 2, multinomial
resampling); ``--PARTICLES 16`` gives its k = 16 row.

Usage::

    python experiments/eval_color_fk.py --__TESTING__ True
    python experiments/eval_color_fk.py --BETA 8.0 --PARTICLES 128
"""
import json
import os
import time

import numpy as np
import torch
import pytorch_lightning as pl
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_adapt.core import (
    DeFoGModel, CrossAttentionAdapter, AdapterComposition, ConditionBranch, FeynmanKacSampler,
)
from defog_megan.zoo import zoo_path
from defog_megan.colorgraphs import (
    RB_PALETTE, distribution_score, load_pool_npz, save_pool_npz,
    structural_report,
)
from defog_megan.rb_motifs import MOTIF_NAMES, count_motifs_pyg, exact_target
from defog_megan.color_energy import ColorTargetEnergy

# ============================================================================
# Parameters
# ============================================================================

POOL_PATH: str = zoo_path("colorgraphs", "data/pool.npz")
BASE_CKPT: str = zoo_path("colorgraphs", "base.ckpt")
# The cross-attention property adapter of Table 1. (PyComex wipes results/<experiment>/debug at the
# start of any debug-mode run, so a trained adapter should be copied out of it before it is used.)
ADAPTER_CKPT: str = zoo_path("colorgraphs", "property_adapter.ckpt")

# :param GUIDANCE_WEIGHT:
#     The colour sweep's selected weight.
GUIDANCE_WEIGHT: float = 2.0

# --- Feynman-Kac ------------------------------------------------------------
# :param BETA:
#     Tilt strength; the potential is ``-beta * energy``. The energy is normalised
#     so a 3-motif miss costs 1.0, so beta reads as a log-weight ratio:
#     2.5 -> ~12x weight ratio between a good and a bad particle, 8.0 -> ~3000x.
BETA: float = 16.0
# :param PARTICLES:
#     K. This IS the sampling batch -- particles compete within one call.
PARTICLES: int = 128
# :param WARMUP_FRAC:
#     No resampling before this fraction of steps. Early in denoising the
#     predicted clean graph is unreliable AND size-biased, so early resampling
#     culls large graphs for being transiently invalid rather than wrong.
WARMUP_FRAC: float = 0.7
RESAMPLE_INTERVAL: int = 0       # 0 -> the sampler's default (steps // 8)
ESS_FRAC: float = 0.0            # 0 -> resample at every checkpoint
# The paper's runs resampled multinomially (the library default is "residual").
RESAMPLE_SCHEME: str = "multinomial"

REQUEST_LEVELS: list = [-3, -2, -1, 0, 1, 2, 3]
# :param N_PER_LEVEL:
#     A multiple of K, so no cell ends on a short batch running a different FK than the rest
#     of that cell. 1024 for the paper runs: a multiple of both K in use (16 and 128), and the
#     same N the plain-adapter arms are sampled at, so novelty and uniqueness are measured over
#     equally sized sets.
N_PER_LEVEL: int = 1024
EVAL_SAMPLE_STEPS: int = 500
ETA: float = 0.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"

TRAIN_SPLIT: float = 0.9
REFERENCE_SIZE: int = 2000
SEED: int = 42

# False: every run gets its own time-stamped folder (True writes to one `debug/` folder, which the next run wipes).
__DEBUG__: bool = False
__TESTING__: bool = False


def _sync_clock(device) -> float:
    """``perf_counter``, but only after CUDA has actually finished. An
    unsynchronised timer would measure kernel-launch overhead rather than
    sampling, which is the whole quantity this arm is being charged for."""
    if torch.cuda.is_available() and getattr(device, "type", str(device)) != "cpu":
        torch.cuda.synchronize()
    return time.perf_counter()


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    e.log(f"FK on the colour adapter -- beta={e.BETA} K={e.PARTICLES} "
          f"w={e.GUIDANCE_WEIGHT}")

    if e.__TESTING__:
        e.REQUEST_LEVELS = [-3, 0, 3]
        e.PARTICLES = 4
        e.N_PER_LEVEL = 8
        e.EVAL_SAMPLE_STEPS = 20
        e.REFERENCE_SIZE = 200

    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if e.N_PER_LEVEL % e.PARTICLES:
        raise ValueError(
            f"N_PER_LEVEL={e.N_PER_LEVEL} is not a multiple of PARTICLES="
            f"{e.PARTICLES}; the last batch would run FK with a different "
            f"population than the rest of the cell")

    graphs, meta = load_pool_npz(e.POOL_PATH)
    labels = np.array([m["target"][0] for m in meta], dtype=np.float64)
    n_train = int(len(graphs) * e.TRAIN_SPLIT)
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(graphs), generator=generator).tolist()
    reference = [graphs[i] for i in perm[n_train:]][: e.REFERENCE_SIZE]
    reference_y = [float(labels[i]) for i in perm[n_train:]][: e.REFERENCE_SIZE]

    e.log(f"resample scheme: {e.RESAMPLE_SCHEME}")

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    if base.num_node_classes != len(RB_PALETTE):
        raise ValueError(f"base has {base.num_node_classes} node classes, palette "
                         f"has {len(RB_PALETTE)} -- wrong base")
    adapter = CrossAttentionAdapter.load(e.ADAPTER_CKPT, device="cpu").to(device).eval()
    if getattr(adapter, "xattn_tokens", 0) == 0:
        raise ValueError("adapter reloaded WITHOUT cross-attention -- wrong adapter file?")
    e.log(f"adapter {sum(p.numel() for p in adapter.parameters()):,} params, "
          f"xattn={adapter.xattn_tokens}, fourier={adapter.cond_fourier}")

    dump = {"weights": [e.GUIDANCE_WEIGHT], "levels": list(e.REQUEST_LEVELS),
            "blend_space": "prob",
            "fk": {"beta": e.BETA, "particles": e.PARTICLES,
                   "warmup_frac": e.WARMUP_FRAC,
                   "resample_interval": e.RESAMPLE_INTERVAL or "auto",
                   "ess_frac": e.ESS_FRAC or "every checkpoint",
                   "resample_scheme": e.RESAMPLE_SCHEME},
            "reference_y": reference_y, "samples": {}, "structural": {}}

    per_level, struct = {}, {}
    # One FK combination per process on one GPU, so seconds_per_sample here is
    # directly comparable to the plain adapter eval's. Graphs are saved so that
    # novelty / uniqueness / validity / KL can be scored for every arm by ONE
    # code path rather than per-arm definitions that would not be comparable.
    timing = {"device": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu"),
              "n_gpus_visible": torch.cuda.device_count(),
              "sample_steps": int(e.EVAL_SAMPLE_STEPS),
              "particles": int(e.PARTICLES), "beta": float(e.BETA),
              "cells": {}}
    e.log(f"  {'req':>5} {'achieved':>9} {'sd':>7} {'hit':>7} {'MAE':>7} {'dist':>7}")
    for level in e.REQUEST_LEVELS:
        cond = torch.tensor([[float(level)]], dtype=torch.float, device=device)
        # A fresh energy per level: the FK contract carries no per-row condition,
        # so the target is fixed at construction.
        energy = ColorTargetEnergy(float(level))
        samples = []
        t0 = _sync_clock(device)
        while len(samples) < e.N_PER_LEVEL:
            cur = e.PARTICLES          # K IS the batch; never a partial chunk
            composition = AdapterComposition(
                [ConditionBranch(adapter, cond.expand(cur, -1), e.GUIDANCE_WEIGHT)],
                base=base)
            sampler = FeynmanKacSampler(
                base, energy_fn=energy, composition=composition,
                beta=e.BETA, warmup_frac=e.WARMUP_FRAC,
                resample_interval=(e.RESAMPLE_INTERVAL or None),
                ess_frac=(e.ESS_FRAC or None),
                eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_SAMPLE_STEPS,
                time_distortion=e.TIME_DISTORTION, resample_scheme=e.RESAMPLE_SCHEME)
            samples += sampler.sample(num_samples=cur, device=device,
                                      show_progress=False)
        cell_seconds = _sync_clock(device) - t0
        timing["cells"][f"lv{level:+d}"] = {
            "seconds": cell_seconds, "n": len(samples),
            "seconds_per_sample": cell_seconds / max(1, len(samples))}

        fname = f"gen_fk_b{e.BETA:g}_k{e.PARTICLES}_lv{level:+d}.npz"
        save_pool_npz(os.path.join(e.path, fname), samples, [{}] * len(samples))

        counts = [count_motifs_pyg(s) for s in samples]
        achieved = np.array([exact_target(c) for c in counts], dtype=np.float64)
        report = structural_report(samples, reference)
        per_level[str(level)] = {
            "n": len(samples),
            "mean_achieved": float(achieved.mean()),
            "sd_achieved": float(achieved.std()),
            "hit_rate": float((achieved == level).mean()),
            "mae": float(np.abs(achieved - level).mean()),
            "mean_motifs": {m: float(np.mean([c[m] for c in counts]))
                            for m in MOTIF_NAMES},
        }
        struct[str(level)] = {
            "distribution_score": distribution_score(report),
            "connected_rate": report["connected_rate"],
            "fk_connected": energy.last_stats.get("connected_rate", float("nan")),
        }
        r, s = per_level[str(level)], struct[str(level)]
        e.log(f"  {level:>5} {r['mean_achieved']:>9.3f} {r['sd_achieved']:>7.3f} "
              f"{r['hit_rate']:>7.3f} {r['mae']:>7.3f} {s['distribution_score']:>7.3f}")

    per_sample = [c["seconds_per_sample"] for c in timing["cells"].values()]
    if per_sample:
        timing["mean_seconds_per_sample"] = float(np.mean(per_sample))
        e.log(f"FK sampling (K={e.PARTICLES}, beta={e.BETA}) on "
              f"{timing['device']}: {timing['mean_seconds_per_sample']:.3f} "
              f"s/graph averaged over {len(per_sample)} levels")

    dump["samples"][str(e.GUIDANCE_WEIGHT)] = per_level
    dump["structural"][str(e.GUIDANCE_WEIGHT)] = struct
    dump["timing"] = timing
    e["eval/timing"] = timing
    out_path = os.path.join(e.path, "conditional_samples.json")
    with open(out_path, "w") as handle:
        json.dump(dump, handle)
    e["eval/structural"] = struct
    e["eval/fk"] = dump["fk"]

    m = np.array([per_level[str(l)]["mean_achieved"] for l in e.REQUEST_LEVELS])
    L = np.array(e.REQUEST_LEVELS, dtype=float)
    slope, icpt = np.polyfit(L, m, 1)
    scores = [struct[str(l)]["distribution_score"] for l in e.REQUEST_LEVELS]
    e["eval/slope"], e["eval/intercept"] = float(slope), float(icpt)
    e["eval/per_level"] = per_level
    e.log(f"beta={e.BETA} K={e.PARTICLES}: slope {slope:.3f} intercept {icpt:+.3f}  "
          f"hit {np.mean([per_level[str(l)]['hit_rate'] for l in e.REQUEST_LEVELS]):.3f}  "
          f"MAE {np.mean([per_level[str(l)]['mae'] for l in e.REQUEST_LEVELS]):.3f}  "
          f"dist {np.mean(scores):.4f}")
    e.log("   (no-FK colour adapter at w=1: slope 0.911, hit 0.676, MAE 0.456)")
    e.log(f"wrote {out_path}")


experiment.run_if_main()
