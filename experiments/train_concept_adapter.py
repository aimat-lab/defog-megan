"""
Train the MEGAN-concept adapter of a molecular rung on its frozen DeFoG base.

One adapter takes the whole ``φ`` vector (42 dims for the ZINC clustering, 29 for the Ames one) as its condition,
trained with the base's own denoising CE loss. At inference a single concept is selected and its target vector steers
generation.

This module deliberately **does not import MEGAN**: everything MEGAN-derived is precomputed and travels as data.

* ``concept_conditions.npz`` -- the per-molecule ``φ`` (training signal), see ``prepare_concept_dataset.py``
* ``concept_targets.npz``    -- the per-concept target vectors (steering), see ``export_concept_targets.py``

The run ends by *dumping the generated SMILES*; scoring them back through MEGAN is a separate step.

The defaults are the ZINC logP rung of the paper (nine neutral elements, kekulized, 50 epochs, learning rate 3e-4, the one
selected from the sweep 1e-4 .. 4e-4). For the Ames rung pass its base, tables and 12 atom classes and ``--EPOCHS 250``
(see ``reproduce/``). The conditions table holds molecules, so it is not part of the zoo's download: build it with
``prepare_concept_dataset.py`` first.

Usage:
    python experiments/train_concept_adapter.py --__TESTING__ True
    python experiments/train_concept_adapter.py --LEARNING_RATE 3e-4
"""
import os
import json

import numpy as np
import torch
import pytorch_lightning as pl
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_adapt.core import (
    DeFoGModel, CrossAttentionAdapter, AdapterModule, AdapterComposition, ConditionBranch,
    AdaptedSampler, Sampler, EMACallback,
)
from defog_megan.data import load_conditions, load_targets
from defog_megan.evaluate import decode_samples
from defog_megan.molecules import KEKULE_BONDS, NEUTRAL_KEK_ATOMS, build_encoders, smiles_to_pyg_data
from defog_megan.zoo import zoo_path

RDLogger.DisableLog("rdApp.*")

# ============================================================================
# Parameters
# ============================================================================

# :param CONDITIONS_PATH / TARGETS_PATH: the two precomputed MEGAN artifacts.
CONDITIONS_PATH: str = zoo_path("zinc-logp", "data/concept_conditions.npz")
TARGETS_PATH: str = zoo_path("zinc-logp", "data/concept_targets.npz")

# :param BASE_CKPT: the frozen unconditional base (the kekulized ZINC base).
BASE_CKPT: str = zoo_path("zinc-logp", "base.ckpt")

# The frozen base's node-class order -- must not be derived from data. A mismatch silently relabels
# sulfur as fluorine rather than raising, so these travel with BASE_CKPT (the zoo's ``base.vocabulary.json`` /
# ``base.schema.json`` record them). The Ames base has the three charged classes N+ O- N- after these nine.
ATOM_TYPES: list = list(NEUTRAL_KEK_ATOMS)
BOND_TYPES: list = list(KEKULE_BONDS)

# :param KEKULIZE: must be True for a vocabulary with no AROMATIC class. Set it
#       together with BOND_TYPES; see prepare_concept_dataset for why it matters.
KEKULIZE: bool = True

# --- Adapter architecture (mirrors the property/fingerprint adapters) ---
H_HIDDEN: int = 256
TIME_CONDITIONED: bool = True
STREAMS: list = ["X", "E", "y"]

# :param XATTN_TOKENS: node->condition cross-attention width (0 is the FiLM-only adapter).
#
#       FiLM computes each layer's {scale, shift, gate} from the condition ALONE and broadcasts one diagonal affine map to
#       every atom. Cross-attention gives the condition its own token bank and lets each atom form a query from its own
#       representation, so different kinds of atom can read different condition tokens and receive qualitatively
#       different corrections. For a motif-valued condition -- which is what phi is -- that is the mechanism the FiLM
#       path cannot express: "put this substructure HERE" rather than "shift everything". The output projection is
#       zero-initialized, so the adapter is still an exact no-op at init and the product-of-experts composition is
#       unaffected. It modulates the X stream only; the E stream still gets FiLM alone.
XATTN_TOKENS: int = 64
XATTN_DIM: int = 128
XATTN_HEADS: int = 16

# --- Training ---
EPOCHS: int = 50
BATCH_SIZE: int = 24              # matches the proven fingerprint-adapter recipe
LEARNING_RATE: float = 3e-4       # selected from the sweep 1e-4 .. 4e-4 (and, for Ames, on concepts outside the cells)
# No condition dropout: the adapter's gate head is zero-initialised, so the unconditional branch IS the frozen base.
MAX_TIME_HOURS: float = 8.0
EMA_DECAY: float = 0.999

# :param TARGET_MODE: 'empirical' (mean φ of the concept's members) or 'ideal'
#       (0.0 at the concept, 1.0 elsewhere). Both are in the npz.
TARGET_MODE: str = "empirical"

# --- Sampling ---
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
# Steering weights of the in-training evaluation (the paper's concept adapters are used at w = 1).
GUIDANCE_WEIGHTS: list = [1.0]
# :param EVAL_CONCEPTS: empty = the N best-supported concepts in the training
#       table (steering to a concept with 30 examples measures the data, not the
#       method).
EVAL_CONCEPTS: list = []
N_EVAL_CONCEPTS: int = 4
N_PER_CONCEPT: int = 64
N_BASELINE: int = 128
EVAL_CHUNK: int = 32
COMPOSE_MODE: str = "product"     # single branch: product == mean

# --- mid-training probe (MEGAN-free: validity/uniqueness only) ---
PROBE_EVERY_K: int = 5
PROBE_N: int = 32
PROBE_STEPS: int = 100
PROBE_WEIGHT: float = 1.0

TRAIN_LIMIT: int = 0              # 0 = all molecules (smoke runs only)
SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False
# no desktop notification when a run ends: on a machine without a desktop session it fails with a (harmless) traceback
__NOTIFY__: bool = False


class GenerationProbe(pl.Callback):
    """Every K epochs: sample under a target concept and report generation health.

    Deliberately MEGAN-free -- it cannot tell you whether the *concept* was hit,
    only whether the adapter is still producing valid, diverse molecules. That is
    the failure mode worth catching early (steering that destroys validity); the
    concept question is answered by the local evaluation afterwards.
    """

    def __init__(self, e, target, atom_decoder, bond_decoder):
        super().__init__()
        self.e, self.target = e, target
        self.ad, self.bd = atom_decoder, bond_decoder

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
                             self.e.PROBE_WEIGHT)],
            base=pl_module.base, mode=self.e.COMPOSE_MODE)
        sampler = AdaptedSampler(pl_module.base, composition, eta=self.e.ETA,
                                 omega=self.e.OMEGA, sample_steps=self.e.PROBE_STEPS,
                                 time_distortion=self.e.TIME_DISTORTION)
        samples, remaining = [], self.e.PROBE_N
        while remaining > 0:
            take = min(self.e.EVAL_CHUNK, remaining)
            samples += sampler.sample(take, device=pl_module.device, show_progress=False)
            remaining -= take
        _, stats = decode_samples(samples, self.ad, self.bd)
        self.e.log(f"[epoch {epoch}] PROBE(w={self.e.PROBE_WEIGHT}): "
                   f"validity={stats['validity']:.2f} uniqueness={stats['uniqueness']:.2f}")


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e.log(f"MEGAN-concept adapter on a frozen DeFoG base (device={device})")

    # -- the two precomputed MEGAN artifacts ---------------------------------
    smiles_list, phi, concept_ids = load_conditions(e.CONDITIONS_PATH)
    if e.TRAIN_LIMIT:
        smiles_list, phi = smiles_list[:e.TRAIN_LIMIT], phi[:e.TRAIN_LIMIT]
    e.log(f"conditions: {len(smiles_list)} molecules x {len(concept_ids)} concepts")

    target_matrix, target_ids = load_targets(e.TARGETS_PATH, mode=e.TARGET_MODE)
    assert target_ids == concept_ids, (
        "target vectors and conditions table disagree on the concept ordering -- "
        "the adapter would be steered with a permuted vector")
    e.log(f"targets: '{e.TARGET_MODE}' {target_matrix.shape}")

    support = (phi < 0.999).sum(axis=0)
    e["data/support"] = {cid: int(s) for cid, s in zip(concept_ids, support)}
    e.log(f"per-concept support: min={support.min()} "
          f"median={int(np.median(support))} max={support.max()}")

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(
        e.ATOM_TYPES, e.BOND_TYPES)

    graphs = []
    for smiles, row in zip(smiles_list, phi):
        data = smiles_to_pyg_data(smiles, atom_encoder, bond_encoder,
                                  kekulize=e.KEKULIZE)
        if data is None:
            continue
        data.cond = torch.tensor(row, dtype=torch.float).reshape(1, -1)   # RAW
        graphs.append(data)
    e.log(f"encoded {len(graphs)} training graphs")
    train_loader = DataLoader(graphs, batch_size=e.BATCH_SIZE, shuffle=True)

    # -- frozen base + adapter ----------------------------------------------
    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0, f"expected an unconditional base, got cond_dim={base.cond_dim}"
    adapter = CrossAttentionAdapter.for_base(
        base, cond_dim=len(concept_ids), hidden=e.H_HIDDEN,
        time_conditioned=e.TIME_CONDITIONED, streams=tuple(e.STREAMS),
        cond_mean=phi.mean(axis=0), cond_std=phi.std(axis=0),
        xattn_tokens=e.XATTN_TOKENS, xattn_dim=e.XATTN_DIM,
        xattn_heads=e.XATTN_HEADS,
        name="concept_adapter", cond_type="megan_concepts")
    # `for_base` supplies the library's default architecture for anything not passed; make sure what was built is
    # what was asked for, rather than a FiLM-only adapter whose run would look like a null result.
    built = int(getattr(adapter, "xattn_tokens", 0))
    assert built == int(e.XATTN_TOKENS), (
        f"asked for xattn_tokens={e.XATTN_TOKENS} but the adapter was built with {built}")
    e["adapter/xattn_tokens"] = built
    e["adapter/num_params"] = int(sum(p.numel() for p in adapter.parameters()))
    e.log(f"adapter: {e['adapter/num_params']:,} params "
          f"(base {sum(p.numel() for p in base.parameters()):,} frozen)")
    e.log(f"architecture: xattn_tokens={built} dim={getattr(adapter, 'xattn_dim', 0)} "
          f"heads={getattr(adapter, 'xattn_heads', 0)} "
          f"cond_fourier={getattr(adapter, 'cond_fourier', 0)}")

    if e.EVAL_CONCEPTS:
        eval_concepts = list(e.EVAL_CONCEPTS)
    else:
        eval_concepts = [concept_ids[i] for i in np.argsort(-support)[:e.N_EVAL_CONCEPTS]]
    e["eval/concepts"] = eval_concepts
    e.log("eval concepts: " + ", ".join(
        f"{c}(n={support[concept_ids.index(c)]})" for c in eval_concepts))
    targets = {c: target_matrix[concept_ids.index(c)] for c in eval_concepts}

    # -- train ---------------------------------------------------------------
    module = AdapterModule(base, adapter, cond_attr="cond", lr=e.LEARNING_RATE)
    probe = GenerationProbe(e, targets[eval_concepts[0]], atom_decoder, bond_decoder)
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS}, accelerator="auto",
        devices=1, enable_progress_bar=False, enable_checkpointing=False, logger=False,
        gradient_clip_val=1.0, callbacks=[probe, EMACallback(decay=e.EMA_DECAY)])
    e.log(f"training: epochs<={e.EPOCHS} batch={e.BATCH_SIZE} lr={e.LEARNING_RATE} "
          f"max_time={e.MAX_TIME_HOURS}h")
    trainer.fit(module, train_dataloaders=train_loader)

    checkpoint = adapter.save(os.path.join(e.path, "concept_adapter"))
    e.log(f"saved adapter -> {checkpoint}")

    # -- sample under each concept, dump SMILES for scoring -------------------
    e.log("=" * 60)
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
    baseline_smiles, baseline_stats = decode_samples(
        sample_with(baseline_sampler, e.N_BASELINE), atom_decoder, bond_decoder)
    e.log(f"  validity={baseline_stats['validity']:.2f} "
          f"uniqueness={baseline_stats['uniqueness']:.2f}")

    dump = {
        "target_mode": e.TARGET_MODE,
        "learning_rate": e.LEARNING_RATE,
        "sampling": {"steps": e.EVAL_STEPS, "eta": e.ETA, "omega": e.OMEGA,
                     "time_distortion": e.TIME_DISTORTION},
        "concept_ids": concept_ids,
        "baseline": {"smiles": baseline_smiles, "stats": baseline_stats},
        "per_concept": {},
    }
    for concept_id in eval_concepts:
        dump["per_concept"][concept_id] = {"per_w": {}}
        for weight in e.GUIDANCE_WEIGHTS:
            composition = AdapterComposition(
                [ConditionBranch(adapter,
                                 torch.tensor(targets[concept_id], dtype=torch.float32),
                                 weight)],
                base=base, mode=e.COMPOSE_MODE)
            sampler = AdaptedSampler(base, composition, eta=e.ETA, omega=e.OMEGA,
                                     sample_steps=e.EVAL_STEPS,
                                     time_distortion=e.TIME_DISTORTION)
            smiles, stats = decode_samples(
                sample_with(sampler, e.N_PER_CONCEPT), atom_decoder, bond_decoder)
            dump["per_concept"][concept_id]["per_w"][str(weight)] = {
                "smiles": smiles, "stats": stats}
            e.log(f"  {concept_id} w={weight}: validity={stats['validity']:.2f} "
                  f"uniqueness={stats['uniqueness']:.2f} n_valid={stats['n_valid']}")

    e.commit_json("generated_samples.json", dump)
    e.log("wrote generated_samples.json -- score the SMILES with the explainer (see defog_megan.evaluate)")
    e.log("Done.")


@experiment.testing
def testing(e: Experiment) -> None:
    e.EPOCHS = 2
    e.TRAIN_LIMIT = 400
    e.BATCH_SIZE = 16
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
    e.GUIDANCE_WEIGHTS = [1.0]


experiment.run_if_main()
