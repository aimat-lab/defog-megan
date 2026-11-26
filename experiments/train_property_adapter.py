"""
Train a scalar property adapter on a frozen DeFoG base (here: the mutagenicity class of the Ames set).

The scalar counterpart to ``train_concept_adapter.py``. Same frozen base, same molecules, same vocabulary -- the condition
is a single number (the class, 0 or 1) instead of a 29-dim MEGAN membership vector, so the difference between the two
adapters is the CONDITION and not the data or the generator.

The architecture is DeFoG's measured property recipe, whose three training defaults only reproduce its accuracy together:

* ``COND_FOURIER = 3``  -- without it the trunk sees the property as ONE raw float while the flow-time gets a 64-dim
  sinusoidal embedding, and the learned map converges to the smooth part of the target long before the part that
  distinguishes close values. Three bands is the ceiling that still keeps neighbouring targets correlated enough to
  interpolate between.
* ``XATTN_TOKENS = 64`` -- each atom queries the condition rather than every atom receiving the same broadcast FiLM
  correction. This is the half that carries the effect.
* ``LEARNING_RATE = 4e-4`` and ``EPOCHS = 80`` -- 20 epochs is badly under-trained for this architecture.

WHAT IS DIFFERENT HERE, and why. The class is not RDKit-computable: logP, TPSA and QED have exact oracles, so a generation
error there is a pure measure of steering, while the only thing that can score a *generated* molecule for mutagenicity is
the MEGAN model. Every number built on it is model-graded, and says so.

This module deliberately **does not import MEGAN** -- the label travels as data (``label_conditions.npz``, built by
``scripts/prepare_ames_defog_label.py``) and scoring is a separate step. The molecules are not part of the zoo.

Usage:
    python experiments/train_property_adapter.py --__TESTING__ True
    python experiments/train_property_adapter.py --LEARNING_RATE 3e-4
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
    AdaptedSampler, EMACallback,
)
from defog_megan.evaluate import decode_samples
from defog_megan.molecules import (
    CHARGED_KEK_V1_ATOMS, KEKULE_BONDS, build_encoders, smiles_to_pyg_data,
)
from defog_megan.sizes import (
    DatasetSizeDistribution, dataset_size_histogram, base_size_histogram)
from defog_megan.zoo import zoo_path

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# Parameters
# ============================================================================

# :param CONDITIONS_PATH: smiles + the property value, built by scripts/prepare_ames_defog_label.py.
# :param VALUE_KEY: the array of that table that holds the value (tables of the research code call it "logs").
CONDITIONS_PATH: str = zoo_path("ames", "data/label_conditions.npz")
VALUE_KEY: str = "value"
# :param SPLIT_PATH: the FROZEN molecule-level split. Training uses `train` only;
#       `dev_targets` and `report_targets` are held out and never trained on.
SPLIT_PATH: str = zoo_path("ames", "data/label_split.json")

BASE_CKPT: str = zoo_path("ames", "base.ckpt")

# Must match the frozen base's node-class order (the zoo's ``base.vocabulary.json`` records it): a mismatch silently
# relabels sulfur as fluorine rather than raising. The charged Ames base has the three classes N+ O- N- after the nine.
ATOM_TYPES: list = list(CHARGED_KEK_V1_ATOMS)
BOND_TYPES: list = list(KEKULE_BONDS)
KEKULIZE: bool = True

# --- Adapter architecture: DeFoG's measured property recipe ---
H_HIDDEN: int = 256
TIME_CONDITIONED: bool = True
STREAMS: list = ["X", "E", "y"]
COND_FOURIER: int = 3
XATTN_TOKENS: int = 64
XATTN_DIM: int = 128
XATTN_HEADS: int = 16

# --- Training ---
EPOCHS: int = 80
BATCH_SIZE: int = 24
LEARNING_RATE: float = 4e-4
# No condition dropout: the adapter's gate head is zero-initialised, so the unconditional branch IS the frozen base.
MAX_TIME_HOURS: float = 10.0
EMA_DECAY: float = 0.999

# --- Sampling / probe ---
# The probe weight is 2.0 because that is where a probability-space adapter ships: probing at 1.0 while shipping at 2.0
# would compare an adapter at one operating point against a model at another.
PROBE_WEIGHT: float = 2.0
EVAL_STEPS: int = 250
ETA: float = 200.0
OMEGA: float = 0.3
TIME_DISTORTION: str = "polydec"
COMPOSE_MODE: str = "product"
EVAL_CHUNK: int = 32

PROBE_EVERY_K: int = 5
PROBE_N: int = 32
PROBE_STEPS: int = 100

# :param SIZE_FROM_DATASET: draw generated graph sizes from the training set's size marginal
#       rather than the base's. Deliberately NOT conditional on the target: a target-conditional
#       size would satisfy much of an extreme request by making the graph big, and the error
#       would credit the adapter for the sampler's work.
SIZE_FROM_DATASET: bool = True

TRAIN_LIMIT: int = 0              # 0 = all molecules (smoke runs only)
SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False
# no desktop notification when a run ends: on a machine without a desktop session it fails with a (harmless) traceback
__NOTIFY__: bool = False


class GenerationProbe(pl.Callback):
    """Every K epochs: generate at a fixed target and report generation health.

    MEGAN-free by construction -- it cannot say whether the property was hit, only whether
    the adapter still produces valid, diverse molecules. That is the failure worth
    catching early; the property question is answered by the scoring afterwards.
    """

    def __init__(self, e, target, atom_decoder, bond_decoder, size_dist):
        super().__init__()
        self.e, self.target = e, target
        self.ad, self.bd = atom_decoder, bond_decoder
        self.size_dist = size_dist

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
        condition = torch.tensor([[self.target]], dtype=torch.float32)
        composition = AdapterComposition(
            [ConditionBranch(pl_module.adapter, condition.reshape(-1),
                             self.e.PROBE_WEIGHT)],
            base=pl_module.base, mode=self.e.COMPOSE_MODE)
        sampler = AdaptedSampler(pl_module.base, composition, eta=self.e.ETA,
                                 omega=self.e.OMEGA, sample_steps=self.e.PROBE_STEPS,
                                 time_distortion=self.e.TIME_DISTORTION)
        samples, remaining = [], self.e.PROBE_N
        while remaining > 0:
            take = min(self.e.EVAL_CHUNK, remaining)
            kwargs = {"size_dist": self.size_dist} if self.size_dist is not None else {}
            samples += sampler.sample(take, device=pl_module.device,
                                      show_progress=False, **kwargs)
            remaining -= take
        _, stats = decode_samples(samples, self.ad, self.bd)
        self.e.log(f"[epoch {epoch}] PROBE(value={self.target:.2f}, w={self.e.PROBE_WEIGHT}): "
                   f"validity={stats['validity']:.2f} uniqueness={stats['uniqueness']:.2f}")


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e.log(f"property adapter on a frozen DeFoG base (device={device})")

    table = np.load(e.CONDITIONS_PATH, allow_pickle=True)
    smiles_all = table["smiles"].tolist()
    values_all = np.asarray(table[e.VALUE_KEY], dtype=np.float32)
    e.log(f"conditions: {len(smiles_all)} molecules with a label")

    with open(e.SPLIT_PATH) as file:
        split = json.load(file)
    train_idx = list(split["train"])
    # A held-out molecule leaking into training would make the report targets
    # memorised rather than generalised, and nothing downstream would notice.
    held = {t["index"] for t in split["dev_targets"]} | {
        t["index"] for t in split["report_targets"]}
    overlap = held & set(train_idx)
    assert not overlap, f"held-out targets present in the training rows: {sorted(overlap)[:5]}"
    if e.__TESTING__ or e.TRAIN_LIMIT:
        train_idx = train_idx[:e.TRAIN_LIMIT] if e.TRAIN_LIMIT else train_idx
    e.log(f"split: {len(train_idx)} train rows, {len(held)} held-out targets "
          f"(frozen {split['created']}, seed {split['seed']})")

    train_smiles = [smiles_all[i] for i in train_idx]
    train_values = values_all[train_idx]
    e.log(f"train values: mean {train_values.mean():.3f} sd {train_values.std():.3f} "
          f"range [{train_values.min():.2f}, {train_values.max():.2f}]")

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(
        e.ATOM_TYPES, e.BOND_TYPES)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0, f"expected an unconditional base, got cond_dim={base.cond_dim}"

    graphs = []
    for smiles, value in zip(train_smiles, train_values):
        data = smiles_to_pyg_data(smiles, atom_encoder, bond_encoder,
                                  kekulize=e.KEKULIZE)
        if data is None:
            continue
        data.cond = torch.tensor([[float(value)]], dtype=torch.float)   # RAW
        graphs.append(data)
    e.log(f"encoded {len(graphs)} training graphs")
    assert graphs, "no training graphs encoded -- check ATOM_TYPES against the base"
    train_loader = DataLoader(graphs, batch_size=e.BATCH_SIZE, shuffle=True)

    # -- the adapter ---------------------------------------------------------
    # cond_dim is 1, which is what makes cond_fourier legal at all: the adapter refuses Fourier bands above 8
    # dims, so the concept adapter cannot have them and this one can. That is the intended asymmetry.
    adapter = CrossAttentionAdapter.for_base(
        base, cond_dim=1, hidden=e.H_HIDDEN,
        time_conditioned=e.TIME_CONDITIONED, streams=tuple(e.STREAMS),
        cond_mean=np.array([train_values.mean()]), cond_std=np.array([train_values.std()]),
        cond_fourier=e.COND_FOURIER,
        xattn_tokens=e.XATTN_TOKENS, xattn_dim=e.XATTN_DIM, xattn_heads=e.XATTN_HEADS,
        name="property_adapter", cond_type="property")
    built_x = int(getattr(adapter, "xattn_tokens", 0))
    built_f = int(getattr(adapter, "cond_fourier", 0))
    # `for_base` fills in the library's defaults for anything not passed; make sure what was built is what was asked
    # for, rather than a FiLM-only adapter whose run would look like a weak result.
    assert built_x == int(e.XATTN_TOKENS), (
        f"asked for xattn_tokens={e.XATTN_TOKENS}, got {built_x}")
    assert built_f == int(e.COND_FOURIER), (
        f"asked for cond_fourier={e.COND_FOURIER}, got {built_f}")
    e["adapter/xattn_tokens"] = built_x
    e["adapter/cond_fourier"] = built_f
    e["adapter/num_params"] = int(sum(p.numel() for p in adapter.parameters()))
    e["cond/mean"] = float(train_values.mean())
    e["cond/std"] = float(train_values.std())
    e.log(f"adapter: {e['adapter/num_params']:,} params "
          f"(base {sum(p.numel() for p in base.parameters()):,} frozen)")
    e.log(f"architecture: xattn_tokens={built_x} dim={getattr(adapter, 'xattn_dim', 0)} "
          f"heads={getattr(adapter, 'xattn_heads', 0)} cond_fourier={built_f}")

    # -- size distribution ---------------------------------------------------
    size_dist = None
    if e.SIZE_FROM_DATASET:
        base_hist = base_size_histogram(base, max_nodes=base.max_nodes)
        base_min = int(base_hist[0].min()) if base_hist is not None else 1
        hist = dataset_size_histogram(train_smiles)
        size_dist = DatasetSizeDistribution(hist, max_nodes=base.max_nodes,
                                            min_nodes=base_min)
        mean_size = float((hist[0] * hist[1]).sum() / hist[1].sum())
        e.log(f"size distribution: training-set marginal, mean {mean_size:.1f} atoms, "
              f"clipped to the base's [{base_min}, {base.max_nodes}]")
        e["sizes/mean_atoms"] = mean_size

    # -- train ---------------------------------------------------------------
    module = AdapterModule(base, adapter, cond_attr="cond", lr=e.LEARNING_RATE)
    probe_target = float(np.percentile(train_values, 10))   # a low-end target
    callbacks = [GenerationProbe(e, probe_target, atom_decoder, bond_decoder, size_dist)]
    if e.EMA_DECAY:
        callbacks.append(EMACallback(decay=e.EMA_DECAY))

    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS}, accelerator="auto",
        devices=1, enable_progress_bar=False, enable_checkpointing=False, logger=False,
        gradient_clip_val=1.0, callbacks=callbacks)
    e.log(f"training: epochs<={e.EPOCHS} batch={e.BATCH_SIZE} lr={e.LEARNING_RATE} "
          f"max_time={e.MAX_TIME_HOURS}h")
    trainer.fit(module, train_loader)

    out = os.path.join(e.path, "property_adapter.ckpt")
    adapter.save(out)
    e.log(f"saved adapter -> {out}")
    e.log("Done.")


@experiment.testing
def testing(e: Experiment) -> None:
    e.EPOCHS = 2
    e.TRAIN_LIMIT = 400
    e.BATCH_SIZE = 16
    e.MAX_TIME_HOURS = 0.2
    e.H_HIDDEN = 32
    e.PROBE_STEPS = 5
    e.PROBE_EVERY_K = 1
    e.PROBE_N = 6
    e.EVAL_CHUNK = 6


experiment.run_if_main()
