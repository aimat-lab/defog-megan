"""
Experiments 3 and 4 (ZINC rung): logP and concept steering on molecules, alone and jointly, with and without Feynman-Kac.

The molecular rung of the paper (logP of the DeFoG-decoded ZINC molecules; a computer-science experiment in which molecules
are one class of graphs). The structure is the colored-graph experiment's (``sample_color_joint.py``) carried over: the
property is exact (the Crippen logP of the decoded molecule, computed by RDKit), the concepts come from the MEGAN trained on
the same decoded molecules, and the cells are (concept, logP level) pairs classed by the pool.

STAGES

    single    Experiment 3 -- each adapter ALONE.
              uncond          frozen base                                  once
              property        property adapter alone, w in P_WEIGHTS       per level
              concept         concept adapter alone,  w in C_WEIGHTS       per concept (all 42)
    probe     Experiment 4 -- both adapters, one row per WEIGHT_GRID entry per cell.
    fk        joint+fk and fk_only at FK_WEIGHTS. MEGAN is the energy of both arms.

The property arm of ``single`` at W_PROPERTY_SOLO doubles as arm B (property + rejection filter) in Experiment 4, so it is
drawn large.

SIZES come from the base's OWN distribution for every arm: a target-conditional size model would leak part of the logP
target into the sampler (bigger molecules have higher logP), so no arm gets one.

SAMPLING is at eta 500, 500 steps, omega 0, polydec, with probability-space blending and the plain sum of the two adapters'
log-weights (``COMPOSE_MODE = "product"``). The defaults are the settings of the paper's runs; the seeds and the sharding of
the paper are in ``reproduce/``.

K IS THE BATCH for FK: particles compete inside one ``sample()`` call, so FK cells are drawn in chunks of exactly PARTICLES
and N_PER_CELL must be a multiple of it.

Writes ``gen_<unit>.json`` (SMILES per generated graph, ``null`` where the graph did not decode, so validity is counted over
everything generated) plus ``manifest.json``. Scoring is a separate step (``scripts/score_mol_joint.py``). Shards of one
stage can run on several GPUs or machines: every shard derives the same deterministic partition of the work units.

Usage:
    python experiments/sample_mol_joint.py --__TESTING__ True
    python experiments/sample_mol_joint.py --STAGE '"single"' --SHARD 0 --NUM_SHARDS 4
"""
import json
import os
import time

import torch
from rdkit import Chem, RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_adapt.core import (
    DeFoGModel, CrossAttentionAdapter, AdapterComposition, ConditionBranch, AdaptedSampler,
    Sampler, FeynmanKacSampler,
)
from defog_megan.bundle import load_space, reference_check
from defog_megan.data import load_targets
from defog_megan.molecules import (
    KEKULE_BONDS, NEUTRAL_KEK_ATOMS, build_encoders, make_domain, mol_to_smiles, pyg_data_to_mol,
)
from defog_megan.zoo import zoo_path

RDLogger.DisableLog("rdApp.*")

# ============================================================================
BASE_CKPT: str = zoo_path("zinc-logp", "base.ckpt")
PROPERTY_ADAPTER: str = zoo_path("zinc-logp", "property_adapter.ckpt")
CONCEPT_ADAPTER: str = zoo_path("zinc-logp", "concept_adapter.ckpt")
TARGETS_PATH: str = zoo_path("zinc-logp", "data/concept_targets.npz")
TARGET_MODE: str = "empirical"
ATOM_TYPES: list = list(NEUTRAL_KEK_ATOMS)
BOND_TYPES: list = list(KEKULE_BONDS)

# The six concepts and five logP levels of the cells (three concepts per channel, classed native / off-native by the pool).
CONCEPTS: list = ["ch0_cl6", "ch0_cl7", "ch0_cl14", "ch1_cl22", "ch1_cl29", "ch1_cl34"]
LEVELS: list = [0.5, 1.5, 2.5, 3.5, 4.5]

# MEGAN, the FK energy. Read only in the fk stage; the scoring settings come from the bundle's own concept table, and the
# explainer is checked against a reference table before it steers anything (GATE_*, 0 switches the gate off).
GATE_N: int = 64
GATE_REFERENCE: str = zoo_path("zinc-logp", "data/reference_check.npz")
GATE_TOLERANCE: float = 1e-5

# :param STAGE: "single", "probe" or "fk" -- selects the arms, see the module docstring.
STAGE: str = "single"

# --- single stage ---
# :param SINGLE_ARMS: which arms of the single stage to draw ("uncond", "property", "concept"). Without "concept"
#       the concept adapter is not even loaded.
SINGLE_ARMS: list = ["uncond", "property", "concept"]
W_PROPERTY_SOLO: float = 2.0      # the logP adapter's measured optimum in probability space; the property arm doubles as arm B
P_WEIGHTS: list = [1.0, 2.0, 3.0]
C_WEIGHTS: list = [1.0, 2.0]
#: concepts the single stage covers; empty means the whole vocabulary (all 42)
SINGLE_CONCEPTS: list = []

# --- probe stage ---
# :param WEIGHT_GRID: (w_concept, w_property) pairs for the joint arm.
WEIGHT_GRID: list = [[1.0, 1.0]]
FK_WEIGHTS: list = [1.0, 1.0]      # (w_concept, w_property) for the fk stage
COMPOSE_MODE: str = "product"

# --- Feynman-Kac (fk stage) ---
FK_BETA: float = 16.0
FK_LAM: float = 1.0
FK_SCALE: float = 1.16             # logP label sd: a miss of this size costs ~1
FK_ENERGY_MODE: str = "sum"
FK_MACRO_WEIGHT: float = 0.5
PARTICLES: int = 16
FK_WARMUP_FRAC: float = 0.7
FK_RESAMPLE_INTERVAL: int = 0      # 0 -> the sampler's default (steps // 8)
FK_ESS_FRAC: float = 0.0           # >0: resample only when the effective sample size < frac * K
# The paper's runs resampled multinomially (the library default is "residual").
FK_RESAMPLE_SCHEME: str = "multinomial"

N_UNCOND: int = 1024
N_PROPERTY: int = 1024             # per level and weight ...
N_PROPERTY_MAIN: int = 1024        # ... except at W_PROPERTY_SOLO (it doubles as arm B)
N_CONCEPT: int = 1024
N_PER_CELL: int = 1024
EVAL_CHUNK: int = 64
EVAL_STEPS: int = 500
ETA: float = 500.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"

SHARD: int = 0
NUM_SHARDS: int = 1
SEED: int = 42
# False: every run gets its own time-stamped folder (True writes to one `debug/` folder, which the next run wipes).
__DEBUG__: bool = False
__TESTING__: bool = False


def _lv(level: float) -> str:
    return f"lv{level:+.1f}"


def work_units(e, target_ids) -> list:
    """Every unit of sampling this stage does, with a cost proxy for sharding.

    The cost is samples x branches evaluated per step; the FK energy's MEGAN calls are
    CPU-side and a handful per trajectory.
    """
    units = []
    if e.STAGE == "single":
        if "uncond" in e.SINGLE_ARMS:
            units.append({"arm": "uncond", "n": e.N_UNCOND, "cost": e.N_UNCOND})
        for w in (e.P_WEIGHTS if "property" in e.SINGLE_ARMS else []):
            n = e.N_PROPERTY_MAIN if float(w) == float(e.W_PROPERTY_SOLO) else e.N_PROPERTY
            for level in e.LEVELS:
                units.append({"arm": "property", "level": float(level), "n": n,
                              "w": [None, float(w)], "cost": n * 2,
                              "key": f"property_w{w:g}_{_lv(level)}"})
        concepts = list(e.SINGLE_CONCEPTS) or list(target_ids)
        for w in (e.C_WEIGHTS if "concept" in e.SINGLE_ARMS else []):
            for concept in concepts:
                units.append({"arm": "concept", "concept": concept, "n": e.N_CONCEPT,
                              "w": [float(w), None], "cost": e.N_CONCEPT * 2,
                              "key": f"concept_{concept}_w{w:g}"})
    elif e.STAGE == "probe":
        for concept in e.CONCEPTS:
            for level in e.LEVELS:
                for wc, wp in e.WEIGHT_GRID:
                    units.append({"arm": "joint", "concept": concept, "level": float(level),
                                  "n": e.N_PER_CELL, "w": [float(wc), float(wp)],
                                  "cost": e.N_PER_CELL * 3,
                                  "key": f"joint_{concept}_{_lv(level)}_wc{wc:g}_wp{wp:g}"})
    elif e.STAGE == "fk":
        wc, wp = (float(x) for x in e.FK_WEIGHTS)
        for concept in e.CONCEPTS:
            for level in e.LEVELS:
                units.append({"arm": "joint+fk", "concept": concept, "level": float(level),
                              "n": e.N_PER_CELL, "w": [wc, wp], "cost": e.N_PER_CELL * 3,
                              "key": f"joint_fk_{concept}_{_lv(level)}_wc{wc:g}_wp{wp:g}"})
                units.append({"arm": "fk_only", "concept": concept, "level": float(level),
                              "n": e.N_PER_CELL, "w": [None, None], "cost": e.N_PER_CELL,
                              "key": f"fk_only_{concept}_{_lv(level)}"})
    else:
        raise ValueError(f"unknown STAGE {e.STAGE!r}")
    for u in units:
        u.setdefault("key", u["arm"])
    return units


def assign_shards(units: list, num_shards: int) -> list:
    """Longest-processing-time greedy assignment. Deterministic, so every shard derives
    the same partition independently and no unit is drawn twice or dropped."""
    loads = [0] * num_shards
    owner = {}
    for u in sorted(units, key=lambda u: (-u["cost"], u["key"])):
        s = loads.index(min(loads))
        owner[u["key"]] = s
        loads[s] += u["cost"]
    return [owner[u["key"]] for u in units]


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    if e.__TESTING__:
        e.CONCEPTS = e.CONCEPTS[:1]
        e.LEVELS = e.LEVELS[:2]
        e.WEIGHT_GRID = e.WEIGHT_GRID[:1]
        e.P_WEIGHTS = e.P_WEIGHTS[:1]
        e.C_WEIGHTS = e.C_WEIGHTS[:1]
        e.SINGLE_CONCEPTS = e.CONCEPTS[:2] if e.CONCEPTS else []
        e.N_UNCOND = e.N_PROPERTY = e.N_PROPERTY_MAIN = e.N_CONCEPT = e.N_PER_CELL = 8
        e.PARTICLES = 4
        e.EVAL_CHUNK = 8
        e.EVAL_STEPS = 20

    torch.manual_seed(e.SEED + e.SHARD)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e.log(f"stage {e.STAGE} | shard {e.SHARD}/{e.NUM_SHARDS} | device {device}")

    if e.STAGE == "fk" and e.N_PER_CELL % e.PARTICLES:
        raise ValueError(f"N_PER_CELL={e.N_PER_CELL} is not a multiple of PARTICLES="
                         f"{e.PARTICLES}; a short final batch would run a different FK")

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(
        e.ATOM_TYPES, e.BOND_TYPES)
    target_matrix, target_ids = load_targets(e.TARGETS_PATH, mode=e.TARGET_MODE)
    target_ids = list(target_ids)
    missing = [c for c in list(e.CONCEPTS) + list(e.SINGLE_CONCEPTS) if c not in target_ids]
    if missing:
        raise ValueError(f"concepts not in the vocabulary: {missing}")

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    if base.num_node_classes != len(e.ATOM_TYPES):
        raise ValueError(f"base has {base.num_node_classes} node classes, ATOM_TYPES has "
                         f"{len(e.ATOM_TYPES)} -- wrong base")
    prop_adapter = CrossAttentionAdapter.load(e.PROPERTY_ADAPTER, device="cpu").to(device).eval()
    need_concept = e.STAGE != "single" or "concept" in e.SINGLE_ARMS
    concept_adapter = (CrossAttentionAdapter.load(e.CONCEPT_ADAPTER, device="cpu").to(device).eval()
                       if need_concept else None)
    for name, adapter, dim in ((("property", prop_adapter, 1),)
                               + ((("concept", concept_adapter, len(target_ids)),) if need_concept else ())):
        # A FiLM-only reload would silently sweep a different model than the one trained.
        if getattr(adapter, "xattn_tokens", 0) == 0:
            raise ValueError(f"{name} adapter reloaded WITHOUT cross-attention -- wrong adapter file?")
        if adapter.cond_dim != dim:
            raise ValueError(f"{name} adapter cond_dim={adapter.cond_dim}, expected {dim}")
        adapter.check_compatible(base)
        e.log(f"{name} adapter: {sum(p.numel() for p in adapter.parameters()):,} params, "
              f"xattn={adapter.xattn_tokens}")

    domain = make_domain(atom_decoder, bond_decoder)
    space = None
    if e.STAGE == "fk":
        from defog_megan.mol_joint_energy import MolJointEnergy
        e.log("loading MEGAN (the FK energy) ...")
        space = load_space("zinc-logp")
        if list(space.concept_ids) != target_ids:
            raise ValueError("the FK energy's clustering orders concepts differently from "
                             "the target file -- FK would drive a different concept than "
                             "the adapter is steered to")
        if e.GATE_N:
            deviation = reference_check(space, "zinc-logp", n=e.GATE_N, tolerance=e.GATE_TOLERANCE,
                                        reference=e.GATE_REFERENCE)
            e.log(f"explainer gate: phi vs the reference table on {e.GATE_N} molecules, max |diff| = {deviation:.3e}")

    units = work_units(e, target_ids)
    owners = assign_shards(units, e.NUM_SHARDS)
    mine = [u for u, s in zip(units, owners) if s == e.SHARD]
    e.log(f"{len(units)} units in this stage; this shard draws {len(mine)} "
          f"({sum(u['n'] for u in mine)} graphs)")

    def branches(unit, n):
        out = []
        wc, wp = unit["w"]
        if wc is not None:
            cvec = torch.tensor(target_matrix[target_ids.index(unit["concept"])],
                                dtype=torch.float32, device=device)
            out.append(ConditionBranch(concept_adapter, cvec.expand(n, -1), float(wc)))
        if wp is not None:
            pvec = torch.tensor([[float(unit["level"])]], dtype=torch.float32, device=device)
            out.append(ConditionBranch(prop_adapter, pvec.expand(n, -1), float(wp)))
        return out

    def decode(samples):
        """One SMILES per generated graph; ``None`` where it does not decode, so validity
        is counted over everything generated."""
        out = []
        for sample in samples:
            mol = pyg_data_to_mol(sample, atom_decoder, bond_decoder)
            smi = mol_to_smiles(mol) if mol is not None else None
            out.append(smi if smi is not None and Chem.MolFromSmiles(smi) is not None else None)
        return out

    def draw(unit):
        fk = unit["arm"] in ("joint+fk", "fk_only")
        chunk = e.PARTICLES if fk else e.EVAL_CHUNK
        samples, energy_calls = [], 0
        while len(samples) < unit["n"]:
            cur = chunk if fk else min(chunk, unit["n"] - len(samples))
            br = [] if unit["arm"] == "uncond" else branches(unit, cur)
            composition = (AdapterComposition(br, base=base, mode=e.COMPOSE_MODE) if br else None)
            if fk:
                energy = MolJointEnergy(
                    space, domain, target_ids.index(unit["concept"]),
                    target_logp=unit["level"], lam=e.FK_LAM, scale=e.FK_SCALE,
                    mode=e.FK_ENERGY_MODE, macro_weight=e.FK_MACRO_WEIGHT)
                sampler = FeynmanKacSampler(
                    base, energy_fn=energy, composition=composition, beta=e.FK_BETA,
                    warmup_frac=e.FK_WARMUP_FRAC,
                    resample_interval=(e.FK_RESAMPLE_INTERVAL or None),
                    ess_frac=(e.FK_ESS_FRAC or None),
                    eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                    time_distortion=e.TIME_DISTORTION, resample_scheme=e.FK_RESAMPLE_SCHEME)
            elif composition is not None:
                sampler = AdaptedSampler(base, composition, eta=e.ETA, omega=e.OMEGA,
                                         sample_steps=e.EVAL_STEPS,
                                         time_distortion=e.TIME_DISTORTION)
            else:
                sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                                  time_distortion=e.TIME_DISTORTION)
            samples += sampler.sample(num_samples=cur, device=device, show_progress=False)
            if fk:
                energy_calls += energy.n_calls
        return decode(samples), energy_calls

    manifest = {
        "stage": e.STAGE, "shard": e.SHARD, "num_shards": e.NUM_SHARDS,
        "concepts": list(e.CONCEPTS), "levels": [float(x) for x in e.LEVELS],
        "concept_ids": target_ids, "target_mode": e.TARGET_MODE,
        "compose_mode": e.COMPOSE_MODE, "blend_space": "prob",
        "weights": {"property_solo": e.W_PROPERTY_SOLO, "p_weights": list(e.P_WEIGHTS),
                    "c_weights": list(e.C_WEIGHTS)},
        "weight_grid": [[float(a), float(b)] for a, b in e.WEIGHT_GRID],
        "fk": {"weights": [float(x) for x in e.FK_WEIGHTS], "beta": e.FK_BETA,
               "lam": e.FK_LAM, "scale": e.FK_SCALE, "energy_mode": e.FK_ENERGY_MODE,
               "particles": e.PARTICLES, "warmup_frac": e.FK_WARMUP_FRAC,
               "ess_frac": e.FK_ESS_FRAC, "resample_scheme": e.FK_RESAMPLE_SCHEME,
               "macro_weight": e.FK_MACRO_WEIGHT},
        "sampling": {"steps": e.EVAL_STEPS, "eta": e.ETA, "omega": e.OMEGA,
                     "time_distortion": e.TIME_DISTORTION,
                     "sizes": "base marginal"},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "units": [],
    }
    for unit in mine:
        t0 = time.perf_counter()
        smiles, energy_calls = draw(unit)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        fname = f"gen_{unit['key']}.json"
        with open(os.path.join(e.path, fname), "w") as f:
            json.dump({"key": unit["key"], "smiles": smiles}, f)
        record = {k: v for k, v in unit.items() if k != "cost"}
        record.update(file=fname, n=len(smiles), seconds=seconds,
                      seconds_per_sample=seconds / max(1, len(smiles)),
                      energy_calls=energy_calls or None)
        manifest["units"].append(record)
        valid = sum(s is not None for s in smiles) / max(1, len(smiles))
        e.log(f"  {unit['key']:<44} n={len(smiles):>5}  valid {valid:.2f}  {seconds:7.1f}s "
              f"({record['seconds_per_sample']:.3f} s/graph)")

    e.commit_json("manifest.json", manifest)
    e.log("Done.")


experiment.run_if_main()
