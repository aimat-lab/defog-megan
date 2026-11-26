"""
Experiment 2: joint property x concept steering on colour graphs, with and without FK.

The manuscript's second experiment. Experiment 1 showed each adapter reaching its own
condition alone; this asks for BOTH at once -- a requested concept c AND a requested exact
motif sum y* -- which is the paper's central claim on the one rung with exact ground truth.

WHY THE CELLS ARE (concept, absolute level) AND NOT OFFSETS. On the molecular rung logS
targets were offsets from each concept's native mean because those means spanned seven
log units. Here the property is an integer in [-3, +3] and the pool is flat over it
(14,285 graphs per level), so an absolute grid is comparable across concepts as it is.
Whether a cell is native, off-native or unsupported is then a property of the POOL, which
the scorer reads off it rather than this module asserting it:

    ch1_cl11 at +3   4,056 real core members   -- native
    ch1_cl12 at -1     933                     -- off-native, satisfiable
    ch0_cl4  at +3       0                     -- unsupported: the negative control

The six concepts are the four the single-concept runs graded (ch0_cl3, ch0_cl4, ch1_cl11,
ch1_cl12), so each has known solo numbers, plus two whose native property runs AGAINST
their channel (ch0_cl7 at +0.88, ch1_cl16 at -1.11) -- the cases where Experiment 1's
channel-suppression effect and the requested property pull in different directions.

ARMS. Every arm draws at Experiment 1's sampling settings (500 steps, eta=0, omega=0,
polydec), so the single-adapter arms reproduce its operating points and every number here
is paired with them:

    uncond         frozen base                                  once
    property       property adapter alone, w=2 (Table 1 row)    once per LEVEL
    concept        concept adapter alone,  w=1 (Experiment 1)   once per CONCEPT
    joint          both adapters, one row per WEIGHT_GRID entry per cell
    joint+fk       both adapters + FK on ColorJointEnergy       per cell
    fk_only        frozen base + FK on ColorJointEnergy         per cell  (was training needed?)

The property arm is drawn large (N_PROPERTY) because it doubles as ARM B, property +
rejection filter: "samples per accepted hit" is 1 / P(concept core AND property match)
under property-only sampling, and for the rarer concepts that rate is small enough to need
the sample size. Arm B is pure post-processing and lives in the scorer.

COMPOSITION. ``mode="product"`` throughout, so a WEIGHT_GRID entry (w_c, w_p) is literally
the two adapters' guidance weights: (1, 2) stacks both Experiment 1 operating points.
``"mean"`` only rescales the same family by 1/N, so it adds no new points.

TWO STAGES, because the FK arms need an operating point and that is what the probe finds:

    probe   uncond, property, concept, joint over WEIGHT_GRID      no MEGAN needed
    fk      joint+fk, fk_only at the chosen (w_c, w_p)            MEGAN is the FK energy

Plus two for the Pareto figure (property match vs concept match), which add points to the
probe/fk results rather than repeat them -- weights or lambdas already drawn are skipped:

    sweep     property at SWEEP_W_PROPERTY, concept at SWEEP_W_CONCEPT, joint over
              SWEEP_GRID, fk_only at each of FK_LAMS                MEGAN for fk_only
    fk_sweep  joint+fk at every FK_SWEEP_WEIGHTS pair x FK_LAMS    MEGAN is the FK energy

FK_LAMS traces FK's own trade-off: the energy is phi_c + lam * property error, so lam is to
FK what (w_c, w_p) is to the adapters.

No size model: like every colour-graph run, sizes come from the base's own distribution.

K IS THE BATCH for FK: particles compete inside one ``sample()`` call, so FK cells are
drawn in chunks of exactly PARTICLES and N_PER_CELL must be a multiple of it.

Writes one ``gen_<unit>.npz`` per work unit plus ``manifest.json``; grading is a separate step
(scripts/score_color_joint.py). Shards of one stage can run on several GPUs or machines: every shard
derives the same deterministic partition of the work units.

Usage:
    python experiments/sample_color_joint.py --__TESTING__ True
    python experiments/sample_color_joint.py --STAGE '"probe"' --SHARD 0 --NUM_SHARDS 4

Checkpoints, request vectors and the MEGAN bundle default to the model zoo (see DEFOG_MEGAN_ZOO).
"""
import json
import os
import time

import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog_adapt.core import (
    DeFoGModel, CrossAttentionAdapter, AdapterComposition, ConditionBranch, AdaptedSampler,
    Sampler, FeynmanKacSampler,
)
from defog_megan.bundle import reference_check
from defog_megan.colorgraphs import RB_PALETTE, save_pool_npz
from defog_megan.data import load_targets
from defog_megan.zoo import zoo_path

# ============================================================================
#: The unconditional base (the exponential-moving-average snapshot of epoch 80).
BASE_CKPT: str = zoo_path("colorgraphs", "base.ckpt")
#: The Table 1 cross-attention property adapter.
PROPERTY_ADAPTER: str = zoo_path("colorgraphs", "property_adapter.ckpt")
#: The Experiment 1 concept adapter.
CONCEPT_ADAPTER: str = zoo_path("colorgraphs", "concept_adapter.ckpt")
TARGETS_PATH: str = zoo_path("colorgraphs", "data/concept_targets.npz")
TARGET_MODE: str = "empirical"

# Only read in the fk stage. Must match how phi is scored everywhere else (Experiment 1,
# the conditions table), or FK optimises a different phi than the evaluation measures.
MEGAN_CKPT: str = zoo_path("colorgraphs", "megan/model.ckpt")
MEGAN_CLUSTERING: str = zoo_path("colorgraphs", "megan/clustering.clu")
MEGAN_PROCESSING: str = zoo_path("colorgraphs", "megan/process.py")
FIDELITY_THRESHOLD: float = 0.5
SCORE_K: int = 5
SHARPEN_TEMPERATURE: float = 0.5
# The gate before any steered run: recompute phi on GATE_N pool graphs and compare with the conditions table the
# concept adapter was trained against (the pool and table default to the zoo). An explainer that disagrees would steer
# FK toward a concept nobody chose, and nothing downstream would show it. 0 switches the gate off.
GATE_N: int = 64
GATE_REFERENCE: str = zoo_path("colorgraphs", "data/concept_conditions.npz")
GATE_POOL: str = zoo_path("colorgraphs", "data/pool.npz")
GATE_TOLERANCE: float = 1e-5

CONCEPTS: list = ["ch0_cl3", "ch0_cl4", "ch0_cl7", "ch1_cl11", "ch1_cl12", "ch1_cl16"]
LEVELS: list = [-3, -1, 0, 1, 3]

# :param STAGE: "probe" or "fk" -- selects the arms, see the module docstring.
STAGE: str = "probe"

# Single-adapter operating points, from Experiment 1.
W_PROPERTY_SOLO: float = 2.0
W_CONCEPT_SOLO: float = 1.0
# :param WEIGHT_GRID: (w_concept, w_property) pairs for the joint arm in the probe.
#       (1, 2) stacks both solo operating points; the other two back each one off, since
#       product mode gives the frozen base a coefficient of 1 - sum(w) and three is
#       further past the single-branch w=2 than anything measured on this rung.
WEIGHT_GRID: list = [[1.0, 2.0], [1.0, 1.0], [0.5, 1.0]]
# :param FK_WEIGHTS: the (w_concept, w_property) the fk stage steers at -- chosen from the
#       probe, so it is a launcher argument rather than a default anyone should trust.
FK_WEIGHTS: list = [1.0, 2.0]
# --- Pareto sweep (stages "sweep" and "fk_sweep") ---
SWEEP_W_PROPERTY: list = [0.5, 1.0, 3.0]
SWEEP_W_CONCEPT: list = [0.5, 2.0, 3.0]
#: The 4x4 joint grid over {0.5, 1, 2, 3} minus the three pairs the probe already drew.
SWEEP_GRID: list = [[wc, wp] for wc in (0.5, 1.0, 2.0, 3.0) for wp in (0.5, 1.0, 2.0, 3.0)
                    if [wc, wp] not in ([1.0, 2.0], [1.0, 1.0], [0.5, 1.0])]
#: fk_only in "sweep"; joint+fk in "fk_sweep". lam=1 is already in the fk stage.
FK_LAMS: list = [0.25, 0.5, 2.0, 4.0]
#: (w_c, w_p) pairs for joint+fk in "fk_sweep" -- read off the sweep's frontier.
FK_SWEEP_WEIGHTS: list = [[1.0, 1.0]]
# "product" is the plain sum of the adapters' log-ratios; the paper's joint runs used it.
COMPOSE_MODE: str = "product"

# --- Feynman-Kac (fk stage) ---
# beta=16, warmup 0.7 are the Table 1 FK rows' settings; K=16 is its k=16 row, which gives
# eight independent populations per 128-graph cell rather than one.
FK_BETA: float = 16.0
FK_LAM: float = 1.0
FK_ENERGY_MODE: str = "sum"
PARTICLES: int = 16
FK_WARMUP_FRAC: float = 0.7
# The paper's Feynman-Kac runs resampled multinomially. Stated here because the library's own default is "residual".
FK_RESAMPLE_SCHEME: str = "multinomial"
FK_RESAMPLE_INTERVAL: int = 0      # 0 -> the sampler's default (steps // 8)

N_UNCOND: int = 1024
N_PROPERTY: int = 1024
N_CONCEPT: int = 128
N_PER_CELL: int = 128
EVAL_CHUNK: int = 64
EVAL_STEPS: int = 500
ETA: float = 0.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"

SHARD: int = 0
NUM_SHARDS: int = 1
SEED: int = 42
__DEBUG__: bool = True
__TESTING__: bool = False


def _wtag(wc: float, wp: float) -> str:
    return f"wc{wc:g}_wp{wp:g}"


def work_units(e) -> list:
    """Every unit of sampling this stage does, with a cost proxy for sharding.

    The cost is samples x branches evaluated per step (FK runs the same network, its
    MEGAN calls are CPU-side and a handful per trajectory), which is close enough to
    balance four GPUs without measuring.
    """
    units = []
    if e.STAGE == "probe":
        units.append({"arm": "uncond", "n": e.N_UNCOND, "cost": e.N_UNCOND * 1})
        for level in e.LEVELS:
            units.append({"arm": "property", "level": level, "n": e.N_PROPERTY,
                          "w": [None, e.W_PROPERTY_SOLO], "cost": e.N_PROPERTY * 2})
        for concept in e.CONCEPTS:
            units.append({"arm": "concept", "concept": concept, "n": e.N_CONCEPT,
                          "w": [e.W_CONCEPT_SOLO, None], "cost": e.N_CONCEPT * 2})
        for concept in e.CONCEPTS:
            for level in e.LEVELS:
                for wc, wp in e.WEIGHT_GRID:
                    units.append({"arm": "joint", "concept": concept, "level": level,
                                  "n": e.N_PER_CELL, "w": [float(wc), float(wp)],
                                  "cost": e.N_PER_CELL * 3})
    elif e.STAGE == "sweep":
        for wp in e.SWEEP_W_PROPERTY:
            for level in e.LEVELS:
                units.append({"arm": "property", "level": level, "n": e.N_PROPERTY,
                              "w": [None, float(wp)], "cost": e.N_PROPERTY * 2})
        for wc in e.SWEEP_W_CONCEPT:
            for concept in e.CONCEPTS:
                units.append({"arm": "concept", "concept": concept, "n": e.N_CONCEPT,
                              "w": [float(wc), None], "cost": e.N_CONCEPT * 2})
        for concept in e.CONCEPTS:
            for level in e.LEVELS:
                for wc, wp in e.SWEEP_GRID:
                    units.append({"arm": "joint", "concept": concept, "level": level,
                                  "n": e.N_PER_CELL, "w": [float(wc), float(wp)],
                                  "cost": e.N_PER_CELL * 3})
                for lam in e.FK_LAMS:
                    units.append({"arm": "fk_only", "concept": concept, "level": level,
                                  "n": e.N_PER_CELL, "w": [None, None], "lam": float(lam),
                                  "cost": e.N_PER_CELL})
    elif e.STAGE == "fk_sweep":
        for concept in e.CONCEPTS:
            for level in e.LEVELS:
                for wc, wp in e.FK_SWEEP_WEIGHTS:
                    for lam in e.FK_LAMS:
                        units.append({"arm": "joint+fk", "concept": concept, "level": level,
                                      "n": e.N_PER_CELL, "w": [float(wc), float(wp)],
                                      "lam": float(lam), "cost": e.N_PER_CELL * 3})
    elif e.STAGE == "fk":
        wc, wp = (float(x) for x in e.FK_WEIGHTS)
        for concept in e.CONCEPTS:
            for level in e.LEVELS:
                units.append({"arm": "joint+fk", "concept": concept, "level": level,
                              "n": e.N_PER_CELL, "w": [wc, wp], "cost": e.N_PER_CELL * 3})
                units.append({"arm": "fk_only", "concept": concept, "level": level,
                              "n": e.N_PER_CELL, "w": [None, None], "cost": e.N_PER_CELL})
    else:
        raise ValueError(f"unknown STAGE {e.STAGE!r}")

    for u in units:
        parts = [u["arm"].replace("+", "_")]
        if "concept" in u:
            parts.append(u["concept"])
        if "level" in u:
            parts.append(f"lv{u['level']:+d}")
        if u["arm"] == "joint" or u["arm"] == "joint+fk":
            parts.append(_wtag(*u["w"]))
        elif e.STAGE == "sweep" and u["arm"] in ("property", "concept"):
            wc, wp = u["w"]
            parts.append(f"wp{wp:g}" if wc is None else f"wc{wc:g}")
        if "lam" in u:
            parts.append(f"lam{u['lam']:g}")
        u["key"] = "_".join(parts)
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
        e.SWEEP_GRID = e.SWEEP_GRID[:1]
        e.SWEEP_W_PROPERTY = e.SWEEP_W_PROPERTY[:1]
        e.SWEEP_W_CONCEPT = e.SWEEP_W_CONCEPT[:1]
        e.FK_LAMS = e.FK_LAMS[:1]
        e.FK_SWEEP_WEIGHTS = e.FK_SWEEP_WEIGHTS[:1]
        e.N_UNCOND = e.N_PROPERTY = e.N_CONCEPT = e.N_PER_CELL = 8
        e.PARTICLES = 4
        e.EVAL_CHUNK = 8
        e.EVAL_STEPS = 20

    torch.manual_seed(e.SEED + e.SHARD)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e.log(f"stage {e.STAGE} | shard {e.SHARD}/{e.NUM_SHARDS} | device {device}")

    if e.STAGE in ("fk", "sweep", "fk_sweep") and e.N_PER_CELL % e.PARTICLES:
        raise ValueError(f"N_PER_CELL={e.N_PER_CELL} is not a multiple of PARTICLES="
                         f"{e.PARTICLES}; a short final batch would run a different FK")

    target_matrix, target_ids = load_targets(e.TARGETS_PATH, mode=e.TARGET_MODE)
    target_ids = list(target_ids)
    missing = [c for c in e.CONCEPTS if c not in target_ids]
    if missing:
        raise ValueError(f"concepts not in the vocabulary: {missing}")

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    if base.num_node_classes != len(RB_PALETTE):
        raise ValueError(f"base has {base.num_node_classes} node classes, palette has "
                         f"{len(RB_PALETTE)} -- wrong base")
    prop_adapter = CrossAttentionAdapter.load(e.PROPERTY_ADAPTER, device="cpu").to(device).eval()
    concept_adapter = CrossAttentionAdapter.load(e.CONCEPT_ADAPTER, device="cpu").to(device).eval()
    for name, adapter, dim in (("property", prop_adapter, 1),
                               ("concept", concept_adapter, len(target_ids))):
        # A FiLM-only reload would silently sweep a different model than Experiment 1's.
        if getattr(adapter, "xattn_tokens", 0) == 0:
            raise ValueError(f"{name} adapter reloaded WITHOUT cross-attention -- wrong adapter file?")
        if adapter.cond_dim != dim:
            raise ValueError(f"{name} adapter cond_dim={adapter.cond_dim}, expected {dim}")
        adapter.check_compatible(base)
        e.log(f"{name} adapter: {sum(p.numel() for p in adapter.parameters()):,} params, "
              f"xattn={adapter.xattn_tokens}")

    space = None
    if e.STAGE in ("fk", "sweep", "fk_sweep"):
        from defog_megan.color_joint_energy import ColorJointEnergy
        from defog_megan.torch_compat import load_concept_space
        e.log("loading MEGAN (the FK energy) ...")
        space = load_concept_space(
            e.MEGAN_CKPT, e.MEGAN_CLUSTERING, e.MEGAN_PROCESSING,
            fidelity_threshold=e.FIDELITY_THRESHOLD, dataset_type="regression",
            score_method="knn", score_k=e.SCORE_K, sharpen="sparsemax",
            sharpen_temperature=e.SHARPEN_TEMPERATURE)
        if list(space.concept_ids) != target_ids:
            raise ValueError("the FK energy's clustering orders concepts differently from "
                             "the target file -- FK would drive a different concept than "
                             "the adapter is steered to")
        if e.GATE_N:
            deviation = reference_check(space, "colorgraphs", n=e.GATE_N, tolerance=e.GATE_TOLERANCE,
                                        reference=e.GATE_REFERENCE, pool=e.GATE_POOL)
            e.log(f"explainer gate: phi vs the conditions table on {e.GATE_N} pool graphs, max |diff| = {deviation:.3e}")

    units = work_units(e)
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

    def draw(unit):
        fk = unit["arm"] in ("joint+fk", "fk_only")
        chunk = e.PARTICLES if fk else e.EVAL_CHUNK
        samples, energy_calls = [], 0
        while len(samples) < unit["n"]:
            cur = chunk if fk else min(chunk, unit["n"] - len(samples))
            br = [] if unit["arm"] == "uncond" else branches(unit, cur)
            composition = (AdapterComposition(br, base=base, mode=e.COMPOSE_MODE) if br else None)
            if fk:
                energy = ColorJointEnergy(
                    space, space.index_of(unit["concept"]), float(unit["level"]),
                    lam=unit.get("lam", e.FK_LAM), mode=e.FK_ENERGY_MODE)
                sampler = FeynmanKacSampler(
                    base, energy_fn=energy, composition=composition, beta=e.FK_BETA,
                    warmup_frac=e.FK_WARMUP_FRAC,
                    resample_interval=(e.FK_RESAMPLE_INTERVAL or None),
                    resample_scheme=e.FK_RESAMPLE_SCHEME,
                    eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                    time_distortion=e.TIME_DISTORTION)
            elif composition is not None:
                sampler = AdaptedSampler(base, composition, eta=e.ETA, omega=e.OMEGA,
                                         sample_steps=e.EVAL_STEPS,
                                         time_distortion=e.TIME_DISTORTION)
            else:
                sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA,
                                  sample_steps=e.EVAL_STEPS,
                                  time_distortion=e.TIME_DISTORTION)
            samples += sampler.sample(num_samples=cur, device=device, show_progress=False)
            if fk:
                energy_calls += energy.n_calls
        return samples, energy_calls

    manifest = {
        "stage": e.STAGE, "shard": e.SHARD, "num_shards": e.NUM_SHARDS,
        "concepts": list(e.CONCEPTS), "levels": list(e.LEVELS),
        "concept_ids": target_ids, "target_mode": e.TARGET_MODE,
        "compose_mode": e.COMPOSE_MODE, "blend_space": "prob",
        "solo_weights": {"property": e.W_PROPERTY_SOLO, "concept": e.W_CONCEPT_SOLO},
        "weight_grid": [[float(a), float(b)] for a, b in e.WEIGHT_GRID],
        "sweep": {"w_property": list(e.SWEEP_W_PROPERTY), "w_concept": list(e.SWEEP_W_CONCEPT),
                  "grid": [[float(a), float(b)] for a, b in e.SWEEP_GRID],
                  "fk_lams": list(e.FK_LAMS),
                  "fk_weights": [[float(a), float(b)] for a, b in e.FK_SWEEP_WEIGHTS]},
        "fk": {"weights": [float(x) for x in e.FK_WEIGHTS], "beta": e.FK_BETA,
               "lam": e.FK_LAM, "energy_mode": e.FK_ENERGY_MODE,
               "particles": e.PARTICLES, "warmup_frac": e.FK_WARMUP_FRAC,
               "resample_scheme": e.FK_RESAMPLE_SCHEME,
               "resample_interval": e.FK_RESAMPLE_INTERVAL or None},
        "sampling": {"steps": e.EVAL_STEPS, "eta": e.ETA, "omega": e.OMEGA,
                     "time_distortion": e.TIME_DISTORTION},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "units": [],
    }
    for unit in mine:
        t0 = time.perf_counter()
        samples, energy_calls = draw(unit)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        fname = f"gen_{unit['key']}.npz"
        save_pool_npz(os.path.join(e.path, fname), samples, [{}] * len(samples))
        record = {k: v for k, v in unit.items() if k != "cost"}
        record.update(file=fname, n=len(samples), seconds=seconds,
                      seconds_per_sample=seconds / max(1, len(samples)),
                      energy_calls=energy_calls or None)
        manifest["units"].append(record)
        e.log(f"  {unit['key']:<40} n={len(samples):>5}  {seconds:7.1f}s "
              f"({record['seconds_per_sample']:.3f} s/graph)")

    e.commit_json("manifest.json", manifest)
    e.log("Done.")


experiment.run_if_main()
