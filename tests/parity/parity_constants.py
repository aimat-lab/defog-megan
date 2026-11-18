"""The definitions of the colored-graph parity cases: which files, which settings, which cases, which tiers.

Kept free of torch and numpy so that the structure tests of the golden file run in the ordinary unit suite. The code
that runs the cases is ``cases.py``.
"""

GROUP = "colorgraphs"

#: logical name -> path inside the zoo group
FILES = {
    "base": "base.ckpt",
    "property_adapter": "property_adapter.ckpt",
    "concept_adapter": "concept_adapter.ckpt",
    "native_model": "native/model.ckpt",
    "native_stats": "native/condition_stats.json",
    "concept_targets": "data/concept_targets.npz",
}

#: the settings the paper's colored-graph arms ran with, plus the reduced step counts of these cases
SETTINGS = {
    "eta": 0.0,
    "omega": 0.0,
    "time_distortion": "polydec",
    "compose_mode": "product",            # the plain sum of the adapters' log-ratios
    "blend_space": "prob",                # probability space; defog_adapt has no other
    "weights": {"property": 2.0, "concept": 1.0, "joint": [1.0, 1.0]},
    "property_level": 1.0,                # the requested exact motif sum
    "concept_index": 3,                   # ch0_cl3, one of the six cells' concepts
    "fk": {"beta": 16.0, "warmup_frac": 0.7, "resample_interval": None, "ess_frac": None,
           "resample_scheme": "multinomial"},
    "native": {"guidance_scale": 1.0},
    "explicit_nodes": 12,
}

#: two sizes of the same cases. ``quick`` runs in about a minute and is what a routine check uses. ``paper`` uses the
#: paper's sampler settings (500 steps, 16 Feynman-Kac particles) and is the release gate. Only the loop length and the
#: batch change; the code that runs is the same.
_QUICK_STEPS = {"base_sampling": 10, "explicit_size": 10, "property_adapter": 10, "concept_adapter": 10,
                "joint_product": 20, "fk_joint": 20, "native_conditioned": 10}
_GRAPHS = {"base_sampling": 4, "explicit_size": 4, "property_adapter": 4, "concept_adapter": 4,
           "joint_product": 4, "fk_joint": 8, "native_conditioned": 4}
TIERS = {
    "quick": {"steps": _QUICK_STEPS, "graphs": _GRAPHS},
    "paper": {"steps": {name: 500 for name in _QUICK_STEPS}, "graphs": {**_GRAPHS, "fk_joint": 16}},
}

#: case name -> (seed, one line of what it pins)
CASES = {
    "base_sampling": (1, "unconditional sampling, node counts from the base's own distribution"),
    "explicit_size": (2, "sampling with an explicit node count (the size-sweep path)"),
    "property_adapter": (3, "the property adapter alone at its paper weight"),
    "concept_adapter": (4, "the concept adapter alone at its paper weight"),
    "joint_product": (5, "both adapters composed as a plain sum at the selected weights (1, 1)"),
    "fk_joint": (6, "Feynman-Kac steering of the composition, multinomial resampling, deterministic stand-in energy"),
    "native_conditioned": (7, "the end-to-end conditioned model with its standardized condition, guidance 1"),
}
