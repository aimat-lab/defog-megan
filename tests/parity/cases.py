"""Parity cases for the synthetic colored graphs: the generator side of the paper's pipeline.

One run of each code path that the paper's colored-graph arms exercise, returned as plain JSON data. The same cases run
against two stacks, selected by package name:

* ``defog``        the research package at the revision the paper's runs used. Its output is the golden file.
* ``defog_adapt``  the library this package builds on. It must reproduce the golden output bit for bit.

So these cases may only use API that exists in BOTH packages (the small differences are handled in ``sampler_kwargs``),
and every setting that could change under a re-tuned default is passed explicitly: the paper's settings were launcher
overrides, never defaults.

What the cases pin, and what they do not
----------------------------------------
Pinned: unconditional sampling, sampling from the explicit-size path, each adapter alone, the two-adapter composition
(plain sum of the log-ratios, probability-space blending), Feynman-Kac steering (multinomial resampling), and the
end-to-end conditioned model, each at two sizes (see ``TIERS``): ``quick`` with short loops, ``paper`` with the paper's
500 steps and 16 particles. The Feynman-Kac energy is a deterministic stand-in that depends on no explainer: the
explainer-backed energy of the paper is a later case, once the explainer's numerics can be held fixed on both sides.

Determinism: single-threaded CPU, float32, every stochastic call preceded by ``manual_seed``. The checkpoints come from
the model zoo (``DEFOG_MEGAN_ZOO``) and are identified by their sha256.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from parity_constants import CASES, FILES, GROUP, SETTINGS, TIERS  # noqa: E402,F401


# --------------------------------------------------------------------------------------------------- plumbing
def load_stack(pkg: str):
    return importlib.import_module(pkg + ".core")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(zoo: Path) -> dict:
    paths = {name: Path(zoo) / GROUP / rel for name, rel in FILES.items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("the model zoo lacks: " + ", ".join(missing))
    return paths


def input_hashes(zoo: Path) -> dict:
    """bundle path -> sha256 of every input, as recorded in the golden file."""
    return {rel: file_digest(Path(zoo) / GROUP / rel) for rel in FILES.values()}


def digest(graphs) -> dict:
    """A digest of every generated graph (node classes, edges, edge classes) and its node counts."""
    h = hashlib.sha256()
    for g in graphs:
        h.update(g.x.argmax(-1).numpy().astype(np.int16).tobytes())
        h.update(g.edge_index.numpy().astype(np.int32).tobytes())
        h.update(g.edge_attr.argmax(-1).numpy().astype(np.int8).tobytes())
    return {"digest": h.hexdigest(), "sizes": [int(g.x.shape[0]) for g in graphs]}


def count_energy(X1, E1, node_mask):
    """The stand-in Feynman-Kac energy: deterministic, tensor-valued, no decoding. How far the number of class-1 nodes
    of the predicted clean graph is from 3."""
    return ((X1[..., 1] * node_mask).sum(-1) - 3.0).pow(2) / 9.0


def composition(core, branches, base):
    """An AdapterComposition of the branches, with the blend space passed where the package still has the choice."""
    sig = inspect.signature(core.AdapterComposition.__init__).parameters
    extra = {"blend_space": SETTINGS["blend_space"]} if "blend_space" in sig else {}
    return core.AdapterComposition(branches, base=base, mode=SETTINGS["compose_mode"], **extra)


def fk_kwargs(core, steps):
    """The Feynman-Kac settings, adapted to each package's signature. The research package at the paper's revision has no
    ``resample_scheme`` (it resamples multinomially) and still accepts the options that defog_adapt removed."""
    fk = SETTINGS["fk"]
    sig = inspect.signature(core.FeynmanKacSampler.__init__).parameters
    kw = {"beta": fk["beta"], "warmup_frac": fk["warmup_frac"], "resample_interval": fk["resample_interval"],
          "ess_frac": fk["ess_frac"], "eta": SETTINGS["eta"], "omega": SETTINGS["omega"], "sample_steps": steps,
          "time_distortion": SETTINGS["time_distortion"]}
    if "resample_scheme" in sig:
        kw["resample_scheme"] = fk["resample_scheme"]
    if "rejuvenate" in sig:
        kw.update(rejuvenate=False, jump_length=None)
    return kw


def native_request_vector(stats: dict, level: float, phi_target) -> torch.Tensor:
    """The condition of the end-to-end model, standardized as in its training."""
    z_prop = (float(level) - stats["property"]["mean"]) / stats["property"]["std"]
    z_phi = (np.asarray(phi_target, dtype=np.float64) - np.asarray(stats["phi"]["mean"])) / np.asarray(stats["phi"]["std"])
    return torch.from_numpy(np.concatenate([[z_prop], z_phi]).astype(np.float32))


# ------------------------------------------------------------------------------------------------------ the cases
class Runner:
    """Loads the inputs once and runs any case under one package."""

    def __init__(self, pkg: str, zoo: Path):
        torch.set_num_threads(1)
        self.pkg = pkg
        self.core = load_stack(pkg)
        paths = input_paths(zoo)
        load = self.core.DeFoGModel.load
        self.base = load(str(paths["base"]), device="cpu").eval()
        adapter_load = self.core.AdaLNAdapter.load
        self.property_adapter = adapter_load(str(paths["property_adapter"]), device="cpu").eval()
        self.concept_adapter = adapter_load(str(paths["concept_adapter"]), device="cpu").eval()
        self.native = load(str(paths["native_model"]), device="cpu").eval()
        self.stats = json.loads(Path(paths["native_stats"]).read_text())
        targets = np.load(paths["concept_targets"])
        self.concept_vector = torch.tensor(targets["targets_empirical"][SETTINGS["concept_index"]], dtype=torch.float32)

    # branches ---------------------------------------------------------------------------------------------
    def _property_branch(self, n, weight):
        level = torch.tensor([[SETTINGS["property_level"]]], dtype=torch.float32).expand(n, -1)
        return self.core.ConditionBranch(self.property_adapter, level, weight)

    def _concept_branch(self, n, weight):
        return self.core.ConditionBranch(self.concept_adapter, self.concept_vector.expand(n, -1), weight)

    def _adapted_sampler(self, branches, steps):
        comp = composition(self.core, branches, self.base)
        return self.core.AdaptedSampler(self.base, comp, eta=SETTINGS["eta"], omega=SETTINGS["omega"],
                                        sample_steps=steps, time_distortion=SETTINGS["time_distortion"])

    # cases ------------------------------------------------------------------------------------------------
    def base_sampling(self, n, steps):
        sampler = self.core.Sampler(self.base, eta=SETTINGS["eta"], omega=SETTINGS["omega"], sample_steps=steps,
                                    time_distortion=SETTINGS["time_distortion"])
        return sampler.sample(num_samples=n, device="cpu", show_progress=False)

    def explicit_size(self, n, steps):
        sampler = self.core.Sampler(self.base, eta=SETTINGS["eta"], omega=SETTINGS["omega"], sample_steps=steps,
                                    time_distortion=SETTINGS["time_distortion"])
        return sampler.sample(num_samples=n, num_nodes=SETTINGS["explicit_nodes"], device="cpu", show_progress=False)

    def property_adapter_case(self, n, steps):
        sampler = self._adapted_sampler([self._property_branch(n, SETTINGS["weights"]["property"])], steps)
        return sampler.sample(num_samples=n, device="cpu", show_progress=False)

    def concept_adapter_case(self, n, steps):
        sampler = self._adapted_sampler([self._concept_branch(n, SETTINGS["weights"]["concept"])], steps)
        return sampler.sample(num_samples=n, device="cpu", show_progress=False)

    def joint_product(self, n, steps):
        wc, wp = SETTINGS["weights"]["joint"]
        sampler = self._adapted_sampler([self._concept_branch(n, wc), self._property_branch(n, wp)], steps)
        return sampler.sample(num_samples=n, device="cpu", show_progress=False)

    def fk_joint(self, n, steps):
        wc, wp = SETTINGS["weights"]["joint"]
        comp = composition(self.core, [self._concept_branch(n, wc), self._property_branch(n, wp)], self.base)
        sampler = self.core.FeynmanKacSampler(self.base, energy_fn=count_energy, composition=comp,
                                              **fk_kwargs(self.core, steps))
        return sampler.sample(num_samples=n, device="cpu", show_progress=False)

    def native_conditioned(self, n, steps):
        vec = native_request_vector(self.stats, SETTINGS["property_level"], self.concept_vector.numpy())
        return self.native.sample(num_samples=n, condition=vec.expand(n, -1),
                                  guidance_scale=SETTINGS["native"]["guidance_scale"], sample_steps=steps,
                                  eta=SETTINGS["eta"], omega=SETTINGS["omega"],
                                  time_distortion=SETTINGS["time_distortion"], device="cpu", show_progress=False)

    def run(self, name: str, tier: str = "quick") -> dict:
        seed, _ = CASES[name]
        method = {"property_adapter": self.property_adapter_case, "concept_adapter": self.concept_adapter_case}.get(
            name, getattr(self, name))
        torch.manual_seed(seed)
        return digest(method(TIERS[tier]["graphs"][name], TIERS[tier]["steps"][name]))
