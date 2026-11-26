"""Feynman-Kac energy for a requested mutagenicity class AND a requested concept, on molecules (Experiment 4).

The counterpart of :mod:`defog_megan.mol_joint_energy` for the real-world rung. Per particle the energy is

    clip(phi_c, 0, 1)  +  lam * min(((p - y*) / scale)^2, cap)

where ``phi_c`` is MEGAN's sharpened distance to the requested concept (0 = core member, 1 = not
assigned), ``p`` is **MEGAN's own predicted probability of the mutagenic class** for the decoded molecule
and ``y*`` the requested class (0 or 1). That is the difference to the logP rung, where the property was
RDKit's Crippen logP and the energy could only be satisfied by actually having the value: mutagenicity is
not computable from structure, so the energy asks the same model that later grades the molecules. A match
under FK therefore measures steerability, not independent correctness, and every number built on it is
called model-graded.

``scale`` is the miss that costs about 1 (0.5 for a probability); ``cap`` keeps a hopeless particle from
drowning the concept term. Particles that do not decode get ``invalid_energy``, and the macrocycle penalty
of the logP energy is kept. ``property_kind`` also allows a regression readout of the property.
"""
import typing as t

import numpy as np
import torch


class AmesJointEnergy:
    """FK energy over (concept membership, MEGAN-predicted property). ``energy_fn(X1, E1, node_mask)``.

    :param space: a loaded :class:`~defog_megan.concepts.ConceptSpace`.
    :param domain: object with ``.decode(pyg_data) -> Optional[Mol]``.
    :param concept_index: column of ``phi`` to drive.
    :param target: requested property value (0 or 1 for the class probability), or ``None`` for a concept-only
        energy.
    :param property_kind: how the property is read out of MEGAN, see ``evaluate.property_readout``.
    :param lam: weight on the property term relative to the concept term.
    :param mode: ``"sum"`` or ``"max"`` of the two terms (the colour rung used "sum").
    """

    def __init__(self, space, domain, concept_index: int,
                 target: t.Optional[float] = None, lam: float = 1.0,
                 scale: float = 0.5, cap: float = 2.0, mode: str = "sum",
                 macro_weight: float = 0.5, invalid_energy: float = 1e3,
                 batch_size: int = 256, property_kind: str = "class_probability"):
        if mode not in ("sum", "max"):
            raise ValueError(f"unknown energy mode {mode!r}")
        self.space = space
        self.domain = domain
        self.concept_index = int(concept_index)
        self.target = None if target is None else float(target)
        self.lam = float(lam)
        self.scale = float(scale)
        self.cap = float(cap)
        self.mode = mode
        self.macro_weight = float(macro_weight)
        self.invalid = float(invalid_energy)
        self.batch_size = int(batch_size)
        self.property_kind = property_kind    # "class_probability" (mutagenicity) or "regression"
        self.n_calls = 0
        self.n_scored = 0
        self.last_stats: dict = {}

    @torch.no_grad()
    def __call__(self, X1, E1, node_mask):
        from rdkit import Chem

        from defog_adapt.core.data import dense_to_pyg
        from .evaluate import property_readout
        from .plausibility import macrocycle_penalty

        counts = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, counts)
        out = X1.new_full((len(datas),), self.invalid)

        # Decode, canonicalise, and re-parse: the energy must see exactly the molecule the
        # grader will see, and MEGAN is fed SMILES, so both readouts come from one string.
        smiles, keep, penalties = [], [], []
        for i, data in enumerate(datas):
            mol = self.domain.decode(data)
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
                parsed = Chem.MolFromSmiles(smi) if smi else None
            except Exception:
                continue
            if parsed is None or "." in smi:                  # undecodable or fragmented
                continue
            smiles.append(smi)
            keep.append(i)
            penalties.append(macrocycle_penalty(parsed, self.macro_weight))

        self.n_calls += 1
        if not smiles:
            self.last_stats = {"n": len(datas), "valid": 0}
            return out
        self.n_scored += len(smiles)

        phi = self.space.phi(smiles, batch_size=self.batch_size)
        if len(phi) != len(smiles):                           # MEGAN could not encode some;
            self.last_stats = {"n": len(datas), "valid": 0}   # never pair a molecule wrongly
            return out
        concept = np.clip(phi[:, self.concept_index], 0.0, 1.0)

        predicted = None
        if self.target is not None:
            predicted = property_readout(self.property_kind)(self.space, smiles, self.batch_size)
            if len(predicted) != len(smiles):                 # same guard as for phi
                self.last_stats = {"n": len(datas), "valid": 0}
                return out

        if self.target is None:
            terms = concept
        else:
            miss = np.minimum(((predicted - self.target) / self.scale) ** 2, self.cap)
            terms = (concept + self.lam * miss if self.mode == "sum"
                     else np.maximum(concept, self.lam * miss))

        for j, i in enumerate(keep):
            out[i] = float(terms[j]) + penalties[j]
        self.last_stats = {"n": len(datas), "valid": len(smiles),
                           "mean_concept": float(concept.mean()),
                           "mean_abs_miss": (None if self.target is None else
                                             float(np.abs(predicted - self.target).mean()))}
        return out
