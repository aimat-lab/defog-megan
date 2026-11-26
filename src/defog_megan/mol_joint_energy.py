"""Feynman-Kac energy for a requested logP AND a requested concept, on molecules.

The molecular counterpart of :mod:`defog_megan.color_joint_energy`, for the logP rung
(Experiment 3). Per particle the energy is

    clip(phi_c, 0, 1)  +  lam * min((|logP - y*| / scale)^2, cap)

where ``phi_c`` is MEGAN's sharpened distance to the requested concept (0 = core member,
1 = not assigned) and ``logP`` is **RDKit's Crippen logP of the decoded molecule**. Nothing
about the property is predicted: the grader and the energy compute the same exact function
of the same decoded graph, so a particle can score well only by actually having the value.
(The Ames energy, :mod:`defog_megan.ames_joint_energy`, has to ask MEGAN for its property,
which is not computable from structure.)

``scale`` defaults to the label's own spread (zinc64k logP: sd 1.16) so that a typical miss
costs about 1 and the two terms are comparable on one ``lam``; ``cap`` keeps a hopeless
particle from drowning the concept term. Particles that do not decode to a molecule get
``invalid_energy`` so their weight collapses, and a macrocycle penalty keeps steering from
rewarding ring artifacts.

The concept term is graded by the very MEGAN that supplies it, so a concept match under FK
measures steerability, not independent correctness; the experiment's write-up checks that
FK's matches carry the motifs real concept members carry.
"""
import typing as t

import numpy as np
import torch

#: Spread of the zinc64k logP labels (decoded molecules, Crippen): the miss that scores ~1.
LOGP_LABEL_SD = 1.16


class MolJointEnergy:
    """FK energy over (concept membership, exact logP). ``energy_fn(X1, E1, node_mask)``.

    :param space: a loaded :class:`~defog_megan.concepts.ConceptSpace`.
    :param domain: object with ``.decode(pyg_data) -> Optional[Mol]``.
    :param concept_index: column of ``phi`` to drive.
    :param target_logp: requested logP, or ``None`` for a concept-only energy.
    :param lam: weight on the logP term relative to the concept term.
    :param mode: ``"sum"`` or ``"max"`` of the two terms (the colour rung used "sum").
    """

    def __init__(self, space, domain, concept_index: int,
                 target_logp: t.Optional[float] = None, lam: float = 1.0,
                 scale: float = LOGP_LABEL_SD, cap: float = 2.0, mode: str = "sum",
                 macro_weight: float = 0.5, invalid_energy: float = 1e3,
                 batch_size: int = 256):
        if mode not in ("sum", "max"):
            raise ValueError(f"unknown energy mode {mode!r}")
        from .molecules import warn_if_rdkit_differs
        warn_if_rdkit_differs("the logP term of the Feynman-Kac energy")
        self.space = space
        self.domain = domain
        self.concept_index = int(concept_index)
        self.target = None if target_logp is None else float(target_logp)
        self.lam = float(lam)
        self.scale = float(scale)
        self.cap = float(cap)
        self.mode = mode
        self.macro_weight = float(macro_weight)
        self.invalid = float(invalid_energy)
        self.batch_size = int(batch_size)
        self.n_calls = 0
        self.n_scored = 0
        self.last_stats: dict = {}

    @torch.no_grad()
    def __call__(self, X1, E1, node_mask):
        from rdkit import Chem
        from rdkit.Chem import Crippen

        from defog_adapt.core.data import dense_to_pyg
        from .plausibility import macrocycle_penalty

        counts = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, counts)
        out = X1.new_full((len(datas),), self.invalid)

        # Decode, canonicalise, and re-parse: the energy must see exactly the molecule the
        # grader will see, and MEGAN is fed SMILES, so both readouts come from one string.
        smiles, keep, logps, penalties = [], [], [], []
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
            logps.append(float(Crippen.MolLogP(parsed)))
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

        if self.target is None:
            terms = concept
        else:
            miss = np.minimum(((np.asarray(logps) - self.target) / self.scale) ** 2, self.cap)
            terms = (concept + self.lam * miss if self.mode == "sum"
                     else np.maximum(concept, self.lam * miss))

        for j, i in enumerate(keep):
            out[i] = float(terms[j]) + penalties[j]
        self.last_stats = {"n": len(datas), "valid": len(smiles),
                           "mean_concept": float(concept.mean()),
                           "mean_abs_miss": (None if self.target is None else
                                             float(np.abs(np.asarray(logps) - self.target).mean()))}
        return out
