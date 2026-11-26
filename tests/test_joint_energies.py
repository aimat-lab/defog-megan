"""The Feynman-Kac energies of the molecular rungs: a requested concept AND a requested property, per particle.

The explainer is faked (a lookup from SMILES to membership and predicted property), the decoding is real: dense one-hot
graphs go through ``dense_to_pyg`` and the molecule codec exactly as in a sampling run.
"""
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import Crippen

from defog_megan import evaluate
from defog_megan.ames_joint_energy import AmesJointEnergy
from defog_megan.mol_joint_energy import MolJointEnergy
from defog_megan.molecules import (CHARGED_KEK_V1_ATOMS, KEKULE_BONDS, NEUTRAL_KEK_ATOMS, build_encoders, make_domain,
                                   smiles_to_pyg_data)

INVALID = 1e3


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def dense_batch(smiles_list, atoms=NEUTRAL_KEK_ATOMS):
    """Dense one-hot (X, E, node_mask) for a list of SMILES, padded to the largest molecule."""
    atom_encoder, _, bond_encoder, _ = build_encoders(atoms, KEKULE_BONDS)
    graphs = [smiles_to_pyg_data(s, atom_encoder, bond_encoder, kekulize=True) for s in smiles_list]
    n = max(g.x.shape[0] for g in graphs)
    dx, de = len(atom_encoder), len(bond_encoder) + 1
    X = torch.zeros(len(graphs), n, dx)
    E = torch.zeros(len(graphs), n, n, de)
    E[..., 0] = 1.0
    mask = torch.zeros(len(graphs), n, dtype=torch.bool)
    for b, g in enumerate(graphs):
        k = g.x.shape[0]
        X[b, :k] = g.x
        mask[b, :k] = True
        for (i, j), cls in zip(g.edge_index.t().tolist(), g.edge_attr.argmax(-1).tolist()):
            E[b, i, j] = 0.0
            E[b, i, j, cls] = 1.0
    return X, E, mask


class FakeSpace:
    """Membership of a molecule in two concepts, by canonical SMILES."""

    def __init__(self, phi_by_smiles):
        self.phi_by = {canon(k): v for k, v in phi_by_smiles.items()}

    def phi(self, smiles, batch_size=256):
        return np.array([self.phi_by[s] for s in smiles], dtype=float)


ETHANOL, PROPANOL, BENZENE = "CCO", "CCCO", "c1ccccc1"
PHI = {ETHANOL: [0.9, 0.2], PROPANOL: [0.9, 0.9], BENZENE: [0.9, 0.0]}


def test_logp_energy_is_the_concept_term_plus_the_capped_squared_miss():
    space = FakeSpace(PHI)
    domain = make_domain(*build_encoders(NEUTRAL_KEK_ATOMS, KEKULE_BONDS)[1::2])
    target = Crippen.MolLogP(Chem.MolFromSmiles(ETHANOL))
    energy = MolJointEnergy(space, domain, concept_index=1, target_logp=target, lam=1.0, scale=1.0, cap=2.0)
    out = energy(*dense_batch([ETHANOL, PROPANOL, BENZENE]))

    miss = lambda s: min(((Crippen.MolLogP(Chem.MolFromSmiles(s)) - target) / 1.0) ** 2, 2.0)
    assert out[0].item() == pytest.approx(0.2 + 0.0, abs=1e-5)                     # the particle on target
    assert out[1].item() == pytest.approx(0.9 + miss(PROPANOL), abs=1e-5)
    assert out[2].item() == pytest.approx(0.0 + miss(BENZENE), abs=1e-5)           # a member of the concept, off target


def test_the_miss_is_capped():
    space = FakeSpace(PHI)
    domain = make_domain(*build_encoders(NEUTRAL_KEK_ATOMS, KEKULE_BONDS)[1::2])
    energy = MolJointEnergy(space, domain, concept_index=1, target_logp=50.0, scale=1.0, cap=2.0)
    out = energy(*dense_batch([ETHANOL]))
    assert out[0].item() == pytest.approx(0.2 + 2.0, abs=1e-5)


def test_without_a_target_the_energy_is_the_concept_term_alone():
    space = FakeSpace(PHI)
    domain = make_domain(*build_encoders(NEUTRAL_KEK_ATOMS, KEKULE_BONDS)[1::2])
    out = MolJointEnergy(space, domain, concept_index=1, target_logp=None)(*dense_batch([ETHANOL, PROPANOL, BENZENE]))
    assert out.tolist() == pytest.approx([0.2, 0.9, 0.0], abs=1e-6)


def test_max_mode_takes_the_worse_of_the_two_terms():
    space = FakeSpace(PHI)
    domain = make_domain(*build_encoders(NEUTRAL_KEK_ATOMS, KEKULE_BONDS)[1::2])
    target = Crippen.MolLogP(Chem.MolFromSmiles(ETHANOL))
    out = MolJointEnergy(space, domain, concept_index=1, target_logp=target, scale=1.0, mode="max")(*dense_batch([ETHANOL, PROPANOL]))
    assert out[0].item() == pytest.approx(0.2, abs=1e-5)
    assert out[1].item() == pytest.approx(max(0.9, ((Crippen.MolLogP(Chem.MolFromSmiles(PROPANOL)) - target)) ** 2), abs=1e-5)


def test_undecodable_and_fragmented_particles_get_the_invalid_energy():
    space = FakeSpace(PHI)
    domain = make_domain(*build_encoders(NEUTRAL_KEK_ATOMS, KEKULE_BONDS)[1::2])
    # a fragmented particle: ethanol with the C-O bond cut, i.e. [CC].[O]
    frag_X, frag_E, frag_mask = dense_batch([ETHANOL])
    frag_E[0, 1, 2] = frag_E[0, 2, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])      # cut the C-O bond: [CC].[O]
    out = MolJointEnergy(space, domain, concept_index=1, target_logp=0.0)(frag_X, frag_E, frag_mask)
    assert out[0].item() == pytest.approx(INVALID)


def test_ames_energy_asks_the_explainer_for_the_property(monkeypatch):
    atom_decoder = build_encoders(CHARGED_KEK_V1_ATOMS, KEKULE_BONDS)[1]
    bond_decoder = build_encoders(CHARGED_KEK_V1_ATOMS, KEKULE_BONDS)[3]
    domain = make_domain(atom_decoder, bond_decoder)
    nitro, toluene = "O=[N+]([O-])c1ccccc1", "Cc1ccccc1"
    space = FakeSpace({nitro: [0.7, 0.1], toluene: [0.05, 0.8]})          # two concepts; the second one is requested
    probability = {canon(nitro): 0.8, canon(toluene): 0.1}
    monkeypatch.setattr(evaluate, "property_readout",
                        lambda kind: (lambda sp, smiles, batch_size: np.array([probability[s] for s in smiles])))

    energy = AmesJointEnergy(space, domain, concept_index=1, target=1.0, lam=1.0, scale=0.5, cap=2.0)
    out = energy(*dense_batch([nitro, toluene], atoms=CHARGED_KEK_V1_ATOMS))
    assert out[0].item() == pytest.approx(0.1 + ((0.8 - 1.0) / 0.5) ** 2, abs=1e-5)             # a nitro compound, mutagenic
    assert out[1].item() == pytest.approx(0.8 + min(((0.1 - 1.0) / 0.5) ** 2, 2.0), abs=1e-5)    # capped
    assert out[0] < out[1]


def test_ames_energy_defaults_are_those_of_the_paper_run():
    energy = AmesJointEnergy(FakeSpace({}), object(), concept_index=0)
    assert energy.property_kind == "class_probability" and energy.scale == 0.5 and energy.mode == "sum"
    assert energy.cap == 2.0 and energy.macro_weight == 0.5 and energy.invalid == INVALID
