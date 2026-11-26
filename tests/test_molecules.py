"""The molecule codec for neutral and charge-aware vocabularies (``defog_megan.molecules``).

A vocabulary without charged classes must behave exactly as in defog-adapt (the charge is dropped on encoding and repaired on
decoding); one with them reads strictly (a charge it cannot express rejects the molecule, and decoding repairs nothing).
"""
import json

import pytest
import torch
from rdkit import Chem

from defog_adapt.domains import molecule as adapt_molecule
from defog_megan.molecules import (
    CHARGED_KEK_V1_ATOMS,
    KEKULE_BONDS,
    NEUTRAL_KEK_ATOMS,
    ChargedMoleculeDomain,
    atom_label,
    atom_label_parts,
    build_encoders,
    is_charged_vocab,
    make_domain,
    mol_to_smiles,
    pyg_data_to_mol,
    smiles_to_pyg_data,
)
from defog_megan.zoo import zoo_root


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)


def round_trip(smiles, vocab):
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(vocab, KEKULE_BONDS)
    data = smiles_to_pyg_data(smiles, atom_encoder, bond_encoder, kekulize=True)
    if data is None:
        return None
    mol = pyg_data_to_mol(data, atom_decoder, bond_decoder)
    return None if mol is None else mol_to_smiles(mol)


def test_the_charged_vocabulary_extends_the_neutral_one_in_channel_order():
    assert CHARGED_KEK_V1_ATOMS[:9] == NEUTRAL_KEK_ATOMS
    assert CHARGED_KEK_V1_ATOMS[9:] == ["N+", "O-", "N-"]
    assert len(CHARGED_KEK_V1_ATOMS) == 12


def test_label_helpers():
    assert atom_label_parts("N+") == ("N", 1) and atom_label_parts("O-") == ("O", -1)
    assert atom_label_parts("Cl") == ("Cl", 0)
    assert atom_label(Chem.MolFromSmiles("C[N+](C)(C)C").GetAtomWithIdx(1)) == "N+"
    assert atom_label(Chem.MolFromSmiles("C[S+](C)[O-]").GetAtomWithIdx(1)) == "S+"
    assert atom_label(Chem.MolFromSmiles("[Fe+2]").GetAtomWithIdx(0)) == "Fe+2"
    assert is_charged_vocab(["C", "N+"]) and not is_charged_vocab(NEUTRAL_KEK_ATOMS)


@pytest.mark.parametrize("smiles", [
    "O=[N+]([O-])c1ccccc1",                   # nitro
    "CN(C)c1ccc(N=[N+]=[N-])cc1",             # azide
    "[O-][n+]1ccccc1",                        # aromatic N-oxide
    "C[N+](C)(C)C",                           # quaternary nitrogen
    "CCO", "c1ccccc1Cl",                      # neutral molecules are untouched
    "c1ccccc1[NH3+]", "O=C([O-])c1ccccc1",    # protonation states are expressible too
])
def test_charged_vocabulary_round_trips(smiles):
    assert canon(round_trip(smiles, CHARGED_KEK_V1_ATOMS)) == canon(smiles)


def test_a_charge_without_a_class_is_rejected_not_changed():
    assert round_trip("C[S+](C)[O-]", CHARGED_KEK_V1_ATOMS) is None
    assert round_trip("[Na+].[Cl-]", CHARGED_KEK_V1_ATOMS) is None


def test_the_neutral_vocabulary_keeps_its_old_reading():
    # the charge is dropped on encoding and the over-valent N is repaired on decoding, which is why a nitro group does not
    # survive the nine-element vocabulary
    assert round_trip("O=[N+]([O-])c1ccccc1", NEUTRAL_KEK_ATOMS) == "O=[N+](O)c1ccccc1"
    assert canon(round_trip("CCO", NEUTRAL_KEK_ATOMS)) == canon("CCO")


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1Cl", "CC(=O)Nc1ccc(O)cc1", "C#CC(=O)OC", "FC(F)(F)c1ccncc1"])
def test_the_charged_encoder_agrees_with_the_neutral_one_on_neutral_molecules(smiles):
    neutral = build_encoders(NEUTRAL_KEK_ATOMS, KEKULE_BONDS)
    charged = build_encoders(CHARGED_KEK_V1_ATOMS, KEKULE_BONDS)
    a = adapt_molecule.smiles_to_pyg_data(smiles, neutral[0], neutral[2], kekulize=True)
    b = smiles_to_pyg_data(smiles, charged[0], charged[2], kekulize=True)
    assert torch.equal(b.x[:, :9], a.x) and not b.x[:, 9:].any()
    assert torch.equal(b.edge_index, a.edge_index) and torch.equal(b.edge_attr, a.edge_attr)


def test_a_neutral_vocabulary_is_delegated_to_defog_adapt_unchanged():
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(NEUTRAL_KEK_ATOMS, KEKULE_BONDS)
    data = smiles_to_pyg_data("O=[N+]([O-])c1ccccc1", atom_encoder, bond_encoder, kekulize=True)
    ours = pyg_data_to_mol(data, atom_decoder, bond_decoder)
    theirs = adapt_molecule.pyg_data_to_mol(data, atom_decoder, bond_decoder, charge_correction=True)
    assert Chem.MolToSmiles(ours) == Chem.MolToSmiles(theirs)


def test_decoding_is_strict_for_a_charged_vocabulary():
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(CHARGED_KEK_V1_ATOMS, KEKULE_BONDS)
    data = smiles_to_pyg_data("O=[N+]([O-])c1ccccc1", atom_encoder, bond_encoder, kekulize=True)
    # turn the nitro N+ into a plain N: a four-bonded neutral nitrogen is a model error and must not be repaired
    nitro = data.x[:, atom_encoder["N+"]] == 1
    data.x[nitro] = 0
    data.x[nitro, atom_encoder["N"]] = 1
    mol = pyg_data_to_mol(data, atom_decoder, bond_decoder)
    assert mol is not None                      # the molecule object is built ...
    assert mol_to_smiles(mol) is None           # ... but does not sanitize
    with pytest.raises(ValueError):
        pyg_data_to_mol(data, atom_decoder, bond_decoder, charge_correction=True)


def test_the_formal_charges_come_from_the_classes():
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(CHARGED_KEK_V1_ATOMS, KEKULE_BONDS)
    data = smiles_to_pyg_data("O=[N+]([O-])c1ccccc1", atom_encoder, bond_encoder, kekulize=True)
    mol = pyg_data_to_mol(data, atom_decoder, bond_decoder)
    assert sorted(a.GetFormalCharge() for a in mol.GetAtoms()) == [-1] + [0] * 7 + [1]


def test_the_charged_domain_decodes_what_the_neutral_domain_cannot():
    _, atom_decoder, _, bond_decoder = build_encoders(CHARGED_KEK_V1_ATOMS, KEKULE_BONDS)
    atom_encoder, _, bond_encoder, _ = build_encoders(CHARGED_KEK_V1_ATOMS, KEKULE_BONDS)
    data = smiles_to_pyg_data("CN(C)c1ccc(N=[N+]=[N-])cc1", atom_encoder, bond_encoder, kekulize=True)

    domain = make_domain(atom_decoder, bond_decoder)
    assert isinstance(domain, ChargedMoleculeDomain)
    assert canon(Chem.MolToSmiles(domain.decode(data))) == canon("CN(C)c1ccc(N=[N+]=[N-])cc1")
    assert domain.is_valid(data) and domain.identity(data) == canon("CN(C)c1ccc(N=[N+]=[N-])cc1")
    with pytest.raises(NotImplementedError):
        domain.metrics([data])


def test_make_domain_gives_the_neutral_domain_for_a_neutral_vocabulary():
    _, atom_decoder, _, bond_decoder = build_encoders(NEUTRAL_KEK_ATOMS, KEKULE_BONDS)
    domain = make_domain(atom_decoder, bond_decoder)
    assert type(domain) is adapt_molecule.MoleculeDomain


# -- against the zoo (skipped without it) ------------------------------------------------------------------------------

def _sidecar(group, name):
    path = zoo_root() / group / name
    if not path.is_file():
        pytest.skip(f"{path} is not available")
    return json.loads(path.read_text())


def test_the_ames_base_vocabulary_is_the_charged_one_of_the_library():
    sidecar = _sidecar("ames", "base.vocabulary.json")
    assert sidecar["atom_decoder"] == CHARGED_KEK_V1_ATOMS
    assert [b.upper() for b in sidecar["bond_decoder"][1:]] == KEKULE_BONDS
    assert sidecar["kekulize"] is True and sidecar["num_node_classes"] == 12 and sidecar["num_edge_classes"] == 4


def test_the_zinc_base_vocabulary_is_the_neutral_one_of_the_library():
    schema = _sidecar("zinc-logp", "base.schema.json")
    assert schema["atom_types"] == NEUTRAL_KEK_ATOMS and schema["bond_types"] == KEKULE_BONDS and schema["kekulize"] is True
