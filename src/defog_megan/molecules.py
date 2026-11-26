"""The molecule codec of the molecular rungs, for neutral and for charge-aware atom vocabularies.

``defog_adapt.domains.molecule`` reads a molecule by element symbol: a formal charge is dropped on encoding, and decoding
repairs an over-valent N, O or S by giving it a charge. That is the right reading for the nine-element ZINC vocabulary
(C N O F P S Cl Br I), and it is what the logP rung (Experiment 3) uses.

The mutagenicity rung (Experiment 4) needs a vocabulary that can write nitro, azide and N-oxide groups, which are 19% of the
Ames set. Its atom classes are the same nine elements plus three charged ones, ``N+``, ``O-`` and ``N-`` (the vocabulary
``charged_kek_v1``). A label that ends in ``+`` or ``-`` is a charged class, and such a vocabulary is read strictly:

* encoding keys an atom on its symbol *and* its formal charge, so a charge without a class (``[S+]``, ``[Na+]``) rejects the
  molecule, instead of silently changing it;
* decoding sets the formal charge from the class and repairs nothing, because the model chooses the charged class itself and
  a repair would paper over its mistakes (a four-bonded neutral nitrogen stays invalid).

Everything here dispatches on the vocabulary, so the same call serves both rungs, and for a neutral vocabulary it is exactly
the call of ``defog_adapt``.
"""
from __future__ import annotations

import typing as t
import warnings

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

from defog_adapt.domains import molecule as _adapt
from defog_adapt.domains.molecule import MoleculeDomain, build_encoders, mol_to_smiles, needs_kekulize

__all__ = [
    "CHARGED_KEK_V1_ATOMS", "NEUTRAL_KEK_ATOMS", "KEKULE_BONDS", "build_encoders", "needs_kekulize", "mol_to_smiles",
    "atom_label", "atom_label_parts", "is_charged_vocab", "smiles_to_pyg_data", "pyg_data_to_mol", "ChargedMoleculeDomain",
    "make_domain",
]

#: the nine elements of the kekulized ZINC base, in its channel order
NEUTRAL_KEK_ATOMS: t.List[str] = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
#: the same nine, then the charged classes: N+ and O- carry nitro, N-oxide and quaternary nitrogen, N- the end of an azide.
#: With these three 99.4% of the Ames set is expressible (charge and element only), against 81.4% without them.
CHARGED_KEK_V1_ATOMS: t.List[str] = NEUTRAL_KEK_ATOMS + ["N+", "O-", "N-"]
#: bond classes of both kekulized vocabularies (class 0 is "no bond" and is not listed)
KEKULE_BONDS: t.List[str] = ["SINGLE", "DOUBLE", "TRIPLE"]


def atom_label_parts(label: str) -> t.Tuple[str, int]:
    """Split a vocabulary label into element symbol and formal charge: ``"N+" -> ("N", 1)``, ``"Cl" -> ("Cl", 0)``."""
    if label.endswith("+"):
        return label[:-1], 1
    if label.endswith("-"):
        return label[:-1], -1
    return label, 0


def atom_label(atom) -> str:
    """The vocabulary label of an RDKit atom: its symbol, plus ``+`` or ``-`` for a formal charge of +1 or -1.

    Any other charge gets an explicit ``+2``-style label, which no vocabulary contains.
    """
    charge = atom.GetFormalCharge()
    symbol = atom.GetSymbol()
    if charge == 0:
        return symbol
    return symbol + ("+" if charge == 1 else "-" if charge == -1 else f"{charge:+d}")


def is_charged_vocab(atom_labels: t.Iterable[str]) -> bool:
    """Whether a vocabulary has charged atom classes (and is therefore read strictly)."""
    return any(label.endswith(("+", "-")) for label in atom_labels)


def smiles_to_pyg_data(smiles: str, atom_encoder, bond_encoder, kekulize: bool = False) -> t.Optional[Data]:
    """Encode a SMILES string as a PyG ``Data`` with one-hot ``x`` and ``edge_attr`` (edge class 0 = no bond).

    For a neutral vocabulary this is ``defog_adapt``'s encoder. For a charged one atoms are keyed on symbol and charge
    (see the module docstring); the rest is the same encoding: bonds in both directions, sorted by ``row * N + col``.

    :returns: ``None`` if the molecule cannot be parsed, has an atom or bond the vocabulary does not contain, or has no bond.
    """
    if not is_charged_vocab(atom_encoder):
        return _adapt.smiles_to_pyg_data(smiles, atom_encoder, bond_encoder, kekulize=kekulize)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if kekulize:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException, ValueError):
            return None

    type_idx = []
    for atom in mol.GetAtoms():
        label = atom_label(atom)
        if label not in atom_encoder:
            return None
        type_idx.append(atom_encoder[label])
    n = len(type_idx)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        if bond_type not in bond_encoder:
            return None
        bond_idx = bond_encoder[bond_type] + 1                      # class 0 stays "no bond"
        row += [start, end]
        col += [end, start]
        edge_type += [bond_idx, bond_idx]
    if not row:
        return None

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = F.one_hot(torch.tensor(edge_type, dtype=torch.long), num_classes=len(bond_encoder) + 1).float()
    order = (edge_index[0] * n + edge_index[1]).argsort()
    x = F.one_hot(torch.tensor(type_idx), num_classes=len(atom_encoder)).float()
    return Data(x=x, edge_index=edge_index[:, order], edge_attr=edge_attr[order])


def pyg_data_to_mol(data, atom_decoder: t.List[str], bond_decoder: t.List,
                    charge_correction: t.Optional[bool] = None) -> t.Optional[Chem.Mol]:
    """Decode a PyG ``Data`` into an (unsanitized) RDKit molecule, or ``None`` if it cannot be built.

    ``charge_correction=None`` is the paper's setting: the relaxed reading (over-valent N, O, S get a +1 charge) for a
    vocabulary without charged classes, and no repair for one with them. For a charged vocabulary the formal charge of each
    atom is the one its class names, and a repair cannot be requested.
    """
    if not is_charged_vocab(atom_decoder):
        return _adapt.pyg_data_to_mol(data, atom_decoder, bond_decoder,
                                      charge_correction=True if charge_correction is None else charge_correction)
    if charge_correction:
        raise ValueError("a vocabulary with charged classes is read strictly: the model chooses the charge, so there is "
                         "no charge correction")

    symbols = [atom_label_parts(label)[0] for label in atom_decoder]
    mol = _adapt.pyg_data_to_mol(data, symbols, bond_decoder, charge_correction=False)
    if mol is None:
        return None
    try:
        data = data.cpu()
        classes = torch.argmax(data.x, dim=-1) if data.x.dim() == 2 else data.x.long()
        for i, cls in enumerate(classes.tolist()):
            charge = atom_label_parts(atom_decoder[cls])[1]
            if charge:
                mol.GetAtomWithIdx(i).SetFormalCharge(charge)
    except (ValueError, RuntimeError, IndexError) as exc:
        warnings.warn(f"pyg_data_to_mol: could not set the formal charges: {exc}")
        return None
    return mol


class ChargedMoleculeDomain(MoleculeDomain):
    """A ``MoleculeDomain`` that decodes with the strict charge-aware reading.

    The paper's Ames experiments pass it where the neutral domain would go: the objects that take a ``domain`` only call
    ``decode``, ``is_valid`` and ``identity``. Encoding a *core* for inpainting is not supported on a charged vocabulary,
    and neither are the neutral domain's distribution metrics (they would silently apply the relaxed repair).
    """

    def _valid_smiles(self, data) -> t.Optional[str]:
        mol = pyg_data_to_mol(data, self.atom_decoder, self.bond_decoder)
        smiles = mol_to_smiles(mol) if mol is not None else None
        if smiles is not None and Chem.MolFromSmiles(smiles) is not None:
            return smiles
        return None

    def _mol_to_core(self, mol, keep=None):
        raise NotImplementedError("inpainting cores are not supported on a vocabulary with charged classes")

    def metrics(self, samples, reference=None, compute_kl: bool = True):
        raise NotImplementedError("the distribution metrics of the neutral domain apply a charge repair that a charged "
                                  "vocabulary does not have; decode and score the samples with defog_megan.evaluate")


def make_domain(atom_decoder, bond_decoder, **kwargs) -> MoleculeDomain:
    """The molecule domain that reads a vocabulary correctly: strict for a charged one, defog-adapt's otherwise."""
    cls = ChargedMoleculeDomain if is_charged_vocab(atom_decoder) else MoleculeDomain
    return cls(atom_decoder=atom_decoder, bond_decoder=bond_decoder, **kwargs)
