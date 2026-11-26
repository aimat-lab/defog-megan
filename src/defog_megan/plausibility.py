"""
Structural plausibility of a generated molecule.

The concept metrics live in MEGAN's explanation-embedding space, and a molecule
can score highly there while being chemically implausible. The visual grids made
this concrete: `ch0_cl7` reached 164x enrichment while producing strained polyaza
macrocycles that look nothing like its cluster members.

Measured over the generated sets, the cleanest discriminator is not a learned
score but ring size:

===========================  ======  =====================
set                            SA     % with a ring > 8
===========================  ======  =====================
real cluster members (672)     3.15   **0.0%**
unconditional baseline (748)   3.33   10.6%
conditioned generation         3.62   15-52%
===========================  ======  =====================

**Zero of 672 real cluster members contain a macrocycle.** So a ring larger than
8 atoms is essentially never a property of a genuine molecule in this vocabulary,
which makes it a near-false-positive-free signal.

SAScore was the obvious alternative and is the weaker instrument here: it is
*absolute*, so it punishes intrinsically complex concepts for their own chemistry
(`ch0_cl7`'s real members already sit at SA 4.06), and an SA <= 4.0 filter would
discard a fifth of the genuine members. It correlates with macrocyclicity only at
Spearman +0.645 -- related, but not the same signal.
"""

import typing as t

from rdkit import Chem

#: A ring larger than this counts as a macrocycle. The ZINC chemistry is
#: dominated by 5- and 6-membered rings; the real cluster members top out at 6.
MACRO_RING_SIZE = 8

#: Penalty is capped: two macrocycles are already decisively implausible, and an
#: uncapped count would let ring geometry swamp the concept signal entirely.
MACRO_CAP = 2


def count_macrocycles(mol, ring_size: int = MACRO_RING_SIZE) -> int:
    """Number of rings larger than ``ring_size`` atoms."""
    if mol is None:
        return 0
    return sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) > ring_size)


def macrocycle_penalty(mol, weight: float, ring_size: int = MACRO_RING_SIZE,
                       cap: int = MACRO_CAP) -> float:
    """``weight * min(n_macrocycles, cap)`` -- a non-negative penalty.

    Graded rather than a hard tier so the search still has a gradient: a molecule
    with one macrocycle should outrank one with three, and a hard gate would make
    every implausible molecule equally bad and give the policy nothing to descend.
    """
    return float(weight) * min(count_macrocycles(mol, ring_size), cap)


#: Rings this small are strained. Three- and four-membered rings do occur in real
#: chemistry (epoxides, azetidines), so this is a soft penalty rather than a gate.
STRAIN_RING_SIZE = 4

#: Cap for the same reason as MACRO_CAP: keep a gradient, avoid swamping the
#: concept signal once a molecule is already decisively strained.
STRAIN_CAP = 3


def count_strained_rings(mol, ring_size: int = STRAIN_RING_SIZE) -> int:
    """Number of rings of ``ring_size`` atoms or fewer."""
    if mol is None:
        return 0
    return sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) <= ring_size)


def count_bridged_atoms(mol) -> int:
    """Atoms shared by three or more rings -- the signature of a caged fusion.

    An atom in two rings is an ordinary fused bicyclic (naphthalene, indole). Three
    or more is a bridgehead in a caged polycycle, which is where the contorted
    structures in the grids come from.
    """
    if mol is None:
        return 0
    info = mol.GetRingInfo()
    return sum(1 for atom in range(mol.GetNumAtoms()) if info.NumAtomRings(atom) >= 3)


def strain_penalty(mol, weight: float, ring_size: int = STRAIN_RING_SIZE,
                   cap: int = STRAIN_CAP) -> float:
    """``weight * min(n_strained_rings + n_bridged_atoms, cap)``.

    Counts small rings and caged bridgeheads together: both are ways of being
    geometrically contorted, and separating them would mean two more weights to
    justify without any evidence that they need different prices.
    """
    if mol is None:
        return 0.0
    count = count_strained_rings(mol, ring_size) + count_bridged_atoms(mol)
    return float(weight) * min(count, cap)


def strain_fraction(smiles_list: t.Sequence[str]) -> float:
    """Fraction of a set carrying a strained ring or a caged bridgehead."""
    total = valid = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        valid += 1
        total += (count_strained_rings(mol) + count_bridged_atoms(mol)) > 0
    return (total / valid) if valid else 0.0


def count_macrocycles_smiles(smiles: str, ring_size: int = MACRO_RING_SIZE) -> int:
    """Convenience wrapper for when only the SMILES is to hand."""
    return count_macrocycles(Chem.MolFromSmiles(smiles), ring_size)


def macrocycle_fraction(smiles_list: t.Sequence[str],
                        ring_size: int = MACRO_RING_SIZE) -> float:
    """Fraction of a set carrying at least one macrocycle (reporting helper)."""
    total = valid = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        valid += 1
        total += count_macrocycles(mol, ring_size) > 0
    return (total / valid) if valid else 0.0
