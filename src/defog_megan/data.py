"""
Serialization for the conditioning table.

Deliberately ``.npz`` rather than parquet: this file has to be readable on machines whose module stack ships pandas
without ``pyarrow`` (an HPC node, an ARM host), and installing a parquet engine into a shared environment to move a table
of floats and strings is a bad trade. ``numpy`` is the one dependency guaranteed everywhere.

SMILES are stored as fixed-width unicode arrays rather than object arrays, so
loading never needs ``allow_pickle=True``.
"""

import typing as t

import numpy as np


def save_conditions(path: str, smiles, phi, concept_ids, smiles_original=None) -> str:
    """Write the ``(molecule, φ)`` table.

    :param path: Destination ``.npz`` path.
    :param smiles: The molecules the conditions describe -- i.e. the ones an
        adapter will actually be trained on.
    :param phi: ``(N, n_concepts)`` membership matrix.
    :param concept_ids: Column layout of ``phi``; stored so a consumer can fail
        loudly on a mismatch instead of training against a permuted vector.
    :param smiles_original: Optional pre-round-trip SMILES, kept for provenance.
    """
    arrays = {
        "smiles": np.asarray(list(smiles), dtype=np.str_),
        "phi": np.asarray(phi, dtype=np.float32),
        "concept_ids": np.asarray(list(concept_ids), dtype=np.str_),
    }
    if smiles_original is not None:
        arrays["smiles_original"] = np.asarray(list(smiles_original), dtype=np.str_)
    np.savez_compressed(path, **arrays)
    return path


def load_conditions(path: str) -> t.Tuple[t.List[str], np.ndarray, t.List[str]]:
    """Read the table back.

    :returns: ``(smiles, phi, concept_ids)``.
    """
    bundle = np.load(path)
    return (
        [str(s) for s in bundle["smiles"]],
        np.asarray(bundle["phi"], dtype=np.float32),
        [str(c) for c in bundle["concept_ids"]],
    )


def load_targets(path: str, mode: str = "empirical") -> t.Tuple[np.ndarray, t.List[str]]:
    """Read the per-concept target vectors exported by ``export_concept_targets``.

    :returns: ``(matrix, concept_ids)`` where row ``i`` is the condition vector
        that asks for ``concept_ids[i]``.
    """
    bundle = np.load(path)
    key = f"targets_{mode}"
    if key not in bundle:
        available = [k[len("targets_"):] for k in bundle.files if k.startswith("targets_")]
        raise KeyError(f"target mode {mode!r} not in {path} (have: {available})")
    return np.asarray(bundle[key], dtype=np.float32), [str(c) for c in bundle["concept_ids"]]
