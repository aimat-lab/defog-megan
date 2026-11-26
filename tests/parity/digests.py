"""Digests of what an experiment generated, comparable across machines that run the same torch build.

A ``gen_*.npz`` file (graphs of the colored-graph experiments) is digested array by array (name, dtype, shape, bytes; NaN
bytes are the same bytes), a ``gen_*.json`` file (SMILES of the molecular experiments) by its list of SMILES.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest_file(path) -> str:
    path = Path(path)
    if path.suffix == ".json":
        return hashlib.sha256(json.dumps(json.loads(path.read_text())["smiles"]).encode()).hexdigest()
    h = hashlib.sha256()
    with np.load(path, allow_pickle=True) as data:
        for name in sorted(data.files):
            array = data[name]
            h.update(name.encode())
            if array.dtype == object:
                h.update(json.dumps(array.tolist(), default=str).encode())
            else:
                h.update(str(array.dtype).encode() + str(array.shape).encode() + np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def digest_run(folder) -> dict:
    """``{file name: digest}`` of every generated file of one result folder."""
    folder = Path(folder)
    return {p.name: digest_file(p) for p in sorted(folder.glob("gen_*")) if p.suffix in (".npz", ".json") and ".phi" not in p.name}
