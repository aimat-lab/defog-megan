"""Run an experiment twice, as the research code and as this repository, and compare what it generated.

The old run uses the research library and the research package at the revision the paper's runs used (both are
extracted outside this repository); the new run uses this repository and ``defog_adapt``. Both run in the SAME Python
environment (same torch build, same explainer stack, one thread) with identical arguments and seeds, so a difference
can only come from the port. Each run happens in a scratch copy of the experiment, so neither writes into a checkout.

    python tests/parity/experiment_parity.py sample_color_joint --old-library <research repo> \\
        --reference <extracted research package> --args "--__TESTING__ True --STAGE \\"'probe'\\""

Compared: every ``gen_*.npz`` array for array (the graph experiments), every ``gen_*.json`` SMILES list (the molecular
ones), and the manifest apart from timings, the device and the keys only one side has.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
VOLATILE = {"seconds", "seconds_per_sample", "device", "duration", "start_time", "end_time"}


def _stable(node):
    """A manifest without the keys that legitimately differ between two runs."""
    if isinstance(node, dict):
        return {k: _stable(v) for k, v in node.items() if k not in VOLATILE}
    if isinstance(node, list):
        return [_stable(v) for v in node]
    return node


def run(script: Path, args: list, scratch: Path, pythonpath: list, zoo: str) -> Path:
    """Run one experiment from a scratch copy; return its result folder."""
    target = scratch / "experiments"
    target.mkdir(parents=True)
    shutil.copy(script, target / script.name)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
           "POLARS_SKIP_CPU_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1", "DEFOG_MEGAN_ZOO": zoo,
           "PYTHONPATH": os.pathsep.join(pythonpath)}
    done = subprocess.run([sys.executable, str(target / script.name), *args], cwd=scratch, env=env,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{script.name} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    folders = sorted((target / "results").glob(f"{script.stem}/*"))
    folders = [f for f in folders if f.is_dir()]
    if len(folders) != 1:
        raise RuntimeError(f"expected one result folder for {script.name}, found {folders}")
    return folders[0]


def _same_array(a, b) -> bool:
    """Exact equality, with NaN equal to NaN (the dumps carry NaN targets)."""
    if a.dtype == object or b.dtype == object:
        return a.shape == b.shape and all(_same_array(np.asarray(x), np.asarray(y)) for x, y in zip(a.ravel(), b.ravel()))
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def compare(old: Path, new: Path) -> list:
    """The differences between two result folders (empty when the runs agree)."""
    problems = []
    old_files = {p.name for p in old.glob("gen_*.npz")} | {p.name for p in old.glob("gen_*.json")}
    new_files = {p.name for p in new.glob("gen_*.npz")} | {p.name for p in new.glob("gen_*.json")}
    if old_files != new_files:
        problems.append(f"different generated files: only old {sorted(old_files - new_files)}, "
                        f"only new {sorted(new_files - old_files)}")
    for name in sorted(n for n in old_files & new_files if n.endswith(".json")):
        a, b = json.loads((old / name).read_text()), json.loads((new / name).read_text())
        if a["smiles"] != b["smiles"]:
            n_diff = sum(x != y for x, y in zip(a["smiles"], b["smiles"])) + abs(len(a["smiles"]) - len(b["smiles"]))
            problems.append(f"{name}: {n_diff} of {len(a['smiles'])} SMILES differ")
    for name in sorted(n for n in old_files & new_files if n.endswith(".npz")):
        a, b = np.load(old / name, allow_pickle=True), np.load(new / name, allow_pickle=True)
        if set(a.files) != set(b.files):
            problems.append(f"{name}: different arrays {sorted(set(a.files) ^ set(b.files))}")
            continue
        bad = [k for k in a.files if not _same_array(a[k], b[k])]
        if bad:
            problems.append(f"{name}: arrays differ: {bad}")
    if (old / "manifest.json").exists() and (new / "manifest.json").exists():
        ma, mb = (_stable(json.loads((d / "manifest.json").read_text())) for d in (old, new))
        for key in sorted(set(ma) & set(mb)):
            if ma.get(key) != mb.get(key) and key not in IGNORED_MANIFEST_KEYS:
                problems.append(f"manifest key {key!r} differs")
    return problems


#: keys the port deliberately changed: the new manifest records the resampling scheme (the old one did not)
IGNORED_MANIFEST_KEYS = {"fk"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", help="file name of the experiment without .py, as in experiments/")
    ap.add_argument("--old-library", required=True, type=Path, help="checkout of the research library (contains experiments/)")
    ap.add_argument("--reference", required=True, type=Path, help="directory holding the extracted research package `defog/`")
    ap.add_argument("--zoo", default=os.environ.get("DEFOG_MEGAN_ZOO", ""), help="the model zoo")
    ap.add_argument("--new-name", default=None, help="file name of the experiment in this repository, if it was renamed")
    ap.add_argument("--args", default="", help="the experiment's arguments, as one shell-quoted string")
    ap.add_argument("--old-args", default=None, help="arguments for the old run only, when its parameter names differ")
    ap.add_argument("--keep", type=Path, help="keep both result folders in this directory")
    ap.add_argument("--control", choices=["old", "new"], default=None,
                    help="run only that side, twice, to measure how deterministic it is on its own")
    opts = ap.parse_args()
    args = shlex.split(opts.args)
    old_args = shlex.split(opts.old_args) if opts.old_args is not None else args
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old_script, old_path = opts.old_library / "experiments" / f"{opts.experiment}.py", [str(opts.reference), str(opts.old_library)]
        new_script, new_path = REPO / "experiments" / f"{opts.new_name or opts.experiment}.py", [str(REPO / "src")]
        if opts.control == "old":
            new_script, new_path, args = old_script, old_path, old_args
        elif opts.control == "new":
            old_script, old_path, old_args = new_script, new_path, args
        old = run(old_script, old_args, tmp / "old", old_path, opts.zoo)
        new = run(new_script, args, tmp / "new", new_path, opts.zoo)
        problems = compare(old, new)
        n = len(list(old.glob("gen_*.npz"))) + len(list(old.glob("gen_*.json")))
        if opts.keep:
            shutil.copytree(old, opts.keep / "old", dirs_exist_ok=True)
            shutil.copytree(new, opts.keep / "new", dirs_exist_ok=True)
    print(f"{opts.experiment}: compared {n} generated files -> " + ("IDENTICAL" if not problems else "DIFFERENT"))
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
