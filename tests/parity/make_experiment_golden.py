"""Write ``golden/experiments.json`` from the result folders of the RESEARCH stack.

Each case is run by ``experiment_parity.py`` (research code and this repository, same environment, same arguments); when it
reports IDENTICAL, the research side's result folder (kept with ``--keep``) is the golden. This turns those folders into digests:

    python tests/parity/make_experiment_golden.py --case colorgraphs/probe=<folder> --case ames/fk=<folder> ...

The digests are bit-exact only within one torch build on one CPU architecture; the file records the torch version, and the gate
skips on another one.
"""
import argparse
import json
import platform
from pathlib import Path

import torch

from cases_experiments import CASES
from digests import digest_run

OUT = Path(__file__).resolve().parent / "golden" / "experiments.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", required=True, metavar="NAME=FOLDER")
    args = ap.parse_args()
    cases = {}
    for spec in args.case:
        name, _, folder = spec.partition("=")
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}; have {sorted(CASES)}")
        files = digest_run(folder)
        if not files:
            raise SystemExit(f"{folder} holds no generated files")
        experiment, arguments = CASES[name]
        manifest = json.loads((Path(folder) / "manifest.json").read_text())
        cases[name] = {"experiment": experiment, "arguments": arguments, "files": files,
                       "energy_calls": {u["key"]: u.get("energy_calls") for u in manifest.get("units", [])
                                        if u.get("energy_calls")}}
    OUT.write_text(json.dumps({
        "format": 1,
        "source": "result folders of the research code (see tests/parity/README.md)",
        "environment": {"torch": torch.__version__, "python": platform.python_version(), "threads": 1, "device": "cpu"},
        "cases": dict(sorted(cases.items())),
    }, indent=1) + "\n")
    print(f"wrote {OUT}: {len(cases)} cases, {sum(len(c['files']) for c in cases.values())} files")


if __name__ == "__main__":
    main()
