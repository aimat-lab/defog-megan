"""Generate (or check) the golden file of the colored-graph generator-side cases.

Run it under the RESEARCH stack to create the golden file, at the revision the paper's runs used:

    PYTHONPATH=<reference stack> <research python> tests/parity/generate_golden.py \\
        --pkg defog --zoo "$DEFOG_MEGAN_ZOO" --revision <revision> --tier quick

or run it under ``defog_adapt`` with ``--check`` to compare against the golden file without writing anything. See
``tests/parity/README.md`` and ``tests/parity/make_golden.sh`` for the whole recipe.

The golden file records, per tier, the cases' digests, and once: the sha256 of every input checkpoint, the explicit
settings and the environment (the digests are bit-exact only within one torch build on one CPU architecture, so the
build is recorded). Generating a second tier into the same file is allowed only from the same revision, inputs and
environment.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases  # noqa: E402  (after the path tweak)

FORMAT = 2


def environment() -> dict:
    import numpy
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
            "threads": 1, "machine": platform.machine(), "system": platform.system()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkg", required=True, choices=["defog", "defog_adapt"])
    parser.add_argument("--zoo", required=True, type=Path, help="root of the model zoo (group folders inside)")
    parser.add_argument("--tier", default="quick", choices=sorted(cases.TIERS))
    parser.add_argument("--golden", type=Path, default=HERE / "golden" / "colorgraphs.json")
    parser.add_argument("--check", action="store_true", help="compare with the golden file instead of writing")
    parser.add_argument("--revision", help="revision of the reference stack (recorded in the golden file)")
    parser.add_argument("--reference-dir", type=Path,
                        help="directory the reference package must be imported from (a guard against importing "
                             "another checkout of the same package name)")
    parser.add_argument("--cases", help="comma-separated subset (check mode only)")
    args = parser.parse_args()

    names = args.cases.split(",") if args.cases else list(cases.CASES)
    unknown = [n for n in names if n not in cases.CASES]
    if unknown:
        parser.error(f"unknown case(s): {unknown}; have {list(cases.CASES)}")
    if args.cases and not args.check:
        parser.error("a golden tier must contain every case; --cases is for --check only")
    if not args.check and args.revision is None:
        parser.error("generation mode needs --revision")

    runner = cases.Runner(args.pkg, args.zoo)
    package_file = Path(sys.modules[args.pkg].__file__).resolve()
    if args.reference_dir is not None and args.reference_dir.resolve() not in package_file.parents:
        print(f"refusing: {args.pkg} was imported from {package_file.parent.parent}, not from "
              f"{args.reference_dir.resolve()}. The wrong checkout trains and samples a different model silently.")
        return 2

    results = {}
    for name in names:
        started = time.time()
        results[name] = runner.run(name, args.tier)
        print(f"[{args.tier}] {name:<20} {results[name]['digest'][:16]}  sizes {results[name]['sizes']}  "
              f"({time.time() - started:.0f} s)", flush=True)

    if args.check:
        golden = json.loads(args.golden.read_text())
        expected = golden["tiers"][args.tier]["cases"]
        bad = [n for n in names if {k: results[n][k] for k in ("digest", "sizes")} != {k: expected[n][k] for k in ("digest", "sizes")}]
        print("MATCH" if not bad else f"MISMATCH in: {bad}")
        return 1 if bad else 0

    env, inputs = environment(), cases.input_hashes(args.zoo)
    if args.golden.exists():
        golden = json.loads(args.golden.read_text())
        same = (golden["environment"] == env and golden["inputs"] == inputs
                and golden["reference"]["revision"] == args.revision and golden["settings"] == cases.SETTINGS)
        if not same:
            print("refusing to merge into the existing golden file: its revision, environment, inputs or settings "
                  "differ. Delete it to regenerate every tier.")
            return 2
    else:
        golden = {"format": FORMAT, "group": cases.GROUP,
                  "reference": {"package": args.pkg, "revision": args.revision,
                                "what": "the research package at the revision the paper's colored-graph runs used"},
                  "environment": env, "inputs": inputs, "settings": cases.SETTINGS, "tiers": {}}
    golden["tiers"][args.tier] = {
        "config": cases.TIERS[args.tier],
        "cases": {n: {**results[n], "pins": cases.CASES[n][1]} for n in cases.CASES},
    }
    args.golden.parent.mkdir(parents=True, exist_ok=True)
    args.golden.write_text(json.dumps(golden, indent=1) + "\n")
    print("wrote", args.golden, "tiers:", sorted(golden["tiers"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
