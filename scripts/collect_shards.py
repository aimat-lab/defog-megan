"""Collect the run folders of one sharded sampling stage into ``<out>/shard<k>/``, the layout the scorers read.

A sampling experiment run with ``--SHARD k --NUM_SHARDS n`` writes one PyComex run folder per shard
(``experiments/results/<experiment>/<time>__<id>/``) holding ``manifest.json`` and the ``gen_*`` files. This copies the
manifest and the generated files of every shard of one stage into ``<out>/shard<k>/``, reading the stage and the shard
number from the manifests, so that concurrent runs of other stages in the same folder are not mixed up.

    python scripts/collect_shards.py sample_color_joint --stage probe --num-shards 4 \\
        --since "$START" --out results/color_joint/probe_n1024

``--since`` (unix time) restricts the search to run folders written after a start time, so that an earlier run of the same
stage is not picked up; without it the newest folder of every shard is taken.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", help="name of the experiment, e.g. sample_ames_joint")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--since", type=float, default=0.0, help="only run folders whose manifest is newer than this unix time")
    ap.add_argument("--results", type=Path, default=Path("experiments/results"))
    args = ap.parse_args()

    found = {}
    for manifest_path in sorted((args.results / args.experiment).glob("*/manifest.json"), key=lambda p: (p.stat().st_mtime, str(p))):
        if manifest_path.stat().st_mtime < args.since:
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("stage") != args.stage or manifest.get("num_shards") != args.num_shards:
            continue
        found[int(manifest["shard"])] = manifest_path.parent          # a later run of the same shard replaces an earlier one

    missing = [k for k in range(args.num_shards) if k not in found]
    if missing:
        print(f"missing shards {missing} of stage {args.stage!r} under {args.results / args.experiment}", file=sys.stderr)
        return 1
    for shard, folder in sorted(found.items()):
        target = args.out / f"shard{shard}"
        target.mkdir(parents=True, exist_ok=True)
        for source in [folder / "manifest.json", *sorted(folder.glob("gen_*"))]:
            shutil.copy2(source, target / source.name)
        print(f"shard {shard}: {folder.name} -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
