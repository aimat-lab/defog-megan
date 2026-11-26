"""``python -m defog_megan.zoo``: list the zoo, verify a local copy, fetch a group, pack a group (maintainers).

    python -m defog_megan.zoo list
    python -m defog_megan.zoo fetch colorgraphs                  # into $DEFOG_MEGAN_ZOO or ~/.cache/defog-megan/zoo
    python -m defog_megan.zoo verify colorgraphs --ship core optional
    python -m defog_megan.zoo pack ames --src /path/to/zoo --out dist/ --version 1.0.0 --update-manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from . import archive
from .layout import zoo_root
from .manifest import Manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m defog_megan.zoo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the groups and what each holds")
    for name, text in (("verify", "check a local copy against the manifest"), ("fetch", "download and unpack a group"),
                       ("pack", "build the archive of a group (maintainers)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("group", nargs="?" if name == "verify" else None)
        p.add_argument("--ship", nargs="+", default=list(archive.DEFAULT_SHIP if name != "verify" else ("core",)),
                       choices=["core", "optional"])
        if name != "pack":
            p.add_argument("--root", default=None, help="the zoo folder (default: $DEFOG_MEGAN_ZOO or the cache)")
        if name == "fetch":
            p.add_argument("--base-url", default=None, help="where the archives are served from (default: the manifest's)")
        if name == "pack":
            p.add_argument("--src", required=True, help="a zoo folder in the bundle layout that holds the files")
            p.add_argument("--out", required=True)
            p.add_argument("--version", required=True)
            p.add_argument("--update-manifest", action="store_true", help="record the archive in the manifest of the package")
    args = ap.parse_args(argv)
    manifest = Manifest.load()

    if args.command == "list":
        for group, title in manifest.groups.items():
            core, opt = manifest.total_bytes(group, ("core",)), manifest.total_bytes(group, ("optional",))
            print(f"{group:<12} {title:<28} core {core / 1e6:7.1f} MB   optional {opt / 1e6:6.1f} MB"
                  f"{'   (archive available)' if group in manifest.archives else ''}")
        return 0
    if args.command == "pack":
        entry = archive.pack(manifest, args.group, args.src, args.out, args.version, args.ship)
        print(json.dumps({args.group: entry}, indent=1))
        if args.update_manifest:
            path = Path(str(resources.files("defog_megan.zoo").joinpath("manifest.json")))
            data = json.loads(path.read_text())
            data.setdefault("archives", {})[args.group] = entry
            path.write_text(json.dumps(data, indent=1) + "\n")
            print(f"recorded in {path}")
        return 0

    root = Path(args.root) if args.root else zoo_root()
    if args.command == "fetch":
        report = archive.fetch(manifest, args.group, root, args.base_url, args.ship)
    else:
        report = manifest.verify(root, args.group, args.ship)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
