"""The manifest of the model zoo: which files make up each group of models, with sizes and sha256 hashes.

A *group* is one experimental setting (synthetic colored graphs, molecules with Crippen logP, Ames mutagenicity). Its
*files* are the checkpoints, the explainer bundle (model, clustering, processing module, reports) and the small data
files that carry contracts (concept tables, request vectors, vocabularies). Each file has a ``ship`` level:

``core``       needed to use the models; part of the default download
``optional``   useful, not needed (illustrated reports, snapshots, controls)
``withheld``   not redistributed, because it holds molecule tables whose terms we do not control

``in_hand`` is False for a file that is part of the paper but not yet in the maintainer's local collection; its size and
hash come from an earlier record and it cannot be verified from here.

The weights are pickled PyTorch checkpoints, so loading one executes code. The manifest exists so that a file can be
checked against a hash *before* it is loaded.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Union

SHIP_LEVELS = ("core", "optional", "withheld")
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1 << 20


class ManifestError(ValueError):
    """The manifest is malformed."""


@dataclass(frozen=True)
class ZooFile:
    group: str
    id: str
    path: str          # relative, POSIX style, inside the group's bundle
    kind: str
    ship: str
    bytes: int
    sha256: str
    in_hand: bool
    note: str = ""


@dataclass
class Verification:
    """The outcome of checking a local copy against the manifest."""

    ok: List[ZooFile] = field(default_factory=list)
    missing: List[ZooFile] = field(default_factory=list)
    wrong_size: List[ZooFile] = field(default_factory=list)
    wrong_hash: List[ZooFile] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.missing or self.wrong_size or self.wrong_hash)

    def summary(self) -> str:
        parts = [f"{len(self.ok)} ok"]
        for label, items in (("missing", self.missing), ("wrong size", self.wrong_size), ("wrong hash", self.wrong_hash)):
            if items:
                parts.append(f"{len(items)} {label}: " + ", ".join(f"{f.group}/{f.path}" for f in items))
        return "; ".join(parts)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_path(path: str, where: str) -> None:
    posix = PurePosixPath(path)
    if not path or posix.is_absolute() or "\\" in path or ".." in posix.parts:
        raise ManifestError(f"{where}: {path!r} must be a relative POSIX path without '..'")


class Manifest:
    """The parsed, validated manifest."""

    def __init__(self, data: dict):
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ManifestError(f"unsupported schema_version {data.get('schema_version')!r}, expected {SCHEMA_VERSION}")
        self._titles: Dict[str, str] = {}
        self._files: List[ZooFile] = []
        #: group -> {"file", "bytes", "sha256"} of the archive that holds the group's core and optional files (see archive.py)
        self.archives: Dict[str, dict] = {}
        #: where the archives are served from, e.g. the assets of a release; may be overridden when fetching
        self.base_url: Optional[str] = (data.get("download") or {}).get("base_url")
        for group, entry in (data.get("archives") or {}).items():
            if not _SHA256.match(str(entry.get("sha256", ""))) or not isinstance(entry.get("bytes"), int):
                raise ManifestError(f"archive of {group}: needs 'bytes' (int) and 'sha256' (64 hex digits)")
            _check_path(entry.get("file", ""), f"archive of {group}")
            self.archives[group] = dict(entry)
        seen_ids, seen_paths = set(), set()
        for group, body in data.get("groups", {}).items():
            self._titles[group] = body.get("title", group)
            for entry in body.get("files", []):
                where = f"{group}/{entry.get('id', '?')}"
                if entry.get("ship") not in SHIP_LEVELS:
                    raise ManifestError(f"{where}: ship must be one of {SHIP_LEVELS}, got {entry.get('ship')!r}")
                if not _SHA256.match(str(entry.get("sha256", ""))):
                    raise ManifestError(f"{where}: sha256 must be 64 lowercase hex digits")
                size = entry.get("bytes")
                if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                    raise ManifestError(f"{where}: bytes must be a positive integer")
                _check_path(entry.get("path", ""), where)
                if (group, entry["id"]) in seen_ids:
                    raise ManifestError(f"{where}: duplicate id")
                if (group, entry["path"]) in seen_paths:
                    raise ManifestError(f"{where}: duplicate path {entry['path']!r}")
                seen_ids.add((group, entry["id"]))
                seen_paths.add((group, entry["path"]))
                self._files.append(ZooFile(group=group, id=entry["id"], path=entry["path"], kind=entry["kind"],
                                           ship=entry["ship"], bytes=size, sha256=entry["sha256"],
                                           in_hand=bool(entry.get("in_hand", True)), note=entry.get("note", "")))

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "Manifest":
        """The manifest shipped with the package, or the one at ``path``."""
        if path is None:
            text = resources.files("defog_megan.zoo").joinpath("manifest.json").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        return cls(json.loads(text))

    @property
    def groups(self) -> Dict[str, str]:
        """Group name -> title."""
        return dict(self._titles)

    def files(self, group: Optional[str] = None, ship: Optional[Iterable[str]] = None) -> List[ZooFile]:
        """The files of a group (or all), optionally only those with one of the given ship levels."""
        if group is not None and group not in self._titles:
            raise KeyError(f"unknown group {group!r}; have {sorted(self._titles)}")
        levels = None if ship is None else set(ship)
        return [f for f in self._files if (group is None or f.group == group) and (levels is None or f.ship in levels)]

    def get(self, group: str, id_: str) -> ZooFile:
        for f in self._files:
            if f.group == group and f.id == id_:
                return f
        raise KeyError(f"no file {id_!r} in group {group!r}")

    def total_bytes(self, group: Optional[str] = None, ship: Sequence[str] = ("core",)) -> int:
        return sum(f.bytes for f in self.files(group, ship))

    def verify(self, root: Union[str, Path], group: Optional[str] = None, ship: Sequence[str] = ("core",),
               check_hash: bool = True) -> Verification:
        """Check ``root/<group>/<path>`` for every selected file: present, the right size, the right sha256.

        The size is compared first because it is free; the hash is read only for files whose size matches.
        """
        root = Path(root)
        result = Verification()
        for f in self.files(group, ship):
            local = root / f.group / f.path
            if not local.is_file():
                result.missing.append(f)
            elif local.stat().st_size != f.bytes:
                result.wrong_size.append(f)
            elif check_hash and _sha256_of(local) != f.sha256:
                result.wrong_hash.append(f)
            else:
                result.ok.append(f)
        return result
