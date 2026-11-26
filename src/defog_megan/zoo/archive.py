"""One archive per group of the zoo: packing it (the maintainer's side) and fetching and unpacking it (the user's side).

The archive is an uncompressed tar (checkpoints do not compress) holding ``<group>/<path>`` for every core and optional file of
the group, written with fixed times and owners so that packing the same files twice gives the same bytes. Its size and sha256
are recorded in the manifest (``archives``), the place it is served from in ``download.base_url``. A fetch checks the archive
against the manifest before it unpacks anything, unpacks only the members the manifest lists (never a path of its own
choosing), and then verifies every file again against the manifest.

The weights are pickled checkpoints and the explainer's processing module is Python code: a hash that matches the manifest
shows that the files are those of the release, not that the release is safe to run.
"""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Sequence, Union

from .manifest import Manifest, Verification, ZooFile

_CHUNK = 1 << 20
DEFAULT_SHIP = ("core", "optional")


class ArchiveError(RuntimeError):
    """An archive is missing from the manifest, does not match it, or holds something the manifest does not list."""


def archive_name(group: str, version: str) -> str:
    return f"defog-megan-zoo-{group}-{version}.tar"


def pack(manifest: Manifest, group: str, src_root: Union[str, Path], out_dir: Union[str, Path], version: str,
         ship: Sequence[str] = DEFAULT_SHIP) -> dict:
    """Write the archive of one group from ``src_root/<group>/<path>``; return its ``{"file", "bytes", "sha256"}`` entry.

    The files are verified against the manifest first, so that only files the manifest describes are packed.
    """
    src_root, out_dir = Path(src_root), Path(out_dir)
    report = manifest.verify(src_root, group, ship)
    if not report.passed:
        raise ArchiveError(f"the files of {group} under {src_root} do not match the manifest: {report.summary()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / archive_name(group, version)
    with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as tar:
        for f in sorted(manifest.files(group, ship), key=lambda f: f.path):
            local = src_root / f.group / f.path
            info = tarfile.TarInfo(f"{f.group}/{f.path}")
            info.size, info.mtime, info.mode = local.stat().st_size, 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(local, "rb") as handle:
                tar.addfile(info, handle)
    return {"file": target.name, "bytes": target.stat().st_size, "sha256": _sha256(target)}


def fetch(manifest: Manifest, group: str, root: Union[str, Path], base_url: Optional[str] = None,
          ship: Sequence[str] = DEFAULT_SHIP) -> Verification:
    """Download the archive of ``group``, check it against the manifest, unpack it into ``root/<group>/`` and verify the files.

    ``base_url`` (default: the manifest's ``download.base_url``) may be a ``file://`` URL, which is how the archives are
    tested and how a mirror on a local disk is used.
    """
    entry = manifest.archives.get(group)
    if entry is None:
        raise ArchiveError(f"the manifest has no archive for group {group!r}; has {sorted(manifest.archives)}")
    base_url = base_url or manifest.base_url
    if not base_url:
        raise ArchiveError("no download location: the manifest has none and no base_url was given")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    wanted = {f"{f.group}/{f.path}": f for f in manifest.files(group, ship)}
    with tempfile.TemporaryDirectory(dir=root, prefix=".fetch-") as tmp:
        archive = Path(tmp) / entry["file"]
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/{entry['file']}") as response, open(archive, "wb") as handle:
            shutil.copyfileobj(response, handle, _CHUNK)
        if archive.stat().st_size != entry["bytes"] or _sha256(archive) != entry["sha256"]:
            raise ArchiveError(f"{entry['file']} does not match the manifest (size or sha256); nothing was unpacked")
        _unpack(archive, root, wanted)
    return manifest.verify(root, group, ship)


def _unpack(archive: Path, root: Path, wanted: "dict[str, ZooFile]") -> None:
    with tarfile.open(archive) as tar:
        names = set()
        for member in tar.getmembers():
            if not member.isreg() or member.name not in wanted:
                raise ArchiveError(f"the archive holds {member.name!r}, which the manifest does not list")
            names.add(member.name)
        missing = sorted(set(wanted) - names)
        if missing:
            raise ArchiveError(f"the archive lacks {missing[:3]}{'...' if len(missing) > 3 else ''}")
        for member in tar.getmembers():
            target = root / member.name                                  # the name is a manifest path, checked above
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink, _CHUNK)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
