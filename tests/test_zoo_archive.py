"""Packing a group of the zoo into one archive, and fetching it back: round trip and refusals."""
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from defog_megan.zoo import Manifest, archive
from defog_megan.zoo.__main__ import main as zoo_main

FILES = {"a.bin": (b"alpha" * 100, "core"), "sub/b.json": (b'{"b": 1}', "core"), "c.pdf": (b"pdf" * 50, "optional"),
         "w.csv": (b"smiles\nCCO\n", "withheld")}


def make_zoo(tmp_path):
    root = tmp_path / "src"
    entries = []
    for i, (path, (data, ship)) in enumerate(sorted(FILES.items())):
        target = root / "g" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entries.append({"id": f"F{i}", "path": path, "kind": "x", "ship": ship, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    return root, {"schema_version": 1, "groups": {"g": {"title": "G", "files": entries}}}


def served(tmp_path, data_manifest, entry_file, content):
    """A folder to download from, holding ``entry_file``, and a manifest that records exactly that file."""
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    (site / entry_file).write_bytes(content)
    data = dict(data_manifest)
    data["archives"] = {"g": {"file": entry_file, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}
    return Manifest(data), site.as_uri()


def test_the_archive_holds_the_core_and_optional_files_and_is_reproducible(tmp_path):
    root, data = make_zoo(tmp_path)
    manifest = Manifest(data)
    first = archive.pack(manifest, "g", root, tmp_path / "out1", "9.9.9")
    second = archive.pack(manifest, "g", root, tmp_path / "out2", "9.9.9")
    assert first == second and first["file"] == "defog-megan-zoo-g-9.9.9.tar"
    with tarfile.open(tmp_path / "out1" / first["file"]) as tar:
        assert sorted(tar.getnames()) == ["g/a.bin", "g/c.pdf", "g/sub/b.json"]       # not the withheld table


def test_a_fetch_unpacks_exactly_what_the_manifest_lists(tmp_path):
    root, data = make_zoo(tmp_path)
    packed = archive.pack(Manifest(data), "g", root, tmp_path / "site", "1")
    data["archives"] = {"g": packed}
    manifest = Manifest(data)
    destination = tmp_path / "zoo"
    report = archive.fetch(manifest, "g", destination, (tmp_path / "site").as_uri())
    assert report.passed and len(report.ok) == 3
    assert (destination / "g" / "sub" / "b.json").read_bytes() == FILES["sub/b.json"][0]
    assert not (destination / "g" / "w.csv").exists()


def test_a_tampered_archive_is_refused_before_anything_is_unpacked(tmp_path):
    root, data = make_zoo(tmp_path)
    packed = archive.pack(Manifest(data), "g", root, tmp_path / "site", "1")
    data["archives"] = {"g": packed}
    path = tmp_path / "site" / packed["file"]
    content = bytearray(path.read_bytes())
    content[2000] ^= 0xFF                                         # one flipped byte, same size
    path.write_bytes(bytes(content))
    destination = tmp_path / "zoo"
    with pytest.raises(archive.ArchiveError, match="does not match the manifest"):
        archive.fetch(Manifest(data), "g", destination, (tmp_path / "site").as_uri())
    assert not (destination / "g").exists()


def _tar_bytes(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize("extra", ["g/not-in-the-manifest.bin", "../escape.bin", "/abs/path.bin"])
def test_a_member_the_manifest_does_not_list_is_refused(tmp_path, extra):
    root, data = make_zoo(tmp_path)
    members = {f"g/{p}": FILES[p][0] for p in ("a.bin", "sub/b.json", "c.pdf")}
    members[extra] = b"payload"
    manifest, url = served(tmp_path, data, "evil.tar", _tar_bytes(members))     # a hash that matches: only the content is wrong
    destination = tmp_path / "zoo"
    with pytest.raises(archive.ArchiveError, match="does not list"):
        archive.fetch(manifest, "g", destination, url)
    assert not (tmp_path / "escape.bin").exists() and not (destination / "g").exists()


def test_an_archive_that_lacks_a_listed_file_is_refused(tmp_path):
    root, data = make_zoo(tmp_path)
    members = {f"g/{p}": FILES[p][0] for p in ("a.bin", "sub/b.json")}              # c.pdf is missing
    manifest, url = served(tmp_path, data, "short.tar", _tar_bytes(members))
    with pytest.raises(archive.ArchiveError, match="lacks"):
        archive.fetch(manifest, "g", tmp_path / "zoo", url)


def test_packing_refuses_a_source_that_does_not_match_the_manifest(tmp_path):
    root, data = make_zoo(tmp_path)
    (root / "g" / "a.bin").write_bytes(b"changed" * 10)
    with pytest.raises(archive.ArchiveError, match="do not match the manifest"):
        archive.pack(Manifest(data), "g", root, tmp_path / "out", "1")


def test_a_fetch_needs_a_location_and_an_archive(tmp_path):
    root, data = make_zoo(tmp_path)
    with pytest.raises(archive.ArchiveError, match="no archive"):
        archive.fetch(Manifest(data), "g", tmp_path / "zoo", "file:///nowhere")
    packed = archive.pack(Manifest(data), "g", root, tmp_path / "site", "1")
    data["archives"] = {"g": packed}
    with pytest.raises(archive.ArchiveError, match="no download location"):
        archive.fetch(Manifest(data), "g", tmp_path / "zoo")


def test_the_command_line_lists_the_groups(capsys):
    assert zoo_main(["list"]) == 0
    out = capsys.readouterr().out
    assert "colorgraphs" in out and "ames" in out and "zinc-logp" in out
