"""The verifier and the manifest's validation, on small synthetic manifests (no real weights involved)."""
import hashlib
import json

import pytest

from defog_megan.zoo import Manifest, ManifestError


def entry(id_, path, content, ship="core", **kw):
    return {"id": id_, "path": path, "kind": "adapter", "ship": ship, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(), "in_hand": True, "note": "", **kw}


def build(tmp_path, files):
    """Write ``files`` (path -> bytes) under tmp_path/g/ and return the manifest that describes them."""
    entries = []
    for i, (path, content) in enumerate(files.items()):
        target = tmp_path / "g" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append(entry(f"F{i}", path, content))
    return Manifest({"schema_version": 1, "groups": {"g": {"title": "G", "requires": [], "files": entries}}})


def test_a_correct_copy_passes(tmp_path):
    manifest = build(tmp_path, {"a.bin": b"alpha", "sub/b.bin": b"beta beta"})
    result = manifest.verify(tmp_path)
    assert result.passed and len(result.ok) == 2 and not (result.missing or result.wrong_size or result.wrong_hash)


def test_a_missing_file_is_reported(tmp_path):
    manifest = build(tmp_path, {"a.bin": b"alpha", "b.bin": b"beta"})
    (tmp_path / "g" / "b.bin").unlink()
    result = manifest.verify(tmp_path)
    assert not result.passed
    assert [f.path for f in result.missing] == ["b.bin"] and [f.path for f in result.ok] == ["a.bin"]


def test_a_file_of_the_wrong_size_is_reported_without_reading_it(tmp_path):
    manifest = build(tmp_path, {"a.bin": b"alpha"})
    (tmp_path / "g" / "a.bin").write_bytes(b"alphabet")
    result = manifest.verify(tmp_path)
    assert [f.path for f in result.wrong_size] == ["a.bin"] and not result.wrong_hash


def test_a_file_of_the_right_size_and_different_content_fails_the_hash(tmp_path):
    """The case a size check cannot catch: a corrupted or swapped file of identical length."""
    manifest = build(tmp_path, {"a.bin": b"alpha"})
    (tmp_path / "g" / "a.bin").write_bytes(b"omega")
    result = manifest.verify(tmp_path)
    assert not result.passed and [f.path for f in result.wrong_hash] == ["a.bin"]
    assert manifest.verify(tmp_path, check_hash=False).passed        # sizes alone cannot see it


def test_only_the_selected_ship_levels_are_checked(tmp_path):
    entries = [entry("A", "core.bin", b"core"), entry("B", "opt.bin", b"opt", ship="optional"),
               entry("C", "held.bin", b"held", ship="withheld")]
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "core.bin").write_bytes(b"core")
    manifest = Manifest({"schema_version": 1, "groups": {"g": {"files": entries}}})
    assert manifest.verify(tmp_path).passed                               # default: core only
    assert not manifest.verify(tmp_path, ship=("core", "optional")).passed
    assert [f.id for f in manifest.files(ship=("withheld",))] == ["C"]
    assert manifest.total_bytes(ship=("core", "optional")) == len(b"core") + len(b"opt")


def test_the_summary_names_what_is_wrong(tmp_path):
    manifest = build(tmp_path, {"a.bin": b"alpha", "b.bin": b"beta"})
    (tmp_path / "g" / "b.bin").unlink()
    assert "1 ok" in manifest.verify(tmp_path).summary() and "g/b.bin" in manifest.verify(tmp_path).summary()


def test_an_unknown_group_is_an_error_not_an_empty_list(tmp_path):
    manifest = build(tmp_path, {"a.bin": b"alpha"})
    with pytest.raises(KeyError):
        manifest.verify(tmp_path, group="nope")


@pytest.mark.parametrize("mutation, message", [
    ({"sha256": "XYZ"}, "sha256"),
    ({"sha256": "A" * 64}, "sha256"),                      # upper case is not what the tools print
    ({"bytes": 0}, "bytes"),
    ({"bytes": True}, "bytes"),
    ({"ship": "everything"}, "ship"),
    ({"path": "/abs/a.bin"}, "relative"),
    ({"path": "../a.bin"}, "relative"),
    ({"path": "dir\\a.bin"}, "relative"),
    ({"path": ""}, "relative"),
])
def test_a_malformed_entry_is_refused(mutation, message):
    good = entry("A", "a.bin", b"alpha")
    bad = {**good, **mutation}
    with pytest.raises(ManifestError, match=message):
        Manifest({"schema_version": 1, "groups": {"g": {"files": [bad]}}})


def test_duplicates_and_a_wrong_schema_version_are_refused():
    a, b = entry("A", "a.bin", b"alpha"), entry("B", "b.bin", b"beta")
    with pytest.raises(ManifestError, match="duplicate id"):
        Manifest({"schema_version": 1, "groups": {"g": {"files": [a, {**b, "id": "A"}]}}})
    with pytest.raises(ManifestError, match="duplicate path"):
        Manifest({"schema_version": 1, "groups": {"g": {"files": [a, {**b, "path": "a.bin"}]}}})
    with pytest.raises(ManifestError, match="schema_version"):
        Manifest({"schema_version": 2, "groups": {}})


def test_the_same_path_in_two_groups_is_fine():
    a = entry("A", "a.bin", b"alpha")
    manifest = Manifest({"schema_version": 1, "groups": {"g1": {"files": [a]}, "g2": {"files": [a]}}})
    assert len(manifest.files()) == 2


def test_a_manifest_can_be_loaded_from_a_file(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema_version": 1, "groups": {"g": {"title": "T", "requires": ["chem"],
                                                                         "files": [entry("A", "a.bin", b"alpha")]}}}))
    manifest = Manifest.load(path)
    assert manifest.groups == {"g": "T"} and manifest.requires("g") == ["chem"]
    assert manifest.get("g", "A").path == "a.bin"
    with pytest.raises(KeyError):
        manifest.get("g", "Z")
