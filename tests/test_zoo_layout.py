"""Where the zoo is, and how a file in it is named."""
from pathlib import Path

from defog_megan.zoo import ENV_VAR, Manifest, zoo_path, zoo_root


def test_the_environment_variable_names_the_root(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert zoo_root() == tmp_path
    assert zoo_path("colorgraphs", "base.ckpt") == str(tmp_path / "colorgraphs" / "base.ckpt")


def test_without_it_the_root_is_a_per_user_cache(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert zoo_root() == Path.home() / ".cache" / "defog-megan" / "zoo"


def test_a_tilde_in_the_variable_is_expanded(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "~/some-zoo")
    assert zoo_root() == Path.home() / "some-zoo"


def test_the_root_is_read_at_the_time_of_the_call_not_at_import(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "a"))
    first = zoo_path("ames", "base.ckpt")
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "b"))
    assert zoo_path("ames", "base.ckpt") != first


def test_a_missing_file_is_not_an_error_here(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert isinstance(zoo_path("zinc-logp", "megan/model.ckpt"), str)        # existence is the caller's business


def test_every_path_in_the_manifest_resolves_below_its_groups_folder(monkeypatch, tmp_path):
    """The manifest and the layout agree on what a file's place is: <root>/<group>/<path>."""
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    for f in Manifest.load().files():
        assert Path(zoo_path(f.group, f.path)) == tmp_path / f.group / f.path
