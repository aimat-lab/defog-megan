"""Sharded sampling: the partition of the work units over shards, and the collection of the shards' run folders.

A unit drawn twice or not at all would corrupt a table without an error, so both halves are tested: every shard derives the
same partition independently (``assign_shards``), and ``scripts/collect_shards.py`` gathers exactly the shards of one stage.
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                      # PyComex inspects the module that a decorator runs in
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ames():
    return load("experiments/sample_ames_joint.py", "sample_ames_joint_for_test")


def settings(**overrides):
    base = dict(STAGE="single", SINGLE_ARMS=["uncond", "property", "concept"], P_WEIGHTS=[1.5], C_WEIGHTS=[1.0],
                W_PROPERTY_SOLO=1.0, LEVELS=[0.0, 1.0], CONCEPTS=["ch0_cl1", "ch1_cl6", "ch1_cl8"], SINGLE_CONCEPTS=[],
                N_UNCOND=64, N_PROPERTY=32, N_PROPERTY_MAIN=64, N_CONCEPT=16, N_PER_CELL=16, WEIGHT_GRID=[[1.0, 1.0], [1.0, 2.0]],
                FK_WEIGHTS=[1.0, 1.0])
    base.update(overrides)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize("stage", ["single", "probe", "fk"])
def test_every_unit_has_a_distinct_key(ames, stage):
    units = ames.work_units(settings(STAGE=stage), [])
    keys = [u["key"] for u in units]
    assert keys and len(keys) == len(set(keys))


@pytest.mark.parametrize("stage", ["single", "probe", "fk"])
@pytest.mark.parametrize("num_shards", [1, 2, 3, 8])
def test_every_unit_goes_to_exactly_one_shard_and_the_partition_is_reproducible(ames, stage, num_shards):
    units = ames.work_units(settings(STAGE=stage), [])
    owners = ames.assign_shards(units, num_shards)
    assert len(owners) == len(units) and set(owners) <= set(range(num_shards))
    assert owners == ames.assign_shards(list(units), num_shards)                        # what every shard computes for itself
    taken = [u["key"] for k in range(num_shards) for u, o in zip(units, owners) if o == k]
    assert sorted(taken) == sorted(u["key"] for u in units)                            # none dropped, none twice


def test_the_load_is_balanced_by_cost(ames):
    units = ames.work_units(settings(STAGE="probe", CONCEPTS=[f"c{i}" for i in range(7)]), [])
    owners = ames.assign_shards(units, 4)
    loads = [sum(u["cost"] for u, o in zip(units, owners) if o == k) for k in range(4)]
    assert max(loads) - min(loads) <= max(u["cost"] for u in units)


def test_the_samplers_of_the_other_settings_partition_the_same_way():
    for path in ("experiments/sample_mol_joint.py", "experiments/sample_color_joint.py"):
        module = load(path, "sampler_" + Path(path).stem)
        units = [{"key": f"u{i}", "cost": 1 + (i % 5), "arm": "x"} for i in range(23)]
        owners = module.assign_shards(units, 4)
        assert sorted(o for o in owners) and set(owners) == {0, 1, 2, 3} and len(owners) == 23, path


# ---- the collector ---------------------------------------------------------------------------------------------------------

@pytest.fixture()
def collect():
    module = load("scripts/collect_shards.py", "collect_shards_for_test")

    def run(*argv):
        old = sys.argv
        sys.argv = ["collect_shards.py", *argv]
        try:
            return module.main()
        finally:
            sys.argv = old
    return run


_clock = iter(range(10_000, 20_000))


def make_run(results, experiment, name, stage, shard, num_shards, mtime=None):
    """A run folder with a manifest; its time is explicit and increasing, so that the order of runs never depends on the
    resolution of the file system's clock."""
    mtime = next(_clock) if mtime is None else mtime
    folder = results / experiment / name
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"stage": stage, "shard": shard, "num_shards": num_shards}))
    (folder / f"gen_{stage}_{shard}.json").write_text(json.dumps({"run": name}))
    (folder / "experiment_out.log").write_text("not collected")
    os.utime(folder / "manifest.json", (mtime, mtime))
    return folder


def test_the_shards_of_one_stage_are_gathered_and_nothing_else(tmp_path, collect):
    results = tmp_path / "results"
    make_run(results, "exp", "a", "probe", 0, 2)
    make_run(results, "exp", "b", "probe", 1, 2)
    make_run(results, "exp", "c", "fk", 0, 2)                  # another stage of the same experiment
    make_run(results, "exp", "d", "probe", 0, 4)               # the same stage, a run of another size
    assert collect("exp", "--stage", "probe", "--num-shards", "2", "--out", str(tmp_path / "out"), "--results", str(results)) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["shard0", "shard1"]
    assert sorted(p.name for p in (tmp_path / "out" / "shard0").iterdir()) == ["gen_probe_0.json", "manifest.json"]
    assert json.loads((tmp_path / "out" / "shard0" / "gen_probe_0.json").read_text()) == {"run": "a"}     # not the 4-shard run d
    assert json.loads((tmp_path / "out" / "shard1" / "gen_probe_1.json").read_text()) == {"run": "b"}


def test_a_missing_shard_is_an_error_and_writes_nothing(tmp_path, collect):
    results = tmp_path / "results"
    make_run(results, "exp", "a", "probe", 0, 2)
    assert collect("exp", "--stage", "probe", "--num-shards", "2", "--out", str(tmp_path / "out"), "--results", str(results)) == 1
    assert not (tmp_path / "out").exists()


def test_a_later_run_of_a_shard_wins_and_since_excludes_the_old_ones(tmp_path, collect):
    results = tmp_path / "results"
    make_run(results, "exp", "old", "probe", 0, 1, mtime=1_000)
    newer = make_run(results, "exp", "new", "probe", 0, 1, mtime=2_000)
    (newer / "gen_probe_0.json").write_text('{"newer": true}')   # the newer run's content
    assert collect("exp", "--stage", "probe", "--num-shards", "1", "--out", str(tmp_path / "out1"), "--results", str(results)) == 0
    assert json.loads((tmp_path / "out1" / "shard0" / "gen_probe_0.json").read_text()) == {"newer": True}
    assert collect("exp", "--stage", "probe", "--num-shards", "1", "--since", "5000", "--out", str(tmp_path / "out2"),
                   "--results", str(results)) == 1                                           # both are older than --since
