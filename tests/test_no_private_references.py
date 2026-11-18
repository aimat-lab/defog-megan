"""Nothing that identifies the machines, accounts or sibling projects this was developed on ships.

The research this package grew out of ran on specific machines, from specific home directories, and fed into other
projects. None of that belongs in a public wheel or a public repository, and it leaks in easily: a comment quoting a job
number, a default path, the name of a downstream tool. This scans every text file that would be published and fails on
the patterns below, so a leak is a test failure, not something found after release.

The patterns that are GENERIC (local paths, scheduler commands, job numbers, e-mail addresses) are listed here. The
patterns that are SPECIFIC (the names of particular clusters, accounts and downstream projects) must not be, because a
published list of private names is itself a leak. They are read from a file named by the environment variable
``DEFOG_MEGAN_PRIVATE_PATTERNS``, one regular expression per line (``#`` starts a comment), which a maintainer keeps
outside the repository. Without it that half of the scan is skipped.

The author's name and address are deliberate and live in a few metadata files (``ALLOWED``). Anything else matching is
either a leak to remove or a pattern to refine, and the allowlist is the one place to say which.
"""

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: what is scanned: everything that would be published or that a contributor reads
SCANNED = ["src", "docs", "tests", ".github", "README.md", "CHANGELOG.md", "CLAUDE.md", "pyproject.toml",
           "NOTICE", "LICENSE", "CITATION.cff"]
SKIP_SUFFIXES = {".json", ".ckpt", ".pt", ".pyc", ".png", ".mp4", ".gz"}

PATTERNS = {
    "a local path": r"(?<![\w.-])/(home|media|mnt|scratch|work)/[A-Za-z0-9_.-]+",
    "a home-directory shorthand": r"~/(Downloads|Programming|Documents)",
    "a scheduler": r"\b(slurm|sbatch|srun|squeue)\b|#SBATCH",
    "a job number": r"\bjobs?\s*#?\s*\d{4,}\b",
    "a person's name": r"\bjonas\b|\bteufel\b|\bthe16thpythonist\b",
    "an email address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
}


def private_patterns():
    """The maintainer's specific patterns, from the file named by DEFOG_MEGAN_PRIVATE_PATTERNS (or none)."""
    path = os.environ.get("DEFOG_MEGAN_PRIVATE_PATTERNS")
    if not path or not Path(path).is_file():
        return {}
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    return {f"a private name ({i})": ln for i, ln in enumerate(l for l in lines if l and not l.startswith("#"))}


#: path (relative, as a suffix) -> patterns that are intended there, with the reason
ALLOWED = {
    "pyproject.toml": {"a person's name", "an email address"},      # package author metadata
    "LICENSE": {"a person's name", "an email address"},              # copyright holder
    "NOTICE": {"a person's name", "an email address"},               # credits (the MEGAN papers are by the author)
    "CITATION.cff": {"a person's name", "an email address"},         # how to cite
}


def candidates():
    for entry in SCANNED:
        path = ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.is_file() and p.suffix not in SKIP_SUFFIXES and "__pycache__" not in p.parts \
                        and "results" not in p.parts and ".venv" not in str(p):
                    yield p


def scan_text(text, rel, patterns, allowed=frozenset()):
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in patterns.items():
            if label in allowed:
                continue
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                yield rel, number, label, match.group(0), line.strip()[:120]


def scan(path: Path, patterns):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    rel = path.relative_to(ROOT).as_posix()
    if rel == "tests/test_no_private_references.py":
        return                                               # this file lists the generic patterns
    allowed = next((v for k, v in ALLOWED.items() if rel.endswith(k)), set())
    yield from scan_text(text, rel, patterns, allowed)


def test_no_private_reference_is_shipped():
    patterns = {**PATTERNS, **private_patterns()}
    leaks = [hit for path in candidates() for hit in scan(path, patterns)]
    assert not leaks, "private references in published files:\n" + "\n".join(
        f"  {rel}:{n}  [{label}]  {found!r}   {line}" for rel, n, label, found, line in leaks[:40])


@pytest.mark.parametrize("label, sample", [
    ("a local path", "cd /home/someone/project"),
    ("a scheduler", "#SBATCH --gpus=4"),
    ("a job number", "see job 1234567"),
    ("a person's name", "written by Jonas"),
    ("an email address", "contact someone@example.org"),
])
def test_the_patterns_actually_match_what_they_are_for(label, sample):
    """A scanner that matches nothing passes forever. Each pattern has to fire on a sample of what it exists to
    catch."""
    assert re.search(PATTERNS[label], sample, flags=re.IGNORECASE), f"{label} did not match {sample!r}"


def test_the_scan_reaches_every_published_directory():
    """A scan that quietly stops reading a directory passes forever. Assert it actually covers each."""
    seen = {p.relative_to(ROOT).parts[0] for p in candidates()}
    for required in ("src", "tests", "docs", "README.md", "pyproject.toml", "LICENSE", "NOTICE"):
        assert required in seen, f"the private-reference scan no longer reads {required}"
    assert sum(1 for _ in candidates()) > 15, "suspiciously few files scanned"


def test_the_maintainers_private_patterns_are_applied_when_a_file_is_given(tmp_path, monkeypatch):
    """The specific names are not in this repository, so the mechanism that loads them has to be shown to work: a
    pattern from the file must catch a planted occurrence, comments and blank lines must be ignored, and without the
    variable nothing is loaded."""
    patterns_file = tmp_path / "patterns.txt"
    patterns_file.write_text("# a comment\n\nzzz-private-name\n")
    monkeypatch.setenv("DEFOG_MEGAN_PRIVATE_PATTERNS", str(patterns_file))
    loaded = private_patterns()
    assert list(loaded.values()) == ["zzz-private-name"]
    hits = list(scan_text("run on the ZZZ-Private-Name cluster", "docs/x.md", {**PATTERNS, **loaded}))
    assert len(hits) == 1 and hits[0][2].startswith("a private name")
    monkeypatch.delenv("DEFOG_MEGAN_PRIVATE_PATTERNS")
    assert private_patterns() == {}
