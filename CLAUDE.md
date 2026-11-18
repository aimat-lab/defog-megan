# CLAUDE.md

Guidance for working in this repository.

## What this is

`defog-megan` (import `defog_megan`) is the reusable package behind a paper on concept-conditioned graph generation: a
frozen DeFoG generator steered by a property adapter and a concept adapter, with the concept condition taken from the
membership vector of a self-explaining GNN (MEGAN), plus Feynman-Kac steering, an evaluation, a command line and a
model zoo. It is built on `defog-adapt` (the generator side) and keeps only what the paper's results use. Circuits,
reinforcement-learning fine-tuning, the solubility rung and the learned concept head are out of scope on purpose.

It is a **port** of research code that stays elsewhere as the archive of the original runs. The inventory of that code
(what the paper path needs, the gaps against `defog-adapt`, the MEGAN stack, the model files) was written before this
repository and is kept outside it; `docs/` carries what a contributor needs.

## Layout

- `src/defog_megan/zoo/` the model-zoo manifest (`manifest.json`) and its loader and verifier
- `docs/` architecture (the plan), zoo (contents, contracts, licences), parity (how a port is accepted)
- `tests/` CPU-only unit tests; `tests/parity/` (not yet there) opt-in comparison against golden outputs

## Commands

Until `defog-adapt` is on PyPI, install it from its checkout first:

```bash
uv venv --python 3.12 .venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e "<path to defog-adapt>[chem]"
uv pip install -e . --no-deps && uv pip install pytest pytest-xdist
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -n 8 -q            # unit tests, no network, a few seconds
DEFOG_MEGAN_ZOO=/path/to/zoo .venv/bin/python -m pytest -m parity   # parity gate (local only; not there yet)
```

## Rules

- **Parity gate.** A ported module is accepted only when it reproduces, bit for bit, golden outputs generated from the
  research stack at the revision the paper ran on, with every setting passed explicitly (never the research defaults;
  the paper's settings were launcher overrides). Any drift must be understood, not tolerated. See `docs/parity.md`.
- **Paper settings are explicit.** The composition mode of the joint runs is `product` (the plain sum of the adapters'
  log-ratios), the Feynman-Kac resampling scheme is `multinomial`, the blend space is probability space. Never rely on a
  default of a dependency for any of them, and write them into every manifest of a run.
- **Checkpoint compatibility.** Existing checkpoints load unchanged; a model bundle (model, clustering, processing module,
  vocabulary, normalisation) is checked against a reference table of memberships before it is used.
- **Test your tests.** Every new test is checked against a deliberately broken implementation, **after** asserting that
  the unbroken one passes. A test that cannot fail is not a test.
- **Never commit** checkpoints, datasets (the molecule tables in particular), experiment output, cluster or account
  names, job IDs or local paths. `tests/test_no_private_references.py` fails on the generic patterns; the specific names
  are read from a file named by `DEFOG_MEGAN_PRIVATE_PATTERNS`, one regex per line, that the maintainer keeps outside the
  repository. Before publishing a history, scan every commit, not only the tip.
- **Do not add a feature without a recorded result** that needs it, and say what the evidence is.
- Changes that belong in `defog-adapt` (a bug, a missing hook) are made there, in a worktree and under its rules, never
  worked around silently here. This package never names projects that depend on it.
- Nothing is pushed or published (GitHub, TestPyPI, PyPI, Zenodo) without an explicit go-ahead.

## Behaviour notes that matter

- A membership vector is a **distance**: 0 is a core member, 1 is not assigned; all ones is the legitimate "no concept"
  condition. Its layout is a contract: concepts are ordered by (channel, cluster index), and every consumer checks it.
- Normalisation constants are baked into each adapter. Never recompute them from data.
- An adapter is paired with its base through a base token stored in both; check it on load.
- The vocabulary of a base is stored in no checkpoint. It travels with the model and is checked on load.
- The explainer pins its own `torch` and `numpy` ranges (conservative, not needed for inference); the explainer's
  version must be at least 1.6.0, because older releases load the checkpoints without error and return wrong
  memberships.
- Feynman-Kac steering runs the explainer in the same process as the generator (it is the energy), on CPU.
