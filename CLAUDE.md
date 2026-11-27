# CLAUDE.md

Guidance for working in this repository.

## What this is

`defog-megan` (import `defog_megan`) is the code and the model zoo of a paper on concept-conditioned graph generation: a
frozen DeFoG generator steered by a property adapter and a concept adapter, with the concept condition taken from the
membership vector of a self-explaining GNN (MEGAN), plus Feynman-Kac steering, the scorers and the experiments of the main
text. It is built on `defog-adapt` (the generator side) and keeps only what the paper's results use. Circuits,
reinforcement-learning fine-tuning, the solubility rung and the learned concept head are out of scope on purpose, and the
experiments of the appendices come in a later release.

It is a **port** of research code that stays elsewhere as the archive of the original runs. The experiments keep their
structure on purpose (module-level parameters, stages, file formats), so that the results of the paper and the code that made
them line up; do not "clean them up" without a parity run.

## Layout

- `src/defog_megan/` the library (`docs/architecture.md` lists the modules); `zoo/` is the manifest, the verifier, the archives
  and the command line
- `experiments/` PyComex experiments; `scripts/` scorers and data preparation; `reproduce/` the paper's settings end to end
- `docs/` architecture, zoo (contents, contracts, licences, hosting), parity (how a port is accepted, what was found)
- `tests/` CPU-only unit tests; `tests/parity/` the golden outputs of the research code (generator side and experiments), the
  tools that make them, and the opt-in gates (`tests/parity/README.md`)

## Commands

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e . pytest pytest-xdist
OMP_NUM_THREADS=2 pytest -q                                   # unit tests, no network, about a minute
DEFOG_MEGAN_ZOO=<zoo> OMP_NUM_THREADS=1 pytest -m parity      # the golden outputs (opt-in, long, needs the zoo and torch 2.5.0)
python -m defog_megan.zoo list | verify <group> | fetch <group> | pack <group> ...
```

## Rules

- **Parity gate.** A ported module is accepted only when it reproduces, bit for bit, the outputs of the research code at the
  revision the paper ran on, with every setting passed explicitly (never the research defaults; the paper's settings were
  launcher overrides). Any drift must be understood, not tolerated. See `docs/parity.md`.
- **Paper settings are explicit.** The composition mode of the joint runs is `product` (the plain sum of the adapters'
  log-ratios), the Feynman-Kac resampling scheme is `multinomial`, the blend space is probability space. Never rely on a
  default of a dependency for any of them, and write them into every manifest of a run.
- **A check must not move what it checks.** The explainer builds a data loader on every call, which draws from the global torch
  generator; anything that calls it before sampling (the reference gate) must leave the random streams as it found them
  (`bundle.preserved_rng`).
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
- Nothing is pushed or published (GitHub, PyPI, Zenodo) without an explicit go-ahead.

## Behaviour notes that matter

- A membership vector is a **distance**: 0 is a core member, 1 is not assigned; all ones is the legitimate "no concept"
  condition. Its layout is a contract: concepts are ordered by (channel, cluster index), and every consumer checks it.
- Normalisation constants are baked into each adapter. Never recompute them from data.
- An adapter is paired with its base through a base token stored in both; check it on load.
- The vocabulary of a base is stored in no checkpoint. It travels with the model (`base.vocabulary.json`,
  `base.schema.json`) and is checked on load. The Ames base has charged atom classes and is read strictly
  (`defog_megan.molecules`); never decode it with the neutral codec.
- The explainer pins its own `torch` and `numpy` ranges (conservative, not needed for inference); the explainer's
  version must be at least 1.6.0, because older releases load the checkpoints without error and return wrong
  memberships.
- Feynman-Kac steering runs the explainer in the same process as the generator (it is the energy), on CPU.
- The archived runs of Experiment 1's Feynman-Kac rows record `resample_scheme: residual`; that was the parameter, not the
  scheme in effect (the pinned sampler ignored it and resampled multinomially). The experiments here record what is used.
