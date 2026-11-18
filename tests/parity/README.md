# Parity tests

A port is accepted when it reproduces the research stack's outputs bit for bit, not when it looks right
(see `docs/parity.md`). This folder holds the golden outputs and the tools to regenerate and check them.

## What is here

| file | what |
|---|---|
| `parity_constants.py` | the definitions: the checkpoint files, the explicit settings, the cases, the two tiers (no torch needed) |
| `cases.py` | the code that runs each case under one package, `defog` (research) or `defog_adapt` (candidate) |
| `generate_golden.py` | generation mode (research stack) and `--check` mode (candidate, writes nothing) |
| `make_golden.sh` | the whole recipe for one tier: extract the reference package from its revision, run the generator |
| `golden/colorgraphs.json` | the golden file: digests per tier, sha256 of every input, settings, environment |
| `test_golden_file.py` | structure tests of the golden file, in the ordinary suite (no torch, no checkpoints) |
| `test_parity.py` | the opt-in gate (`-m parity`): the candidate reproduces every digest |

## The cases (colored graphs, generator side)

Unconditional sampling, explicit-size sampling, the property adapter alone, the concept adapter alone, the two adapters
composed as a plain sum at weights (1, 1), Feynman-Kac steering of that composition with multinomial resampling and a
deterministic stand-in energy, and the end-to-end conditioned model. Each case seeds the generator, samples a few
graphs, and digests every graph (node classes, edges, edge classes).

Every setting is passed explicitly, and the paper's were overrides, not defaults: eta 0, omega 0, polydec time
distortion, plain-sum composition in probability space, 16 particles with beta 16 and warmup 0.7, multinomial
resampling. The two tiers differ only in the loop length and the batch:

| tier | steps | Feynman-Kac particles | takes |
|---|---|---|---|
| `quick` | 10 to 20 | 8 | a few minutes |
| `paper` | 500 | 16 | about two hours per stack, on one CPU core |

## Not covered yet

- The explainer: membership vectors and the explainer-backed Feynman-Kac energy of the paper. They need the explainer's
  numerics fixed on both sides; the membership reference table in the zoo is the golden for the vectors themselves.
- The molecular groups (the base, the logP and Ames adapters, the charge-aware vocabulary) and the inpainting arm.
- The statistical results of the paper (hit rates over cells).

## Regenerate a tier (research stack)

```bash
export DEFOG_FORK=<git checkout of the research package>     # contains defog/
export REFERENCE_REV=<revision the paper's colored-graph runs used>
export RESEARCH_PYTHON=<interpreter of the research environment>
export DEFOG_MEGAN_ZOO=<the model zoo>
tests/parity/make_golden.sh quick        # or: paper
```

The reference package is extracted from the revision with `git archive`, never read from a working tree, and the
generator refuses to run if `defog` is imported from anywhere else (the wrong checkout loads a different model
silently). A second tier is merged into an existing golden file only from the same revision, inputs, settings and
environment; otherwise delete the file and regenerate every tier.

## Run the gate (candidate)

```bash
DEFOG_MEGAN_ZOO=<zoo> OMP_NUM_THREADS=1 PYTHONPATH=<defog-adapt>/src:<this repo>/src \
    <research python> -m pytest -m parity tests/parity/test_parity.py
DEFOG_MEGAN_PARITY_TIER=paper ...        # the paper's settings
```

Bit-exactness holds within one torch build on one CPU architecture, so run the candidate in the environment the golden
file records (the test skips with that reason otherwise). Run it on a snapshot of the commit under test
(`git archive`), not a working tree.

## How the gate itself was checked

A test that cannot fail is not a test. The gate was run against candidates broken on purpose: the composition
averaging the adapters instead of summing them (the two-adapter and the steered cases fail, the single-adapter case
passes), and the multinomial resampler returning the particles in another order (only the steered case fails).
