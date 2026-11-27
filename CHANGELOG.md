# Changelog

## Unreleased

- README: a banner animation of what a concept is above a recording of the sampler (`docs/assets/banner/banner.gif`), a results
  chart of the paper's two joint experiments (`docs/assets/results.json` and `results.py`), a tested quickstart
  (`examples/quickstart.py`), and the sections reordered after those of `defog-adapt`. The content of the old README is kept.
- Installation: `pip install -e .` installs everything. The extras `chem`, `megan`, `experiments`, `dev` and `all` are gone and
  their dependencies are now dependencies of the package; the tests need `pytest` (and `pytest-xdist` for `-n`), installed by hand.
  The zoo manifest's unused `requires` field, which named the extras, and `Manifest.requires()` are removed.

## 1.0.0 (2026-11-26)

The code of the paper *Generating Graphs for a Reason: Explanation-Conditioned Adapters for Discrete Flow Matching*: the four
experiments of the main text, their scorers, and the model zoo. Built on `defog-adapt` 0.1.0.

### Included

- `experiments/`: the colored-graph base, the property-adapter evaluation (with and without Feynman-Kac steering), the concept
  tables and the concept-adapter sampling and training, the property-adapter and concept-adapter training of the molecular
  settings, and the three joint samplers (`sample_color_joint`, `sample_mol_joint`, `sample_ames_joint`; stages `single`,
  `probe`, `fk`, sharded over devices). The paper's settings are the defaults or are passed by `reproduce/`.
- `scripts/`: the scorers of the colored arms, the concept arm, the colored joint rungs, the logP rung and the Ames rung; the
  Ames and ZINC data preparation; `collect_shards.py`; the learning-rate selection; the builder of the public reference tables.
- `reproduce/`: one script per experiment with the seeds and sizes of the paper's runs, and the preparation of the molecule tables.
- `src/defog_megan/`: the membership contract and explainer runtime, the bundle check, the synthetic domain and its exact
  property, the molecule codec with a strict charge-aware reading (vocabulary of the Ames base), the four Feynman-Kac energies, the
  evaluation helpers, and the zoo (manifest, verifier, archives, command line).
- The model zoo: three groups, 563 MB of core files, each with a model card, and for the molecular groups a public reference
  table that checks a copy of the explainer stack.
- `tests/parity/`: the golden outputs of the research code, for the generator side (seven cases, two tiers) and for every
  experiment (eight cases), with the tools that make them.

### Dependencies

`defog-adapt>=0.1.0`; the explainer stack (`graph_attention_student>=1.6.0`, `megan_global_explanations>=0.6.0`, `visual_graph_datasets>=0.17.0`);
`rdkit==2026.3.4`, the release of the paper's runs (the logP rung's exact property depends on it); PyComex for the experiments.

### How it was checked

`defog-adapt` 0.1.0 reproduces the golden outputs of the generator side at both tiers (quick, and the paper's 500 steps with 16
particles). Every experiment and script also runs to completion in its tiny testing mode on a CPU and on a GPU (the reproduction
scripts of Experiments 2 to 4 included). The table-preparation chains rebuild the paper's tables from the public data: the Ames
rung exactly, and the ZINC molecules and their logP byte for byte.

Every ported experiment was run next to the research code in one environment with identical arguments, and the generated
graphs and SMILES lists compared: identical for all three stages of the molecular settings and for the colored joint sampler. The
scorers reproduce the paper's reports from the paper's own raw samples. The tables the molecular experiments need are rebuilt
from the public data and agree with the paper's (the molecule tables byte for byte; the memberships bitwise equal in the CPU
checks, for Ames and for ZINC). `docs/parity.md` has the
details, including two findings along the way: the explainer check ahead of a sampling run moved the random stream of the run, and
the resampling scheme recorded in the archived Feynman-Kac runs (`residual`) was not the one in effect (`multinomial`).

### Not included

The experiments of the appendices (end-to-end baselines, prototype size sweep, inpainting, per-cell tables, figures), which come
in a later release; the nitrenium study, which is a separate project.
