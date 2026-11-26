# Parity: how a port is accepted

The package replaces research code whose outputs are the paper's results. A port is accepted when it reproduces those
outputs, not when it looks right. Four kinds of evidence were used, from the generator up to the tables.

## 1. Golden outputs of the generator side

For each ported path the research stack, at the revision the paper's runs used, produces a golden file on CPU with one
thread, every setting passed explicitly. The port must reproduce it **bit for bit**. The research defaults are not used:
the paper's settings were passed as overrides, so a golden run replays those overrides.

Each golden case stores a digest of every generated graph, so a change of a single node class fails. Digests are
bit-exact only within one torch build on one CPU architecture; the golden file records that environment, and the gate
skips with a reason in another one.

Seven colored-graph cases, in a quick tier and in the paper's tier (500 steps, 16 particles): unconditional sampling,
explicit-size sampling, each adapter alone, the two-adapter composition (plain sum at weights (1, 1)), Feynman-Kac steering
with multinomial resampling, and the end-to-end conditioned model. `defog-adapt` 0.1.0 reproduces **both tiers**, case by
case (the paper tier took about three hours on one CPU core, the steered case 2.3 of them). A composition that averages the
adapters instead of summing them fails the two-adapter and the steered cases and not the single-adapter ones (checked: the
paper's runs summed), and a resampler that draws in another order fails the steered case.

## 2. The experiments, side by side with the research code

The cases above use a stand-in energy. The experiments are checked as wholes: each was run in the tiny testing mode (a few
graphs, 20 steps) as the research code and as this repository, in **one environment** (one torch build, one thread, CPU), with
identical arguments and seeds, and every generated graph array and SMILES list was compared, together with the manifest and
the number of explainer calls of the steered units. This covers the adapters and their composition, the molecule codecs
(including the strict charge-aware reading of the Ames base), the energies with the explainer in the loop, and the sharding.

| experiment | stages | result |
|---|---|---|
| colored joint sampler (`sample_color_joint`) | probe, fk | identical (probe: 6 files; fk: 4 files and the explainer calls) |
| logP sampler (`sample_mol_joint`) | single, probe, fk | identical (the single stage compared on its files; probe and fk with the weights of the paper's runs on both sides) |
| Ames sampler (`sample_ames_joint`) | single, probe, fk | identical |

Two controls guard the comparison: running the research code twice and this repository twice each give identical outputs
(shown on the Feynman-Kac stage of the colored joint sampler, the one with the most moving parts), so the experiments are
deterministic on their own and a difference between the two would be a difference of the port.
The outputs of the research side are stored as digests (`tests/parity/golden/experiments.json`), and the opt-in gate
`pytest -m parity tests/parity/test_experiment_golden.py` repeats the check without the research code.

## 3. The scorers, on the paper's own raw samples

A scorer is an exact function of the samples. The paper's raw samples of the colored rung were scored with the ported
scorers, with the explainer recomputing every membership vector, and the outputs compared with the reports of the research code:

| result | outcome |
|---|---|
| Table 1: base, adapter at w = 1 and 2, Feynman-Kac with 16 and 128 particles, the FiLM-only control | identical (every row, every column) |
| Table 2: the concept arm on 19 concepts (match, lift, property shift) | identical, summary and per-concept table |
| Table 3: six arms over 30 cells (1024 graphs each) | identical: the summary, all 180 rows of the cell table, and the motif mechanisms |
| Table 4: the logP rung (190 dumps of molecules; the property is the exact Crippen logP) | identical (summary, cell table, the Experiment 3 report) with the paper's RDKit release, see below |
| the Ames rung (52 dumps; model-graded) | identical (summary, cell table, the Experiment 4 report) |

## 4. The data the experiments need

- The request vectors and the concept tables, re-exported with the explainer of the zoo, are **identical** to the zoo's (0.0
  difference, identical specs) for all three groups. The memberships of the first 400 colored graphs, recomputed, agree to 6e-8.
- The molecular membership tables, rebuilt from the paper's source molecules by `experiments/prepare_concept_dataset.py`: the
  decoded SMILES of the first 200 molecules of the ZINC and of the Ames table are identical to the paper's tables (for Ames,
  including its 35 charged molecules, read with the strict codec), the memberships agree to 2e-7 and exactly, respectively.
- The Ames data scripts rebuild the paper's tables from the public dataset **byte for byte** (the molecule table, its metadata,
  the size histogram), and the coverage reports, the label table and the frozen split match.
- **The ZINC pool is rebuilt from the public dataset as well.** The paper's pool is not the dataset's strings but what the base
  makes of them: each molecule of `zinc64k` encoded into the base's vocabulary and decoded (which drops charges and
  stereochemistry), each decoded molecule kept once. `scripts/prepare_zinc_defog.py` does that, and the table it writes (65,434
  of the dataset's 65,529 molecules, 95 repeats dropped, with the exact Crippen logP) is identical to the paper's `zinc64k_defog.csv`
  byte for byte. The memberships of that pool, recomputed by the chain on a CPU, are bitwise equal to the paper's (all 65,434 x 42
  values, so no molecule changes its core or periphery status), and so is the logP of every molecule.

## 5. That the experiments run

Parity says the outputs are the research code's. This says that a user gets them from a clean checkout.

- **Every experiment and script in its tiny testing mode** (a few graphs, 20 steps) exits cleanly on a CPU, and the samplers,
  evaluations and trainings also on a GPU, with the device recorded in the manifests. The reproduction scripts of Experiments 2 to 4
  (sampling, shard collection, scoring; `NUM_SHARDS=1`) run through on a GPU. `scripts/predict_pool_property.py` and
  `scripts/pool_logp.py` give the zoo's property tables exactly, and `scripts/score_ames_lr_selection.py` prints what the
  research script printed on the same samples.
- **One unit of each sampler at the paper's settings** (500 steps, 16 particles, chunks of 64, the explainer in the loop) ran on
  an RTX 3060 and produced only valid graphs: 128 colored graphs in 10 minutes, 256 ZINC molecules in 14 (their logP centred on
  the requested 0.5, mean 0.45) and 256 Ames molecules in 12; the GPU memory in use stayed under 2 GB.
- **The gate of section 2 on a second machine.** `pytest -m parity tests/parity/test_experiment_golden.py` passes (8 cases, 14
  minutes on one thread) on a different CPU model (a Ryzen 9 3900X, the digests were made on a 5900X) with Python 3.11 instead of
  3.10, against the public zoo alone, so the digests are not tied to one machine of one architecture family.
- **Installation** of `[chem,megan,experiments]` from PyPI into new environments worked on Python 3.10 (the development
  environment), 3.11 (CPU and CUDA torch, on a second machine) and 3.12 (with `pyarrow<26`); the unit tests pass in them, and the table
  chain of the molecular groups ran in the second machine's environment.

## What was found on the way

- **A check that moves the random stream.** The explainer builds a data loader on every call, and a data loader draws its base
  seed from the global torch generator. The explainer gate ahead of a steered run therefore changed which random numbers the run
  used, and the Feynman-Kac stage of the ported joint sampler differed from the research code's (the probe stage, which has no
  explainer, did not). Found with the two control runs and a trace of every energy call (the new run's generator state before its
  first call equalled the old run's state after its first). The gate now restores the global random states; a test pins that.
- **A label that was not the setting.** The archived Feynman-Kac runs of Experiment 1 record `resample_scheme: residual`, the
  parameter of the evaluation script. The sampler that ran did not accept the parameter and resampled multinomially, which is what
  the experiments here use and record.
- **The exact property depends on the RDKit release.** Scoring the paper's raw logP samples with a newer RDKit (2026.09.2 instead
  of the paper's 2026.03.4) changed the Crippen logP of 11 of 6,833 sampled molecules by up to 1.4 and with it the error and the
  property match of the rung in the fourth decimal; with 2026.03.4 the scorer's summary, cell table and per-experiment report
  are identical to the paper's. `chem` therefore pins 2026.03.4, the manifests record the release, and the code warns when another
  one is installed. The membership vectors and the Ames rung do not depend on it.
- **Defaults that were overrides.** The research code's file defaults for the molecular runs (eta 25, joint weights (1, 2), 128
  graphs) were not what the paper ran (eta 500, weights (1, 1), 1024 graphs). The experiments here default to the paper's settings.

## What is not bit-exact, and what is not covered

- Runs on a GPU, and the statistical results of the paper (hit rates over cells, the cost per hit), which are compared with a
  looser check, since single steered cells varied by up to 0.43 between independent runs of the same setting, and other arms by
  about 0.08. That check catches a broken pipeline, not a rounding difference.
- The membership vectors come from the explainer, an external dependency with its own numerics. They are checked against the
  reference tables of the zoo within the tolerance the research pipeline itself gated on (1e-5; the observed difference is
  below 2e-7), not bit for bit.
- The training experiments are smoke-run, and the six adapters of the zoo (two colored, four molecular) are built by their recipes
  with exactly the parameter counts and architectures of the paper's checkpoints; they are not retrained and compared.
- The experiments of the appendices, which are not part of this release.

## Known traps the checks must not hide

- The explainer silently returns different memberships under an older release; a reference table gates every bundle.
- The generator's default resampling scheme differs from the one the paper used; every experiment passes it explicitly.
- A library that stops calling an end-of-loop hook makes a constrained sampler lose its exact core without an error; the
  inpainting experiments (a later release) assert the core.
