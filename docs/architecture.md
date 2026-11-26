# Architecture

How the repository is put together and why. It is the code of one paper; it keeps only what the paper's results use.

## The pipeline the paper uses

Every experiment of the paper is one chain, applied to three settings (synthetic colored graphs, molecules with Crippen
logP, Ames mutagenicity):

1. An explainer (MEGAN) and a concept clustering of its explanation embeddings define a vocabulary of concepts. Each
   graph gets a **membership vector**: for every concept a distance (0 is a core member, 1 is not assigned).
2. A frozen unconditional DeFoG base is conditioned by two separately trained adapters: a **property adapter** (a
   scalar or a class) and a **concept adapter** (the membership vector).
3. A joint sampler composes the adapters at sampling time and, optionally, adds **Feynman-Kac particle steering**
   with an energy built from the membership of the requested concept and the error of the requested property. The
   explainer is the energy, so it runs in the sampling process.
4. A scorer decodes the samples and grades them: validity, property match, concept match, joint match, distinct hits,
   cost per distinct hit, the cell classes of the training pool, and the lift over the unconditional base.

The generator (the base, the adapters, their composition, the samplers, the Feynman-Kac sampler) is `defog-adapt`. This
repository is everything around it.

## Layout

```
src/defog_megan/
  concepts.py           the membership contract: ConceptSpace turns graphs or molecules into membership vectors, builds the
                        request vectors, and records the scoring settings
  torch_compat.py       loads the explainer's checkpoints under the torch build of the paper
  bundle.py             an explainer bundle of the zoo, loaded with the scoring settings its own table records, and the
                        reference check that a copy of the explainer reproduces the stored memberships
  data.py               the tables of memberships and request vectors (.npz, no pyarrow needed to read them)
  colorgraphs.py        the synthetic domain: the palette, encoding and decoding of graphs, structural metrics, the pool file
  rb_motifs.py          the planted motifs and the exact property (a function of the structure)
  molecules.py          the molecule codec: defog-adapt's for a neutral vocabulary, a strict charge-aware reading for the
                        Ames base (N+, O-, N-), and the domain that uses it
  color_energy.py, color_joint_energy.py, mol_joint_energy.py, ames_joint_energy.py
                        the Feynman-Kac energies of the three settings (concept term + property term + validity tier)
  plausibility.py       the ring penalty of the molecular energies
  evaluate.py           decoding and scoring helpers (hit rates, property shift)
  sizes.py, gpu.py      dataset size distribution; memory probe for training
  zoo/                  the manifest, the verifier, the archives, the command line (python -m defog_megan.zoo)
experiments/            one PyComex module per experiment: base and adapter training, the evaluations, the concept tables,
                        the three joint samplers (stages single, probe, fk, sharded over devices)
scripts/                the scorers, the data preparation of the Ames set, collect_shards.py, the reference-table builder
reproduce/              the paper's settings, one script per experiment, and the table preparation
tests/                  unit tests; tests/parity/ holds the golden outputs of the research code and the gates
```

## Decisions

- **The explainer is optional, but not for steering.** Adapter training and sampling without steering need only the
  membership tables and request vectors (data). Steering, grading and membership tables for new data need the
  explainer, in the same process as the generator, on the CPU.
- **Paper settings are explicit.** The composition mode of the joint runs is the plain sum of the adapters'
  log-ratios (`product`), the Feynman-Kac resampling scheme is `multinomial`, the blend space is probability space. The
  experiments pass them explicitly and write them into every manifest. A dependency changing a default must not change a
  result (the library's own default scheme is `residual`).
- **Vocabularies are files, not conventions.** No checkpoint stores the atom list of its base; the zoo ships it as a
  sidecar file, and the codec reads a vocabulary with charged classes strictly.
- **A bundle is checked before it is used.** Model, clustering and processing module are verified against a reference
  membership table (the explainer's releases differ in ways that change memberships silently). The check is a gate inside
  the experiments, and it leaves the global random streams alone: the explainer builds a data loader per call, which draws
  from the global generator, so a check that ran before sampling used to shift the run it guarded.
- **Experiments are the research code's experiments.** They keep their parameters as module-level constants, their
  stages and their file formats, so that the results of the paper and the code that made them line up. Only what no result
  needed was removed (the trace sampler, rejuvenation, the diversity stage, options that were no-ops in every run).

## Not in this release

The experiments of the appendices (the end-to-end baselines, the prototype size sweep, inpainting, the per-cell tables and
the figures built from them), the circuit setting, reinforcement-learning fine-tuning, the solubility pilot and the learned
concept head (no result of the paper uses it). The nitrenium DFT study of the appendix is a separate project.
