# Architecture (the plan)

This is the planned shape of the package. **Only the zoo module exists today**; everything else is a proposal that the
port will refine. Nothing is added without a recorded result that needs it.

## The pipeline the paper uses

Every experiment of the paper is one chain, applied to three settings (synthetic colored graphs, molecules with
Crippen logP, Ames mutagenicity):

1. An explainer (MEGAN) and a concept clustering of its explanation embeddings define a vocabulary of concepts. Each
   graph gets a **membership vector**: for every concept a distance (0 is a core member, 1 is not assigned).
2. A frozen unconditional DeFoG base is conditioned by two separately trained adapters: a **property adapter** (a
   scalar or a class) and a **concept adapter** (the membership vector).
3. A joint sampler composes the adapters at sampling time and, optionally, adds **Feynman-Kac particle steering**
   with an energy built from the membership of the requested concept and the error of the requested property. The
   explainer is the energy, so it runs in the sampling process.
4. A scorer decodes the samples and grades them: validity, property match, concept match, joint match, distinct hits,
   cost per distinct hit, the cell classes of the training pool, and the lift over the unconditional base.

## Planned modules

```
defog_megan/
  concepts/     the scoring contract of a membership vector as a spec file that is actually read; a numpy-only
                clustering scorer (no explainer import); request vectors; one table format for memberships
  megan/        optional extra: the explainer runtime (memberships from graphs, property and class readout), model
                bundles checked against a reference membership table
  domains/      colored graphs (with the motif grader), molecules (kekulized and charge-aware vocabularies);
                encode, decode and a vocabulary file that travels with the model
  energies/     one parametrised joint energy (concept term, property term, invalid-sample tier, ring penalty);
                the helpers of the inpainting arms
  sampling/     one joint sampler (single, probe, steered and inpainting stages, sharding, a manifest that records
                every setting); Feynman-Kac with pinned, recorded settings
  training/     adapter recipes (scalar, class, concept), base training, the native-conditioning baseline
  evaluation/   one scorer for all three settings
  zoo/          the manifest, the verifier (exists), download and model cards
  presets/      the paper's arms as explicit data
  cli/          train, sample, score and zoo commands driven by YAML configs
```

Depends on `defog-adapt` for the generator, on the optional `megan` extra for the explainer, and on the `chem` extra for
molecules.

## Decisions already taken

- **The explainer is optional, but not for steering.** Adapter training and sampling without steering need only the
  membership tables and request vectors (data). Steering, grading and membership tables for new data need the
  explainer, in the same process as the generator.
- **Paper settings are data, not defaults.** The composition mode of the joint runs is the plain sum of the adapters'
  log-ratios, the Feynman-Kac resampling scheme is multinomial, the blend space is probability space; every sampler
  writes all settings into its manifest. A dependency changing a default must not change a result.
- **One implementation per concept.** The research code has three near-copies of the joint sampler, the scorer and the
  energy, and re-specifies the membership scoring contract at dozens of places. The package has one of each, and the
  contract is a file that is read.
- **Vocabularies are files, not conventions.** No checkpoint stores the atom list of its base; each model bundle ships
  it, and loading checks it.
- **A bundle is checked before it is used.** Model, clustering and processing module are verified against a reference
  membership table (the explainer's releases differ in ways that change memberships silently).

## Out of scope

Circuits, reinforcement-learning fine-tuning, the solubility setting, the learned concept head (it did not steer
reliably and no paper result uses it), learned size models, tables and figures of the paper itself.
