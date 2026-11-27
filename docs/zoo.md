# The model zoo

The zoo holds the trained models behind the paper, so that they can be tried and checked without retraining. It is
described by one manifest (`defog_megan.zoo.Manifest`, data in `src/defog_megan/zoo/manifest.json`): every file with its
size, its sha256 hash and a ship level. A group is distributed as **one archive** (a tar of its core and optional files,
recorded in the manifest with its size and hash and the place it is served from); `python -m defog_megan.zoo fetch <group>`
downloads it, checks it against the manifest, unpacks only the files the manifest lists and verifies each of them.

## Groups

| group | what it is | core | optional | withheld |
|---|---|---|---|---|
| `colorgraphs` | Synthetic colored graphs with planted motifs (paper Experiments 1 and 2, the end-to-end baseline) | 14 files, 254.9 MB | 2 files, 44.4 MB | none |
| `zinc-logp` | Molecules steered towards Crippen logP and concepts (paper Experiment 3) | 13 files, 162.2 MB | 1 file, 21.4 MB | 3 files, 7.7 MB |
| `ames` | Mutagenicity class and concepts (paper Experiment 4, qualitative) | 16 files, 145.9 MB | 2 files, 14.5 MB | 5 files, 1.3 MB |

About 563 MB of core files in total. Each group contains:

- the **base** (unconditional DeFoG model), and for the colored graphs also the end-to-end conditioned model;
- the **adapters** (property or class, and concept);
- the **explainer bundle**: the self-explaining GNN, its concept clustering, the processing module it was trained with
  and its quality reports;
- the small **data files that carry contracts**: request vectors, the concept table (the scoring settings the memberships
  were computed with), the vocabulary of the base, size histograms, and for the colored graphs the dataset itself and a
  reference table of memberships;
- for the molecular groups a **public reference table** (`data/reference_check.npz`): about fifty well-known molecules with the
  memberships (and for Ames the class probabilities) that the explainer computes for them. The paper's own reference tables
  hold SMILES of the datasets and are withheld; this one is what a copy of the explainer stack is checked against;
- a **`MODEL_CARD.md`** that says what the models are, the settings of the paper's runs and what to know before using them.

`ship` levels: `core` is needed to use the models, `optional` is useful but not needed, `withheld` is not redistributed.
Every file of the manifest has been collected and checked against its recorded hash (`in_hand` is true throughout). Three
files were sanitized before release: the concept table of the Ames group and the reference table of the colored graphs
record the names of the artifacts, not the paths of the machine that produced them, and the schema file of the ZINC base
records the model name, not the name of the package it was exported from. The hashes are those of the released files.

## Checking a copy

```python
from defog_megan.zoo import Manifest

manifest = Manifest.load()
report = manifest.verify("/path/to/zoo", group="colorgraphs")     # root/<group>/<path>, sizes first, then sha256
assert report.passed, report.summary()
```

**Weights are pickles.** A PyTorch checkpoint can run code when it is loaded, and so can a MEGAN processing module (it is
a Python file). Verify the hash against the manifest before loading anything, and only take files from a source you trust.

## Contracts that must travel with a model

A checkpoint alone is not a usable model. These are stored nowhere inside it, or are easy to get silently wrong:

- **Concept order.** Memberships are ordered by (channel, cluster index). An adapter trained against one order is
  meaningless under another.
- **What a membership is.** A distance: 0 is a core member, 1 is not assigned; the all-ones vector is the legitimate
  "no concept" condition.
- **Normalisation of the condition**, baked into each adapter (`cond_mean`, `cond_std`). It must not be recomputed.
- **The pairing with the base.** An adapter stores a base token, and so does its base; they must match.
- **The vocabulary of the base** (atom and bond classes, their order): in no checkpoint. The molecular groups ship it as a
  file. The charge-aware Ames base has twelve atom classes including the charged ones, which a neutral-only vocabulary
  cannot write.
- **The scoring settings of the memberships** (fidelity threshold, neighbour count, sharpening): in the concept table.
  The threshold differs between the colored graphs and the molecules.
- **The processing module and the explainer release.** Memberships computed with an older explainer release come out
  wrong without any error. The bundle is checked against the reference membership table before use.

## Things a model card must say

- The molecular **base is not a plain unconditional model**: it went through two rounds of structural-sanity
  reinforcement learning after its pretraining.
- The training split of the logP adapter is not documented by a training log. Its stored normalisation (mean 2.825, sd 1.158)
  matches the training split of the recipe it was made with, and the model card says so.
- The colored-graph **base is the exponential-moving-average snapshot from epoch 80** of a 150-epoch run, selected on
  validity, not a final-epoch model.
- The Ames group steers towards a **predicted** class of a weak classifier, not towards experimental mutagenicity; its
  intended use and its limits are stated.

## Data behind the models and what it allows (not legal advice)

| data | position |
|---|---|
| Synthetic colored graphs | generated by code; no third-party terms. The dataset ships. |
| ZINC-250k | free to use; redistribution of significant subsets needs written permission. Weights look publishable; molecule tables are withheld and rebuilt from the public data (`reproduce/prepare_tables.sh zinc`). |
| ChEMBL release 37 | CC BY-SA 3.0, attribution and share-alike. Whether weights count as an adaptation is untested; the corpus is not shipped. |
| Ames benchmark | the supporting-information record of the original paper is CC BY-NC 4.0. Models derived from it are treated as non-commercial; the tables are withheld and scripts that download the data are preferred. |

Everything that holds SMILES (the membership tables of the molecular groups, the label tables, the target lists in the
split files) is `withheld`; the tests enforce it. `reproduce/prepare_tables.sh` rebuilds these tables from the public datasets: the
molecule and label tables come out identical to the paper's files, and the memberships bitwise equal in the checks (CPU runs).

## Hosting

The archives are assets of the GitHub release of the tag (`download.base_url` of the manifest); `fetch --base-url` takes any
other place, a `file://` folder included. The same archives are kept in a citable Zenodo record ([10.5281/zenodo.24583881](https://doi.org/10.5281/zenodo.24583881), all versions), which is
planned to become the default location.

For maintainers: `python -m defog_megan.zoo pack <group> --src <zoo> --out <dir> --version <v> --update-manifest` builds the
archive of one group (reproducibly: fixed times and owners) after verifying the source against the manifest and records its
size and hash. `scripts/build_reference_check.py` builds the public reference table of a molecular group.
