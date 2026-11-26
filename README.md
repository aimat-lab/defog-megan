# defog-megan

**Generating graphs for a reason.** Code and models of the paper *Generating Graphs for a Reason: Explanation-Conditioned
Adapters for Discrete Flow Matching*. A frozen [DeFoG](https://arxiv.org/abs/2410.04263) generator is steered by two small
adapters, one for a **property** (a number or a class) and one for a **concept**: the concept-membership vector of a
self-explaining GNN ([MEGAN](https://github.com/aimat-lab/graph_attention_student)). The two compose at sampling time, and
Feynman-Kac particle steering can sharpen the result with the explainer itself as the energy. The generator side lives in
[`defog-adapt`](https://github.com/aimat-lab/defog-adapt); this repository adds the concept layer, the joint property and
concept pipeline, the scorers, the experiments of the paper and a **model zoo** with the trained models.

> **Status: v1.0, the code of the paper.** The four experiments of the main text (colored graphs: property alone, then
> property and concept together; molecules steered towards Crippen logP; the Ames mutagenicity class) run from this
> repository, and the numbers of the paper can be re-derived from its raw samples. The experiments of the appendices
> (end-to-end baselines, prototype size sweep, inpainting, per-cell tables) are not included yet and come in a later
> release. The code is research code, kept as close as possible to what produced the paper's numbers; it is **not** a
> general-purpose library, and its interfaces may change.
>
> The training code of some models lives elsewhere: the two molecular bases and the logP property adapter were trained with the
> DeFoG implementation and the MEGAN models with the MEGAN repositories. This repository trains the colored-graph base and the
> adapters of the paper and uses the trained files of the others from the zoo.

## Install

Python 3.10 to 3.12. The explainer stack pins `torch<2.5.1` and `numpy<2`; the paper's runs used torch 2.5.0.

```bash
git clone https://github.com/aimat-lab/defog-megan && cd defog-megan
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu      # or the CUDA build of 2.5.0
pip install -e ".[chem,megan,experiments]"
```

`chem` adds RDKit (the molecular experiments), `megan` the explainer stack (steering, grading and membership tables), and
`experiments` PyComex, which runs the experiments. Python 3.12 additionally needs `pyarrow<26`, which the `megan` extra
requests.

**RDKit is pinned to the release of the paper's runs (2026.03.4).** The property of the logP rung is RDKit's Crippen logP of the
decoded molecule, and that value differs between releases for a small share of molecules (11 of 6,833 sampled molecules
between 2026.03.4 and 2026.09.2, by up to 1.4), which moves the rung's error in the fourth decimal. With the pinned release the
scorers reproduce the paper's numbers exactly; the code warns once if another release is installed.

## Get the models

The trained models and the small data files that carry their contracts are in the **model zoo**, three groups of files
(`colorgraphs`, `zinc-logp`, `ames`; about 560 MB of core files in all), described by a manifest with the size and sha256 of
every file.

```bash
export DEFOG_MEGAN_ZOO=$HOME/defog-megan-zoo                  # the default is ~/.cache/defog-megan/zoo
python -m defog_megan.zoo fetch colorgraphs                   # downloads the group's archive from the release page,
python -m defog_megan.zoo fetch zinc-logp                     # checks it against the manifest, unpacks it
python -m defog_megan.zoo fetch ames
python -m defog_megan.zoo verify colorgraphs --ship core optional
```

Each group contains a `MODEL_CARD.md` that says what the models are, what they were trained on and what to know before using
them (the molecular base went through reinforcement learning; the colored-graph base is the epoch-80 snapshot of a 150-epoch
run; **every number of the Ames rung is model-graded**, and the Ames models derive from non-commercial data).

**The weights are pickles, and the explainer's processing module is Python code**: loading them runs code. The hashes in
the manifest show that a file is the one of the release; they do not show that it is safe. See [`docs/zoo.md`](docs/zoo.md).

## Reproduce the results of the paper

The scripts in [`reproduce/`](reproduce) run the paper's settings (500 steps, multinomial resampling, the seeds of the paper's
runs, 1024 graphs per cell) on the models of the zoo, split over the visible GPUs, and score the result.

| result | script | scorer | what it computes |
|---|---|---|---|
| Table 1 (Experiment 1) | `reproduce/exp1_colored_property.sh` | `scripts/score_color_arms.py` | the property adapter against the base, the FiLM-only control and Feynman-Kac steering: MAE of the requested property, validity, uniqueness, novelty, KL to the data |
| Table 2 (Experiment 1) | the same script | `scripts/score_color_concepts.py` | the concept adapter on the 19 concepts: match rate, lift over the base, shift of the property |
| Table 3 (Experiment 2) | `reproduce/exp2_colored_joint.sh` | `scripts/score_color_joint.py` | six arms over 30 cells: validity, property match, concept match, joint match, distinct hits |
| Table 4, Figure 2 (Experiment 3) | `reproduce/exp3_zinc_logp.sh` | `scripts/score_mol_joint.py` | the same for molecules, with the exact Crippen logP |
| Figure 3 and its tables (Experiment 4) | `reproduce/exp4_ames.sh` | `scripts/score_ames_joint.py` | the same for the Ames class, model-graded |

The colored-graph scripts need only the zoo. The molecular scorers also need tables of the molecules (their memberships, the
logP or class probability of the pool, the training molecules for the novelty column), which hold SMILES and are therefore not
in the zoo: `reproduce/prepare_tables.sh ames` and `reproduce/prepare_tables.sh zinc` build them from the public data, and they
are the paper's tables. The molecules, their labels and the exact logP come out identical to the paper's files, byte for byte
(the corpus steps `scripts/prepare_ames_defog.py` and `scripts/prepare_zinc_defog.py` decode the public molecules through the
base's vocabulary and keep each decoded molecule once). The memberships are recomputed by the explainer and came out bitwise
equal to the paper's in the checks (CPU runs); another machine's float arithmetic can differ in the last digits.

```bash
export DEFOG_MEGAN_ZOO=$HOME/defog-megan-zoo
reproduce/exp2_colored_joint.sh                                  # long on one GPU; NUM_SHARDS splits it over several
EXTRA="--__TESTING__ True" NUM_SHARDS=1 reproduce/exp2_colored_joint.sh     # a tiny run of the whole chain, minutes on a CPU
```

**What a rerun gives.** Sampling is stochastic and a GPU run is not bit-reproducible across hardware, so a rerun agrees with the
paper within sampling error, not exactly. The scorers, on the other hand, are exact functions of the samples: given the paper's
own samples they give the paper's numbers (checked: Tables 1 and 2 come out identical to the reports of the research code,
for example a mean concept match of 0.723 against an unconditional 0.064, a property shift of 1.215, which is 20.3% of the
range; the arms of Table 1 give a property MAE of 0.148 for the adapter at weight 2 against 0.026 with Feynman-Kac steering).

## What is in the repository

| path | what |
|---|---|
| `experiments/` | the experiments, one [PyComex](https://pypi.org/project/pycomex/) module each: training the colored base and the adapters, the evaluations, the concept tables, and the joint samplers of the three settings (`sample_color_joint`, `sample_mol_joint`, `sample_ames_joint`). Parameters are module-level constants with the paper's settings as defaults; `--__TESTING__ True` runs a tiny version. |
| `scripts/` | the scorers, the data preparation of the Ames set, `collect_shards.py`, the zoo tools |
| `reproduce/` | the scripts that run the paper's settings end to end |
| `src/defog_megan/` | `concepts` (the membership contract and the explainer runtime), `bundle` (an explainer bundle and its reference check), `colorgraphs` and `rb_motifs` (the synthetic domain and its exact property), `molecules` (the molecule codec, with a strict charge-aware reading for the Ames base), the Feynman-Kac energies, `evaluate`, `sizes`, and `zoo` |
| `tests/` | unit tests (CPU, a minute) and `tests/parity/`, the golden outputs of the research code |
| `docs/` | the model zoo, the architecture, and how the port was proven equal to the research code |

## Tests and parity

```bash
pytest                                     # unit tests; the zoo-dependent ones skip without the zoo
DEFOG_MEGAN_ZOO=$HOME/defog-megan-zoo pytest -m parity      # the golden outputs of the research code (opt-in, long)
```

The code was ported from research code that produced the paper's numbers, and a port was accepted when it reproduced
that code's outputs **bit for bit** (CPU, one thread, same torch build), not when it looked right: the generator side
with golden digests, every experiment with identical SMILES lists and graph arrays, and the scorers with the paper's own
raw samples. [`docs/parity.md`](docs/parity.md) says what was compared and what was found on the way.

## Relationship to DeFoG, citation and licence

The generator follows the upstream [DeFoG](https://github.com/qym7/DeFoG) implementation (Qin, Madeira, Thanou, Frossard;
ICML 2025) through `defog-adapt`. This package is not affiliated with or endorsed by the DeFoG authors. See
[`NOTICE`](NOTICE) for the credits and [`CITATION.cff`](CITATION.cff) for how to cite; the reference of the paper is added
there when it is public. The code is MIT, see [`LICENSE`](LICENSE). **The data behind the models carries its own terms**,
stated in the model cards: the colored graphs are generated by code and free, ZINC is free to use with redistribution of large
subsets subject to permission, and the Ames data is non-commercial, so models derived from it are treated that way.
