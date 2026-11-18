# defog-megan

**Concept-conditioned graph generation.** Generate graphs for a requested property *and* for the structural concept
through which a graph reaches it. The condition is the concept-membership vector of a self-explaining GNN
([MEGAN](https://github.com/aimat-lab/graph_attention_student)); the generator is a frozen
[DeFoG](https://arxiv.org/abs/2410.04263) model steered by small adapters and, optionally, by Feynman-Kac particle
steering. The generator side lives in [`defog-adapt`](https://github.com/aimat-lab/defog-adapt); this package adds the
concept layer, the joint property + concept pipeline, the evaluation, a command line, and a **model zoo** with the models
of the accompanying paper.

> **Status: pre-alpha (1.0.0.dev0), a skeleton.** The packaging, CI, the model-zoo manifest with its verifier and the
> plan are in place. The generation pipeline is being ported from the research code and is **not** in the package yet,
> so there is nothing to generate with today. Read [`docs/architecture.md`](docs/architecture.md) for what is planned
> and [`docs/parity.md`](docs/parity.md) for how a port is accepted.

## What is in it today

| | |
|---|---|
| `defog_megan.zoo` | The manifest of the paper's models and data files (size, sha256, what ships and what is withheld) and a verifier for a local copy. Three groups: synthetic colored graphs, molecules with Crippen logP, Ames mutagenicity. |
| `docs/zoo.md` | What each group contains, the contracts that must travel with a model (concept order, normalisation, vocabularies), and the licence position of the data behind them. |
| `docs/architecture.md` | The planned layout: concepts, MEGAN runtime (optional), domains, energies, sampling, training, evaluation, zoo, presets, command line. |
| `docs/parity.md` | How the port is proven: golden outputs from the research stack, reproduced bit for bit. |

## Install (planned)

```bash
pip install defog-megan                 # the generator side and the zoo tools
pip install "defog-megan[chem]"         # + RDKit: the molecular domains
pip install "defog-megan[megan]"        # + the MEGAN stack: steering, grading, membership tables for new data
```

Until `defog-adapt` is released, install it from its repository first and this package without dependencies (see
[`CLAUDE.md`](CLAUDE.md)). The `megan` extra can be left out for training adapters and for sampling without steering.

## The model zoo

```python
from defog_megan.zoo import Manifest

manifest = Manifest.load()
print(manifest.total_bytes(ship=("core",)) / 1e6, "MB of core files")
report = manifest.verify("/path/to/zoo", group="colorgraphs")   # sizes and sha256 against the manifest
print(report.summary())
```

The weights are pickled PyTorch checkpoints, so **loading one executes code**: only load files whose hashes match the
manifest from a source you trust. The molecule tables behind the models are not redistributed; see
[`docs/zoo.md`](docs/zoo.md).

## Relationship to DeFoG, citation and licence

The generator follows the upstream [DeFoG](https://github.com/qym7/DeFoG) implementation (Qin, Madeira, Thanou,
Frossard; ICML 2025) through `defog-adapt`. This package is not affiliated with or endorsed by the DeFoG authors. See
[`NOTICE`](NOTICE) for the credits and [`CITATION.cff`](CITATION.cff) for how to cite. MIT, see [`LICENSE`](LICENSE). The
data behind the zoo models carries its own terms; they are stated in the model cards.

## Development

[`CLAUDE.md`](CLAUDE.md) has the layout, the commands and the rules (parity gate, no private references, nothing pushed
or published without a go-ahead).
