# Parity: how a port is accepted

The package replaces research code whose outputs are the paper's results. A port is accepted when it reproduces those
outputs, not when it looks right.

## Golden outputs

For each ported path the research stack, at the revision the paper's runs used, produces a small golden file on CPU with
one thread, every setting passed explicitly. The port must reproduce it **bit for bit**. The research defaults are not
used: the paper's settings were passed as overrides, so a golden run replays those overrides.

Planned cases, per setting (colored graphs first, then the molecular ones):

| case | what it pins |
|---|---|
| adapter forward pass | the adapter's output for a fixed state, against the stored base |
| membership vectors | the explainer plus the clustering on a fixed set of graphs, against the stored reference table |
| fixed-seed sampling | a few graphs from the base, from one adapter, from the two-adapter composition (plain sum of the log-ratios) |
| steered sampling | one Feynman-Kac run with multinomial resampling and the joint energy |
| explicit sizes | sampling with the size taken from the dataset's size histogram |
| clamp arm | an inpainting arm with steering, asserting that the exact core survives in every returned graph |

Each case stores a digest of every generated graph, so a change of a single node class fails.

## What is not bit-exact

The statistical results of the paper (hit rates over cells, the cost per hit) are compared with a looser check, since
single steered cells varied by up to 0.43 between independent runs of the same setting, and other arms by about 0.08.
That check catches a broken pipeline, not a rounding difference.

## Where the reference lives

The golden generator runs against the research stack and the shipped models; neither is part of this repository.
The generator and the golden files are added with the first ported module. The tests that compare against them are
marked `parity` and are opt-in: they need the model zoo (`DEFOG_MEGAN_ZOO`) and are not run in CI.

## Known traps the golden outputs must not hide

- The explainer silently returns different memberships under an older release; a reference table gates every bundle.
- The generator's default resampling scheme differs from the one the paper used; a golden case without the explicit
  setting would pass on the wrong scheme.
- A library that stops calling an end-of-loop hook makes a constrained sampler lose its exact core without an error;
  the clamp case asserts the core.
