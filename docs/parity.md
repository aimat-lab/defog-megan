# Parity: how a port is accepted

The package replaces research code whose outputs are the paper's results. A port is accepted when it reproduces those
outputs, not when it looks right.

## Golden outputs

For each ported path the research stack, at the revision the paper's runs used, produces a golden file on CPU with one
thread, every setting passed explicitly. The port must reproduce it **bit for bit**. The research defaults are not used:
the paper's settings were passed as overrides, so a golden run replays those overrides.

Each golden case stores a digest of every generated graph, so a change of a single node class fails. Digests are
bit-exact only within one torch build on one CPU architecture; the golden file records that environment, and the gate
skips with a reason in another one.

## Status

| setting | generator side | explainer side |
|---|---|---|
| synthetic colored graphs | **7 cases, quick tier generated and reproduced** by `defog-adapt`; paper tier (500 steps, 16 particles) generated separately | planned |
| molecules (logP), Ames | planned | planned |

The seven colored-graph cases: unconditional sampling, explicit-size sampling, each adapter alone, the two-adapter
composition (plain sum at weights (1, 1)), Feynman-Kac steering with multinomial resampling, and the end-to-end
conditioned model. See `tests/parity/README.md` for the recipe, the tiers and what is not covered.

## What is not bit-exact

The statistical results of the paper (hit rates over cells, the cost per hit) are compared with a looser check, since
single steered cells varied by up to 0.43 between independent runs of the same setting, and other arms by about 0.08.
That check catches a broken pipeline, not a rounding difference.

The membership vectors come from the explainer, an external dependency with its own numerics. They are checked against
the reference table in the zoo within the tolerance the research pipeline itself gated on (1e-5), not bit for bit, and
the explainer-backed Feynman-Kac energy is added as a case once the explainer's code is held fixed on both sides.

## Known traps the golden outputs must not hide

- The explainer silently returns different memberships under an older release; a reference table gates every bundle.
- The generator's default resampling scheme differs from the one the paper used; every case passes it explicitly.
- A composition that averages the adapters instead of summing them fails the two-adapter and the steered cases and not
  the single-adapter ones (checked); the paper's runs summed.
- A library that stops calling an end-of-loop hook makes a constrained sampler lose its exact core without an error;
  the clamp case (not yet written) asserts the core.
