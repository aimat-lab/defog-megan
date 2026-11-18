# Changelog

## 1.0.0 (unreleased)

Pre-alpha: the repository skeleton and the model-zoo manifest. Nothing of the generation pipeline is ported yet.

### Included

- Packaging (`defog-megan`, import `defog_megan`), CI, the private-reference scan, the repository rules (`CLAUDE.md`).
- `defog_megan.zoo`: the manifest of the paper's models and data files (sizes, sha256, what ships), and a verifier that
  checks a local copy against it.
- `docs/`: the architecture plan, the model zoo (contents, contracts, licences) and the parity plan.
- `tests/parity/`: the golden outputs of the research stack for the colored-graph generator side (seven cases, two
  tiers), the generator that makes them, structure tests of the golden file, and the opt-in gate. The gate was
  checked against candidates broken on purpose.

### Planned

See `docs/architecture.md` and `docs/parity.md`.
