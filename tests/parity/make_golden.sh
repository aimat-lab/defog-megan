#!/usr/bin/env bash
# Regenerate one tier of the golden file of the colored-graph generator-side cases from the research stack.
#
#   tests/parity/make_golden.sh quick      # about a minute or two
#   tests/parity/make_golden.sh paper      # the paper's 500 steps and 16 particles; much longer
#
# Needs (all outside this repository):
#   DEFOG_FORK        a git checkout of the research package (it contains the `defog/` directory)
#   REFERENCE_REV     the revision the paper's colored-graph runs used
#   RESEARCH_PYTHON   the interpreter of the research environment (the same torch build as the candidate runs)
#   DEFOG_MEGAN_ZOO   the model zoo (group folders with the checkpoints inside)
#
# The reference package is extracted from the revision itself (git archive), never taken from a working tree, and
# the generator refuses to run if `defog` is imported from anywhere else. A second tier is merged into an existing
# golden file only from the same revision, inputs and environment.
set -euo pipefail
TIER="${1:?usage: make_golden.sh quick|paper}"
: "${DEFOG_FORK:?}" "${REFERENCE_REV:?}" "${RESEARCH_PYTHON:?}" "${DEFOG_MEGAN_ZOO:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
REF="$REPO/.work/reference/defog_$REFERENCE_REV"

rm -rf "$REF" && mkdir -p "$REF"
git -C "$DEFOG_FORK" archive "$REFERENCE_REV" defog | tar -x -C "$REF"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REF" \
  "$RESEARCH_PYTHON" "$HERE/generate_golden.py" --pkg defog --zoo "$DEFOG_MEGAN_ZOO" \
  --revision "$REFERENCE_REV" --reference-dir "$REF" --tier "$TIER"
