#!/usr/bin/env bash
# Experiment 2 (Table 3): property and concept steering together on the colored graphs, 6 arms x 30 cells x 1024 graphs.
# The seeds are those of the paper's runs.
source "$(dirname "$0")/common.sh"

# uncond, property (w 2), concept (w 1) and joint (1, 1): no explainer needed
run_sharded sample_color_joint probe color_joint/probe_n1024 \
    --SEED 2000 --N_CONCEPT 1024 --N_PER_CELL 1024 --WEIGHT_GRID "[[1.0,1.0]]"
# joint+fk and fk_only: the explainer is the Feynman-Kac energy (checked against the zoo's reference first)
run_sharded sample_color_joint fk color_joint/fk_n1024 \
    --SEED 2100 --N_PER_CELL 1024 --FK_WEIGHTS "[1.0,1.0]"

python scripts/score_color_joint.py --root "$OUT/color_joint" --stages probe_n1024 fk_n1024 --out "$OUT/scores/color_joint"
