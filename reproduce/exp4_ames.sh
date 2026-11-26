#!/usr/bin/env bash
# Experiment 4 (Figure 3 and the tables of the real-world rung): mutagenicity class and concept steering, model-graded.
# Scoring needs the tables of the molecules, which are not in the zoo: build them first with reproduce/prepare_tables.sh ames,
# and point TABLES at the folder it wrote.
source "$(dirname "$0")/common.sh"
: "${TABLES:?TABLES: the folder written by reproduce/prepare_tables.sh ames}"

run_sharded sample_ames_joint single ames_joint/single_n1024 --SEED 2500
run_sharded sample_ames_joint probe  ames_joint/probe_n1024  --SEED 2600
run_sharded sample_ames_joint fk     ames_joint/fk_n1024     --SEED 2700

python scripts/score_ames_joint.py --root "$OUT/ames_joint" --stages single_n1024 probe_n1024 fk_n1024 \
    --conditions "$TABLES/concept_conditions.npz" --logp "$TABLES/prob_pred.npy" --train-csv "$TABLES/ames_defog.csv" \
    --out "$OUT/scores/ames_joint"
