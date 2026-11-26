#!/usr/bin/env bash
# Experiment 3 (Table 4, Figure 2): logP and concept steering on ZINC molecules, alone and together.
# The sampling needs only the zoo. Scoring needs the tables of the molecules, which are not in the zoo: build them first with
# reproduce/prepare_tables.sh zinc, and point TABLES at the folder it wrote.
source "$(dirname "$0")/common.sh"
: "${TABLES:?TABLES: the folder written by reproduce/prepare_tables.sh zinc}"

run_sharded sample_mol_joint single mol_joint/single_n1024 --SEED 2200
run_sharded sample_mol_joint probe  mol_joint/probe_n1024  --SEED 2300
run_sharded sample_mol_joint fk     mol_joint/fk_n1024     --SEED 2400

python scripts/score_mol_joint.py --root "$OUT/mol_joint" --stages single_n1024 probe_n1024 fk_n1024 \
    --conditions "$TABLES/concept_conditions.npz" --logp "$TABLES/logp.npy" --train-csv "$TABLES/molecules.csv" \
    --out "$OUT/scores/mol_joint"
