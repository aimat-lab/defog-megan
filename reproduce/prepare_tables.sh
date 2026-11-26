#!/usr/bin/env bash
# Build, from the public data, the tables of the molecular groups that are not in the zoo because they hold molecules.
#
#   reproduce/prepare_tables.sh ames   # the Ames set, its memberships, label table, split and predicted class probabilities
#   reproduce/prepare_tables.sh zinc   # the ZINC molecules, their memberships and Crippen logP
#
# The folder it writes (TABLES, default $OUT/tables/<group>) is what exp3_zinc_logp.sh and exp4_ames.sh score against.
#
# For Ames the chain reproduces the tables of the paper exactly. For ZINC it does not: the paper's pool was built from a table
# of the 65,434 molecules whose source strings went through a step that the public dataset alone does not determine (the
# strings of the dataset and of the pool agree up to stereochemistry and charge notation for most, not all, molecules). A pool
# rebuilt here is of the same kind and size, so the cell classes and the numbers agree closely but not to the last digit; the
# paper's own pool table holds SMILES of ZINC molecules and is not redistributed (see docs/zoo.md).
source "$(dirname "$0")/common.sh"
GROUP="${1:?usage: prepare_tables.sh ames|zinc}"
: "${TABLES:=$OUT/tables/$GROUP}"
mkdir -p "$TABLES"

case "$GROUP" in
ames)
    ZOO="$DEFOG_MEGAN_ZOO/ames"
    python scripts/prepare_ames_defog.py --out "$TABLES" ${DATASET:+--dataset "$DATASET"}
    python scripts/prepare_ames_defog_label.py --meta "$TABLES/ames_defog_meta.csv" --out "$TABLES"
    run_one prepare_concept_dataset --SOURCE_CSV "'$TABLES/ames_defog.csv'" \
        --MEGAN_CKPT "'$ZOO/megan/model.ckpt'" --CLUSTERING_PATH "'$ZOO/megan/clustering.clu'" --PROCESSING_PATH "'$ZOO/megan/process.py'" \
        --DATASET_TYPE "'classification'" --ATOM_TYPES "['C','N','O','F','P','S','Cl','Br','I','N+','O-','N-']" --MAX_NODES 0 --MIN_NODES 0
    cp "$(newest_run prepare_concept_dataset)/concept_conditions.npz" "$TABLES/concept_conditions.npz"
    python scripts/predict_pool_property.py --group ames --kind class_probability \
        --conditions "$TABLES/concept_conditions.npz" --out "$TABLES/prob_pred.npy"
    ;;
zinc)
    python scripts/export_vgd_smiles.py --name zinc64k --out "$TABLES/molecules.csv" ${DATASET:+--dataset "$DATASET"}
    run_one prepare_concept_dataset --SOURCE_CSV "'$TABLES/molecules.csv'"
    cp "$(newest_run prepare_concept_dataset)/concept_conditions.npz" "$TABLES/concept_conditions.npz"
    python scripts/pool_logp.py --conditions "$TABLES/concept_conditions.npz" --out "$TABLES/logp.npy"
    ;;
*)
    echo "unknown group '$GROUP' (ames or zinc)" >&2; exit 1 ;;
esac
echo "tables written to $TABLES"
