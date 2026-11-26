#!/usr/bin/env bash
# Build, from the public data, the tables of the molecular groups that are not in the zoo because they hold molecules.
#
#   reproduce/prepare_tables.sh ames   # the Ames set, its memberships, label table, split and predicted class probabilities
#   reproduce/prepare_tables.sh zinc   # the ZINC molecules, their memberships and Crippen logP
#
# The folder it writes (TABLES, default $OUT/tables/<group>) is what exp3_zinc_logp.sh and exp4_ames.sh score against.
#
# Both chains rebuild the tables of the paper from the public data. A corpus step comes first (scripts/prepare_ames_defog.py,
# scripts/prepare_zinc_defog.py): it keeps the molecules the base represents, as the base decodes them, each once. Its table
# (the Ames set with its labels; the 65,434 ZINC molecules with their exact Crippen logP) is identical to the paper's, byte for
# byte. The memberships are then computed by the explainer and came out bitwise equal to the paper's in the checks (CPU runs);
# another machine's float arithmetic can differ in the last digits. The paper's own tables hold SMILES of the molecules and are
# not redistributed (see docs/zoo.md); these are the same tables.
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
    python scripts/prepare_zinc_defog.py --out "$TABLES" ${DATASET:+--dataset "$DATASET"}
    run_one prepare_concept_dataset --SOURCE_CSV "'$TABLES/molecules.csv'"
    cp "$(newest_run prepare_concept_dataset)/concept_conditions.npz" "$TABLES/concept_conditions.npz"
    python scripts/pool_logp.py --conditions "$TABLES/concept_conditions.npz" --out "$TABLES/logp.npy"
    ;;
*)
    echo "unknown group '$GROUP' (ames or zinc)" >&2; exit 1 ;;
esac
echo "tables written to $TABLES"
