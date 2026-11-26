#!/usr/bin/env bash
# Experiment 1 (Tables 1 and 2): the property adapter, its Feynman-Kac steering and the concept adapter, on the colored graphs.
# Uses the models of the zoo (no training). 1024 graphs per request level and per concept.
source "$(dirname "$0")/common.sh"
ZOO="$DEFOG_MEGAN_ZOO/colorgraphs"

# --- sampling ---------------------------------------------------------------------------------------------------------
# Table 1. The base, and the cross-attention adapter at w = 1 and 2 (the base is the unconditional sample of the same run).
run_one eval_target_adapter --EXPECT_XATTN 1
ADAPTER_RUN="$(newest_run eval_target_adapter)"
# the control: the same adapter with FiLM only
run_one eval_target_adapter --ADAPTER_CKPT "'$ZOO/property_adapter_film_only.ckpt'" --EXPECT_XATTN 0
FILM_RUN="$(newest_run eval_target_adapter)"
# Feynman-Kac on the adapter at w = 2 (beta 16): 16 particles, and 128
run_one eval_color_fk --PARTICLES 16
FK16_RUN="$(newest_run eval_color_fk)"
run_one eval_color_fk --PARTICLES 128
FK128_RUN="$(newest_run eval_color_fk)"
# Table 2. The concept adapter on all 19 concepts, 1024 graphs each (shard 0 also draws the unconditional baseline).
run_sharded sample_color_concept_adapter concepts color_concepts --N_PER_CONCEPT 1024 --N_BASELINE 1024

# The tiny testing mode samples one weight and two concepts: the full tables cannot be scored from it.
if [[ "$EXTRA" == *__TESTING__* ]]; then echo "testing mode: sampled, not scored"; exit 0; fi

# --- scoring ----------------------------------------------------------------------------------------------------------
python scripts/score_color_arms.py \
    --arm "Base=$ADAPTER_RUN:baseline" \
    --arm "+xattn (w=1.0)=$ADAPTER_RUN:w1.0" --arm "+xattn (w=2.0)=$ADAPTER_RUN:w2.0" \
    --arm "+FiLM (w=1.0)=$FILM_RUN:w1.0" --arm "+FiLM (w=2.0)=$FILM_RUN:w2.0" \
    --arm "+xattn+FK (K=16)=$FK16_RUN" --arm "+xattn+FK (K=128)=$FK128_RUN" \
    --out "$OUT/scores/color_arms"
python scripts/score_color_concepts.py "$OUT/color_concepts" --out "$OUT/scores/color_concepts"
