#!/usr/bin/env bash
# Shared by the reproduction scripts. Source it; it runs nothing by itself.
#
#   DEFOG_MEGAN_ZOO  where the model zoo is (default ~/.cache/defog-megan/zoo)
#   OUT              where the collected samples, logs and scores go (default ./results)
#   NUM_SHARDS       how many processes a sampling stage is split over (default: one per visible GPU, or 1 on the CPU)
#   EXTRA            extra arguments for every experiment, e.g. EXTRA="--__TESTING__ True" for a tiny smoke run
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${DEFOG_MEGAN_ZOO:=$HOME/.cache/defog-megan/zoo}"
export DEFOG_MEGAN_ZOO
: "${OUT:=$REPO/results}"
: "${EXTRA:=}"
export POLARS_SKIP_CPU_CHECK=1          # the explainer stack imports polars, which some CPUs reject without this
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
: "${NUM_SHARDS:=$GPUS}"
[ "$NUM_SHARDS" -ge 1 ] || NUM_SHARDS=1
mkdir -p "$OUT/logs"

# run_sharded <experiment> <stage> <collect-into> [experiment arguments...]
#   Runs one process per shard (shard k on GPU k when there are GPUs), waits for all of them, and gathers the run folders
#   into $OUT/<collect-into>/shard<k>/, the layout the scorers read.
run_sharded() {
    local experiment="$1" stage="$2" dest="$3"; shift 3
    local start pids=() failed=0 k
    start="$(date +%s)"
    for ((k = 0; k < NUM_SHARDS; k++)); do
        local env_gpu=()
        [ "$GPUS" -ge 1 ] && env_gpu=(CUDA_VISIBLE_DEVICES="$k")
        env ${env_gpu[@]+"${env_gpu[@]}"} python -u "experiments/$experiment.py" --__DEBUG__ False \
            --STAGE "'$stage'" --SHARD "$k" --NUM_SHARDS "$NUM_SHARDS" "$@" $EXTRA \
            > "$OUT/logs/${experiment}_${stage}_shard$k.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [ "$failed" = 0 ] || { echo "a shard of $experiment/$stage failed; see $OUT/logs" >&2; return 1; }
    python scripts/collect_shards.py "$experiment" --stage "$stage" --num-shards "$NUM_SHARDS" --since "$start" --out "$OUT/$dest"
}

# run_one <experiment> [experiment arguments...]: one process on the first GPU; prints nothing, see newest_run
run_one() {
    local experiment="$1"; shift
    python -u "experiments/$experiment.py" --__DEBUG__ False "$@" $EXTRA > "$OUT/logs/${experiment}.log" 2>&1
}

# newest_run <experiment>: the newest run folder of an experiment (call it right after run_one)
newest_run() {
    ls -dt "$REPO/experiments/results/$1"/*/ | head -1 | sed 's:/$::'
}
