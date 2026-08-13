#!/bin/bash
# Submit the whole HF router pipeline as dependent SLURM jobs.
#
#   bash hf/slurm/e2e.sh                 # everything, in dependency order
#   bash hf/slurm/e2e.sh --dry-run       # print the sbatch commands, submit nothing
#   bash hf/slurm/e2e.sh --check         # preflight only
#   bash hf/slurm/e2e.sh --only train    # one stage, no dependencies
#
# Stages: data train baseline b0 heal evals figures
#
# Long stages are submitted as several identical jobs chained with
# --dependency=afterany. Each one resumes from the last checkpoint and exits in
# seconds once the run is DONE, so a stage that needs more than one 24h slot
# just uses the next job in its chain, and over-submitting is free. afterany
# (not afterok) is deliberate: a wall-clock kill is a job failure, and it is
# exactly the case the next segment exists to continue.
set -euo pipefail

CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
source "$CONFIG"
source "$REPO_DIR/hf/runs.sh"
export CONFIG

STAGES=(data train baseline b0 heal evals figures)
ONLY=""
DRY_RUN=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --only)    ONLY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --check)   CHECK_ONLY=true; shift ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$ONLY" ]] && [[ ! " ${STAGES[*]} " == *" $ONLY "* ]]; then
    echo "unknown stage '$ONLY'; expected one of: ${STAGES[*]}" >&2
    exit 1
fi

want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

# ── preflight ────────────────────────────────────────────────────────────────

preflight() {
    local problems=0
    say() { echo "  $*"; }
    warn() { echo "  WARNING: $*"; problems=$((problems + 1)); }

    echo "config:"
    say "partition   $PARTITION${ACCOUNT:+  account $ACCOUNT}"
    say "big nodes   $BIG_NODES"
    say "repo        $REPO_DIR"
    say "data        $DATA_DIR"
    say "phase A out $OUT_DIR"
    say "phase B out $HEAL_DIR"
    say "segments    A=$SEGMENTS_A B=$SEGMENTS_B"
    echo "checks:"

    [[ -d "$VENV" ]] || warn "no venv at $VENV — run: uv sync --extra gpu --group hf"
    command -v sbatch >/dev/null || warn "sbatch not on PATH; is this a login node?"
    if command -v sinfo >/dev/null; then
        sinfo -h -p "$PARTITION" >/dev/null 2>&1 || warn "partition '$PARTITION' not visible to sinfo"
    fi

    if [[ -f "$DATA_DIR/manifest.json" ]]; then
        say "data shards present"
    else
        local hours=$((DATA_TOKENS / 1000000 / 3600))
        say "data not built yet (~${hours}h at 1M tok/s, and prepare_data is not resumable)"
        [[ $hours -gt 20 ]] && warn "DATA_TOKENS=$DATA_TOKENS risks the 24h cap; lower it"
    fi

    # ~450GB phase B (pruned) + ~150GB phase A + shards at 2 bytes/token
    local need_gb=$((600 + DATA_TOKENS * 2 / 1000000000))
    local root="${OUT_DIR%/*}"
    if [[ -d "$root" ]]; then
        local free_gb
        free_gb=$(df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc '0-9') || free_gb=""
        if [[ -n "$free_gb" ]]; then
            say "disk: ${free_gb}G free at $root, need ~${need_gb}G"
            [[ "$free_gb" -lt "$need_gb" ]] && warn "not enough free disk"
        fi
    else
        warn "$root does not exist"
    fi

    say "phase A runs: ${#PHASE_A_RUNS[@]}   phase B conditions: ${#PHASE_B_RUNS[@]}"
    echo
    [[ $problems -eq 0 ]] && echo "preflight OK" || echo "preflight finished with $problems warning(s)"
}

preflight
$CHECK_ONLY && exit 0
echo

$DRY_RUN || mkdir -p "$LOG_DIR" "$OUT_DIR" "$HEAL_DIR"

# ── submission helpers ───────────────────────────────────────────────────────

# submit <script> [extra sbatch args…] -> job id on stdout
submit() {
    local script="$1"; shift
    local cmd=(sbatch --parsable --chdir="$LOG_DIR" --partition="$PARTITION")
    [[ -n "$ACCOUNT" ]] && cmd+=(--account="$ACCOUNT")
    cmd+=("$@" "$REPO_DIR/hf/slurm/$script")
    if $DRY_RUN; then
        echo "${cmd[*]}" >&2
        echo "DRY"
    else
        "${cmd[@]}"
    fi
}

# chain <segments> <script> [extra sbatch args…] -> id of the last job
# The first segment inherits $dep (may be empty); the rest wait on their
# predecessor with afterany.
chain() {
    local segments="$1" script="$2" dep="$3"; shift 3
    local jid=""
    for ((i = 0; i < segments; i++)); do
        jid=$(submit "$script" ${dep:+--dependency="$dep"} "$@")
        dep="afterany:$jid"
    done
    echo "$jid"
}

announce() { echo "$1: $2"; }

# Note there is no special handling for --only: a skipped stage never sets its
# job id, so the downstream ${JOB:+--dependency=…} simply expands to nothing.

BIG=(--nodelist="$BIG_NODES" "$GRES_MULTI")
ONE=("$GRES_ONE")
A_LAST=$((${#PHASE_A_RUNS[@]} - 1))
B_LAST=$((${#PHASE_B_RUNS[@]} - 1))

# ── the pipeline ─────────────────────────────────────────────────────────────

DATA_JOB=""; TRAIN_JOB=""; BASE_JOB=""; B0_JOB=""; HEAL_JOB=""; EVAL_JOB=""

if want data; then
    DATA_JOB=$(submit 01_data.sbatch)
    announce data "$DATA_JOB"
fi

if want train; then
    TRAIN_JOB=$(chain "$SEGMENTS_A" 02_train.sbatch "${DATA_JOB:+afterok:$DATA_JOB}" \
        --array="0-${A_LAST}%2" "${BIG[@]}")
    announce train "$TRAIN_JOB  (${#PHASE_A_RUNS[@]} runs x $SEGMENTS_A segments)"
fi

if want baseline; then
    BASE_JOB=$(submit 03_baseline.sbatch ${DATA_JOB:+--dependency=afterok:$DATA_JOB} "${ONE[@]}")
    announce baseline "$BASE_JOB"
fi

if want b0; then
    B0_JOB=$(chain 2 04_diagnose.sbatch "${DATA_JOB:+afterok:$DATA_JOB}" "${ONE[@]}")
    announce b0 "$B0_JOB"
fi

if want heal; then
    # gate on the baseline eval as well as B0: a swap result means nothing until
    # the unmodified checkpoint reproduces its published numbers.
    upstream=""
    [[ -n "$B0_JOB$BASE_JOB" ]] && upstream="afterok:${B0_JOB:+$B0_JOB:}${BASE_JOB}"
    upstream="${upstream%:}"
    HEAL_JOB=$(chain "$SEGMENTS_B" 05_heal.sbatch "$upstream" \
        --array="0-${B_LAST}%2" "${BIG[@]}")
    announce heal "$HEAL_JOB  (${#PHASE_B_RUNS[@]} conditions x $SEGMENTS_B segments)"
fi

if want evals; then
    EVAL_JOB=$(submit 06_evals.sbatch ${HEAL_JOB:+--dependency=afterany:$HEAL_JOB} \
        --array="0-${B_LAST}%2" "${ONE[@]}")
    announce evals "$EVAL_JOB"
fi

if want figures; then
    last="${EVAL_JOB:-$TRAIN_JOB}"
    submit 07_figures.sbatch ${last:+--dependency=afterany:$last} "${ONE[@]}" >/dev/null
    announce figures "submitted"
fi

echo
echo "logs: $LOG_DIR    queue: squeue -u $(whoami)"
