#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════════════
# Unified MoE Training Speedrun
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces all speedrun_moe_*.sh scripts with a single parameterized version.
#
# Usage:
#   bash speedrun.sh --router vsa_fpe --depth 8
#   bash speedrun.sh --router hash --depth 4
#   bash speedrun.sh --router direct_fpe --depth 10 --batch-size 1
#   bash speedrun.sh --router clifford_quat_fpe --depth 8
#
# All options:
#   --router TYPE      Router type: linear, hash, vsa_random, vsa_fpe, direct_fpe,
#                      clifford_quat_fpe, clifford_quat_random, clifford_complex_fpe
#                      (default: linear)
#   --depth N          Model depth: 4, 8, 10, 12, 16, 20 (default: 8)
#   --gpus N           Number of GPUs (default: 8)
#   --batch-size N     Device batch size (default: 2)
#   --tag TAG          Override model tag (default: auto-generated from depth+router)
#   --resume STEP      Resume training from this step
#   --save-every N     Checkpoint interval (default: 250)
#   --experts N        Number of experts (default: 8)
#   --top-k K          Experts per token (default: 2)
#   --shards N         Data shards to download (default: auto from depth)
#   --skip-setup       Skip venv/tokenizer/data setup (already done)
#   --skip-data        Skip data download only
#   --only STAGE       Run only: pretrain, eval, midtrain, sft, report
#   --wandb RUN        Wandb run name (default: dummy)
#
# Examples:
#   # Full pipeline, d8, FPE router
#   bash speedrun.sh --router vsa_fpe --depth 8
#
#   # Just pretrain d10 FPE, resume from crash
#   bash speedrun.sh --router vsa_fpe --depth 10 --resume 5750 --only pretrain
#
#   # Quick d4 sweep of all routers
#   for r in hash linear vsa_random vsa_fpe direct_fpe; do
#       bash speedrun.sh --router $r --depth 4
#   done
#
#   # Clifford experiments
#   bash speedrun.sh --router clifford_quat_fpe --depth 8
#
#   # 32 experts with top-4
#   bash speedrun.sh --router vsa_fpe --depth 8 --experts 32 --top-k 4

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────────────

ROUTER="linear"
DEPTH=8
GPUS=8
BATCH_SIZE=2
TAG=""
RESUME=""
SAVE_EVERY=250
NUM_EXPERTS=8
NUM_EXPERTS_PER_TOK=2
SHARDS=""
SKIP_SETUP=false
SKIP_DATA=false
ONLY=""
WANDB_RUN="dummy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --router)       ROUTER="$2"; shift 2 ;;
        --depth)        DEPTH="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --resume)       RESUME="$2"; shift 2 ;;
        --save-every)   SAVE_EVERY="$2"; shift 2 ;;
        --experts)      NUM_EXPERTS="$2"; shift 2 ;;
        --top-k)        NUM_EXPERTS_PER_TOK="$2"; shift 2 ;;
        --shards)       SHARDS="$2"; shift 2 ;;
        --skip-setup)   SKIP_SETUP=true; shift ;;
        --skip-data)    SKIP_DATA=true; shift ;;
        --only)         ONLY="$2"; shift 2 ;;
        --wandb)        WANDB_RUN="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Derive model tag if not specified
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "$TAG" ]; then
    case $ROUTER in
        linear)                 TAG="d${DEPTH}" ;;
        hash)                   TAG="d${DEPTH}_hash" ;;
        vsa_random)             TAG="d${DEPTH}_random_vsa" ;;
        vsa_fpe)                TAG="d${DEPTH}_fpe_vsa" ;;
        direct_fpe)             TAG="d${DEPTH}_fpe_direct" ;;
        clifford_quat_fpe)      TAG="d${DEPTH}_clifford_quat_fpe" ;;
        clifford_quat_random)   TAG="d${DEPTH}_clifford_quat_random" ;;
        clifford_complex_fpe)   TAG="d${DEPTH}_clifford_complex_fpe" ;;
        clifford_complex_random) TAG="d${DEPTH}_clifford_complex_random" ;;
        *)                      TAG="d${DEPTH}_${ROUTER}" ;;
    esac
fi

# Auto-detect shards needed from depth if not specified
# ~295 * depth^3 params, 20:1 ratio, 4.8 chars/tok, 250M chars/shard
if [ -z "$SHARDS" ]; then
    case $DEPTH in
        4)  SHARDS=8 ;;
        8)  SHARDS=24 ;;
        10) SHARDS=48 ;;
        12) SHARDS=80 ;;
        16) SHARDS=200 ;;
        20) SHARDS=240 ;;
        *)  SHARDS=240 ;;
    esac
fi

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/data/$(whoami)/.cache/nanochat"
export UV_CACHE_DIR="/data/$(whoami)/.cache/uv"
mkdir -p $NANOCHAT_BASE_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Print config
# ─────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "  MoE Speedrun"
echo "═══════════════════════════════════════════════════════════════"
echo "  Router:      $ROUTER"
echo "  Depth:       $DEPTH"
echo "  Tag:         $TAG"
echo "  GPUs:        $GPUS"
echo "  Batch size:  $BATCH_SIZE"
echo "  Experts:     $NUM_EXPERTS (top-$NUM_EXPERTS_PER_TOK)"
echo "  Data shards: $SHARDS"
echo "  Save every:  $SAVE_EVERY"
[ -n "$RESUME" ] && echo "  Resume from: step $RESUME"
[ -n "$ONLY" ]   && echo "  Only stage:  $ONLY"
echo "═══════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# Build the torchrun prefix and router args
# ─────────────────────────────────────────────────────────────────────────────

RUN="torchrun --standalone --nproc_per_node=$GPUS"

# Router arg (linear is the default, so omit it)
ROUTER_ARG=""
if [ "$ROUTER" != "linear" ]; then
    ROUTER_ARG="--moe_router_type=$ROUTER"
fi

# Resume arg
RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume_from_step $RESUME"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Setup (venv, tokenizer, data)
# ─────────────────────────────────────────────────────────────────────────────

if [ "$SKIP_SETUP" = false ]; then
    # Python venv
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi

source .venv/bin/activate

if [ "$SKIP_SETUP" = false ]; then
    # Report header
    python -m nanochat.report reset

    # Tokenizer (only if not already trained)
    if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.json" ]; then
        python -m nanochat.dataset -n 8
        python -m scripts.tok_train --max_chars=2000000000 --vocab_size=65536
        python -m scripts.tok_eval
    fi
fi

# Data download
if [ "$SKIP_DATA" = false ] && [ "$SKIP_SETUP" = false ]; then
    echo "Downloading $SHARDS data shards..."
    python -m nanochat.dataset -n $SHARDS
fi

# Identity conversations for midtraining
curl -sL -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Stage: Pretrain
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "$ONLY" ] || [ "$ONLY" = "pretrain" ]; then
    echo ""
    echo ">>> PRETRAIN: $TAG"
    $RUN -m scripts.base_train -- \
        --depth=$DEPTH \
        --target_param_data_ratio=20 \
        --moe_layer_freq=2 \
        --num_experts=$NUM_EXPERTS \
        --num_experts_per_tok=$NUM_EXPERTS_PER_TOK \
        --device_batch_size=$BATCH_SIZE \
        $ROUTER_ARG \
        --run=$WANDB_RUN \
        --save_every $SAVE_EVERY \
        --model_tag=$TAG \
        $RESUME_ARG
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage: Base eval
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "$ONLY" ] || [ "$ONLY" = "eval" ]; then
    echo ""
    echo ">>> BASE LOSS: $TAG"
    $RUN -m scripts.base_loss --device_batch_size=$BATCH_SIZE --model_tag=$TAG

    echo ""
    echo ">>> BASE EVAL: $TAG"
    $RUN -m scripts.base_eval --model-tag=$TAG
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage: Midtrain
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "$ONLY" ] || [ "$ONLY" = "midtrain" ]; then
    echo ""
    echo ">>> MIDTRAIN: $TAG"
    $RUN -m scripts.mid_train -- --run=$WANDB_RUN --device_batch_size=$BATCH_SIZE --model_tag=$TAG

    echo ""
    echo ">>> MID EVAL: $TAG"
    $RUN -m scripts.chat_eval -- -i mid --model-tag=$TAG
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage: SFT
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "$ONLY" ] || [ "$ONLY" = "sft" ]; then
    echo ""
    echo ">>> SFT: $TAG"
    $RUN -m scripts.chat_sft -- --run=$WANDB_RUN --device_batch_size=$BATCH_SIZE --model_tag=$TAG

    echo ""
    echo ">>> SFT EVAL: $TAG"
    $RUN -m scripts.chat_eval -- -i sft --model-tag=$TAG
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage: Report
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "$ONLY" ] || [ "$ONLY" = "report" ]; then
    echo ""
    echo ">>> REPORT: $TAG"
    # Backup report before regenerating
    [ -f "$NANOCHAT_BASE_DIR/report.md" ] && cp "$NANOCHAT_BASE_DIR/report.md" "$NANOCHAT_BASE_DIR/report_${TAG}_backup.md"
    python -m nanochat.report generate
    cp "$NANOCHAT_BASE_DIR/report.md" "$NANOCHAT_BASE_DIR/report_${TAG}.md"
    echo "Report saved to $NANOCHAT_BASE_DIR/report_${TAG}.md"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done: $TAG"
echo "═══════════════════════════════════════════════════════════════"
