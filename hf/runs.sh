#!/bin/bash
# Single source of truth for the HF experiment matrices.
#
# Sourced by hf/run_phase_a.sh, hf/run_phase_b.sh and the SLURM jobs in
# hf/slurm/, so the matrix is defined exactly once. Each Phase A entry is
#   "<run-name> <size> <router> <seed> [extra flags…]"
# and the run name is explicit rather than derived, because ablations differ
# from matrix runs only by a flag and would otherwise collide.
#
# Sizes and token budgets come from docs/HF_PORT_PLAN.md §5:
#   S ~110M total / 2.2B tokens, M ~275M / 5.5B, L ~573M / 11.5B (Chinchilla 20:1)

# size -> HIDDEN LAYERS HEADS LR DEV_BS  (device batch retuned at the pilot)
size_params() {
    case "$1" in
        S) HIDDEN=512;  LAYERS=8;  HEADS=8;  LR=6e-4; DEV_BS=16 ;;
        M) HIDDEN=768;  LAYERS=12; HEADS=12; LR=5e-4; DEV_BS=8  ;;
        L) HIDDEN=1024; LAYERS=16; HEADS=16; LR=4e-4; DEV_BS=4  ;;
        *) echo "unknown size: $1" >&2; return 1 ;;
    esac
}

ROUTERS_DEFAULT=(linear hash vsa_random vsa_fpe direct_fpe)

# ── Phase A ──────────────────────────────────────────────────────────────────
# The matrix: 5 routers x (S and M at 2 seeds, L at 1) = 25 runs.
PHASE_A_MATRIX=()
for size in S M L; do
    if [[ $size == L ]]; then seeds=(0); else seeds=(0 1); fi
    for router in "${ROUTERS_DEFAULT[@]}"; do
        for seed in "${seeds[@]}"; do
            PHASE_A_MATRIX+=("${size}_${router}_s${seed} $size $router $seed")
        done
    done
done

# Ablations. Names carry an explicit suffix so they never collide with the
# matrix entry that shares their size/router/seed — without one, a name-derived
# scheme silently reuses (and then skips) the matrix run.
#   *_aux0     : load-balancing aux loss off, isolating its effect on utilization
#   *_normtopk : norm_topk_prob=True, which renormalizes the selected gate
#                weights to sum to 1. This is the probe for the gate-mass
#                confound — with norm_topk_prob=False (the host default) a
#                low-variance fixed router leaves the selected mass near
#                top_k/E, down-scaling the MoE branch relative to a learned
#                gate. Run for both linear and vsa_fpe so the gap is measurable.
PHASE_A_ABLATIONS=(
    "S_linear_s0_aux0       S linear  0 --aux-loss-coeff 0.0"
    "S_vsa_fpe_s0_aux0      S vsa_fpe 0 --aux-loss-coeff 0.0"
    "S_linear_s0_normtopk   S linear  0 --norm-topk-prob"
    "S_vsa_fpe_s0_normtopk  S vsa_fpe 0 --norm-topk-prob"
    "M_clifford_quat_fpe_s0 M clifford_quat_fpe 0"
)

# What the SLURM array in hf/slurm/02_train.sbatch indexes into.
PHASE_A_RUNS=("${PHASE_A_MATRIX[@]}" "${PHASE_A_ABLATIONS[@]}")

# The pilot: measure real tok/s on the cheapest size before committing the rest.
PHASE_A_PILOT=(S_linear_s0 S_vsa_fpe_s0)

# ── Phase B1 ─────────────────────────────────────────────────────────────────
# "<run-name> <condition> <router>". The control's router is unused.
PHASE_B_RUNS=(
    "heal_control           control   linear"
    "heal_blend_vsa_fpe     blend     vsa_fpe"
    "heal_blend_vsa_random  blend     vsa_random"
    "heal_hard_swap_vsa_fpe hard_swap vsa_fpe"
)
