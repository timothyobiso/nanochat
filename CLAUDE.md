# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) (branched from upstream commit `201d705`) — a minimal full-stack ChatGPT clone (tokenizer → pretraining → midtraining → SFT → optional RL → eval → inference/serving) designed to run end-to-end on one 8XH100 node. This fork adds **Mixture-of-Experts support with experimental router types** (VSA/HRR, FPE, Clifford algebra, hash) for a research paper; `scripts/paper_analysis.py` generates the figures in `paper_figures/`.

Upstream's philosophy applies: single, cohesive, minimal, hackable codebase. No config objects, no model factories, no framework abstractions.

## Commands

```bash
# Environment (uv-managed; pick exactly one torch extra — they conflict)
uv sync --extra gpu     # CUDA 12.8 wheels
uv sync --extra cpu     # CPU/MPS wheels
source .venv/bin/activate

# Tests
python -m pytest tests/test_engine.py -v -s
python -m pytest tests/test_engine.py::test_kv_cache_resize -v   # single test
python -m pytest -m "not slow"                                    # skip slow marker

# Full MoE pipeline (unified, parameterized — supersedes old_speedruns/*)
bash speedrun.sh --router vsa_fpe --depth 8
bash speedrun.sh --router vsa_fpe --depth 10 --resume 5750 --only pretrain
# Key flags: --router, --depth, --experts, --top-k, --gpus, --batch-size,
#            --only {pretrain,eval,midtrain,sft,report}, --skip-setup, --skip-data, --tag

# Individual stages (note the `--` separator before script args under torchrun)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=8 --moe_layer_freq=2 --moe_router_type=vsa_fpe --model_tag=d8_fpe_vsa
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --model_tag=d8_fpe_vsa
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --model_tag=d8_fpe_vsa
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft --model-tag=d8_fpe_vsa

# Single GPU / CPU / MPS: drop torchrun (`python -m scripts.base_train ...`);
# device autodetects, gradient accumulation compensates. See dev/runcpu.sh for tiny-scale flags.

# Talk to a trained model
python -m scripts.chat_cli
python -m scripts.chat_web   # ChatGPT-style UI on port 8000

# MoE analysis
torchrun --standalone --nproc_per_node=8 -m scripts.analyze_experts -- --model-tag d8_fpe_vsa
python -m scripts.paper_analysis   # loss curves, utilization heatmaps, router latency figures
```

Wandb logging is off by default (`--run=dummy` disables it); pass `--run=<name>` to enable.

## Data & checkpoints layout

Everything lives under `$NANOCHAT_BASE_DIR` (defaults to `~/.cache/nanochat`; `speedrun.sh` sets it to `/data/$(whoami)/.cache/nanochat`): tokenizer, fineweb data shards, `base_checkpoints/`, `mid_checkpoints/`, `chatsft_checkpoints/`, and `report.md`.

- Each checkpoint dir is keyed by **model_tag** (e.g. `d8`, `d8_fpe_vsa`, `d10_hash`) — tag encodes depth + router type; `speedrun.sh` derives it automatically. `checkpoint_manager.load_model("base"|"mid"|"sft", ...)` guesses the largest/most-recent tag if none is given, so pass `--model_tag` explicitly when multiple runs coexist.
- Download data shards with `python -m nanochat.dataset -n N`. Shard count must cover ~20 tokens/param (Chinchilla) or the loader silently loops epochs; `speedrun.sh` auto-picks shards per depth.
- `checkpoint_manager._patch_missing_config_keys/_patch_missing_keys` backfill defaults so old checkpoints keep loading — extend those when adding new config keys or parameters.

## Architecture

**Pipeline stages** map 1:1 to `scripts/`: `tok_train`/`tok_eval` (rustbpe BPE tokenizer) → `base_train`/`base_loss`/`base_eval` (pretraining, CORE score) → `mid_train` (midtraining on conversations/tool use) → `chat_sft` → optional `chat_rl` (GSM8K) → `chat_eval` → `chat_cli`/`chat_web` (inference via `nanochat/engine.py`, KV cache, Python-interpreter tool calls through `nanochat/execution.py`). Each stage appends to the report card (`nanochat/report.py`). Eval/training task definitions live in `tasks/` (`TaskMixture`/`TaskSequence` in `tasks/common.py`).

**Model** (`nanochat/gpt.py`): GPT Transformer with rotary embeddings, and learnable per-layer scalars — `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` (skip connection back to the normalized input embedding). Sizing has a single knob: `--depth`; `model_dim = depth * aspect_ratio` (default 64).

**MoE** (this fork's core addition, all in `gpt.py`):
- `moe_layer_freq` controls which layers use MoE (0 = dense, 2 = every other layer — the speedrun default); `num_experts` (8) / `num_experts_per_tok` (2); load-balancing aux loss weighted by `moe_aux_loss_coeff`.
- `moe_router_type` selects the router: `linear` (standard learned), `hash` (parameter-free, position-based), `vsa_random`/`vsa_fpe` (`VSARouter`, holographic HRR bind/bundle memory), `direct_fpe` (`DirectFPERouter`), and `clifford_{quat,complex}_{fpe,random}` (`CliffordRouter`). Non-linear routers hold fixed buffers, not learned weights.
- `MoELayer.forward` is decorated `@torch.compiler.disable` because per-expert token counts are dynamic; the rest of the model compiles with `dynamic=False`. `base_train` keeps `orig_model` (uncompiled) for checkpoint saving and eval, where input shapes vary.
- MoE runs need much smaller `--device_batch_size` (speedrun default 2, vs 32 for dense upstream).

**Meta-device footgun**: the model `__init__` runs in a meta-device context, so tensor values assigned there are fake. All real initialization must happen in `init_weights()` — routers implement an `init_buffers()` hook called from there.

**Optimizers**: gradient sync happens *inside* the optimizers, not via DDP. `nanochat/muon.py` (matrix params: Polar Express orthogonalization, Adafactor-style variance reduction, cautious weight decay scaled ∝ 1/width²) and `nanochat/adamw.py` (embeddings + scalars; small params <1024 elements use `all_reduce` instead of reduce_scatter). `--weight_decay` applies to Muon only; AdamW is hardcoded to 0.

## HF-ecosystem experiments (`hf/`)

Port of the router comparison to the HuggingFace stack plus router-swap experiments on pretrained OLMoE-1B-7B (branch `hf-experiments`). **Full plan and design rationale: `docs/HF_PORT_PLAN.md`**; condensed run matrices in the 2026-08-09 `dev/LOG.md` entry.

- **transformers is pinned `>=4.57.3,<5`**: the `OlmoeSparseMoeBlock.gate` contract flips in 5.x (gate returns a 3-tuple, aux-loss collection matches on `isinstance(OlmoeTopKRouter)`). Migration notes in `hf/patch_olmoe.py`'s docstring.
- Install: `uv sync --extra gpu --group hf` (Mac dev: `--extra cpu --group hf`).
- Routers are **imported from `nanochat/gpt.py`**, never duplicated. `hf/routers.py:build_router` seeds init deterministically (per-layer seed = base + 1000·layer); `FixedRouterGate`/`BlendedRouterGate` replace the `nn.Linear` gate instance and pin their buffers to fp32 under model-wide bf16 casts.
- **Patched checkpoints must load via `hf.patch_olmoe.load_router_olmoe`** — bare `from_pretrained` silently rebuilds an unpatched model with a random gate. `hf/eval_lm.py` handles this automatically.

```bash
python -m hf.prepare_data --data-dir <dir> --num-tokens 30000000000   # FineWeb-Edu -> uint16 shards (OLMoE tokenizer)
torchrun --standalone --nproc_per_node=8 -m hf.train_olmoe -- --data-dir <dir> --out-dir <runs> --run-name S_vsa_fpe --router vsa_fpe --hidden-size 512 --num-layers 8 --num-heads 8   # Phase A
python -m hf.diagnose_router --data-dir <dir> --out b0_report.json    # B0: agreement/seeds/swap-ppl/distill (1 GPU)
torchrun --standalone --nproc_per_node=8 -m hf.heal_olmoe -- --data-dir <dir> --out-dir <runs> --run-name heal_vsa --condition blend --router vsa_fpe --b0-report b0_report.json   # B1
python -m hf.eval_lm --model <ckpt-or-name> --out results.json        # lm-eval (baseline gate: match published OLMoE numbers first)
python -m hf.analysis --mode phase_a --runs <run1>,<run2>             # figures mirroring paper_analysis.py
python -m pytest tests/test_hf_routers.py tests/test_hf_data.py tests/test_hf_diagnostics.py -v   # all CPU
```

## Experiment conventions

- `dev/LOG.md` is the running experiment log — findings, sweeps, and rationale for optimizer/architecture changes are documented there. Add entries for substantive experiments.
- `old_speedruns/` holds the superseded per-router scripts; `miniseries.sh` (depth series), `scaling_laws.sh` (FLOPs-budget sweeps), and `run1000.sh` (the $1000 d32 run) are upstream batch scripts that write CSV results under the base dir.
- Upstream PR policy is LLM-contribution disclosure; keep changes minimal and readable in that spirit.
