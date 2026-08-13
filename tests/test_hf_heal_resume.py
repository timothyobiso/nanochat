"""Phase B1 resume must be exact.

The SLURM jobs in hf/slurm/ split a 5B-token healing run across several
wall-clock-limited jobs, each continuing from the last checkpoint. If resume
drifts, the split silently changes the experiment — and it costs node-days to
notice. This runs a tiny model for four steps, kills it right after the
step-2 checkpoint, resumes, and requires the remaining steps to match the
uninterrupted run exactly.

CPU only, seconds to run.
"""

import json
import sys

import numpy as np
import pytest
import torch
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from hf import heal_olmoe

SEQ_LEN, DEVICE_BATCH, TOTAL_BATCH = 32, 2, 128   # grad_accum 2 at world=1
NUM_ITERATIONS = 4


class _Killed(Exception):
    """Stand-in for SLURM killing the job at its wall-clock limit."""


def make_shards(path, tokens_per_split={"train": 4096, "val": 1024}):
    manifest = {"splits": {}}
    for split, n in tokens_per_split.items():
        rng = np.random.default_rng(0)
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(path / f"{split}_00000.bin")
        manifest["splits"][split] = {"shards": [{"file": f"{split}_00000.bin", "num_tokens": n}],
                                     "num_tokens": n, "num_bytes": 2 * n}
    (path / "manifest.json").write_text(json.dumps(manifest))


def make_base_model(path):
    torch.manual_seed(0)
    config = OlmoeConfig(
        vocab_size=256, hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        num_experts=8, num_experts_per_tok=2, max_position_embeddings=SEQ_LEN,
        tie_word_embeddings=False, router_aux_loss_coef=0.01, use_cache=False,
    )
    OlmoeForCausalLM(config).float().save_pretrained(path)


def run_heal(monkeypatch, model_dir, data_dir, out_dir, run_name):
    """Invoke heal_olmoe.main() as the sbatch job does — always --resume auto."""
    monkeypatch.setattr(sys, "argv", [
        "heal_olmoe",
        "--model", str(model_dir),
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
        "--run-name", run_name,
        "--condition", "blend",
        "--router", "vsa_fpe",
        "--num-tokens", str(NUM_ITERATIONS * TOTAL_BATCH),
        "--total-batch-size", str(TOTAL_BATCH),
        "--seq-len", str(SEQ_LEN),
        "--device-batch-size", str(DEVICE_BATCH),
        "--anneal-steps", "2",
        "--warmup-steps", "1",
        "--calib-tokens", "64",
        "--eval-every", "100",       # only step 0 and the last step
        "--telemetry-every", "100",
        "--eval-tokens", "128",
        "--save-every", "2",
        "--resume", "auto",
        "--device", "cpu",           # MPS lacks the torch.fft coverage VSA needs
    ])
    heal_olmoe.main()


def train_losses(run_dir):
    """step -> train_loss, keeping the last record for a step if one repeats."""
    losses = {}
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            record = json.loads(line)
            if "train_loss" in record:
                losses[record["step"]] = record["train_loss"]
    return losses


def test_heal_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    data_dir, model_dir, out_dir = tmp_path / "data", tmp_path / "base", tmp_path / "runs"
    data_dir.mkdir()
    make_shards(data_dir)
    make_base_model(model_dir)

    # (1) reference: four steps straight through
    run_heal(monkeypatch, model_dir, data_dir, out_dir, "ref")
    reference = train_losses(out_dir / "ref")
    assert sorted(reference) == [0, 1, 2, 3]
    assert (out_dir / "ref" / "DONE").exists()

    # (2) same run, killed immediately after the step-2 checkpoint is written.
    # prune_checkpoints is the last thing the save block does, so raising from it
    # leaves a complete checkpoint behind, exactly like a wall-clock kill would.
    real_prune = heal_olmoe.prune_checkpoints

    def prune_then_die(run_dir, keep_models=()):
        real_prune(run_dir, keep_models)
        raise _Killed

    monkeypatch.setattr(heal_olmoe, "prune_checkpoints", prune_then_die)
    with pytest.raises(_Killed):
        run_heal(monkeypatch, model_dir, data_dir, out_dir, "seg")
    monkeypatch.setattr(heal_olmoe, "prune_checkpoints", real_prune)

    assert (out_dir / "seg" / "step_000002" / "trainer_state.pt").exists()
    assert not (out_dir / "seg" / "DONE").exists(), "a killed run must not look complete"
    assert sorted(train_losses(out_dir / "seg")) == [0, 1]

    # (3) resume and finish
    run_heal(monkeypatch, model_dir, data_dir, out_dir, "seg")
    resumed = train_losses(out_dir / "seg")

    assert (out_dir / "seg" / "DONE").exists()
    assert sorted(resumed) == [0, 1, 2, 3]
    for step in (2, 3):
        assert resumed[step] == reference[step], (
            f"step {step} diverged after resume: {resumed[step]} != {reference[step]}"
        )


def test_resume_restores_drift_reference(tmp_path, monkeypatch):
    """The drift reference is the pre-swap learned-gate routing. Recomputing it
    on resume would re-baseline against a gate that has already moved, so it is
    carried in trainer_state.pt."""
    data_dir, model_dir, out_dir = tmp_path / "data", tmp_path / "base", tmp_path / "runs"
    data_dir.mkdir()
    make_shards(data_dir)
    make_base_model(model_dir)

    run_heal(monkeypatch, model_dir, data_dir, out_dir, "run")
    state = torch.load(out_dir / "run" / "step_000004" / "trainer_state.pt", weights_only=True)

    assert state["step"] == NUM_ITERATIONS
    assert state["reference"], "no drift reference in the checkpoint"
    for layer_logits in state["reference"].values():
        assert layer_logits.shape[-1] == 2   # num_experts_per_tok
