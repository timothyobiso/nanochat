"""Tests for hf/checkpoints.py.

prune_checkpoints deletes training output, and the SLURM jobs use DONE and
latest_checkpoint to decide whether to resubmit and where to resume from. A
mistake here either loses a run or silently restarts it, so the rules are
pinned: the newest checkpoint is never touched, milestones keep their model,
everything else goes.
"""

from hf.checkpoints import (
    checkpoint_steps,
    is_done,
    latest_checkpoint,
    mark_done,
    prune_checkpoints,
)


def make_checkpoints(run_dir, steps):
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        ckpt = run_dir / f"step_{step:06d}"
        (ckpt / "model").mkdir(parents=True)
        (ckpt / "model" / "model.safetensors").write_text("weights")
        (ckpt / "trainer_state.pt").write_text("optimizer")
    return run_dir


def test_checkpoint_discovery_ignores_other_entries(tmp_path):
    run_dir = make_checkpoints(tmp_path / "run", [250, 500, 1250])
    (run_dir / "metrics.jsonl").write_text("{}\n")
    (run_dir / "step_notanumber").mkdir()

    assert checkpoint_steps(run_dir) == [250, 500, 1250]
    assert latest_checkpoint(run_dir) == str(run_dir / "step_001250")


def test_no_checkpoints_yet(tmp_path):
    assert checkpoint_steps(tmp_path / "missing") == []
    assert latest_checkpoint(tmp_path / "missing") is None


def test_done_sentinel(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert not is_done(run_dir)
    mark_done(run_dir)
    assert is_done(run_dir)


def test_prune_keeps_newest_whole_and_milestone_models(tmp_path):
    run_dir = make_checkpoints(tmp_path / "run", [250, 500, 750, 1000])
    prune_checkpoints(run_dir, keep_models=[500, 1000])

    # newest survives intact — it is the only one a resume can start from
    assert (run_dir / "step_001000" / "trainer_state.pt").exists()
    assert (run_dir / "step_001000" / "model").is_dir()
    # milestone keeps its model for lm-eval but drops the optimizer state
    assert (run_dir / "step_000500" / "model").is_dir()
    assert not (run_dir / "step_000500" / "trainer_state.pt").exists()
    # non-milestones are gone
    assert not (run_dir / "step_000250").exists()
    assert not (run_dir / "step_000750").exists()


def test_prune_is_idempotent_and_safe_on_a_single_checkpoint(tmp_path):
    run_dir = make_checkpoints(tmp_path / "run", [250])
    for _ in range(2):
        prune_checkpoints(run_dir, keep_models=[1000])
        assert (run_dir / "step_000250" / "trainer_state.pt").exists()
