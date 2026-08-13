"""Checkpoint housekeeping shared by the Phase A and Phase B trainers.

Both trainers write ``<run_dir>/step_NNNNNN/{model/,trainer_state.pt}`` and, on
clean completion, a ``DONE`` sentinel. The SLURM jobs in ``hf/slurm/`` rely on
both: ``DONE`` means "finished, do not resubmit", and the newest ``step_*`` dir
is what the next segment passes to ``--resume``. Existence of ``metrics.jsonl``
must never be used as the completion test — a run killed at 5% has one.

Retention: only the newest checkpoint keeps its optimizer state, because that is
the only one a resume can start from. Older checkpoints listed in
``keep_models`` (the lm-eval milestones) keep ``model/`` and lose
``trainer_state.pt``; the rest are deleted. At OLMoE scale that is the
difference between ~800 GB and ~110 GB per healing condition.
"""

import os
import shutil

DONE = "DONE"


def step_of(name):
    """Step number encoded in a `step_NNNNNN` dir name, or None."""
    if not name.startswith("step_"):
        return None
    try:
        return int(name[len("step_"):])
    except ValueError:
        return None


def checkpoint_steps(run_dir):
    """Steps of the checkpoints present in run_dir, ascending."""
    if not os.path.isdir(run_dir):
        return []
    steps = (step_of(n) for n in os.listdir(run_dir))
    return sorted(s for s in steps if s is not None)


def checkpoint_path(run_dir, step):
    return os.path.join(run_dir, f"step_{step:06d}")


def latest_checkpoint(run_dir):
    """Newest checkpoint dir, or None. This is what --resume wants."""
    steps = checkpoint_steps(run_dir)
    return checkpoint_path(run_dir, steps[-1]) if steps else None


def is_done(run_dir):
    return os.path.exists(os.path.join(run_dir, DONE))


def mark_done(run_dir):
    with open(os.path.join(run_dir, DONE), "w") as f:
        f.write("ok\n")


def prune_checkpoints(run_dir, keep_models=()):
    """Keep the newest checkpoint whole; reduce milestones to model-only; drop
    everything else. Safe to call after every save."""
    keep_models = set(keep_models)
    steps = checkpoint_steps(run_dir)
    for step in steps[:-1]:  # never touch the newest — resume depends on it
        path = checkpoint_path(run_dir, step)
        if step in keep_models:
            state = os.path.join(path, "trainer_state.pt")
            if os.path.exists(state):
                os.remove(state)
        else:
            shutil.rmtree(path, ignore_errors=True)
