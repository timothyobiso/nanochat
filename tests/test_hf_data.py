"""Tests for hf/data.py: shard indexing, batch tiling, exact resume."""

import json

import numpy as np
import torch

from hf.data import ShardIndex, distributed_data_loader, tokens_per_byte


def make_dataset(tmp_path, split_tokens={"train": 1000, "val": 100}):
    manifest = {"splits": {}}
    offset = 0
    for split, n in split_tokens.items():
        # two uneven shards per split, tokens = global arange so windows are checkable
        sizes = [n // 3, n - n // 3]
        shards = []
        for i, size in enumerate(sizes):
            fname = f"{split}_{i:05d}.bin"
            np.arange(offset, offset + size, dtype=np.uint16).tofile(tmp_path / fname)
            shards.append({"file": fname, "num_tokens": size})
            offset += size
        manifest["splits"][split] = {"shards": shards, "num_tokens": n, "num_bytes": 4 * n}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_shard_index_read_crosses_boundaries_and_wraps(tmp_path):
    make_dataset(tmp_path)
    index = ShardIndex(str(tmp_path), "train")
    assert index.total_tokens == 1000
    # crossing the shard boundary at 333
    assert np.array_equal(index.read(330, 6), np.arange(330, 336, dtype=np.uint16))
    # wrapping the end of the stream
    got = index.read(998, 4)
    assert np.array_equal(got, np.array([998, 999, 0, 1], dtype=np.uint16))


def test_loader_tiles_stream_without_gaps(tmp_path):
    make_dataset(tmp_path)
    B, T, world = 2, 8, 2
    loaders = [distributed_data_loader(str(tmp_path), "train", B, T, rank=r, world_size=world) for r in range(world)]
    for step in range(5):
        for r, loader in enumerate(loaders):
            inputs, targets, state = next(loader)
            start = ((step * world + r) * B * T) % 999
            expect = np.arange(start, start + B * T + 1) % 1000
            assert torch.equal(inputs.flatten(), torch.from_numpy(expect[:-1]))
            assert torch.equal(targets.flatten(), torch.from_numpy(expect[1:]))
            assert state == {"step": step + 1}


def test_loader_resume_is_exact(tmp_path):
    make_dataset(tmp_path)
    loader = distributed_data_loader(str(tmp_path), "train", 2, 8, rank=1, world_size=2)
    batches = [next(loader) for _ in range(6)]
    resume_state = batches[2][2]  # state after step index 2
    resumed = distributed_data_loader(str(tmp_path), "train", 2, 8, rank=1, world_size=2,
                                      start_step=resume_state["step"])
    for orig in batches[3:]:
        inputs, targets, _ = next(resumed)
        assert torch.equal(inputs, orig[0]) and torch.equal(targets, orig[1])


def test_tokens_per_byte(tmp_path):
    make_dataset(tmp_path)
    assert tokens_per_byte(str(tmp_path), "val") == 0.25
