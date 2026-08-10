"""
Deterministic distributed loader over pre-tokenized uint16 shards.

Shards are flat little-endian uint16 token streams written by hf/prepare_data.py
with a manifest.json alongside. The concatenated stream is addressed by a
global position; rank r at step s reads the half-open window

    pos = ((s * world_size + r) * B * T) mod (total_tokens - 1),  length B*T + 1

so consecutive batches tile the stream with a 1-token overlap (the target of a
window's last token is the next window's first token), nothing is skipped, and
resume is exact: the loader's entire state is the integer step count.
"""

import json
import os

import numpy as np
import torch


def load_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        return json.load(f)


class ShardIndex:
    """Memory-mapped view over the concatenation of a split's shards."""

    def __init__(self, data_dir, split):
        manifest = load_manifest(data_dir)
        self.manifest = manifest
        entries = manifest["splits"][split]["shards"]
        if not entries:
            raise ValueError(f"No shards for split '{split}' in {data_dir}")
        self.mmaps = []
        self.sizes = []
        for entry in entries:
            m = np.memmap(os.path.join(data_dir, entry["file"]), dtype=np.uint16, mode="r")
            assert len(m) == entry["num_tokens"], (
                f"{entry['file']}: {len(m)} tokens on disk, manifest says {entry['num_tokens']}"
            )
            self.mmaps.append(m)
            self.sizes.append(len(m))
        self.cum = np.cumsum([0] + self.sizes)
        self.total_tokens = int(self.cum[-1])

    def read(self, pos, n):
        """Read n tokens starting at pos, wrapping modulo the stream length."""
        pos = pos % self.total_tokens
        out = np.empty(n, dtype=np.uint16)
        filled = 0
        while filled < n:
            shard = int(np.searchsorted(self.cum, pos, side="right")) - 1
            offset = pos - self.cum[shard]
            take = min(n - filled, self.sizes[shard] - offset)
            out[filled:filled + take] = self.mmaps[shard][offset:offset + take]
            filled += take
            pos = (pos + take) % self.total_tokens
        return out


def distributed_data_loader(data_dir, split, B, T, rank=0, world_size=1,
                            device="cpu", start_step=0):
    """Yield (inputs, targets, state) forever. state = {'step': next_step}: pass
    it back as start_step to resume exactly where a run left off."""
    index = ShardIndex(data_dir, split)
    stride = B * T
    if index.total_tokens < stride + 1:
        raise ValueError(f"Split '{split}' has {index.total_tokens} tokens < one batch ({stride + 1})")
    use_cuda = torch.device(device).type == "cuda"
    step = start_step
    while True:
        pos = ((step * world_size + rank) * stride) % (index.total_tokens - 1)
        window = index.read(pos, stride + 1).astype(np.int64)
        scratch = torch.from_numpy(window)
        if use_cuda:
            scratch = scratch.pin_memory()
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        step += 1
        yield inputs, targets, {"step": step}


def tokens_per_byte(data_dir, split):
    """tokens/byte ratio of a split (recorded at preparation time), used to
    convert mean cross-entropy in nats/token into bits/byte:
    bpb = loss / ln(2) * tokens_per_byte."""
    info = load_manifest(data_dir)["splits"][split]
    return info["num_tokens"] / info["num_bytes"]
