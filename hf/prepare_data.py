"""
Tokenize a HF text dataset into flat uint16 shards for hf/train_olmoe.py.

One-time cost per corpus (rerun from scratch on interruption — shards are
written sequentially and the manifest only lands at the end). Streaming mode
avoids downloading the full parquet set; the fast tokenizer is internally
parallel, so throughput is tokenizer-bound at roughly 1-3M tokens/s.

Example (Phase A corpus, on the node):
  python -m hf.prepare_data --data-dir /data/$USER/hf_data/fineweb_edu_olmoe \
      --num-tokens 30000000000 --val-tokens 50000000

Documents are joined as [doc tokens][EOS], matching standard pretraining
packing. Byte counts per split are recorded so evals can report bits-per-byte.
"""

import argparse
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, required=True, help="output directory for shards + manifest")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", type=str, default="sample-100BT")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--num-tokens", type=int, required=True, help="train-split tokens to write")
    parser.add_argument("--val-tokens", type=int, default=50_000_000, help="val-split tokens (written first, from the head of the stream)")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per shard file")
    parser.add_argument("--doc-batch", type=int, default=1000, help="documents per tokenizer call")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    manifest_path = os.path.join(args.data_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        raise SystemExit(f"{manifest_path} exists; pass --overwrite to redo from scratch")
    os.makedirs(args.data_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token_id
    assert eos is not None, "tokenizer must define an EOS token"
    assert len(tokenizer) < 2**16, f"vocab {len(tokenizer)} does not fit uint16"

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, streaming=True)

    def token_stream():
        """Yield (np.uint16 tokens, num_utf8_bytes) per document batch."""
        batch = []
        for example in dataset:
            batch.append(example[args.text_column])
            if len(batch) == args.doc_batch:
                yield from encode_batch(batch)
                batch = []
        if batch:
            yield from encode_batch(batch)

    def encode_batch(texts):
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for text, ids in zip(texts, encoded):
            ids.append(eos)
            yield np.asarray(ids, dtype=np.uint16), len(text.encode("utf-8"))

    stream = token_stream()
    splits = {}
    # val is carved off the head of the stream so train never sees it
    for split, target in [("val", args.val_tokens), ("train", args.num_tokens)]:
        shards, written, num_bytes = [], 0, 0
        buffer, buffered = [], 0
        while written < target:
            try:
                ids, doc_bytes = next(stream)
            except StopIteration:
                raise SystemExit(
                    f"Dataset exhausted at {written + buffered:,} {split} tokens "
                    f"(wanted {target:,}); reduce --num-tokens or pick a larger config"
                )
            buffer.append(ids)
            buffered += len(ids)
            num_bytes += doc_bytes
            if buffered >= args.shard_size or written + buffered >= target:
                take = min(buffered, target - written)
                flat = np.concatenate(buffer)[:take]
                fname = f"{split}_{len(shards):05d}.bin"
                flat.tofile(os.path.join(args.data_dir, fname))
                shards.append({"file": fname, "num_tokens": int(len(flat))})
                written += len(flat)
                buffer, buffered = [], 0
                print(f"[{split}] wrote {fname}: {written:,}/{target:,} tokens", flush=True)
        splits[split] = {"shards": shards, "num_tokens": written, "num_bytes": num_bytes}

    manifest = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "tokenizer": args.tokenizer,
        "vocab_size": len(tokenizer),
        "eos_token_id": eos,
        "splits": splits,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Manifest at {manifest_path}")


if __name__ == "__main__":
    main()
