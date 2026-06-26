# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.40",
#   "datasets>=2.14",
#   "numpy>=1.24",
# ]
# ///
"""
prepare_olmo3_mini.py

Downloads the OLMo-3 tokenizer from HuggingFace and tokenizes WikiText-103
into uint32 binary files for olmo3_mini_train.py.

Run once on a machine with internet access before training:

    uv run python prepare_olmo3_mini.py --data_dir ./olmo3_data

OLMo-3 tokenizer vocab size is 100,278 — larger than uint16 can hold (65,535),
so tokens are stored as uint32 (4 bytes each).

Outputs:
    {data_dir}/train.bin        WikiText-103 train  (~500 MB)
    {data_dir}/validation.bin   WikiText-103 validation
    {data_dir}/test.bin         WikiText-103 test
    {data_dir}/tokenizer/       HuggingFace tokenizer files (reused at training time)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID   = "allenai/OLMo-3-7B-Instruct"
VOCAB_SIZE = 100_278

_SOURCES = {
    "train":      ("wikitext", "wikitext-103-raw-v1", "train"),
    "validation": ("wikitext", "wikitext-103-raw-v1", "validation"),
    "test":       ("wikitext", "wikitext-103-raw-v1", "test"),
}


def prepare_split(split: str, data_dir: Path, tokenizer) -> None:
    out = data_dir / f"{split}.bin"
    if out.exists():
        print(f"  {split}: already exists, skipping ({out})")
        return

    ds_name, ds_config, ds_split = _SOURCES[split]
    print(f"  {split}: loading {ds_name} / {ds_config} …", flush=True)
    ds = load_dataset(ds_name, ds_config, split=ds_split, trust_remote_code=False)

    # Tokenize in batches; concatenate all tokens into one flat array
    all_ids: list[int] = []

    def tokenize_batch(batch):
        texts = [t for t in batch["text"] if t.strip()]
        enc   = tokenizer(texts, add_special_tokens=False)
        for ids in enc["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)

    batch_size = 512
    for start in range(0, len(ds), batch_size):
        tokenize_batch(ds[start:start + batch_size])
        if start % 10_000 == 0:
            print(f"    … {start:,}/{len(ds):,} examples  "
                  f"({len(all_ids)/1e6:.1f}M tokens)", flush=True)

    arr = np.array(all_ids, dtype=np.uint32)
    arr.tofile(out)
    print(f"  {split}: {len(arr):,} tokens → {out}  "
          f"({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./olmo3_data")
    p.add_argument("--splits",   default="train,validation,test")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = data_dir / "tokenizer"
    if tok_dir.exists():
        print(f"Tokenizer already downloaded at {tok_dir}")
    else:
        print(f"Downloading OLMo-3 tokenizer from {MODEL_ID} …")
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        tok.save_pretrained(tok_dir)
        print(f"  Saved to {tok_dir}  (vocab size = {tok.vocab_size})")

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    assert tokenizer.vocab_size == VOCAB_SIZE, (
        f"Expected vocab {VOCAB_SIZE}, got {tokenizer.vocab_size}")

    print(f"\nTokenizing WikiText-103 → {data_dir}")
    for split in args.splits.split(","):
        prepare_split(split.strip(), data_dir, tokenizer)

    print("\nDone.")


if __name__ == "__main__":
    main()
