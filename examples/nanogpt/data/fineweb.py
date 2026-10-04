"""Download FineWeb and write GPT-2 token shards for nanoGPT.

Adapted from https://github.com/KellerJordan/modded-nanogpt/blob/master/data/fineweb.py
Copyright (c) 2024 Keller Jordan. See LICENSE in this directory.

Run from the repository root:
    python -m examples.nanogpt.data.fineweb
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def write_datafile(filename, tokens):
    """Write a 256-int32 header followed by uint16 GPT-2 token IDs."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 1 or len(tokens) >= 2**31:
        raise ValueError("Expected fewer than 2**31 tokens in a one-dimensional array")
    if not np.issubdtype(tokens.dtype, np.integer) or not ((0 <= tokens) & (tokens < 2**16)).all():
        raise ValueError("Token IDs must be integers in the uint16 range")
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    print(f"Writing {len(tokens):,} tokens to {filename}")
    with open(filename, "wb") as file:
        file.write(header.tobytes())
        file.write(tokens.astype("<u2", copy=False).tobytes())


def init_tokenizer():
    """Initialize once per worker, including under spawn and forkserver."""
    import tiktoken

    global tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")


def tokenize(document):
    tokens = [tokenizer.eot_token]
    tokens.extend(tokenizer.encode_ordinary(document["text"]))
    if any(token < 0 or token >= 2**16 for token in tokens):
        raise ValueError("Token dictionary is too large for uint16")
    return np.asarray(tokens, dtype=np.uint16)


def write_shards(documents, output_dir, shard_size):
    """Use the first shard for validation and subsequent shards for training."""
    if not 0 < shard_size < 2**31:
        raise ValueError("shard_size must be between 1 and 2**31 - 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = np.empty(shard_size, dtype=np.uint16)
    shard_index = token_count = 0
    progress = None

    def save_shard():
        split = "val" if shard_index == 0 else "train"
        path = output_dir / f"fineweb_{split}_{shard_index:06d}.bin"
        write_datafile(path, buffer[:token_count])

    try:
        for tokens in documents:
            position = 0
            while position < len(tokens):
                if progress is None:
                    progress = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                count = min(shard_size - token_count, len(tokens) - position)
                buffer[token_count:token_count + count] = tokens[position:position + count]
                token_count += count
                position += count
                progress.update(count)
                if token_count == shard_size:
                    save_shard()
                    shard_index += 1
                    token_count = 0
                    progress.close()
                    progress = None
        if token_count:
            save_shard()
    finally:
        if progress is not None:
            progress.close()


def main():
    parser = argparse.ArgumentParser(description="Download and tokenize FineWeb for nanoGPT")
    parser.add_argument("-v", "--version", choices=("10B", "100B"), default="10B")
    parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Tokens per shard")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    args = parser.parse_args()
    if not 0 < args.shard_size < 2**31:
        parser.error("--shard_size must be between 1 and 2**31 - 1")
    if args.workers < 1:
        parser.error("--workers must be positive")

    from datasets import load_dataset

    output_dir = Path(__file__).resolve().parent / f"fineweb{args.version}"
    dataset = load_dataset("HuggingFaceFW/fineweb", name=f"sample-{args.version}T", split="train")
    with mp.Pool(args.workers, initializer=init_tokenizer) as pool:
        documents = pool.imap(tokenize, dataset, chunksize=16)
        write_shards(documents, output_dir, args.shard_size)


if __name__ == "__main__":
    main()
