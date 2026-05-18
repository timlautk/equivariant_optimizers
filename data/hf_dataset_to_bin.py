#!/usr/bin/env python3
"""
Generic Hugging Face dataset -> binary shard preprocessor.

Inspired by:
https://github.com/microsoft/dion/blob/main/data/fineweb.py

Example:
    python hf_dataset_to_bin.py \
        --dataset HuggingFaceFW/fineweb-edu \
        --dataset-config sample-10BT \
        --split train \
        --text-column text \
        --tokenizer openai/gpt-oss-20b \
        --output-dir ./fineweb_edu_10B_gpt-oss \
        --val-first-shard
"""

import os
import json
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


MAGIC = 20260317
HEADER_SIZE_INT32 = 256
VERSION = 2

# Header layout:
# header[0] = magic
# header[1] = version
# header[2] = number of tokens
# header[3] = bytes per token (2 or 4)
# header[4] = reserved: delimiter token id, or -1 if none
# header[5] = reserved: number of prepended delimiter tokens per doc
# header[6] = reserved: number of appended delimiter tokens per doc
# header[7:] reserved for future use


def choose_token_dtype(max_token_id: int):
    if max_token_id < 2**16:
        return np.uint16
    if max_token_id < 2**32:
        return np.uint32
    raise ValueError(
        f"Token id space too large: max_token_id={max_token_id} does not fit in uint32."
    )


def write_datafile(filename: str, toks: np.ndarray, bytes_per_token: int, delimiter_id: int, n_prepend: int, n_append: int):
    assert len(toks) < 2**31, "token count too large"

    header = np.zeros(HEADER_SIZE_INT32, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(toks)
    header[3] = bytes_per_token
    header[4] = delimiter_id if delimiter_id is not None else -1
    header[5] = n_prepend
    header[6] = n_append

    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def resolve_delimiter_id(tokenizer, mode: str):
    """
    mode:
      - eos
      - bos
      - sep
      - none
    """
    if mode == "none":
        return None

    token_id = None
    if mode == "eos":
        token_id = tokenizer.eos_token_id
    elif mode == "bos":
        token_id = tokenizer.bos_token_id
    elif mode == "sep":
        token_id = tokenizer.sep_token_id
    else:
        raise ValueError(f"Unknown delimiter mode: {mode}")

    if token_id is None:
        raise ValueError(
            f"Tokenizer '{tokenizer.name_or_path}' does not define {mode}_token_id. "
            f"Use another mode or set --prepend-token none / --append-token none."
        )
    return int(token_id)


# Globals populated inside worker init
_TOKENIZER = None
_TEXT_COLUMN = None
_PREPEND_ID = None
_APPEND_ID = None
_OUT_DTYPE = None


def worker_init(tokenizer_name: str, text_column: str, prepend_id, append_id, out_dtype_str: str, use_fast: bool):
    global _TOKENIZER, _TEXT_COLUMN, _PREPEND_ID, _APPEND_ID, _OUT_DTYPE
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)
    _TOKENIZER.model_max_length = int(1e30)  # disable length warning during preprocessing
    _TEXT_COLUMN = text_column
    _PREPEND_ID = prepend_id
    _APPEND_ID = append_id
    _OUT_DTYPE = np.dtype(out_dtype_str)


def tokenize_example(example):
    text = example[_TEXT_COLUMN]
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    ids = _TOKENIZER(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    out = []
    if _PREPEND_ID is not None:
        out.append(_PREPEND_ID)
    out.extend(ids)
    if _APPEND_ID is not None:
        out.append(_APPEND_ID)

    arr = np.asarray(out, dtype=np.int64)
    if arr.size:
        if arr.min() < 0:
            raise ValueError("Negative token id encountered.")
        max_allowed = np.iinfo(_OUT_DTYPE).max
        if arr.max() > max_allowed:
            raise ValueError(
                f"Token id {arr.max()} exceeds dtype limit {max_allowed} for {_OUT_DTYPE}."
            )

    return arr.astype(_OUT_DTYPE, copy=False)


def dataset_iter(
    dataset_name: str,
    dataset_config: str,
    split: str,
    streaming: bool,
    data_files: str | None,
    revision: str | None,
    cache_dir: str | None = None,
):
    kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": streaming,
        "cache_dir": cache_dir,
    }
    if dataset_config:
        kwargs["name"] = dataset_config
    if data_files:
        # Accept JSON string or a plain path pattern.
        try:
            kwargs["data_files"] = json.loads(data_files)
        except json.JSONDecodeError:
            kwargs["data_files"] = data_files
    if revision:
        kwargs["revision"] = revision

    return load_dataset(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Generic HF dataset preprocessing")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset path, e.g. HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", type=str, default="sample-10BT", help="Optional dataset config/name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--revision", type=str, default=None, help="Optional dataset revision")
    parser.add_argument("--data-files", type=str, default=None, help="Optional JSON string or path/pattern for local/remote files")
    parser.add_argument("--text-column", type=str, default="text", help="Column containing text")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer name/path")
    parser.add_argument("--use-fast", action="store_true", help="Use fast tokenizer backend")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for cache and lock files used by load_dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output shards")
    parser.add_argument("--prefix", type=str, default="dataset", help="Output file prefix")
    parser.add_argument("--shard-size", type=int, default=10**8, help="Tokens per shard")
    parser.add_argument("--streaming", action="store_true", help="Use streaming=True in load_dataset")
    parser.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 1) - 2), help="Worker count")
    parser.add_argument(
        "--prepend-token",
        choices=["eos", "bos", "sep", "none"],
        default="eos",
        help="Token to prepend to each document",
    )
    parser.add_argument(
        "--append-token",
        choices=["eos", "bos", "sep", "none"],
        default="none",
        help="Token to append to each document",
    )
    parser.add_argument(
        "--val-first-shard",
        action="store_true",
        help="Name shard 0 as val and later shards as train, matching the original fineweb.py convention",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=args.use_fast)
    tokenizer.model_max_length = int(1e30)

    prepend_id = resolve_delimiter_id(tokenizer, args.prepend_token)
    append_id = resolve_delimiter_id(tokenizer, args.append_token)

    # Use len(tokenizer)-1 instead of vocab_size-1, because added tokens may extend IDs.
    max_token_id = len(tokenizer) - 1
    if prepend_id is not None:
        max_token_id = max(max_token_id, prepend_id)
    if append_id is not None:
        max_token_id = max(max_token_id, append_id)

    out_dtype = choose_token_dtype(max_token_id)
    bytes_per_token = np.dtype(out_dtype).itemsize

    ds = dataset_iter(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        streaming=args.streaming,
        data_files=args.data_files,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=out_dtype)
    token_count = 0
    progress_bar = None

    with mp.Pool(
        processes=args.num_proc,
        initializer=worker_init,
        initargs=(
            args.tokenizer,
            args.text_column,
            prepend_id,
            append_id,
            np.dtype(out_dtype).name,
            args.use_fast,
        ),
    ) as pool:
        for tokens in pool.imap(tokenize_example, ds, chunksize=16):
            n = len(tokens)

            if token_count + n < args.shard_size:
                all_tokens_np[token_count : token_count + n] = tokens
                token_count += n

                if progress_bar is None:
                    progress_bar = tqdm(
                        total=args.shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_index}",
                    )
                progress_bar.update(n)
            else:
                split_name = "val" if (args.val_first_shard and shard_index == 0) else "train"
                filename = os.path.join(
                    args.output_dir,
                    f"{args.prefix}_{split_name}_{shard_index:06d}.bin",
                )

                remainder = args.shard_size - token_count
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=args.shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_index}",
                    )
                progress_bar.update(remainder)

                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(
                    filename,
                    all_tokens_np,
                    bytes_per_token=bytes_per_token,
                    delimiter_id=prepend_id if prepend_id is not None else append_id,
                    n_prepend=0 if prepend_id is None else 1,
                    n_append=0 if append_id is None else 1,
                )

                shard_index += 1
                progress_bar.close()
                progress_bar = None

                leftover = n - remainder
                all_tokens_np[:leftover] = tokens[remainder:]
                token_count = leftover

        if progress_bar is not None:
            progress_bar.close()

    if token_count != 0:
        split_name = "val" if (args.val_first_shard and shard_index == 0) else "train"
        filename = os.path.join(
            args.output_dir,
            f"{args.prefix}_{split_name}_{shard_index:06d}.bin",
        )
        write_datafile(
            filename,
            all_tokens_np[:token_count],
            bytes_per_token=bytes_per_token,
            delimiter_id=prepend_id if prepend_id is not None else append_id,
            n_prepend=0 if prepend_id is None else 1,
            n_append=0 if append_id is None else 1,
        )


if __name__ == "__main__":
    main()