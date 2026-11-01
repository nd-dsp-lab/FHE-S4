"""SCOTUS (LexGLUE), prepared exactly as in Hyunjung's 1014_Meeting.pdf, sec. 2.2.

    coastalcph/lex_glue, config scotus: 13-way single-label classification of U.S.
    Supreme Court opinions; 5,000 train / 1,400 validation / 1,400 test.

    "tokenized with the GPT-NeoX tokenizer without special tokens and an appended
    EOS token, truncated on the right to L = 512 tokens and padded on the right to
    exactly 512 tokens with the EOS token as padding token."

Note the order: EOS is appended BEFORE truncation, so a long document loses its
EOS. We keep that to match her numbers. The classifier reads the last NON-padding
position, t* = (number of real tokens) - 1.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch

from real_mamba.data import get_tokenizer

REPO = "coastalcph/lex_glue"
N_CLASSES = 13


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"scotus/{split}-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p)
    return t.column("text").to_pylist(), t.column("label").to_pylist()


def encode(texts, L: int = 512):
    tok = get_tokenizer()
    eos = tok.eos_token_id
    ids = torch.full((len(texts), L), eos, dtype=torch.long)
    last = torch.empty(len(texts), dtype=torch.long)
    cut = 16 * L   # chars: >=4x what L tokens need, so the first L tokens are unchanged
    for i, text in enumerate(texts):
        # Tokenize a prefix only: SCOTUS opinions are long, and BPE is local, so the
        # first L tokens of text[:cut] equal those of the full text. A document that
        # long has its EOS truncated away either way, so this changes nothing.
        t = tok(text[:cut], add_special_tokens=False)["input_ids"] + [eos]
        t = t[:L]
        ids[i, : len(t)] = torch.tensor(t)
        last[i] = len(t) - 1
    return ids, last


def get(split: str, L: int = 512, cache_dir: Path | None = None, max_docs: int | None = None):
    """(ids (N, L), last_idx (N,), labels (N,)), cached as a .pt per split and L."""
    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"scotus_{split}_L{L}.pt"
        if cache.exists():
            d = torch.load(cache)
            ids, last, y = d["ids"], d["last"], d["labels"]
            return (ids[:max_docs], last[:max_docs], y[:max_docs]) if max_docs else (ids, last, y)
    texts, labels = load_split(split)
    ids, last = encode(texts, L)
    y = torch.tensor(labels, dtype=torch.long)
    if cache is not None:
        torch.save({"ids": ids, "last": last, "labels": y}, cache)
    return (ids[:max_docs], last[:max_docs], y[:max_docs]) if max_docs else (ids, last, y)
