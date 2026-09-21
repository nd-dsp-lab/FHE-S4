"""Generic-text data and the GPT-NeoX tokenizer that Mamba was trained with.

Mamba's public checkpoints use the `EleutherAI/gpt-neox-20b` tokenizer (vocab
50277, padded to 50288 in the checkpoint). Using any other tokenizer would make
every perplexity number meaningless, so we hard-default to it.

Sources, chosen with `--data`:
    wikitext2        WikiText-2 raw (default). Standard, small, generic English.
                     Pulled from the Hugging Face hub as parquet.
    wikitext103      WikiText-103 raw. Same thing, ~50x bigger; use for Part 9's
                     larger token budgets.
    file:<path>      any local UTF-8 text file
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

TOKENIZER = "EleutherAI/gpt-neox-20b"

_WIKITEXT = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
}

# Held-out corpora, deliberately chosen to be UNLIKE wikitext, because the
# per-head intervals are measured on one corpus and must survive others. If they
# only work on the domain they were fitted on, the whole per-head result is an
# artefact of the eval set rather than a property of the model.
#
#   pile      NeelNanda/pile-10k -- a 10k-document sample of The Pile: web text,
#             code, papers, dialogue. The most diverse thing that is small.
#   lambada   EleutherAI/lambada_openai -- narrative passages, long-range
#             dependency by construction. Also gives us a zero-shot ACCURACY
#             metric, which perplexity alone cannot provide.
#   pubmed    scientific abstracts, a genuinely different register.
_OTHER = {
    "pile": ("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet", "text"),
    "lambada": ("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", "text"),
}


def get_tokenizer(name: str = TOKENIZER):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _wikitext_shards(repo: str, config: str, split: str) -> list[str]:
    """Resolve every parquet shard of a wikitext split.

    wikitext-2 splits are a single `-00000-of-00001.parquet`, but wikitext-103's
    TRAIN split is `-of-00002`. Hardcoding `-of-00001` silently worked for every
    corpus we used until wikitext-103 train, where it failed with a confusing
    `LocalEntryNotFoundError` that looks like a network problem. We probe the
    shard count instead. Works offline, because hf_hub_download resolves from the
    cache and simply raises for a name that was never fetched.
    """
    from huggingface_hub import hf_hub_download
    for total in (1, 2, 4, 8, 16, 32):
        names = [f"{config}/{split}-{i:05d}-of-{total:05d}.parquet" for i in range(total)]
        try:
            return [hf_hub_download(repo, n, repo_type="dataset") for n in names]
        except Exception:                                    # noqa: BLE001
            continue
    raise FileNotFoundError(
        f"no parquet shards found for {repo}:{config}/{split}. If running with "
        f"HF_HUB_OFFLINE=1, pre-fetch them from the login node first."
    )


def _read_parquet_text(repo: str, config: str, split: str) -> str:
    import pyarrow.parquet as pq
    parts = []
    for path in _wikitext_shards(repo, config, split):
        parts.append("".join(pq.read_table(path, columns=["text"]).column("text").to_pylist()))
    return "".join(parts)


def _read_flat(repo: str, path: str, column: str, max_docs: int | None = None) -> list[str]:
    """Read a single parquet/jsonl file from a dataset repo into a list of docs."""
    import json as _json

    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo, path, repo_type="dataset")
    if local.endswith(".parquet"):
        import pyarrow.parquet as pq
        docs = pq.read_table(local, columns=[column]).column(column).to_pylist()
    else:
        docs = []
        with open(local, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(_json.loads(line)[column])
    return docs[:max_docs] if max_docs else docs


def load_docs(source: str, split: str = "validation", max_docs: int | None = None) -> list[str]:
    """Return a list of documents. `load_text` joins these; task evals need them
    kept apart (LAMBADA scores the last word of each passage independently)."""
    if source.startswith("file:"):
        return [Path(source[5:]).read_text(encoding="utf-8")]
    if source in _WIKITEXT:
        repo, config = _WIKITEXT[source]
        import pyarrow.parquet as pq
        docs = []
        for local in _wikitext_shards(repo, config, split):
            docs += pq.read_table(local, columns=["text"]).column("text").to_pylist()
        return docs
    if source in _OTHER:
        repo, path, col = _OTHER[source]
        return _read_flat(repo, path, col, max_docs)
    raise ValueError(f"unknown --data {source!r}; expected one of "
                     f"{sorted(list(_WIKITEXT) + list(_OTHER))} or file:<path>")


def load_text(source: str = "wikitext2", split: str = "validation") -> str:
    """Return one big string of generic text."""
    if source.startswith("file:"):
        return Path(source[5:]).read_text(encoding="utf-8")
    if source in _WIKITEXT:
        repo, config = _WIKITEXT[source]
        return _read_parquet_text(repo, config, split)
    if source in _OTHER:
        # join with blank lines so documents do not run into each other
        return "\n\n".join(load_docs(source, split))
    raise ValueError(f"unknown --data {source!r}; expected one of "
                     f"{sorted(list(_WIKITEXT) + list(_OTHER))} or file:<path>")


DATASETS = sorted(list(_WIKITEXT) + list(_OTHER))


def tokenize_to_blocks(text: str, tokenizer, seq_len: int, max_tokens: int | None = None,
                       drop_last: bool = True) -> torch.Tensor:
    """Concatenate-and-chunk: the standard way to measure LM perplexity.

    Returns (n_blocks, seq_len) int64. No padding, no attention mask -- Mamba is
    causal and we never cross a block boundary, so every token is predicted from
    real context only (except the first token of each block, which is the usual
    and universally-accepted small pessimism).
    """
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    n = len(ids) // seq_len if drop_last else -(-len(ids) // seq_len)
    if n == 0:
        raise ValueError(f"only {len(ids)} tokens, need at least seq_len={seq_len}")
    ids = ids[: n * seq_len]
    return torch.tensor(ids, dtype=torch.long).view(n, seq_len)


def get_blocks(source: str = "wikitext2", split: str = "validation", seq_len: int = 1024,
               max_tokens: int | None = None, tokenizer=None, verbose: bool = True):
    """(blocks, tokenizer, info) -- the one function the scripts call."""
    tok = tokenizer or get_tokenizer()
    text = load_text(source, split)
    blocks = tokenize_to_blocks(text, tok, seq_len, max_tokens)
    info = {
        "data": source,
        "split": split,
        "seq_len": seq_len,
        "n_blocks": int(blocks.shape[0]),
        "n_tokens": int(blocks.numel()),
        "tokenizer": TOKENIZER,
        "text_sha256_12": hashlib.sha256(text.encode()).hexdigest()[:12],
    }
    if verbose:
        print(f"[data] {source}/{split}: {info['n_tokens']:,} tokens "
              f"-> {info['n_blocks']} blocks of {seq_len} (sha {info['text_sha256_12']})")
    return blocks, tok, info
