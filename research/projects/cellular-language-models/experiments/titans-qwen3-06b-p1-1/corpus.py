"""Reproduce P1 prefixes and select a disjoint hash-priority holdout."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p1-meta"))
from prepare_data import articles, CORPUS, SOURCE


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def select(train_rows, validation_rows, encode, *, tokens=2048,
           train_size=200, holdout_size=64, compat_size=8):
    train, holdout, compat = [], [], []
    seen_prefixes = set()
    seen_docs = set()

    def candidates(rows, split):
        for index, text in enumerate(articles(rows)):
            prefix = encode(text)[:tokens]
            if len(prefix) != tokens:
                continue
            doc_hash = hashlib.sha256(text.encode()).hexdigest()
            token_hash = digest(prefix)
            if token_hash in seen_prefixes or doc_hash in seen_docs:
                continue
            seen_prefixes.add(token_hash)
            seen_docs.add(doc_hash)
            yield ({"id": f"{split}:{index}", "title": text.splitlines()[0],
                    "source_split": split, "token_count": len(prefix),
                    "document_sha256": doc_hash, "prefix_sha256": token_hash}, prefix)

    for row in candidates(train_rows, "train"):
        if len(train) < train_size:
            train.append(row)
        else:
            holdout.append(row)
            holdout.sort(key=lambda item: item[0]["document_sha256"])
            del holdout[holdout_size:]
    for row in candidates(validation_rows, "validation"):
        compat.append(row)
        if len(compat) == compat_size:
            break
    partitions = {"train": train, "primary_holdout": holdout, "p1_compat_validation": compat}
    if tuple(map(len, partitions.values())) != (train_size, holdout_size, compat_size):
        raise ValueError(f"Insufficient qualifying documents: { {k: len(v) for k, v in partitions.items()} }")
    metadata = {}
    for split, rows in partitions.items():
        documents = [dict(row, experiment_split=split) for row, _ in rows]
        metadata[split] = {"documents": documents, "sha256": digest(documents)}
    return {split: [prefix for _, prefix in rows] for split, rows in partitions.items()}, {
        "dataset": SOURCE, "dataset_config": "wikitext-103-raw-v1",
        "dataset_revision": CORPUS, "episode_tokens": tokens, "chunk_size": 512,
        "splits": metadata,
    }
