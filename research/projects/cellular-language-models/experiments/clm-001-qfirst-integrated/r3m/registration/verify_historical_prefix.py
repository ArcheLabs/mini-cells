import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import argparse

p = argparse.ArgumentParser(
    description="Reproduce only historical token bytes; no model or training"
)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
base = a.root
old = base / "research/projects/cellular-computing/experiments/007-minicells-30m/artifacts"
meta = json.loads((old / "corpus-manifest.json").read_text())
tokpath = old / "tokenizer.json"
assert hashlib.sha256(tokpath.read_bytes()).hexdigest() == meta["tokenizer_sha256"]
tok = Tokenizer.from_file(str(tokpath))
eos = tok.token_to_id("<eos>")
digest = hashlib.sha256()
count = 0
tokens = 0
rows = 0
raw_index = -1
empty = 0
begin = time.time()
for batch in pq.ParquetFile(a.source).iter_batches(batch_size=1024, columns=["text"]):
    texts = batch.column(0).to_pylist()
    valid = [(i, t.strip()) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    encoded = tok.encode_batch([t for i, t in valid])
    for (i, t), enc in zip(valid, encoded):
        raw_index = rows + i
        ids = enc.ids
        if not ids:
            continue
        ids.append(eos)
        take = min(len(ids), meta["train_stream_tokens"] - tokens)
        digest.update(np.asarray(ids[:take], dtype=np.uint16).tobytes())
        tokens += take
        count += 1
        if tokens == meta["train_stream_tokens"]:
            break
    rows += len(texts)
    if count and rows % 51200 == 0:
        print(
            json.dumps(
                {"stories": count, "tokens": tokens, "seconds": round(time.time() - begin, 1)}
            ),
            flush=True,
        )
    if tokens == meta["train_stream_tokens"]:
        break
result = {
    "expected_sha256": meta["train_raw_sha256"],
    "actual_sha256": digest.hexdigest(),
    "expected_stories": meta["train_stories_consumed"],
    "actual_stories": count,
    "last_raw_index": raw_index,
    "tokens": tokens,
    "seconds": time.time() - begin,
    "source_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    "original_revision_recorded": False,
    "historical_tokenizer_sha256": meta["tokenizer_sha256"],
    "scope": "READ_ONLY_HISTORICAL_TRAIN_STREAM_REPRODUCTION_NO_MODEL_NO_TRAINING",
}
result["status"] = (
    "EXACT_HISTORICAL_TRAIN_PREFIX_REPRODUCED"
    if result["actual_sha256"] == result["expected_sha256"]
    and count == meta["train_stories_consumed"]
    else "HISTORICAL_REVISION_IDENTITY_UNRESOLVED"
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
