import subprocess
import json
import struct
import io
import hashlib
from pathlib import Path
import pyarrow.parquet as pq
import argparse

p = argparse.ArgumentParser()
p.add_argument("--tree", type=Path, required=True)
p.add_argument("--first-shard", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--proxy")
a = p.parse_args()
root = a.cache
root.mkdir(parents=True, exist_ok=True)
rev = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
entries = json.loads(a.tree.read_text())
out = []
cursor = 0
for n, e in enumerate(x for x in entries if "/train-" in x["path"]):
    if n == 0:
        if hashlib.sha256(a.first_shard.read_bytes()).hexdigest() != e["lfs"]["oid"]:
            raise RuntimeError("FIRST_SHARD_HASH_CHANGED")
        m = pq.read_metadata(a.first_shard)
        evidence = "FULL_FILE_SHA256_VERIFIED"
    else:
        size = e["size"]
        url = (
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
            + rev
            + "/"
            + e["path"]
        )

        def get(start, end, suffix):
            dest = root / (str(n) + "-" + suffix)
            subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--fail",
                    "-L",
                    *(["-x", a.proxy] if a.proxy else []),
                    "--connect-timeout",
                    "8",
                    "--max-time",
                    "35",
                    "-r",
                    str(start) + "-" + str(end),
                    url + "?audit_footer=" + suffix,
                    "-o",
                    str(dest),
                ],
                check=True,
            )
            data = dest.read_bytes()
            if len(data) != end - start + 1:
                raise RuntimeError("RANGE_RESPONSE_SIZE_INVALID")
            return data

        tail = get(size - 8, size - 1, "tail")
        length = struct.unpack("<I", tail[:4])[0]
        footer = get(size - 8 - length, size - 1, "metadata")
        assert footer[-4:] == b"PAR1"
        m = pq.read_metadata(io.BytesIO(b"PAR1" + footer))
        evidence = "PINNED_REVISION_HTTP_RANGE_FOOTER_ONLY_NOT_FULL_FILE_SHA256"
        e["footer_sha256"] = hashlib.sha256(footer).hexdigest()
    out.append(
        dict(
            path=e["path"],
            rows=m.num_rows,
            global_start=cursor,
            global_end=cursor + m.num_rows,
            lfs_sha256=e["lfs"]["oid"],
            bytes=e["size"],
            verification=evidence,
            footer_sha256=e.get("footer_sha256"),
        )
    )
    cursor += m.num_rows
    print(out[-1], flush=True)
a.output.write_text(
    json.dumps(
        dict(
            revision=rev,
            split="train",
            ordering="LEXICOGRAPHIC_FROZEN_TRAIN_PARQUET_PATHS",
            total_rows=cursor,
            shards=out,
        ),
        indent=2,
    )
    + "\n"
)
