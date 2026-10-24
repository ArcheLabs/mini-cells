"""Small R3m-E adapters over the repository's immutable runtime helpers."""

from pathlib import Path
import hashlib
import json
import os
import tempfile

from minicells_r3e.common import canonical, digest, fingerprint, read, source_sha

BASE = Path(__file__).resolve().parents[2]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()
