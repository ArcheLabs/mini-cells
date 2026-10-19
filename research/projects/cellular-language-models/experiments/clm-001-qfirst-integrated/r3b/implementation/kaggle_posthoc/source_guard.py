"""Standard-library attestation of the actual immutable measurement checkout."""

from pathlib import Path
import hashlib
import subprocess

ENGINE_SHA = "ed2accbc90f541325ca2f6c53229e9c81475768d"


def verify_engine(root, expected=ENGINE_SHA):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != expected:
        raise RuntimeError("PINNED_SCIENTIFIC_ENGINE_HEAD_MISMATCH")
    tree = subprocess.check_output(["git", "ls-tree", "-rz", "HEAD"], cwd=root).split(b"\0")
    verified = {}
    for entry in tree:
        if not entry:
            continue
        description, encoded = entry.split(b"\t", 1)
        mode, kind, expected_blob = description.split()
        path = root / encoded.decode()
        if not path.exists():
            continue
        if path.is_symlink() or kind != b"blob" or not path.is_file():
            raise RuntimeError("PINNED_SCIENTIFIC_ENGINE_UNSAFE_FILE")
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != expected_blob.decode():
            raise RuntimeError("PINNED_SCIENTIFIC_ENGINE_BYTES_MISMATCH")
        verified[encoded.decode()] = hashlib.sha256(data).hexdigest()
    if not verified:
        raise RuntimeError("PINNED_SCIENTIFIC_ENGINE_MISSING")
    import json

    return {
        "measurement_engine_source_sha": head,
        "materialized_files_verified": len(verified),
        "source_files_sha256": hashlib.sha256(
            json.dumps(verified, sort_keys=True).encode()
        ).hexdigest(),
    }
