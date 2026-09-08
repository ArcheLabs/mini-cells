from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "research" / "validations" / "jam-knowledge-mutation-001" / "protocol.json"
MANIFEST_PATH = ROOT / "research" / "datasets" / "jam-knowledge-v0.1" / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest_identity(
    protocol: Mapping[str, Any],
    manifest_path: Path = MANIFEST_PATH,
) -> str:
    expected = str(protocol["dataset"]["manifest_sha256"])
    actual = sha256_file(manifest_path)
    if actual != expected:
        raise RuntimeError(
            "JAM dataset manifest identity mismatch: "
            f"expected={expected} actual={actual} path={manifest_path}"
        )
    return actual


def verify_frozen_dataset_manifest(
    protocol_path: Path = PROTOCOL_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> str:
    return verify_manifest_identity(load_protocol(protocol_path), manifest_path)


def main() -> int:
    actual = verify_frozen_dataset_manifest()
    print(json.dumps({"dataset_manifest_sha256": actual}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
