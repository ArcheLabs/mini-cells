#!/usr/bin/env python3
"""Create the separate pre-formal lock for the Shadow v2 dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ID = "shadow-cell-validation-001-v2-developmental-maturation"
VALIDATION_DIR = ROOT / "research/validations" / VALIDATION_ID
PROTOCOL = VALIDATION_DIR / "protocol.json"
LOCK = VALIDATION_DIR / "protocol-lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-dataset", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit("checkpoint must exist")
    datasets: dict[str, Path] = {}
    for value in args.seed_dataset:
        try:
            seed, raw_path = value.split("=", 1)
            datasets[str(int(seed))] = Path(raw_path)
        except ValueError as exc:
            raise SystemExit(f"invalid --seed-dataset {value!r}; use SEED=PATH") from exc
    expected_seeds = {"95311", "95312", "95313"}
    if set(datasets) != expected_seeds or any(not path.is_file() for path in datasets.values()):
        raise SystemExit("provide an existing --seed-dataset for each formal seed")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(args.checkpoint)
    dataset_shas = {seed: sha256_file(path) for seed, path in sorted(datasets.items())}
    protocol["model"]["canonical_checkpoint_sha256"] = checkpoint_sha
    protocol["formal_dataset"]["sha256"] = dataset_shas
    protocol_text = json.dumps(protocol, indent=2, sort_keys=False) + "\n"
    protocol_sha = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    lock = {
        "format": "minicells.shadow-cell-validation-001-v2.protocol-lock.v1",
        "validation_id": VALIDATION_ID,
        "status": "FROZEN",
        "protocol_sha256": protocol_sha,
        "canonical_checkpoint_sha256": checkpoint_sha,
        "formal_dataset_sha256": dataset_shas,
        "formal_seeds": [95311, 95312, 95313],
    }
    print(json.dumps({"protocol_sha256": protocol_sha, "canonical_checkpoint_sha256": checkpoint_sha, "formal_dataset_sha256": dataset_shas, "write": args.write}, indent=2, sort_keys=True))
    if args.write:
        PROTOCOL.write_text(protocol_text, encoding="utf-8")
        LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote pre-formal protocol lock: {LOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
