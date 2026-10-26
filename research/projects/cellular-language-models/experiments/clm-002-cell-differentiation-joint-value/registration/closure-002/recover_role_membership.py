"""Reconcile six published role memberships; no model or candidate selection access."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recover(source, output):
    import pyarrow.parquet as pq

    base = Path(__file__).resolve().parent
    bindings = json.loads((base / "RECOVERY_BINDINGS.json").read_text())
    if digest(source) != bindings["source_shard_sha256"]:
        raise RuntimeError("FROZEN_SOURCE_SHARD_HASH_MISMATCH")
    evidence = base / "source-evidence"
    for row in bindings["evidence"]:
        path = evidence / row["file"]
        if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise RuntimeError("REMOTE_EVIDENCE_BYTES_CHANGED")
    proofs = {}
    expectations = {}
    for experiment in ("r3k", "r3l"):
        boundary_path = evidence / (experiment + "_DATA_BOUNDARY_PROOF.json")
        manifest = json.loads((evidence / (experiment + "_RUN_MANIFEST.json")).read_text())
        item = next(r for r in manifest["artifacts"] if r["path"] == "DATA_BOUNDARY_PROOF.json")
        if item["bytes"] != boundary_path.stat().st_size or item["sha256"] != digest(boundary_path):
            raise RuntimeError("PUBLISHED_MANIFEST_BOUNDARY_HASH_MISMATCH")
        proof = json.loads(boundary_path.read_text())
        if proof["revision"] != bindings["revision"] or proof["confirmation"] != "SEALED":
            raise RuntimeError("PUBLISHED_SOURCE_IDENTITY_MISMATCH")
        for role, metadata in proof["roles"].items():
            claim = bindings["claim_bindings"][experiment + "/" + role]
            if (
                metadata["requested_range"] != claim["requested_range"]
                or manifest["identity"]["config_sha256"] != claim["config_sha256"]
            ):
                raise RuntimeError("PUBLISHED_ROLE_AND_ORIGINAL_CLAIM_IDENTITY_DIFFER")
            rows = metadata["accepted"] + metadata["excluded"]
            if sorted(r["story_index"] for r in rows) != list(range(*metadata["requested_range"])):
                raise RuntimeError("PUBLISHED_ROLE_MEMBERSHIP_INCOMPLETE")
            for row in rows:
                index = row["story_index"]
                if index in expectations and expectations[index] != row["text_sha256"]:
                    raise RuntimeError("PUBLISHED_STORY_HASH_CONFLICT")
                expectations[index] = row["text_sha256"]
            proofs[experiment + "/" + role] = {
                "original_claim_id": claim["claim_id"],
                "requested_range": metadata["requested_range"],
                "accepted_count": len(metadata["accepted"]),
                "excluded_count": len(metadata["excluded"]),
                "membership_sha256": hashlib.sha256(
                    json.dumps(metadata, sort_keys=True).encode()
                ).hexdigest(),
                "historical_identity": manifest["identity"],
                "boundary_evidence_sha256": digest(boundary_path),
                "original_asset_arrays_verified": False,
                "exact_evaluation_windows_verified": False,
            }
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != bindings["source_shard_rows"]:
        raise RuntimeError("FROZEN_SHARD_GLOBAL_INDEX_LAYOUT_CHANGED")
    matched = set()
    position = bindings["source_shard_global_start"]
    for batch in parquet.iter_batches(batch_size=4096, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if position in expectations:
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != expectations[position]:
                    raise RuntimeError("HISTORICAL_ROLE_TEXT_HASH_MISMATCH:" + str(position))
                matched.add(position)
            position += 1
        if position > max(expectations):
            break
    if matched != set(expectations) or len(proofs) != 6:
        raise RuntimeError("HISTORICAL_ROLE_SOURCE_COVERAGE_INCOMPLETE")
    result = {
        "status": "SIX_PUBLISHED_ROLE_MEMBERSHIPS_MATCH_FROZEN_RAW_SOURCE",
        "bindings_sha256": digest(base / "RECOVERY_BINDINGS.json"),
        "auditor_sha256": digest(Path(__file__)),
        "source_shard_sha256": digest(source),
        "revision": bindings["revision"],
        "split": bindings["split"],
        "roles": proofs,
        "distinct_raw_records_verified": len(matched),
        "published_boundary_hashes_manifest_verified": True,
        "original_token_arrays_recovered": False,
        "exact_historical_evaluation_windows_recovered": False,
        "new_confirmation_isolation_proven": False,
        "candidate_selection_performed": False,
        "model_loaded": False,
        "gpu_used": False,
        "optimizer_updates": 0,
        "historical_registration_rewritten": False,
        "scientific_registration": "BLOCKED",
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "distinct_raw_records_verified", "scientific_registration")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    recover(args.source_parquet, args.output)
