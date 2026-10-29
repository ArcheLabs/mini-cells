"""Isolated CPU integrity auditor: no model imports, no registration writes, no raw-text output."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

from audit_registration import digest, write, REVISION, TOKENIZER

RANGES = {
    "trigger": [500000, 500256],
    "pilot_report": [500256, 500512],
    "main_report": [500512, 501024],
}


def hashes(text):
    nfkc = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    words = " ".join(re.findall(r"\w+", nfkc))
    return {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in (
            ("raw_utf8", text),
            ("nfkc_casefold_whitespace", nfkc),
            ("nfkc_word_sequence", words),
        )
    }


def locate(layout, index):
    for item in layout["shards"]:
        if item["global_start"] <= index < item["global_end"]:
            return dict(path=item["path"], local_row=index - item["global_start"], global_row=index)
    raise ValueError("GLOBAL_RAW_INDEX_OUT_OF_BOUNDS")


def spans(layout, start, end):
    if not 0 <= start < end <= layout["total_rows"]:
        raise ValueError("INVALID_GLOBAL_RANGE")
    result = []
    for item in layout["shards"]:
        lo, hi = max(start, item["global_start"]), min(end, item["global_end"])
        if lo < hi:
            result.append(
                dict(
                    path=item["path"],
                    global_range=[lo, hi],
                    local_range=[lo - item["global_start"], hi - item["global_start"]],
                )
            )
    if sum(x["global_range"][1] - x["global_range"][0] for x in result) != end - start:
        raise RuntimeError("SHARD_MAPPING_HAS_GAP")
    return result


def iter_text(source, limit=None):
    import pyarrow.parquet as pq

    cursor = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=4096, columns=["text"]):
        for value in batch.column(0).to_pylist():
            if limit is not None and cursor >= limit:
                return
            if not isinstance(value, str):
                raise ValueError("NON_TEXT_RAW_RECORD")
            yield cursor, value
            cursor += 1


def validation_prefix(source, tokenizer_path, manifest):
    from tokenizers import Tokenizer
    import numpy as np

    if digest(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise RuntimeError("HISTORICAL_TOKENIZER_CHANGED")
    tok = Tokenizer.from_file(str(tokenizer_path))
    eos = tok.token_to_id("<eos>")
    sha = hashlib.sha256()
    count = 0
    tokens = 0
    last = -1
    for index, text in iter_text(source):
        if not text.strip():
            continue
        ids = tok.encode(text.strip()).ids
        if not ids:
            continue
        ids.append(eos)
        take = min(len(ids), manifest["validation_stream_tokens"] - tokens)
        sha.update(np.asarray(ids[:take], dtype=np.uint16).tobytes())
        tokens += take
        count += 1
        last = index
        if tokens == manifest["validation_stream_tokens"]:
            break
    if (
        sha.hexdigest() != manifest["validation_raw_sha256"]
        or count != manifest["validation_stories_consumed"]
    ):
        raise RuntimeError("HISTORICAL_VALIDATION_REVISION_NOT_REPRODUCED")
    return dict(
        status="EXACT_HISTORICAL_VALIDATION_PREFIX_REPRODUCED",
        sha256=sha.hexdigest(),
        consumed_nonempty_stories=count,
        last_raw_index=last,
        tokens=tokens,
    )


def audit(
    root, train_source, validation_source, layout_path, tree_path, output, tokenizer_path=None
):
    base = Path(__file__).resolve().parent
    frozen_config_sha = digest(base / "config/pilot.json")
    lock_sha = digest(base / "config/releases.lock")
    ledger_path = base / "registration/GLOBAL_DATASET_USAGE_LEDGER.json"
    ledger = json.loads(ledger_path.read_text())
    for item in ledger["parsed_metadata_files"]:
        if digest(root / item["path"]) != item["sha256"]:
            raise RuntimeError("HISTORICAL_USAGE_LEDGER_SOURCE_CHANGED:" + item["path"])
    existing = output / "DATA_BOUNDARY_PROOF.json"
    if existing.exists():
        cached = json.loads(existing.read_text())
        same = (
            cached.get("auditor", {}).get("code_sha256") == digest(Path(__file__))
            and cached["configuration_after"]
            == dict(config_sha256=frozen_config_sha, lock_sha256=lock_sha)
            and cached["registry_audit"]["ledger_sha256"] == digest(ledger_path)
            and cached["source_metadata_inputs"]
            == dict(layout_sha256=digest(layout_path), tree_sha256=digest(tree_path))
            and cached["source_files"][0]["sha256"] == digest(train_source)
            and cached["source_files"][1]["sha256"] == digest(validation_source)
        )
        for candidate in cached["candidate_ranges"]:
            same = same and candidate["global_range"] == RANGES[candidate["role"]]
        if not same:
            raise RuntimeError("IMMUTABLE_AUDIT_REQUEST_EXISTS_USE_NEW_REQUEST_DIRECTORY")
        amendment = json.loads((output / "STORY_RANGE_AMENDMENT_002.json").read_text())
        if amendment["proof_sha256"] != digest(existing):
            raise RuntimeError("CACHED_AUDIT_PROOF_CHANGED")
        print(json.dumps(dict(status=cached["status"], reused_verified_audit=True)))
        return cached
    layout = json.loads(layout_path.read_text())
    tree = json.loads(tree_path.read_text())
    if layout["revision"] != REVISION or layout["split"] != "train":
        raise RuntimeError("SOURCE_SPLIT_OR_REVISION_CHANGED")
    first = layout["shards"][0]
    if digest(train_source) != first["lfs_sha256"]:
        raise RuntimeError("TRAIN_SHARD_HASH_CHANGED")
    val = next(x for x in tree if "/validation-" in x["path"])
    if digest(validation_source) != val["lfs"]["oid"]:
        raise RuntimeError("VALIDATION_SHARD_HASH_CHANGED")
    prior = json.loads((base / "registration/DATA_BOUNDARY_PROOF.json").read_text())
    if prior["source_sha256"] != digest(train_source):
        raise RuntimeError("HISTORICAL_PREFIX_SOURCE_CHANGED")
    prefix = prior["historical_stream_verification"]
    old = root / "research/projects/cellular-computing/experiments/007-minicells-30m/artifacts"
    oldmeta = json.loads((old / "corpus-manifest.json").read_text())
    if (
        prefix["actual_sha256"] != oldmeta["train_raw_sha256"]
        or prefix["status"] != "EXACT_HISTORICAL_TRAIN_PREFIX_REPRODUCED"
    ):
        raise RuntimeError("COMPLETE_HISTORICAL_TRAIN_IDENTITY_UNVERIFIED")
    historical_end = prefix["last_raw_index"] + 1
    validation = validation_prefix(validation_source, old / "tokenizer.json", oldmeta)
    candidates = []
    lookup = {key: {} for key in hashes("")}
    for role, (start, end) in RANGES.items():
        # Current allocation is in shard zero, but global mapping is not assumed to be local.
        mapping = spans(layout, start, end)
        if any(x["path"] != first["path"] for x in mapping):
            raise RuntimeError("CANDIDATE_REQUIRES_ADDITIONAL_FULL_SHARD_FILES")
    for i, text in iter_text(train_source, max(b for a, b in RANGES.values())):
        role = next((r for r, (a, b) in RANGES.items() if a <= i < b), None)
        if role is None:
            continue
        row = dict(role=role, **locate(layout, i), hashes=hashes(text), empty=not text.strip())
        candidates.append(row)
        for method, sha in row["hashes"].items():
            lookup[method].setdefault(sha, []).append(i)
    if len(candidates) != 1024:
        raise RuntimeError("CANDIDATE_RAW_RECORDS_MISSING")
    intraduplicates = []
    for method, table in lookup.items():
        intraduplicates += [
            dict(method=method, sha256=h, global_rows=ids)
            for h, ids in table.items()
            if len(ids) > 1
        ]
    matches = []
    roots = {}
    counts = {}
    for split, source, end in [
        ("train", train_source, historical_end),
        ("validation", validation_source, validation["last_raw_index"] + 1),
    ]:
        aggregate = hashlib.sha256()
        nonempty = 0
        for index, text in iter_text(source, end):
            if not text.strip():
                continue
            nonempty += 1
            rowhash = hashes(text)
            aggregate.update(
                json.dumps([index, rowhash], sort_keys=True, separators=(",", ":")).encode()
            )
            for method, sha in rowhash.items():
                for candidate in lookup[method].get(sha, []):
                    matches.append(
                        dict(
                            method=method,
                            candidate_global_row=candidate,
                            historical_split=split,
                            historical_global_row=index,
                            sha256=sha,
                            source="Cellular Computing Experiment007",
                            manifest_path=str(old.relative_to(root) / "corpus-manifest.json"),
                            manifest_sha256=digest(old / "corpus-manifest.json"),
                        )
                    )
        roots[split] = aggregate.hexdigest()
        counts[split] = nonempty
    if counts["train"] != oldmeta["train_stories_consumed"]:
        raise RuntimeError("INCOMPLETE_EXPERIMENT007_TEXT_SCAN")
    indexed = [
        dict(role=r, global_range=interval, parquet_mapping=spans(layout, *interval))
        for r, interval in RANGES.items()
    ]
    range_conflicts = []
    for role, (lo, hi) in RANGES.items():
        for record in ledger["intervals"]:
            a, b = record["range"]
            if record["split"] == "train" and lo < b and a < hi:
                range_conflicts.append(dict(role=role, historical=record))
    # These historical prefixes cover all explicit ranges in the current ledger. Unknown
    # reservations and unpinned identities are retained, never silently certified clear.
    uncovered = [
        r
        for r in ledger["intervals"]
        if r["range"][1]
        > (historical_end if r["split"] == "train" else validation["last_raw_index"] + 1)
    ]
    affected = sorted({r["candidate_global_row"] for r in matches})
    summary = {
        role: dict(
            raw_records=sum(r["role"] == role for r in candidates),
            exact_historical_duplicates=len(
                {
                    m["candidate_global_row"]
                    for m in matches
                    if m["method"] == "raw_utf8"
                    and any(
                        c["global_row"] == m["candidate_global_row"] and c["role"] == role
                        for c in candidates
                    )
                }
            ),
            normalized_historical_duplicates=len(
                {
                    m["candidate_global_row"]
                    for m in matches
                    if m["method"] == "nfkc_casefold_whitespace"
                    and any(
                        c["global_row"] == m["candidate_global_row"] and c["role"] == role
                        for c in candidates
                    )
                }
            ),
        )
        for role in RANGES
    }
    tokenizer = dict(required_sha256=TOKENIZER, changed=False, available=False)
    if tokenizer_path:
        if digest(tokenizer_path) != TOKENIZER:
            raise RuntimeError("ORIGINAL_R3M_TOKENIZER_REQUIRED")
        tokenizer["available"] = True
    # Do not claim tensor-window verification without the original artifact and observed starts.
    windows = dict(
        status="BLOCKED_TEXT_CONTENT_LEAK_CONFIRMED" if matches else "NOT_VERIFIED",
        exact_r3m_token_window_ids_verified=False,
        original_tokenizer_artifact_available=tokenizer["available"],
        historical_observed_window_start_log_available=False,
        reason="Identical complete stories already occur in the verified historical training corpus. No new Report assets are materialized; the exact R3m tensor-window gate cannot pass.",
        eligible_corpus_vs_observed_optimizer_windows="KEPT_DISTINCT; no claim that every corpus window was sampled by an optimizer",
    )
    proof = dict(
        schema="minicells.r3m.data-boundary-proof.v2",
        auditor=dict(
            code_sha256=digest(Path(__file__)),
            repository_baseline_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            mode="ISOLATED_CPU_READONLY",
        ),
        source_metadata_inputs=dict(
            layout_sha256=digest(layout_path), tree_sha256=digest(tree_path)
        ),
        status="REGISTRATION_BLOCKED_TEXT_LEAKAGE"
        if matches or intraduplicates
        else "REGISTRATION_BLOCKED_INCOMPLETE_WINDOW_PROOF",
        dataset="roneneldan/TinyStories",
        revision=REVISION,
        split="train",
        source_files=[
            dict(
                path=first["path"], sha256=digest(train_source), bytes=train_source.stat().st_size
            ),
            dict(
                path=val["path"],
                sha256=digest(validation_source),
                bytes=validation_source.stat().st_size,
            ),
        ],
        shard_layout=layout,
        candidate_ranges=indexed,
        candidate_records=candidates,
        summary=summary,
        candidate_internal_duplicates=intraduplicates,
        historical_text_matches=matches,
        historical_affected_candidate_count=len(affected),
        historical_complete_scan=dict(
            nonempty_records=counts,
            ordered_hash_index_sha256=roots,
            train_prefix_identity=prefix,
            validation_prefix_identity=validation,
        ),
        registry_audit=dict(
            ledger_sha256=digest(ledger_path),
            source_documents_verified=len(ledger["parsed_metadata_files"]),
            intervals=len(ledger["intervals"]),
            unpinned_historical_claims=sum(r["revision"] is None for r in ledger["intervals"]),
            coverage_limits="Other unpinned historical snapshots are not certified globally clean by an index comparison. Experiment007 identity is reproduced explicitly; its text collision already blocks registration.",
            range_conflicts=range_conflicts,
            uncovered_declared_intervals=uncovered,
            reservation_markers=ledger["reservation_markers"],
        ),
        tokenizer=tokenizer,
        evaluation_windows=windows,
        registration_changed=False,
        gpu_training_started=False,
        access_boundary=dict(
            model_imports=0,
            model_evaluations=0,
            optimizer_updates=0,
            main_report_use="ISOLATED_IDENTITY_AND_LEAKAGE_AUDIT_ONLY",
            raw_text_persisted=False,
            confirmation_model_access=False,
        ),
        configuration_before=dict(config_sha256=frozen_config_sha, lock_sha256=lock_sha),
        configuration_after=dict(
            config_sha256=digest(base / "config/pilot.json"),
            lock_sha256=digest(base / "config/releases.lock"),
        ),
    )
    if proof["configuration_before"] != proof["configuration_after"]:
        raise RuntimeError("READONLY_AUDITOR_CHANGED_REGISTRATION")
    write(output / "DATA_BOUNDARY_PROOF.json", proof)
    write(
        output / "STORY_RANGE_AMENDMENT_002.json",
        dict(
            status="REJECTED_NOT_REGISTERED",
            candidate_ranges=RANGES,
            proof_sha256=digest(output / "DATA_BOUNDARY_PROOF.json"),
            prior_proof_retained="registration/DATA_BOUNDARY_PROOF.json",
            no_replacement_selected=True,
        ),
    )
    print(
        json.dumps(
            dict(
                status=proof["status"],
                summary=summary,
                matches=len(matches),
                internal_duplicates=len(intraduplicates),
                records_scanned=counts,
                output=str(output),
            )
        )
    )
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--train-source", type=Path, required=True)
    p.add_argument("--validation-source", type=Path, required=True)
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    a = p.parse_args()
    audit(a.root, a.train_source, a.validation_source, a.layout, a.tree, a.output, a.tokenizer)
