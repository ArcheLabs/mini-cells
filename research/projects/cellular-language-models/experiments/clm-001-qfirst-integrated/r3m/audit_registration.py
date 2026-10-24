"""Read-only repository range inventory and candidate text hashes. Never changes registration."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER = "612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff"
CANDIDATES = {
    "R3M_STORY_TRIGGER_CALIBRATION": [100000, 100256],
    "R3M_STORY_PILOT_REPORT": [100256, 100512],
    "R3M_STORY_MAIN_REPORT": [100512, 101024],
}
OLD = {
    "R3M_STORY_TRIGGER_CALIBRATION": [10048, 10304],
    "R3M_STORY_PILOT_REPORT": [10304, 10560],
    "R3M_STORY_MAIN_REPORT": [10560, 11072],
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inventory(root):
    names = subprocess.check_output(
        ["git", "ls-files", "research"], cwd=root, text=True
    ).splitlines()
    records = []
    evidence = []
    unknown = []
    for name in names:
        path = root / name
        if (
            "/r3m/" in name
            or path.suffix not in (".json", ".yaml", ".yml")
            or path.stat().st_size > 5_000_000
        ):
            continue
        text = path.read_text(errors="replace")
        if (
            "TinyStories" not in text
            and "tinystories" not in text
            and not (
                "clm-001-qfirst-integrated/r3l/config/" in name
                or "clm-001-qfirst-integrated/r3k/config/" in name
            )
        ):
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            import yaml

            try:
                obj = yaml.safe_load(text)
            except yaml.YAMLError:
                continue
        sha = digest(path)
        evidence.append(dict(path=name, sha256=sha))

        def walk(value, location="", split="train", revision=None):
            if isinstance(value, dict):
                split = value.get("split", split)
                if split not in ("train", "validation", "test"):
                    split = "train"
                revision = value.get("dataset_revision", value.get("revision", revision))
                for key, item in value.items():
                    role = key.lower()
                    location2 = location + "/" + key
                    if (
                        (
                            key.endswith("_range")
                            or location.endswith("/ranges")
                            or key == "story_indices"
                        )
                        and isinstance(item, list)
                        and len(item) == 2
                        and all(type(x) is int for x in item)
                        and item[0] < item[1]
                    ):
                        if any(
                            w in role + location.lower()
                            for w in (
                                "train",
                                "calibration",
                                "admission",
                                "development",
                                "holdout",
                                "confirmation",
                                "probe",
                                "search",
                                "select",
                                "report",
                                "diagnostic",
                            )
                        ):
                            records.append(
                                dict(
                                    path=name,
                                    sha256=sha,
                                    json_pointer=location2,
                                    role=role,
                                    split=split,
                                    range=item,
                                    revision=revision,
                                    evidence="EXPLICIT_HALF_OPEN_RANGE",
                                )
                            )
                    elif (
                        key
                        in (
                            "train_stories",
                            "train_stories_consumed",
                            "tokenizer_stories",
                            "validation_stories_consumed",
                        )
                        and type(item) is int
                        and item > 0
                    ):
                        records.append(
                            dict(
                                path=name,
                                sha256=sha,
                                json_pointer=location2,
                                role=key,
                                split="validation" if key.startswith("validation") else "train",
                                range=[0, item],
                                revision=revision,
                                evidence="PREFIX_COUNT_FILTERED_STREAM_REQUIRES_SOURCE_VERIFICATION"
                                if key.endswith("consumed")
                                else "DECLARED_TRAIN_PREFIX",
                            )
                        )
                    elif key == "reserved_boundary":
                        unknown.append(
                            dict(
                                path=name,
                                json_pointer=location2,
                                value=item,
                                scope="UPPER_RANGE_ENDPOINT_NOT_RECORDED; NOT_ASSUMED_INFINITE_CONFIRMATION",
                            )
                        )
                    walk(item, location2, split, revision)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, (dict, list)):
                        walk(item, location + "/" + str(i), split, revision)

        walk(obj)
    return dict(
        tracked_research_file_count=len(names),
        parsed_metadata_files=evidence,
        intervals=records,
        reservation_markers=unknown,
        scope="ALL_TRACKED_RESEARCH_JSON_YAML_DECLARATIONS_INCLUDING_HISTORICAL_CORPUS_MANIFESTS_AND_ARTIFACTS",
        notes="Runtime code/manual selections without published metadata are not certified absent. A detected conflict is sufficient to stop registration.",
    )


def run(root, source, output, prefix_verification=None):
    ledger = inventory(root)
    conflicts = []
    for role, (lo, hi) in CANDIDATES.items():
        for row in ledger["intervals"]:
            a, b = row["range"]
            if row["split"] == "train" and lo < b and a < hi:
                conflicts.append(
                    dict(
                        candidate_role=role,
                        candidate_range=[lo, hi],
                        overlap=[max(lo, a), min(hi, b)],
                        historical=row,
                    )
                )
    import pyarrow.parquet as pq

    source_hash = digest(source)
    hashes = {role: [] for role in CANDIDATES}
    seen = {}
    duplicates = []
    offset = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=4096, columns=["text"]):
        for j, text in enumerate(batch.column(0).to_pylist()):
            i = offset + j
            role = next((r for r, (a, b) in CANDIDATES.items() if a <= i < b), None)
            if role is None:
                continue
            raw = hashlib.sha256(text.encode()).hexdigest()
            normalized = hashlib.sha256(
                re.sub(r"\s+", " ", text.strip().lower()).encode()
            ).hexdigest()
            item = dict(story_index=i, text_sha256=raw, normalized_text_sha256=normalized)
            hashes[role].append(item)
            if normalized in seen:
                duplicates.append(
                    dict(
                        first=seen[normalized],
                        second=dict(role=role, story_index=i),
                        normalized_sha256=normalized,
                    )
                )
            else:
                seen[normalized] = dict(role=role, story_index=i)
        offset += batch.num_rows
        if offset >= max(b for a, b in CANDIDATES.values()):
            break
    prefix = json.loads(prefix_verification.read_text()) if prefix_verification else None
    proof = dict(
        schema="minicells.r3m.data-boundary-proof.v1",
        status="REGISTRATION_BLOCKED_HISTORICAL_RANGE_CONFLICT"
        if conflicts
        else "RANGE_CLEAR_CONTENT_AND_WINDOW_AUDIT_PENDING",
        dataset="roneneldan/TinyStories",
        split="train",
        revision=REVISION,
        source_file="data/train-00000-of-00004-2d5a1467fff1081b.parquet",
        source_sha256=source_hash,
        tokenizer_sha256_required=TOKENIZER,
        tokenizer_changed=False,
        old_proposal=OLD,
        candidate_proposal=CANDIDATES,
        per_story_hashes=hashes,
        within_candidate_text_duplicates=duplicates,
        range_conflicts=conflicts,
        historical_stream_verification=prefix,
        evaluation_windows=dict(
            status="BLOCKED_ALL_WINDOWS_DERIVE_FROM_PREVIOUS_TRAIN_STORIES; NO_NEW_TOKENIZER_OR_MODEL_EVALUATION"
        ),
        cross_history_text_test=dict(
            status="CANDIDATE_INDICES_ALREADY_IN_HISTORICAL_TRAIN_PREFIX"
            if prefix and prefix["status"] == "EXACT_HISTORICAL_TRAIN_PREFIX_REPRODUCED"
            else "HISTORICAL_REVISION_VERIFICATION_PENDING"
        ),
        confirmation=dict(
            model_evaluations=0,
            quality_evaluations=0,
            raw_text_persisted=False,
            integrity_auditor_streamed_registered_confirmation_indices=bool(prefix),
            scope="READ_ONLY_HISTORICAL_STREAM_INTEGRITY_NO_MODEL_ACCESS_OR_SELECTION",
        ),
        formal_config_updated=False,
        gpu_started=False,
        suggested_next_candidates={
            "trigger": [500000, 500256],
            "pilot_report": [500256, 500512],
            "main_report": [500512, 501024],
        },
        suggestion_status="UNREGISTERED_INDEX_CANDIDATE_ONLY_REQUIRES_NEW_REVIEW_AND_TEXT_WINDOW_AUDIT; NOT_APPLIED",
    )
    write(output / "GLOBAL_DATASET_USAGE_LEDGER.json", ledger)
    write(output / "DATA_BOUNDARY_PROOF.json", proof)
    write(
        output / "STORY_RANGE_AMENDMENT_001.json",
        dict(
            old=OLD,
            requested=CANDIDATES,
            status="NOT_REGISTERED",
            reason=proof["status"],
            proof_sha256=digest(output / "DATA_BOUNDARY_PROOF.json"),
        ),
    )
    print(
        json.dumps(
            dict(
                status=proof["status"],
                conflicts=len(conflicts),
                candidate_duplicates=len(duplicates),
                metadata_files=len(ledger["parsed_metadata_files"]),
                output=str(output),
            )
        )
    )
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix-verification", type=Path)
    args = p.parse_args()
    run(args.root, args.source, args.output, args.prefix_verification)
