"""Execution-scoped evidence, independent of recoverable scientific run identity."""

from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
import sys
from .common import atomic, digest, read


def execution_directory(parent):
    path = Path(parent) / "executions" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def seal(output, identity, *, status, exit_code, publication, started_at):
    output = Path(output)
    trial = output / "reports/SINGLE_SEED_TRIAL.json"
    files = []
    for path in sorted((output / "reports").glob("*")):
        if not path.is_file() or path.name == "RUN_MANIFEST.json":
            continue
        if path.suffix == ".json":
            read(path)
        elif path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    json.loads(line)
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    manifest = {
        **identity,
        "execution_id": output.name,
        "output_directory": str(output.resolve()),
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "run_status": status,
        "trial_status": read(trial).get("status") if trial.exists() else "NOT_RUN",
        "seed": 151001301,
        "exit_code": exit_code,
        "artifact_files": files,
        "artifact_scope": "AGGREGATE_REPORTS; private checkpoint integrity uses Store receipts",
        "command": sys.argv,
        "publication_status": publication.get("status", "NOT_ATTEMPTED"),
    }
    if publication.get("status") == "PUBLISHED" and publication.get("remote_commit_sha"):
        manifest["remote_commit_sha"] = publication["remote_commit_sha"]
    atomic(output / "reports/RUN_MANIFEST.json", manifest)
    return manifest
