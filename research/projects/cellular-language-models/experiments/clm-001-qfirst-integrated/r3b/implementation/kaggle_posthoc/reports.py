"""Public projections and cumulative private snapshots; no new scientific estimator."""

from copy import deepcopy
from pathlib import Path
import json
import re
import shutil
import time
from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError

PUBLIC_NAMES = {
    "experiment-manifest.json",
    "metrics-summary.json",
    "gradient-audit-summary.json",
    "routing-longitudinal-summary.json",
    "specialization-summary.json",
    "statistical-contrast.json",
    "diagnostic-resource-ledger.json",
    "analysis-index.json",
    "coverage-ledger.json",
    "kaggle-output-input-reference.json",
    "kaggle-version-inventory.json",
    "recovery-ledger.json",
    "performance-summary.json",
    "session-resource-ledger.json",
    "analysis-resource-ledger.json",
    "checkpoint-validation.json",
}
ROW_FIELDS = {
    "slot_key",
    "task_key",
    "arm",
    "step",
    "phase",
    "stage",
    "policies",
    "role",
    "status",
    "reason",
    "result_data_verified",
    "cross_session_persistence_verified",
    "persistence",
    "result_sha256",
    "elapsed_seconds",
    "gpu_stream_seconds",
    "sample_count",
    "coverage_status",
    "device",
    "persist_seconds",
    "started_at",
    "ended_at",
    "reused",
    "computed_this_session",
    "recovery_fact",
    "compute_timings",
    "task_loop_wall_seconds",
}


def project_rows(rows):
    return [{k: v for k, v in r.items() if k in ROW_FIELDS} for r in rows]


def public_projection(values):
    from clm001_r3b.publication import reject_private_payload

    out = {k: deepcopy(v) for k, v in values.items() if k in PUBLIC_NAMES}
    for name in ("coverage-ledger.json", "analysis-index.json", "recovery-ledger.json"):
        if name in out:
            out[name]["tasks"] = project_rows(out[name].get("tasks", []))
    manifest = out.get("experiment-manifest.json", {})
    if "missing" in manifest:
        manifest["missing"] = project_rows(manifest["missing"])
    if "kaggle-version-inventory.json" in out:
        inv = out["kaggle-version-inventory.json"]
        inv["versions"] = [
            {
                k: v
                for k, v in row.items()
                if k not in ("files", "metadata", "missing_training_files")
            }
            for row in inv["versions"]
        ]
    if "recovery-ledger.json" in out:
        out["recovery-ledger.json"]["conflicts"] = [
            {k: v for k, v in r.items() if k in ("reason", "task_key", "file_sha256")}
            for r in out["recovery-ledger.json"].get("conflicts", [])
        ]
    reference = out.get("kaggle-output-input-reference.json", {})
    if "input_checkpoint_ref" in reference:
        reference["input_checkpoint_ref"] = {
            k: v
            for k, v in reference["input_checkpoint_ref"].items()
            if k in ("hf_repo", "hf_repo_type", "hf_revision", "bundle_sha256")
        }
    performance = out.get("performance-summary.json", {})
    if "kaggle_input_transfer" in performance:
        transfers = performance["kaggle_input_transfer"]
        for row in [transfers["training"], *transfers["prior_analysis"]]:
            hashes = row.pop("file_sha256", {})
            row.update(file_count=len(hashes), verified_files_sha256=canonical(hashes))
    for value in out.values():
        reject_private_payload(value)
    return out


def scientific_findings(values, counts):
    complete = counts["complete"] == 102 and not any(
        r.get("coverage_status") == "INSUFFICIENT_COVERAGE"
        for r in values.get("coverage-ledger.json", {}).get("tasks", [])
    )
    text = [
        "# CLM-001-R3b — Scientific findings",
        "",
        "## Evidence boundary",
        "",
        "Single-seed Development only. Confirmation SEALED. MULTISEED_NOT_TESTED. CAUSAL_FEEDBACK_NOT_TESTED.",
        "Frozen initial Probe states; no online-state resampling claim. QS is a zero-competition-gradient Sham, not an independent random-direction-gradient control.",
        "Coverage: `"
        + str(counts)
        + "`. "
        + (
            "All task observations present."
            if complete
            else "Partial observations: missing values are not zero and do not establish a null result."
        ),
        "",
        "H1: "
        + (
            "EXPLORATORY_ONLY; inspect matched DID/CI95 below."
            if complete
            else "INCONCLUSIVE until required matched observations are present."
        ),
        "H2: EXPLORATORY_ONLY where functional evidence exists. H3/H4: NOT_TESTED as independent confirmatory hypotheses.",
        "",
        "## Quality (frozen engine summaries)",
        "",
        "| Arm | Step | Phase | Policy | Mean NLL | Mean path length |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for row in values["metrics-summary.json"]["rows"]:
        for policy, q in row["quality"].items():
            text.append(
                f"| {row['arm']} | {row['step']} | {row['phase']} | {policy} | {q.get('mean_nll')} | {q.get('mean_path_length')} |"
            )
    text += [
        "",
        "## Matched primary comparisons",
        "",
        "Step400 pre_event and post_event are separate checkpoints; do not merge them into one longitudinal point.",
    ]
    for row in values["statistical-contrast.json"]["rows"]:
        text += [
            "",
            "### Step " + str(row["step"]) + " / " + row["phase"],
            "",
            "```json",
            json.dumps(row, indent=2, allow_nan=False),
            "```",
        ]
    text += [
        "",
        "## Functional and Action",
        "",
        "Cell conditional advantage, crossover, paired regret, rank correlation and descriptive lag are recorded in specialization-summary.json; they are not inferred from visit counts.",
        "Router agreement and STOP/CTX margins by depth/context are in routing-longitudinal-summary.json. Quality includes forced-end and context-use measurements where observed.",
        "Coverage by Functional role, arm and step must be checked before comparing specialization or temporal change.",
        "No causal self-reinforcement or formal confirmation claim follows from this single-seed study.",
    ]
    for filename in ("specialization-summary.json", "routing-longitudinal-summary.json"):
        summary = values.get(filename, {})
        text += [
            "",
            "## Observed " + filename,
            "",
            "```json",
            json.dumps(summary, indent=2, allow_nan=False),
            "```",
        ]
    return "\n".join(text) + "\n"


def finalize_snapshot(output, provenance):
    output = Path(output)
    files = {}
    # Cumulative raw results plus receipts/index. Training weights/cache/code are
    # outside /working and are never copied into the next Output package.
    for path in sorted((output / "private-results").rglob("*.json")):
        if path.is_symlink():
            raise ContinuationError("UNSAFE_RESULT_PATH")
        files[path.relative_to(output).as_posix()] = digest(path)
    manifest = {
        "schema": "minicells.r3b.output-snapshot.v1",
        "measurement_engine_source_sha": provenance["measurement_engine_source_sha"],
        "execution_orchestrator_source_sha": provenance["execution_orchestrator_source_sha"],
        "storage_provenance": provenance,
        "files": files,
        "package_bytes": sum((output / name).stat().st_size for name in files),
        "output_version": "OUTPUT_VERSION_PENDING_SAVE",
        "training_updates": 0,
    }
    atomic(output / "snapshot-manifest.json", manifest)
    return manifest


def publication_scope(orchestrator, protocol):
    return {
        "execution_orchestrator_source_sha": orchestrator,
        "analysis_protocol_sha256": protocol,
        "transport": "KAGGLE_OUTPUT",
    }


def check_publication(root, orchestrator, protocol):
    from clm001_r3b.publication import preflight

    root = Path(root)
    lock = read(root / "experiment.lock.json")
    return preflight(
        root.parent, lock["source_sha"], lock["run_id"], publication_scope(orchestrator, protocol)
    )


def publish(root, values, output, orchestrator, protocol):
    """Reuse existing authenticated main-descended Git machinery, with a new explicit allowlist."""
    from clm001_r3b.publication import namespace, reject_private_payload
    from clm001_r3 import publication as existing

    root, output = Path(root), Path(output)
    lock = read(root / "experiment.lock.json")
    scope = publication_scope(orchestrator, protocol)
    begin = time.perf_counter()
    checkout, branch = check_publication(root, orchestrator, protocol)
    directory = (
        checkout
        / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/artifacts"
        / lock["source_sha"]
        / canonical(namespace(lock["run_id"], scope))[:12]
    )
    directory.mkdir(parents=True, exist_ok=True)
    staged = []
    for name, value in values.items():
        if name not in PUBLIC_NAMES:
            raise ContinuationError("PUBLICATION_NOT_WHITELISTED")
        reject_private_payload(value)
        if len(json.dumps(value, allow_nan=False).encode()) > 2 * 1024 * 1024:
            raise ContinuationError("PUBLICATION_FILE_TOO_LARGE")
        atomic(directory / name, value)
        staged.append((directory / name).relative_to(checkout).as_posix())
    attachments = [
        output / "public" / n
        for n in ("SCIENTIFIC-FINDINGS.md", "CLM-001-R3b-POSTHOC-PERFORMANCE.md")
    ]
    attachments += [
        output / "figures" / n
        for n in ("coverage.png", "quality-pre_event.png", "quality-post_event.png")
    ]
    for path in attachments:
        if not path.is_file() or path.is_symlink():
            raise ContinuationError("PUBLICATION_ATTACHMENT_MISSING")
        data = path.read_bytes()
        if len(data) > 2 * 1024 * 1024 or re.search(
            rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
            data,
        ):
            raise ContinuationError("PRIVATE_PUBLICATION_PAYLOAD_REJECTED")
        if path.suffix == ".png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ContinuationError("PUBLICATION_IMAGE_INVALID")
        shutil.copyfile(path, directory / path.name)
        staged.append((directory / path.name).relative_to(checkout).as_posix())
    existing.git(checkout, "add", "--", *staged)
    changed = existing.git(checkout, "diff", "--cached", "--name-only").stdout.splitlines()
    if set(changed) - set(staged):
        raise ContinuationError("NON_METADATA_PUBLICATION_REJECTED")
    if changed:
        existing.git(checkout, "commit", "-m", "results: verified Kaggle-output R3b posthoc")
    helper = existing.helper()
    helper.ROOT = checkout
    helper.push_branch(branch, "GITHUB_TOKEN")
    content = existing.git(checkout, "rev-parse", "HEAD").stdout.strip()
    if (
        existing.git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()[0]
        != content
    ):
        raise ContinuationError("GITHUB_REMOTE_READBACK_FAILED")
    receipt = {
        "status": "PUBLISHED",
        "branch": branch,
        "content_commit_sha": content,
        "files": {name: digest(checkout / name) for name in staged},
        "seconds_before_receipt": time.perf_counter() - begin,
        "scientific_decision": False,
    }
    atomic(directory / "publication.json", receipt)
    rel = (directory / "publication.json").relative_to(checkout).as_posix()
    existing.git(checkout, "add", "--", rel)
    if existing.git(checkout, "diff", "--cached", "--name-only").stdout.strip():
        existing.git(checkout, "commit", "-m", "results: Kaggle-output publication receipt")
    helper.push_branch(branch, "GITHUB_TOKEN")
    head = existing.git(checkout, "rev-parse", "HEAD").stdout.strip()
    if (
        existing.git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()[0]
        != head
    ):
        raise ContinuationError("GITHUB_REMOTE_READBACK_FAILED")
    atomic(
        output / "publication.json",
        {**receipt, "receipt_commit_sha": head, "publication_seconds": time.perf_counter() - begin},
    )
    return receipt
