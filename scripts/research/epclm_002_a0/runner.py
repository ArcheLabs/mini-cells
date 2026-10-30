"""Run isolated Writer/Reader episodes and produce auditable summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .task import Episode, generate_episode, generate_queries


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = ROOT / "research/validations/epclm-002-a0-persistent-state-recall/protocol.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(HERE.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    _json_write(temporary, payload)
    temporary.replace(path)


def _load_episode_progress(
    path: Path,
    *,
    episodes: int,
    seed: int,
    fact_count: int,
    distractor_count: int,
    overwrite_count: int,
) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    records: dict[int, dict[str, Any]] = {}
    valid_lines: list[str] = []
    needs_rewrite = False
    for line_index, line in enumerate(lines):
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError:
            if line_index != len(lines) - 1:
                raise ValueError(f"corrupt episode progress before final line: {path}:{line_index + 1}")
            break  # A killed append may leave only its last JSONL record incomplete.
        if not isinstance(envelope, dict):
            raise ValueError(f"invalid episode progress envelope: {path}:{line_index + 1}")
        record = envelope.get("record")
        if not isinstance(record, dict):
            raise ValueError(f"invalid episode progress record envelope: {path}:{line_index + 1}")
        expected_record_sha = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if envelope.get("record_sha256") != expected_record_sha:
            raise ValueError(f"episode progress record checksum mismatch: {path}:{line_index + 1}")
        index = int(record.get("episode_index", -1))
        if not 0 <= index < episodes or index in records:
            raise ValueError(f"duplicate or out-of-range episode progress index {index}: {path}")
        episode_seed = seed + index * 104729
        episode = generate_episode(
            episode_seed,
            fact_count=fact_count,
            distractor_count=distractor_count,
            overwrite_count=overwrite_count,
        )
        expected_queries = generate_queries(episode, seed=episode_seed ^ 0x5A17)
        if record.get("episode_id") != episode.episode_id:
            raise ValueError(f"episode progress does not match deterministic task identity: {path}:{index}")
        if record.get("query_keys") != expected_queries:
            raise ValueError(f"episode progress query mapping changed: {path}:{index}")
        if record.get("targets") != [episode.final_facts[key] for key in expected_queries]:
            raise ValueError(f"episode progress targets changed: {path}:{index}")
        records[index] = record
        needs_rewrite = needs_rewrite or not line.endswith("\n")
        valid_lines.append(line if line.endswith("\n") else line + "\n")
    if needs_rewrite or len(valid_lines) != len(lines):
        path.write_text("".join(valid_lines), encoding="utf-8")
    return records


def _child_env(forbidden_root: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + current if current else "")
    env.pop("PYTHONSTARTUP", None)
    if forbidden_root is not None:
        env["EPCLM_FORBIDDEN_ROOT"] = str(forbidden_root.resolve())
    else:
        env.pop("EPCLM_FORBIDDEN_ROOT", None)
    return env


def _worker(
    mode: str,
    *,
    cwd: Path,
    backend: str,
    state: Path | None = None,
    input_path: Path | None = None,
    queries: Path | None = None,
    result: Path | None = None,
    checkpoint: Path | None = None,
    forbidden_root: Path | None = None,
    deadline: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], int]:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(f"{mode} worker was not started because the run budget expired")
    command = [sys.executable, "-m", "scripts.research.epclm_002_a0.worker", mode, "--backend", backend]
    for flag, path in (
        ("--input", input_path),
        ("--state", state),
        ("--queries", queries),
        ("--result", result),
        ("--checkpoint", checkpoint),
    ):
        if path is not None:
            command.extend((flag, str(path.resolve())))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=_child_env(forbidden_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    pid = process.pid
    try:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        stdout, stderr = process.communicate(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise TimeoutError(f"{mode} worker exceeded the run wall-time budget") from exc
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} subprocess failed ({completed.returncode}): {completed.stderr[-3000:]}"
        )
    return completed, pid


def _run_episode(
    episode: Episode,
    *,
    query_seed: int,
    backend: str,
    checkpoint: Path | None,
    work_root: Path,
    episode_index: int,
    deadline: float | None = None,
) -> dict[str, Any]:
    case = work_root / f"case-{episode_index:06d}"
    if case.exists():
        shutil.rmtree(case)
    raw_root = case / "raw"
    sandbox = case / "reader-sandbox"
    raw_root.mkdir(parents=True)
    sandbox.mkdir(parents=True)
    writer_input = raw_root / "writer-input.json"
    state_path = sandbox / ("state.json" if backend == "direct-address" else "state.pt")
    result_path = sandbox / "reader-result.json"
    _json_write(writer_input, episode.writer_payload())

    writer_started_ns = time.monotonic_ns()
    _writer_result, writer_pid = _worker(
        "writer",
        cwd=sandbox,
        backend=backend,
        input_path=writer_input,
        state=state_path,
        checkpoint=checkpoint,
        deadline=deadline,
    )
    writer_completed_ns = time.monotonic_ns()
    input_sha256 = _sha256(writer_input)
    writer_input.unlink()
    shutil.rmtree(raw_root)
    raw_removal_ok = not raw_root.exists() and not writer_input.exists()

    # Sampling after Writer termination is the protocol boundary: Writer never sees queries.
    queries = generate_queries(episode, seed=query_seed)
    query_path = sandbox / "queries.json"
    _json_write(query_path, {"episode_id": episode.episode_id, "query_keys": queries})
    state_sha_before = _sha256(state_path)
    reader_started_ns = time.monotonic_ns()
    _reader_result, reader_pid = _worker(
        "reader",
        cwd=sandbox,
        backend=backend,
        state=state_path,
        queries=query_path,
        result=result_path,
        checkpoint=checkpoint,
        forbidden_root=raw_root,
        deadline=deadline,
    )
    state_sha_after = _sha256(state_path)
    reader_completed_ns = time.monotonic_ns()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = result["predictions"]
    targets = [episode.final_facts[key] for key in queries]
    exact = [prediction == target for prediction, target in zip(predictions, targets)]
    no_memory_predictions = [
        int.from_bytes(hashlib.sha256(f"{query_seed}:{query}".encode()).digest()[:4], "big") % (1 << 16)
        for query in queries
    ]
    no_memory_exact = [
        prediction == target
        for prediction, target in zip(no_memory_predictions, targets)
    ]
    overwritten = [key in episode.overwritten_values for key in queries]
    unchanged = [not value for value in overwritten]
    stale = [
        prediction == episode.overwritten_values[key]
        for key, prediction, was_overwritten in zip(queries, predictions, overwritten)
        if was_overwritten
    ]
    exact_overwritten = [value for value, changed in zip(exact, overwritten) if changed]
    exact_unchanged = [value for value, same in zip(exact, unchanged) if same]
    data_files = {Path(path).resolve() for path in result["files_read"]}
    expected_files = {query_path.resolve(), state_path.resolve()}
    if checkpoint is not None:
        expected_files.add(checkpoint.resolve())
    reader_data_allowlist_ok = data_files == expected_files
    return {
        "episode_id": episode.episode_id,
        "episode_index": episode_index,
        "fact_count": len(episode.initial_facts),
        "distractor_count": sum(event["kind"] == "distractor" for event in episode.events),
        "overwrite_count": len(episode.overwritten_values),
        "query_count": len(queries),
        "exact_matches": sum(exact),
        "exact_accuracy": sum(exact) / len(exact),
        "no_memory_accuracy": sum(no_memory_exact) / len(no_memory_exact),
        "full_context_oracle_accuracy": 1.0,
        "overwritten_accuracy": sum(exact_overwritten) / len(exact_overwritten) if exact_overwritten else None,
        "unchanged_accuracy": sum(exact_unchanged) / len(exact_unchanged) if exact_unchanged else None,
        "stale_old_value_rate": sum(stale) / len(stale) if stale else None,
        "writer_pid": writer_pid,
        "writer_time_ms": (writer_completed_ns - writer_started_ns) / 1_000_000,
        "writer_returncode": _writer_result.returncode,
        "writer_completed_monotonic_ns": writer_completed_ns,
        "reader_started_monotonic_ns": reader_started_ns,
        "writer_terminated_before_reader": _writer_result.returncode == 0 and writer_completed_ns < reader_started_ns,
        "writer_input_sha256": input_sha256,
        "raw_history_removed_before_reader": raw_removal_ok,
        "reader_pid": int(result["reader_pid"]),
        "reader_time_ms": (reader_completed_ns - reader_started_ns) / 1_000_000,
        "reader_returncode": _reader_result.returncode,
        "reader_status": result["status"],
        "reader_received_raw_history": bool(result["raw_history_or_kv_received"]),
        "reader_raw_root_denied": bool(result["raw_data_root_denied"]),
        "reader_filesystem_allowlist_enforced": bool(result["filesystem_allowlist_enforced"]),
        "reader_child_process_and_network_guard_enforced": bool(
            result["child_process_and_network_guard_enforced"]
        ),
        "reader_data_allowlist_ok": reader_data_allowlist_ok,
        "reader_data_files": sorted(str(path) for path in data_files),
        "state_sha256_before_reader": state_sha_before,
        "state_sha256_after_reader": state_sha_after,
        "state_unchanged": state_sha_before == state_sha_after,
        "state_bytes": state_path.stat().st_size,
        "logical_state_bytes": int(result["logical_state_bytes"]),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint is not None else 0,
        "full_scan": any(result["full_scan"]),
        "candidate_cells_max": max(result["candidate_cells"], default=0),
        "read_cells_mean": sum(result["read_cells"]) / max(1, len(result["read_cells"])),
        "predictions": predictions,
        "query_keys": queries,
        "targets": targets,
    }


def _run_negative_controls(
    *, seed: int, work_root: Path, count: int = 16, deadline: float | None = None
) -> dict[str, Any]:
    """Exercise no-state abstention and independent wrong-episode state controls."""
    no_state_case = work_root / "negative-no-state"
    no_state_case.mkdir(parents=True)
    query_path = no_state_case / "queries.json"
    result_path = no_state_case / "result.json"
    no_state_episode = generate_episode(seed ^ 0xA51, fact_count=8)
    no_state_queries = generate_queries(no_state_episode, seed=seed ^ 0xB71, query_count=8)
    _json_write(query_path, {"episode_id": no_state_episode.episode_id, "query_keys": no_state_queries})
    _worker(
        "reader",
        cwd=no_state_case,
        backend="direct-address",
        state=None,
        queries=query_path,
        result=result_path,
        forbidden_root=no_state_case / "raw",
        deadline=deadline,
    )
    no_state_result = json.loads(result_path.read_text(encoding="utf-8"))
    no_state_abstains = (
        no_state_result["status"] == "abstain_no_state"
        and all(value is None for value in no_state_result["predictions"])
    )
    no_state_allowlist_ok = (
        {Path(path).resolve() for path in no_state_result["files_read"]} == {query_path.resolve()}
        and no_state_result["filesystem_allowlist_enforced"]
        and no_state_result["child_process_and_network_guard_enforced"]
    )

    wrong_rows: list[dict[str, Any]] = []
    for index in range(count):
        case = work_root / f"negative-wrong-state-{index:04d}"
        raw = case / "raw"
        sandbox = case / "reader-sandbox"
        raw.mkdir(parents=True)
        sandbox.mkdir()
        target_episode = generate_episode(seed + 200003 + index * 2, fact_count=8)
        wrong_episode = generate_episode(seed + 200004 + index * 2, fact_count=8)
        writer_input = raw / "writer-input.json"
        wrong_state = sandbox / "wrong-episode-state.json"
        _json_write(writer_input, wrong_episode.writer_payload())
        _worker(
            "writer", cwd=sandbox, backend="direct-address", input_path=writer_input,
            state=wrong_state, deadline=deadline,
        )
        writer_input.unlink()
        shutil.rmtree(raw)
        keys = generate_queries(target_episode, seed=seed ^ (index + 991), query_count=8)
        wrong_query_path = sandbox / "queries.json"
        wrong_result_path = sandbox / "result.json"
        _json_write(wrong_query_path, {"episode_id": target_episode.episode_id, "query_keys": keys})
        _worker(
            "reader",
            cwd=sandbox,
            backend="direct-address",
            state=wrong_state,
            queries=wrong_query_path,
            result=wrong_result_path,
            forbidden_root=raw,
            deadline=deadline,
        )
        wrong_result = json.loads(wrong_result_path.read_text(encoding="utf-8"))
        wrong_expected_files = {wrong_state.resolve(), wrong_query_path.resolve()}
        wrong_allowlist_ok = (
            {Path(path).resolve() for path in wrong_result["files_read"]} == wrong_expected_files
            and wrong_result["filesystem_allowlist_enforced"]
            and wrong_result["raw_data_root_denied"]
            and not wrong_result["raw_history_or_kv_received"]
        )
        targets = [target_episode.final_facts[key] for key in keys]
        exact_count = sum(
            prediction == target
        for prediction, target in zip(wrong_result["predictions"], targets)
        )
        wrong_rows.append({
            "episode_id": target_episode.episode_id,
            "correct": exact_count,
            "queries": len(keys),
            "reader_allowlist_pass": wrong_allowlist_ok,
        })
    wrong_accuracy = sum(row["correct"] for row in wrong_rows) / max(1, sum(row["queries"] for row in wrong_rows))
    return {
        "no_state_reader_abstained": no_state_abstains,
        "no_state_reader_allowlist_pass": no_state_allowlist_ok,
        "wrong_episode_state_accuracy": wrong_accuracy,
        "wrong_episode_state_query_count": sum(row["queries"] for row in wrong_rows),
        "wrong_episode_state_audit_pass": all(row["reader_allowlist_pass"] for row in wrong_rows),
        "wrong_episode_state_cases": wrong_rows,
    }


def run_condition(
    *,
    backend: str,
    episodes: int,
    seed: int,
    fact_count: int,
    distractor_count: int,
    overwrite_count: int = 0,
    negative_control_count: int = 16,
    checkpoint: Path | None = None,
    formal_run: bool = False,
    max_runtime_seconds: int = 7200,
    output: Path,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 1 <= max_runtime_seconds <= 7200:
        raise ValueError("max_runtime_seconds must be between 1 and 7200")
    if backend != "direct-address" and checkpoint is None:
        raise ValueError("learned model backends require --checkpoint")
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    protocol_hash = _sha256(PROTOCOL)
    protocol_payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    formal_eval_seeds = set(protocol_payload["task"]["splits"]["formal_evaluation_seeds"])
    if seed in formal_eval_seeds and not formal_run:
        raise ValueError("a reserved formal evaluation seed requires --formal")
    if formal_run and (seed not in formal_eval_seeds or episodes < 1000):
        raise ValueError("formal runs require a registered evaluation seed and at least 1000 episodes")
    checkpoint_metadata: dict[str, Any] | None = None
    if checkpoint is not None:
        metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint metadata is required: {metadata_path}")
        checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if checkpoint_metadata.get("model_name") != backend:
            raise ValueError("checkpoint backend does not match requested backend")
        if checkpoint_metadata.get("protocol_sha256") != protocol_hash:
            raise ValueError("checkpoint was trained against a different protocol hash")
        if checkpoint_metadata.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError("checkpoint metadata hash does not match checkpoint bytes")
        if formal_run:
            matching_pairs = [
                row for row in protocol_payload["task"]["splits"]["formal_seed_pairing"]
                if row["evaluation_seed"] == seed
            ]
            if not matching_pairs or checkpoint_metadata.get("training_seed") != matching_pairs[0]["model_seed"]:
                raise ValueError("formal checkpoint/evaluation seed pairing does not match protocol")
    output.mkdir(parents=True, exist_ok=True)
    implementation_hash = _implementation_sha256()
    identity = {
        "format": "minicells.epclm-002-a0.condition-identity.v1",
        "backend": backend,
        "episodes": episodes,
        "seed": seed,
        "condition": {
            "fact_count": fact_count,
            "distractor_count": distractor_count,
            "overwrite_count": overwrite_count,
        },
        "negative_control_count": negative_control_count,
        "formal_run": formal_run,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": implementation_hash,
        "model_checkpoint_sha256": _sha256(checkpoint) if checkpoint is not None else None,
    }
    identity_path = output / "condition-identity.json"
    if identity_path.is_file() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("existing condition identity differs; refusing to mix or overwrite evidence")
    if not identity_path.exists():
        _atomic_json_write(identity_path, identity)
    final_path = output / "result.json"
    if final_path.is_file():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing.get("condition_identity_sha256") != hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest():
            raise ValueError("completed condition result does not match its immutable identity")
        if len(existing.get("records", [])) != episodes:
            raise ValueError("completed condition result has an incomplete episode list")
        completed_episodes = output / "episodes.jsonl"
        if not completed_episodes.is_file() or len(
            completed_episodes.read_text(encoding="utf-8").splitlines()
        ) != episodes:
            raise ValueError("completed condition episode evidence is missing or incomplete")
        partial_path = output / "episodes.partial.jsonl"
        partial_path.unlink(missing_ok=True)
        (output / "partial-status.json").unlink(missing_ok=True)
        return existing

    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    partial_path = output / "episodes.partial.jsonl"
    progress = _load_episode_progress(
        partial_path,
        episodes=episodes,
        seed=seed,
        fact_count=fact_count,
        distractor_count=distractor_count,
        overwrite_count=overwrite_count,
    )
    deadline = time.monotonic() + max_runtime_seconds
    try:
        with tempfile.TemporaryDirectory(prefix="epclm-002-a0-") as work_dir:
            work_root = Path(work_dir)
            progress_interval = max(1, min(100, episodes // 10 or 1))
            for index in range(episodes):
                if index in progress:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("condition reached its registered wall-time budget")
                episode_seed = seed + index * 104729
                episode = generate_episode(
                    episode_seed,
                    fact_count=fact_count,
                    distractor_count=distractor_count,
                    overwrite_count=overwrite_count,
                )
                record = _run_episode(
                    episode,
                    query_seed=episode_seed ^ 0x5A17,
                    backend=backend,
                    checkpoint=checkpoint,
                    work_root=work_root,
                    episode_index=index,
                    deadline=deadline,
                )
                with partial_path.open("a", encoding="utf-8") as handle:
                    record_sha = hashlib.sha256(
                        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    handle.write(
                        json.dumps(
                            {"record": record, "record_sha256": record_sha}, sort_keys=True
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                progress[index] = record
                if (index + 1) % progress_interval == 0 or index + 1 == episodes:
                    print(
                        f"EPCLM-002-A0 {backend} facts={fact_count} distractors={distractor_count} "
                        f"updates={overwrite_count}: {len(progress)}/{episodes} episodes",
                        flush=True,
                    )
            controls_path = output / "negative-controls.json"
            if controls_path.is_file():
                negative_controls = json.loads(controls_path.read_text(encoding="utf-8"))
            else:
                negative_controls = _run_negative_controls(
                    seed=seed,
                    work_root=work_root,
                    count=negative_control_count,
                    deadline=deadline,
                )
                _atomic_json_write(controls_path, negative_controls)
    except TimeoutError:
        _atomic_json_write(
            output / "partial-status.json",
            {
                "format": "minicells.epclm-002-a0.partial-status.v1",
                "status": "FORMAL_INCOMPLETE_TIMEOUT" if formal_run else "DEVELOPMENT_INCOMPLETE_TIMEOUT",
                "scientific_decision": False,
                "formal_run": formal_run,
                "condition_identity_sha256": identity_digest,
                "completed_episodes": len(progress),
                "expected_episodes": episodes,
                "negative_controls_complete": (output / "negative-controls.json").is_file(),
                "resume_from": "episodes.partial.jsonl",
            },
        )
        raise
    records = [progress[index] for index in range(episodes)]

    all_correct = sum(record["exact_matches"] for record in records)
    all_queries = sum(record["query_count"] for record in records)
    valid_isolation = all(
        record["writer_terminated_before_reader"]
        and record["raw_history_removed_before_reader"]
        and not record["reader_received_raw_history"]
        and record["reader_raw_root_denied"]
        and record["reader_filesystem_allowlist_enforced"]
        and record["reader_child_process_and_network_guard_enforced"]
        and record["reader_data_allowlist_ok"]
        and record["state_unchanged"]
        for record in records
    ) and negative_controls["no_state_reader_abstained"] and (
        negative_controls["wrong_episode_state_accuracy"] <= 0.05
    ) and negative_controls["no_state_reader_allowlist_pass"] and negative_controls[
        "wrong_episode_state_audit_pass"
    ]
    payload = {
        "format": "minicells.epclm-002-a0.condition-result.v1",
        "status": "FORMAL_CONDITION" if formal_run else ("PARTIAL" if backend == "direct-address" else "PILOT"),
        "experiment_id": "epclm-002-a0-persistent-state-recall",
        "backend": backend,
        "scientific_role": "engineering_reference" if backend == "direct-address" else (
            "learned_candidate_formal" if formal_run else "learned_candidate_pilot"
        ),
        "formal_run": formal_run,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": implementation_hash,
        "condition_identity_sha256": identity_digest,
        "model_checkpoint_sha256": _sha256(checkpoint) if checkpoint is not None else None,
        "training_seed": checkpoint_metadata.get("training_seed") if checkpoint_metadata else None,
        "model_config": checkpoint_metadata.get("model_config") if checkpoint_metadata else None,
        "model_parameter_count": checkpoint_metadata.get("model_parameter_count") if checkpoint_metadata else None,
        "code_version": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "git_branch": subprocess.run(
            ["git", "-C", str(ROOT), "branch", "--show-current"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "run_command": list(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "reader_device": "cpu",
            "torch_version": checkpoint_metadata.get("torch_version") if checkpoint_metadata else None,
        },
        "seed": seed,
        "episodes": episodes,
        "condition": {"fact_count": fact_count, "distractor_count": distractor_count, "overwrite_count": overwrite_count},
        "metrics": {
            "exact_match_accuracy": all_correct / max(1, all_queries),
            "no_memory_accuracy": sum(record["no_memory_accuracy"] * record["query_count"] for record in records) / max(1, all_queries),
            "full_context_oracle_accuracy": 1.0,
            "exact_matches": all_correct,
            "query_count": all_queries,
            "mean_serialized_state_bytes": sum(record["state_bytes"] for record in records) / len(records),
            "mean_logical_state_bytes": sum(record["logical_state_bytes"] for record in records) / len(records),
            "mean_writer_time_ms": sum(record["writer_time_ms"] for record in records) / len(records),
            "mean_reader_time_ms": sum(record["reader_time_ms"] for record in records) / len(records),
            "checkpoint_bytes": max(record["checkpoint_bytes"] for record in records),
            "isolation_audit_pass": valid_isolation and negative_controls["no_state_reader_abstained"],
            "no_state_reader_abstained": negative_controls["no_state_reader_abstained"],
            "no_state_reader_allowlist_pass": negative_controls["no_state_reader_allowlist_pass"],
            "wrong_episode_state_accuracy": negative_controls["wrong_episode_state_accuracy"],
            "wrong_episode_state_query_count": negative_controls["wrong_episode_state_query_count"],
            "wrong_episode_state_audit_pass": negative_controls["wrong_episode_state_audit_pass"],
            "wrong_episode_negative_control_pass": negative_controls["wrong_episode_state_accuracy"] <= 0.05,
            "full_scan": any(record["full_scan"] for record in records),
            "mean_read_cells": sum(record["read_cells_mean"] for record in records) / len(records),
            "max_candidate_cells": max(record["candidate_cells_max"] for record in records),
            "mean_overwritten_accuracy": _mean(record["overwritten_accuracy"] for record in records),
            "mean_unchanged_accuracy": _mean(record["unchanged_accuracy"] for record in records),
            "mean_stale_old_value_rate": _mean(record["stale_old_value_rate"] for record in records),
        },
        "records": records,
        "negative_controls": negative_controls,
        "scientific_interpretation": (
            "Direct-address is an engineering upper-bound reference; it does not support H2."
            if backend == "direct-address"
            else "A learned-model result is pilot-only until all registered seeds and gates are run."
        ),
    }
    final_episodes = output / "episodes.jsonl"
    temporary_episodes = final_episodes.with_name(final_episodes.name + ".tmp")
    with temporary_episodes.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_episodes.replace(final_episodes)
    _atomic_json_write(output / "result.json", payload)
    partial_path.unlink(missing_ok=True)
    (output / "partial-status.json").unlink(missing_ok=True)
    return payload


def _mean(values: Any) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return sum(selected) / len(selected) if selected else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("direct-address", "learned-cells", "flat-recurrent"), default="direct-address")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=80100, help="development/pilot seed; formal seeds are separately registered")
    parser.add_argument("--facts", type=int, choices=(1, 2, 4, 8, 16, 32), default=8)
    parser.add_argument("--distractors", type=int, choices=(0, 16, 64), default=0)
    parser.add_argument("--overwrites", type=int, default=0)
    parser.add_argument("--negative-control-count", type=int, default=16)
    parser.add_argument("--formal", action="store_true", help="authorize one frozen formal evaluation condition")
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_condition(
        backend=args.backend,
        episodes=args.episodes,
        seed=args.seed,
        fact_count=args.facts,
        distractor_count=args.distractors,
        overwrite_count=args.overwrites,
        negative_control_count=args.negative_control_count,
        checkpoint=args.checkpoint,
        formal_run=args.formal,
        max_runtime_seconds=args.max_runtime_seconds,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
