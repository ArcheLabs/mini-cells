from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "cow-clm-001"
PROTOCOL = ROOT / "research" / "validations" / EXPERIMENT / "protocol.json"
RESULTS = ROOT / "results" / EXPERIMENT


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def _git_blob(path: str) -> str:
    return subprocess.check_output(["git", "hash-object", path], cwd=ROOT, text=True).strip()


def _verify_frozen_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment") != "COW_CLM_001":
        raise RuntimeError("protocol experiment identity mismatch")
    if protocol.get("status") != "FORMAL_PROTOCOL_FROZEN_GPU_PENDING":
        raise RuntimeError("COW-CLM-001 is not a frozen GPU-pending protocol")
    expected = protocol.get("implementation_git_blobs", {})
    actual = {path: _git_blob(path) for path in expected}
    if actual != expected:
        mismatched = [path for path in expected if actual.get(path) != expected[path]]
        raise RuntimeError(f"frozen implementation Git blob mismatch: {mismatched[:1]}")


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8").splitlines()[-120:]
        print("\n".join(tail), file=sys.stderr)
        raise RuntimeError(f"subprocess failed ({result.returncode}): {' '.join(command)}")


def _launch_pair(commands: dict[str, list[str]], log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    processes: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    for name, command in commands.items():
        path = log_root / f"{name}.log"
        handle = path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[name] = (process, handle)
    failures: list[str] = []
    for name, (process, handle) in processes.items():
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append(name)
    if failures:
        for name in failures:
            path = log_root / f"{name}.log"
            print(f"=== {name} log tail ===", file=sys.stderr)
            print("\n".join(path.read_text(encoding="utf-8").splitlines()[-120:]), file=sys.stderr)
        raise RuntimeError(f"parallel COW-CLM subprocesses failed: {failures}")


def _finalize(protocol: dict[str, Any], result_root: Path) -> dict[str, Any]:
    tracks: dict[str, Any] = {}
    all_supported = True
    for track in ("knowledge", "capability"):
        result = _load_json(result_root / track / "result.json")
        reload_result = _load_json(result_root / track / "reload.json")
        minimum = result.get("minimum_supported_capacity")
        best_private_fraction = None
        if minimum is not None:
            selected = next(
                row
                for row in result["capacity_results"]
                if int(row["capacity_sites"]) == int(minimum)
            )
            best_private_fraction = float(selected["private_fraction"])
        supported = result["status"] == "PASS" and bool(reload_result.get("verified"))
        all_supported = all_supported and supported
        tracks[track] = {
            "status": result["status"],
            "reload_status": reload_result["status"],
            "supported": supported,
            "minimum_supported_capacity": minimum,
            "minimum_private_fraction": best_private_fraction,
        }

    scientific_status = (
        "COW_MINIMAL_FUNCTIONAL_FORK_SUPPORTED"
        if all_supported
        else "COW_MINIMAL_FUNCTIONAL_FORK_NOT_YET_SUPPORTED"
    )
    summary = {
        "experiment": "COW_CLM_001",
        "status": "PASS" if all_supported else "FAIL",
        "scientific_status": scientific_status,
        "seed": int(protocol["seed"]),
        "protocol_sha256": _protocol_sha256(),
        "implementation_git_blobs": protocol["implementation_git_blobs"],
        "tracks": tracks,
        "hf_token_loaded": bool(os.environ.get("HF_TOKEN")),
        "does_not_rewrite_prior_decisions": True,
        "nonclaims": protocol["nonclaims"],
    }
    _write_json(result_root / "seed_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen COW-CLM-001 end-to-end")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capability-device")
    parser.add_argument("--parallel-tracks", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    protocol = _load_json(PROTOCOL)
    _verify_frozen_protocol(protocol)

    result_root = RESULTS / f"seed-{int(protocol['seed'])}"
    if result_root.exists():
        shutil.rmtree(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    logs = result_root / "logs"
    python = sys.executable

    capability_device = args.capability_device or args.device
    track_commands = {
        "knowledge": [python, "scripts/research/cow_clm_001/run.py", "--track", "knowledge", "--device", args.device, "--output-dir", str(result_root)],
        "capability": [python, "scripts/research/cow_clm_001/run.py", "--track", "capability", "--device", capability_device, "--output-dir", str(result_root)],
    }
    if args.parallel_tracks:
        _launch_pair(track_commands, logs)
    else:
        for name, command in track_commands.items():
            _run_logged(command, logs / f"{name}.log")

    verify_commands = {
        "knowledge-reload": [python, "scripts/research/cow_clm_001/verify_reload.py", "--track", "knowledge", "--device", args.device, "--result-root", str(result_root)],
        "capability-reload": [python, "scripts/research/cow_clm_001/verify_reload.py", "--track", "capability", "--device", capability_device, "--result-root", str(result_root)],
    }
    if args.parallel_tracks:
        _launch_pair(verify_commands, logs)
    else:
        for name, command in verify_commands.items():
            _run_logged(command, logs / f"{name}.log")

    summary = _finalize(protocol, result_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
