from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "research" / "validations" / "clm-conversion-kill-test-001" / "protocol.json"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = b"blob " + str(len(data)).encode() + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def _load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def verify_implementation(protocol: dict[str, Any]) -> dict[str, str]:
    expected = protocol.get("implementation_git_blobs")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("frozen protocol has no implementation_git_blobs map")
    observed: dict[str, str] = {}
    for relative_path, expected_sha in expected.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(f"missing frozen implementation file: {relative_path}")
        observed_sha = _git_blob_sha(path)
        observed[relative_path] = observed_sha
        if observed_sha != expected_sha:
            raise RuntimeError(
                "implementation identity mismatch for "
                f"{relative_path}: {observed_sha} != {expected_sha}"
            )
    return observed


def _augment_result(seed: int, implementation: dict[str, str]) -> None:
    seed_root = ROOT / "results" / "clm-conversion-kill-test-001" / f"seed-{seed}"
    for name in ("result.json", "seed_summary.json"):
        path = seed_root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["implementation_git_blobs"] = implementation
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verified formal entrypoint for CLM Conversion Kill Test 001 v1.2"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    protocol = _load_protocol()
    if protocol.get("protocol_version") != 1.2:
        raise RuntimeError("formal_entrypoint_v12.py requires protocol_version 1.2")
    if protocol.get("status") != "PROTOCOL_FROZEN_GPU_PENDING":
        raise RuntimeError("formal protocol is not in frozen GPU-pending state")
    implementation = verify_implementation(protocol)

    module_root = ROOT / "scripts" / "research" / "clm_conversion_kill_test_001"
    for path in (ROOT, module_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from scripts.research.clm_conversion_kill_test_001.run_seed_v12 import run

    run(args.seed, args.device)
    _augment_result(args.seed, implementation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
