from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
HYBRID_ROOT = ROOT / "scripts" / "research" / "granite_hybrid_clm_v01"
PROTOCOL_PATH = (
    ROOT / "research" / "validations" / "hybrid-clm-prompt-address-001" / "protocol.json"
)
RESULTS_ROOT = ROOT / "results" / "hybrid-clm-prompt-address-001"
for path in (SRC_ROOT, HYBRID_ROOT):
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from run_milestone import run as run_milestone  # noqa: E402
from verify_reload import run as verify_reload  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _verify_implementation(protocol: dict[str, Any]) -> None:
    expected = protocol.get("implementation_git_blobs")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("protocol has no registered implementation Git blobs")
    for relative, digest in expected.items():
        path = ROOT / relative
        observed = _git_blob_sha(path)
        if observed != digest:
            raise RuntimeError(
                f"implementation identity mismatch for {relative}: {observed} != {digest}"
            )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _environment(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "requested_device": device,
        "hf_token_loaded": bool(os.environ.get("HF_TOKEN")),
        "git_head": _git_head(),
    }
    if torch.cuda.is_available():
        payload["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return payload


def run(*, device: str) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    if protocol.get("experiment") != "HYBRID_CLM_PROMPT_ADDRESS_001":
        raise RuntimeError("unexpected protocol experiment identity")
    if protocol.get("status") != "DIAGNOSTIC_PROTOCOL_FROZEN_GPU_PENDING":
        raise RuntimeError("prompt-address diagnostic protocol is not frozen/pending")
    hosted = protocol.get("hosted_environment", {})
    if bool(hosted.get("require_hf_token")) and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required by the frozen hosted-run protocol")
    _verify_implementation(protocol)

    seed = int(protocol["seed"])
    smoke = protocol["smoke"]
    output_dir = RESULTS_ROOT / f"seed-{seed}"
    result = run_milestone(
        device=device,
        fact_count=int(smoke["facts"]),
        seed=seed,
        address_steps=int(smoke["address_steps"]),
        transform_steps=int(smoke["transform_steps"]),
        output_dir=output_dir,
    )

    reload_status = "SKIPPED_RUNNER_NOT_SUPPORTED"
    if result.get("status") == "GRANITE_HYBRID_CLM_V01_SUPPORTED":
        reload_report = verify_reload(result_dir=output_dir, device=device)
        reload_status = str(reload_report["status"])

    cells = list(result.get("cells", []))
    address_passes = sum(bool(cell.get("address", {}).get("passed")) for cell in cells)
    heldout_passes = sum(
        float(cell.get("address", {}).get("heldout_positive_recall", 0.0)) == 1.0
        for cell in cells
    )
    history_address_passes = sum(
        float(cell.get("address", {}).get("history_false_positive_rate", 1.0)) == 0.0
        for cell in cells
    )
    committed = int(result.get("committed_facts", 0))
    passed = (
        result.get("status") == "GRANITE_HYBRID_CLM_V01_SUPPORTED"
        and reload_status == "GRANITE_HYBRID_CLM_V01_RELOAD_VERIFIED"
    )
    summary = {
        "experiment": "HYBRID_CLM_PROMPT_ADDRESS_001",
        "status": "PASS" if passed else "FAIL",
        "scientific_status": (
            "PROMPT_SCOPED_HYBRID_SMOKE_SUPPORTED"
            if passed
            else "PROMPT_SCOPED_HYBRID_SMOKE_NOT_YET_SUPPORTED"
        ),
        "seed": seed,
        "protocol_sha256": _protocol_sha256(),
        "implementation_git_blobs": protocol["implementation_git_blobs"],
        "routing": result.get("routing"),
        "address_passes": address_passes,
        "heldout_address_passes": heldout_passes,
        "history_address_passes": history_address_passes,
        "requested_facts": int(result.get("requested_facts", 0)),
        "committed_facts": committed,
        "retention_choice_accuracy": float(result.get("retention_choice_accuracy", 0.0)),
        "contextual_child_status": result.get("contextual_child", {}).get("status"),
        "milestone_result_status": result.get("status"),
        "reload_status": reload_status,
        "environment": _environment(device),
    }
    _write_json(output_dir / "seed_summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "scientific_status",
                    "address_passes",
                    "history_address_passes",
                    "committed_facts",
                    "retention_choice_accuracy",
                    "reload_status",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hybrid CLM Prompt Address 001")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(device=args.device)


if __name__ == "__main__":
    main()
