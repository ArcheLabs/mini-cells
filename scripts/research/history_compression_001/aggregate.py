from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "research" / "validations" / "history-compression-001" / "protocol.json"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "history-compression-001"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _environment_signature(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "torch": environment.get("torch"),
        "transformers": environment.get("transformers"),
        "huggingface_hub": environment.get("huggingface_hub"),
        "safetensors": environment.get("safetensors"),
        "cuda_device_name": environment.get("cuda_device_name"),
        "dtype": environment.get("dtype"),
    }


def _write_summary_csv(rows: list[dict[str, Any]]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "mode",
        "history_prompt_count",
        "status",
        "heldout_nll_gain",
        "history_evaluation_mean_kl",
        "history_evaluation_top1_identity",
        "delta_l2_norm",
        "expert_index",
        "group_index",
        "target_router_topk_identity",
    ]
    with (ARTIFACTS / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def aggregate() -> dict[str, Any]:
    protocol = _load(PROTOCOL_PATH)
    protocol_sha = _sha256(PROTOCOL_PATH)
    formal_seeds = [int(value) for value in protocol["formal_seeds"]]
    modes = [dict(value) for value in protocol["compression_modes"]]
    mode_ids = [str(mode["id"]) for mode in modes]
    budgets = {str(mode["id"]): int(mode["history_prompt_count"]) for mode in modes}

    rows: list[dict[str, Any]] = []
    completed_seeds: list[int] = []
    environment_signatures: dict[str, dict[str, Any]] = {}
    observed_hashes: set[str] = set()
    malformed: list[str] = []

    for seed in formal_seeds:
        seed_root = ARTIFACTS / f"seed-{seed}"
        seed_complete = True
        seed_environment: dict[str, Any] | None = None
        for mode_id in mode_ids:
            path = seed_root / mode_id / "result.json"
            if not path.is_file():
                seed_complete = False
                continue
            result = _load(path)
            if result.get("experiment") != protocol["experiment"]:
                malformed.append(f"seed={seed} mode={mode_id}: experiment mismatch")
                continue
            if int(result.get("seed", -1)) != seed or result.get("mode") != mode_id:
                malformed.append(f"seed={seed} mode={mode_id}: result identity mismatch")
                continue
            observed_hashes.add(str(result.get("protocol_sha256")))
            environment = dict(result.get("environment") or {})
            if seed_environment is None:
                seed_environment = environment
            metrics = result["metrics"]
            selection = result["selection"]
            rows.append(
                {
                    "seed": seed,
                    "mode": mode_id,
                    "history_prompt_count": int(result["history_prompt_count"]),
                    "status": result["status"],
                    "heldout_nll_gain": float(metrics["heldout_nll_gain"]),
                    "history_evaluation_mean_kl": float(
                        metrics["history_evaluation_mean_kl"]
                    ),
                    "history_evaluation_top1_identity": float(
                        metrics["history_evaluation_top1_identity"]
                    ),
                    "delta_l2_norm": float(metrics["delta_l2_norm"]),
                    "expert_index": int(selection["expert_index"]),
                    "group_index": int(selection["group_index"]),
                    "target_router_topk_identity": float(
                        metrics["target_router_topk_identity"]
                    ),
                }
            )
        if seed_complete and seed_environment is not None:
            completed_seeds.append(seed)
            environment_signatures[str(seed)] = _environment_signature(seed_environment)

    _write_summary_csv(rows)
    missing_seeds = [seed for seed in formal_seeds if seed not in completed_seeds]
    protocol_consistent = observed_hashes in (set(), {protocol_sha})
    unique_envs = {
        json.dumps(value, sort_keys=True) for value in environment_signatures.values()
    }
    environment_consistent = len(unique_envs) <= 1

    per_mode: dict[str, dict[str, Any]] = {}
    for mode_id in mode_ids:
        mode_rows = [row for row in rows if row["mode"] == mode_id]
        passed = sorted(int(row["seed"]) for row in mode_rows if row["status"] == "PASS")
        failed = sorted(int(row["seed"]) for row in mode_rows if row["status"] == "FAIL")
        complete = len(mode_rows) == len(formal_seeds)
        supported = complete and len(passed) >= 2
        per_mode[mode_id] = {
            "history_prompt_count": budgets[mode_id],
            "passed_seeds": passed,
            "failed_seeds": failed,
            "supported": supported,
        }

    full_supported = bool(per_mode.get("full_32", {}).get("supported"))
    supported_modes = [mode_id for mode_id in mode_ids if per_mode[mode_id]["supported"]]
    minimum_budget = (
        min(budgets[mode_id] for mode_id in supported_modes)
        if full_supported and supported_modes
        else None
    )
    ascending = sorted(((budgets[mode_id], per_mode[mode_id]["supported"]) for mode_id in mode_ids))
    seen_supported = False
    monotonic = True
    for _budget, supported in ascending:
        if supported:
            seen_supported = True
        elif seen_supported:
            monotonic = False

    complete = not missing_seeds and not malformed
    consistent = protocol_consistent and environment_consistent
    if not complete:
        status = "HISTORY_COMPRESSION_INCOMPLETE"
        scientific_decision = False
    elif not consistent:
        status = "HISTORY_COMPRESSION_PROTOCOL_OR_ENVIRONMENT_MISMATCH"
        scientific_decision = False
    elif not full_supported:
        status = "HISTORY_COMPRESSION_POSITIVE_CONTROL_FAILED"
        scientific_decision = True
    elif not monotonic:
        status = "HISTORY_COMPRESSION_NON_MONOTONIC"
        scientific_decision = True
    elif minimum_budget == 0:
        status = "HISTORY_COMPRESSION_ZERO_HISTORY_SUPPORTED"
        scientific_decision = True
    elif minimum_budget == 2:
        status = "HISTORY_COMPRESSION_TO_2_SUPPORTED"
        scientific_decision = True
    elif minimum_budget == 8:
        status = "HISTORY_COMPRESSION_TO_8_SUPPORTED"
        scientific_decision = True
    else:
        status = "HISTORY_COMPRESSION_BEYOND_FULL_NOT_SUPPORTED"
        scientific_decision = True

    decision = {
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol_sha,
        "formal_seeds": formal_seeds,
        "completed_seeds": completed_seeds,
        "missing_seeds": missing_seeds,
        "protocol_consistent": protocol_consistent,
        "observed_protocol_hashes": sorted(observed_hashes),
        "environment_consistent": environment_consistent,
        "environment_signatures": environment_signatures,
        "malformed_results": malformed,
        "per_mode": per_mode,
        "positive_control_supported": full_supported,
        "minimum_observed_supported_history_prompts": minimum_budget,
        "support_monotone_with_history_budget": monotonic,
        "scientific_decision": scientific_decision,
        "status": status,
        "not_claimed": list(protocol["decision"]["not_claimed"]),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return decision


def main() -> int:
    decision = aggregate()
    print(
        json.dumps(
            {
                "status": decision["status"],
                "completed_seeds": decision["completed_seeds"],
                "minimum_observed_supported_history_prompts": decision[
                    "minimum_observed_supported_history_prompts"
                ],
                "per_mode": decision["per_mode"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
