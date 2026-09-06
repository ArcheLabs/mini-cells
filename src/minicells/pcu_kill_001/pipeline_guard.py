"""Fail-fast audit artifacts that must exist even when the science pipeline kills early."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable

from .governance import git_provenance, write_json
from .synthetic import POSITIVE_CONTROL_VERSION


REGISTERED_THRESHOLDS = {
    "g0_top1_token_agreement": 1.0,
    "cache_top1_token_agreement": 1.0,
    "context_oracle_accuracy": 0.90,
    "context_oracle_retrieval_accuracy": 0.90,
    "context_oracle_composition_accuracy": 0.90,
    "direct_accuracy": 0.80,
    "merge_retention": 0.90,
    "composition_accuracy": 0.50,
    "composition_synergy": 0.30,
    "anchor_regression": 0.01,
    "matched_lora_parameter_tolerance": 0.10,
}


def _equivalence_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    g0 = list(kwargs["g0"])
    g0_full_moe = kwargs["g0_full_moe"]
    g0_e2e = kwargs["g0_e2e"]
    cache_gate = kwargs["cache_gate"]
    g0_exact = bool(
        all(item.passed for item in g0)
        and (g0_full_moe is None or g0_full_moe.passed)
        and g0_e2e.passed
    )
    return {
        "schema": "minicells.pcu-kill-001.equivalence.v2",
        "g0_expert": {str(index): item.to_dict() for index, item in enumerate(g0)},
        "g0_full_moe": g0_full_moe.to_dict() if g0_full_moe is not None else None,
        "g0_end_to_end": g0_e2e.to_dict(),
        "cache": cache_gate.to_dict(),
        "g0_exact_embedding": g0_exact,
        "scientific_evidence": kwargs["phase"] == "formal",
    }


def persist_pre_science_evidence(**kwargs: Any) -> None:
    """Persist immutable gates before context-oracle/allocation early exits can occur."""
    output = Path(kwargs["output"])
    output.mkdir(parents=True, exist_ok=True)
    world = kwargs["world"]
    audit = kwargs["audit"]
    manifest = dict(kwargs["manifest"])
    inspector = kwargs["inspector"]
    cache_gate = kwargs["cache_gate"]
    phase = str(kwargs["phase"])
    seed = int(kwargs["seed"])
    root = Path(__file__).resolve().parents[3]
    source = git_provenance(root)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-kill-001.run-identity.v1",
        "experiment": "PCU-KILL-001",
        "phase": phase,
        "seed": seed,
        "run_id": output.name,
        "positive_control_version": POSITIVE_CONTROL_VERSION,
        "formal_execution_not_started": phase != "formal",
        "source": source,
    })
    write_json(output / "MODEL_MANIFEST.json", {
        **manifest,
        "architecture": asdict(inspector),
    })
    write_json(output / "DATASET_MANIFEST.json", {
        **world.to_manifest(),
        "scientific_evidence": phase == "formal",
    })
    write_json(output / "DATASET_AUDIT.json", {
        **audit.to_dict(),
        "scientific_evidence": phase == "formal",
    })
    write_json(output / "EQUIVALENCE.json", _equivalence_payload(kwargs))
    write_json(output / "CACHE_EQUIVALENCE.json", cache_gate.to_dict())


def _augment_decision_with_oracle_v2(output: Path) -> None:
    oracle_path = output / "CONTEXT_ORACLE.json"
    if not oracle_path.is_file():
        return
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    for name in ("DECISION.json", "ENGINEERING_DECISION.json"):
        path = output / name
        if not path.is_file():
            continue
        decision = json.loads(path.read_text(encoding="utf-8"))
        metrics = dict(decision.get("metrics", {}))
        metrics.update({
            "context_oracle_accuracy": float(oracle.get("accuracy", 0.0)),
            "context_oracle_retrieval_a_accuracy": float(oracle.get("retrieval_a_accuracy", 0.0)),
            "context_oracle_retrieval_b_accuracy": float(oracle.get("retrieval_b_accuracy", 0.0)),
            "context_oracle_composition_accuracy": float(oracle.get("composition_accuracy", 0.0)),
        })
        diagnostic = oracle.get("free_generation_diagnostic") or {}
        if diagnostic:
            metrics["context_oracle_free_generation_diagnostic_accuracy"] = float(
                diagnostic.get("accuracy", 0.0)
            )
        thresholds = dict(decision.get("thresholds", {}))
        for key, value in REGISTERED_THRESHOLDS.items():
            thresholds.setdefault(key, value)
        decision["metrics"] = metrics
        decision["thresholds"] = thresholds
        decision["positive_control"] = {
            "version": oracle.get("positive_control_version"),
            "mode": oracle.get("mode"),
            "candidate_pool_size": oracle.get("candidate_pool_size"),
            "passed": oracle.get("passed"),
            "retrieval_a_accuracy": oracle.get("retrieval_a_accuracy"),
            "retrieval_b_accuracy": oracle.get("retrieval_b_accuracy"),
            "composition_accuracy": oracle.get("composition_accuracy"),
            "free_generation_gate": False,
        }
        write_json(path, decision)


def install_pipeline_guard(experiment_module: Any) -> Callable[..., Any]:
    """Wrap the shared worker once, preserving all scientific behavior."""
    current = experiment_module._run_shared_scientific_pipeline
    if getattr(current, "_pcu_fail_fast_audit_guard", False):
        return current

    def guarded_shared_pipeline(**kwargs: Any) -> Any:
        persist_pre_science_evidence(**kwargs)
        result = current(**kwargs)
        _augment_decision_with_oracle_v2(Path(kwargs["output"]))
        return result

    guarded_shared_pipeline._pcu_fail_fast_audit_guard = True  # type: ignore[attr-defined]
    guarded_shared_pipeline._pcu_original_pipeline = current  # type: ignore[attr-defined]
    experiment_module._run_shared_scientific_pipeline = guarded_shared_pipeline
    return guarded_shared_pipeline
