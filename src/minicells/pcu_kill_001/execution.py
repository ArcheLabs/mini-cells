"""Fail-fast entry points for PCU-KILL-001 engineering and formal execution.

The large experiment module owns the shared scientific pipeline.  These entry
points add the protocol's first kill gate *before* any synthetic world, cache,
allocation, mutation training, LoRA training, or task evaluation is created.
A G0 failure is a valid scientific outcome, not an invalid runtime failure.
"""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from . import experiment as _experiment
from .cellular import GraniteArchitectureInspector
from .equivalence import verify_end_to_end, verify_expert_algebra, verify_full_moe
from .governance import (
    DEVELOPMENT_SEED,
    EXPERIMENT_ID,
    assert_engineering_seed,
    assert_seed_registry,
    git_provenance,
    runtime_provenance,
    set_deterministic_seeds,
    write_json,
)
from .model import MODEL_ID, cellularize_model, load_granite, target_module
from .registry import module_tensor_hash


def _resolved_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _g0_preflight(
    *,
    seed: int,
    model_repo: str,
    revision: str | None,
    device: str,
    probe_prefix: str,
) -> dict[str, Any]:
    """Load the exact foundation and evaluate every registered G0 level."""
    set_deterministic_seeds(seed)
    tokenizer, original, manifest = load_granite(model_repo, revision=revision, device=device)
    inspector = GraniteArchitectureInspector.inspect(original, require_granite=True)
    cellular, _ = cellularize_model(original, inspector)
    original_target = target_module(original, inspector.target_path)
    cellular_target = target_module(cellular, inspector.target_path)
    probe_texts = [f"{probe_prefix} {index:03d}." for index in range(128)]
    probe_inputs = _experiment._token_batch(tokenizer, probe_texts, device)
    g0_expert = [
        verify_expert_algebra(
            original_target.experts,
            index,
            inspector.partition,
            vectors=1024,
            seed=seed,
        )
        for index in range(inspector.local_experts)
    ]
    moe_probe = torch.randn(
        128,
        inspector.hidden_size,
        generator=torch.Generator(device="cpu").manual_seed(seed + 1),
    ).to(device)
    g0_full_moe = verify_full_moe(original_target, cellular_target, moe_probe)
    g0_e2e = verify_end_to_end(original, cellular, probe_inputs)
    passed = bool(
        all(item.passed for item in g0_expert)
        and g0_full_moe.passed
        and g0_e2e.passed
    )
    manifest = {
        **manifest,
        "architecture": asdict(inspector),
        "foundation_tensor_sha256": module_tensor_hash(original),
    }
    return {
        "passed": passed,
        "tokenizer": tokenizer,
        "model": original,
        "cellular": cellular,
        "manifest": manifest,
        "inspector": inspector,
        "metrics": {
            "g0_expert": {str(index): item.to_dict() for index, item in enumerate(g0_expert)},
            "g0_full_moe": g0_full_moe.to_dict(),
            "g0_end_to_end": g0_e2e.to_dict(),
            "g0_exact_embedding": passed,
        },
    }


def _release_g0_preflight(preflight: dict[str, Any]) -> None:
    """Drop the duplicate full models before the shared worker reloads them."""
    for key in ("tokenizer", "model", "cellular", "inspector"):
        preflight.pop(key, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_g0_failure(
    output: Path,
    *,
    seed: int,
    phase: str,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    scientific_evidence = phase == "formal"
    valid_formal_run = phase == "formal"
    source = git_provenance(Path(__file__).resolve().parents[3])
    output.mkdir(parents=True, exist_ok=True)
    decision = {
        "schema": "minicells.pcu-kill-001.decision.v1",
        "experiment": EXPERIMENT_ID,
        "phase": phase,
        "status": "EXACT_CELL_EMBEDDING_FAILED",
        "scientific_evidence": scientific_evidence,
        "scientific_decision": True,
        "valid_run": True,
        "valid_formal_run": valid_formal_run,
        "formal_protocol_ready": False,
        "formal_ready": False,
        "gates": {"g0": False},
        "metrics": dict(metrics),
        "foundation": dict(manifest),
        "source": source,
        "reason": "G0 equivalence failed; protocol stopped before task generation or mutation training",
        "formal_execution_not_started": phase != "formal",
    }
    write_json(output / "MODEL_MANIFEST.json", dict(manifest))
    write_json(output / "EQUIVALENCE.json", {**dict(metrics), "scientific_evidence": scientific_evidence})
    write_json(output / "DECISION.json", decision)
    if phase == "engineering":
        write_json(output / "ENGINEERING_DECISION.json", decision)
    write_json(
        output / "RUN_MANIFEST.json",
        {
            "schema": "minicells.pcu-kill-001.run-manifest.v3",
            "experiment": EXPERIMENT_ID,
            "phase": phase,
            "seed": seed,
            "backend": "granite",
            "scientific_evidence": scientific_evidence,
            "source": source,
            "runtime": runtime_provenance(device),
            "stopped_at": "G0",
        },
    )
    return {
        "status": decision["status"],
        "scientific_evidence": scientific_evidence,
        "scientific_decision": True,
        "valid_run": True,
        "valid_formal_run": valid_formal_run,
        "formal_ready": False,
        "g0": False,
        "output": str(output),
    }


def run_engineering(
    seed: int = DEVELOPMENT_SEED,
    backend: str = "granite",
    output: Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Run engineering with a true G0 kill gate before the shared pipeline."""
    if backend == "toy":
        return _experiment.run_engineering(seed=seed, backend=backend, output=output, device=device)
    if backend != "granite":
        raise ValueError("backend must be granite or toy")
    assert_engineering_seed(seed)
    assert_seed_registry(Path(__file__).resolve().parents[3] / "research/formal_seed_registry.json")
    output = output or Path("artifacts/research/pcu-kill-001/engineering") / str(seed)
    chosen_device = _resolved_device(device)
    g0 = _g0_preflight(
        seed=seed,
        model_repo=MODEL_ID,
        revision=None,
        device=chosen_device,
        probe_prefix="PCU-KILL-001 immutable engineering probe",
    )
    if not g0["passed"]:
        return _write_g0_failure(
            output,
            seed=seed,
            phase="engineering",
            manifest=g0["manifest"],
            metrics=g0["metrics"],
            device=chosen_device,
        )
    # The shared implementation rechecks G0 as part of its provenance.  Free
    # the preflight's duplicate models first so the second load does not double
    # resident GPU memory.
    _release_g0_preflight(g0)
    return _experiment.run_engineering(seed=seed, backend=backend, output=output, device=chosen_device)


def run_formal_execution(
    seed: int,
    protocol_path: Path,
    output: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one formal seed with G0 fail-fast before any formal task data."""
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_FORMAL":
        raise RuntimeError("formal worker requires FROZEN_BEFORE_FORMAL protocol")
    if int(seed) not in tuple(int(value) for value in payload.get("formal_seeds", [])):
        raise ValueError(f"seed {seed} is not listed in the frozen protocol")
    model_info = payload["model"]
    chosen_device = _resolved_device(device)
    g0 = _g0_preflight(
        seed=seed,
        model_repo=str(model_info["model_repo"]),
        revision=str(model_info["model_revision"]),
        device=chosen_device,
        probe_prefix=f"PCU-KILL-001 formal seed {seed} probe",
    )
    manifest = g0["manifest"]
    for key in (
        "model_repo",
        "model_revision",
        "config_sha256",
        "weight_file_sha256",
        "tokenizer_sha256",
        "foundation_tensor_sha256",
    ):
        expected = model_info.get(key)
        if expected not in (None, "", []) and manifest.get(key) != expected:
            raise RuntimeError(f"formal foundation identity mismatch before G0 decision: {key}")
    if not g0["passed"]:
        return _write_g0_failure(
            output,
            seed=seed,
            phase="formal",
            manifest=manifest,
            metrics=g0["metrics"],
            device=chosen_device,
        )
    _release_g0_preflight(g0)
    return _experiment.run_formal_execution(seed, protocol_path, output, chosen_device)
