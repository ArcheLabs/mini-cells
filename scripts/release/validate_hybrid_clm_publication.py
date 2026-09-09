#!/usr/bin/env python3
"""Validate a HybridCLM mutation before any Hugging Face publication call."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


FORMAL_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_REGISTRY_SHA256 = "d86da9656ef18c75bab8fa41a9053a68cf4dbb64fbba1a93c80850c29b42e06a"
REQUIRED = ("mutation.safetensors", "cell_config.json", "manifest.json", "provenance.json", "evaluation.json")


def _json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"ARTIFACT_VALIDATION_FAILED: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def _formal_guard(config: dict) -> None:
    if config.get("formal", {}).get("execution_started") is not False:
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: release config crosses formal boundary")
    raw = FORMAL_REGISTRY.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FORMAL_REGISTRY_SHA256:
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: formal seed registry changed")
    states = {int(row["seed"]): row["state"] for row in json.loads(raw).get("seeds", [])}
    forbidden = {int(seed) for seed in config["formal"]["forbidden_seeds"]}
    if any(states.get(seed) != "RESERVED_UNTOUCHED" for seed in forbidden):
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: forbidden seed state changed")
    if int(config["mutation"]["engineering_seed"]) in forbidden:
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: engineering seed is forbidden")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_metrics(config: dict, evaluation: dict) -> bool:
    expected = config.get("expected_metrics", {})
    keys = ("ranking_off", "ranking_on", "ranking_gain", "answer_margin_gain", "answer_nll_gain", "b_control_answer_nll_increase")
    return all(key in evaluation and abs(float(evaluation[key]) - float(expected[key])) <= 1e-12 for key in keys)


def validate(config: dict, mutation_root: Path, *, skip_model: bool = False, device: str = "cpu", local_files_only: bool = False) -> dict:
    from minicells.hybrid.mutation import CellMutation

    missing = [name for name in REQUIRED if not (mutation_root / name).is_file()]
    if missing:
        raise RuntimeError(f"ARTIFACT_VALIDATION_FAILED: missing {missing}")
    if any(path.suffix in {".pkl", ".pickle", ".pt", ".pth"} for path in mutation_root.rglob("*")):
        raise RuntimeError("ARTIFACT_VALIDATION_FAILED: pickle/checkpoint payloads are forbidden")
    manifest = _json(mutation_root / "manifest.json")
    provenance = _json(mutation_root / "provenance.json")
    evaluation = _json(mutation_root / "evaluation.json")
    mutation = CellMutation.from_pretrained(mutation_root)
    foundation = config["foundation"]
    if manifest.get("base_model") != foundation["repo_id"] or manifest.get("base_revision") != foundation["revision"]:
        raise RuntimeError("BASE_COMPATIBILITY_FAILED: manifest foundation identity differs")
    if int(provenance.get("engineering_seed", -1)) != int(config["mutation"]["engineering_seed"]):
        raise RuntimeError("RELEASE_IDENTITY_MISMATCH: engineering seed differs")
    if provenance.get("formal_execution_started") is not False or provenance.get("formal_status") != "NOT_STARTED":
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: mutation provenance crossed formal boundary")
    if _sha256(mutation_root / "mutation.safetensors") != manifest.get("mutation_sha256"):
        raise RuntimeError("ARTIFACT_VALIDATION_FAILED: mutation SHA256 mismatch")
    metrics_match = _check_metrics(config, evaluation)
    if not metrics_match:
        raise RuntimeError("ARTIFACT_VALIDATION_FAILED: frozen engineering metrics do not match")

    compatibility = False
    lifecycle = False
    if not skip_model:
        from minicells import HybridCLM

        hybrid = HybridCLM.from_pretrained(
            foundation["repo_id"],
            revision=foundation["revision"],
            device=device,
            local_files_only=local_files_only,
        )
        if not mutation.placements:
            raise RuntimeError("BASE_COMPATIBILITY_FAILED: mutation has no explicit placement")
        hybrid.cellularize(mutation.placements)
        mutation.validate_against(hybrid.model, inspection=hybrid.inspect(), base_model=foundation["repo_id"], base_revision=foundation["revision"])
        hybrid.attach(mutation)
        hybrid.set_alpha(mutation, 0.0)
        hybrid.set_alpha(mutation, 1.0)
        hybrid.detach(mutation)
        lifecycle = hybrid.verify_restoration().passed
        compatibility = lifecycle
    report = {
        "status": "READY_FOR_PUBLICATION" if metrics_match and (compatibility or skip_model) else "ABORT_PUBLICATION",
        "release": config["package"]["version"],
        "release_tag": config["package"]["tag"],
        "release_commit": config["source"]["commit"],
        "base_model": foundation["repo_id"],
        "base_revision": foundation["revision"],
        "mutation_digest": mutation.digest(),
        "artifact_sha256": manifest["mutation_sha256"],
        "engineering_seed": int(config["mutation"]["engineering_seed"]),
        "metrics_match": metrics_match,
        "artifact_valid": True,
        "compatibility_valid": compatibility,
        "lifecycle_valid": lifecycle,
        "formal_execution_started": False,
        "formal_seeds_untouched": True,
        "model_validation_skipped": skip_model,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--mutation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("PUBLICATION_VALIDATION.json"))
    parser.add_argument("--skip-model", action="store_true", help="only for offline dry runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        config = _json(args.release_config)
        if config["source"]["commit"].startswith("<") or config["source"]["commit"] != _head():
            raise RuntimeError("RELEASE_IDENTITY_MISMATCH: checkout does not match release config commit")
        _formal_guard(config)
        report = validate(config, args.mutation.resolve(), skip_model=args.skip_model, device=args.device, local_files_only=args.local_files_only)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        report = {
            "status": "ABORT_PUBLICATION",
            "error": str(exc),
            "formal_execution_started": False,
            "formal_seeds_untouched": True,
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY_FOR_PUBLICATION" else 1


if __name__ == "__main__":
    sys.exit(main())
