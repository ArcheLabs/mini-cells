#!/usr/bin/env python3
"""Export a previously validated engineering Cell mutation for publication.

The exporter deliberately does not train or execute a formal protocol.  A
source mutation must be supplied by the frozen engineering run (or by the
release configuration); otherwise publication stops with an explicit error.
"""

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
REQUIRED_ARTIFACTS = ("mutation.safetensors", "cell_config.json", "manifest.json")


def _json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"ARTIFACT_VALIDATION_FAILED: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("RELEASE_IDENTITY_MISMATCH: cannot resolve git HEAD")
    return result.stdout.strip()


def _formal_guard(config: dict) -> None:
    if config.get("formal", {}).get("execution_started") is not False:
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: release config crosses formal boundary")
    registry_bytes = FORMAL_REGISTRY.read_bytes()
    if hashlib.sha256(registry_bytes).hexdigest() != FORMAL_REGISTRY_SHA256:
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: formal seed registry changed")
    registry = json.loads(registry_bytes)
    states = {int(row["seed"]): row["state"] for row in registry.get("seeds", [])}
    forbidden = {int(seed) for seed in config["formal"]["forbidden_seeds"]}
    if any(states.get(seed) != "RESERVED_UNTOUCHED" for seed in forbidden):
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: a forbidden seed is not reserved")
    if int(config["mutation"]["engineering_seed"]) in forbidden:
        raise RuntimeError("FORMAL_SEED_GUARD_FAILED: engineering seed is forbidden")


def _source_path(config: dict, override: Path | None) -> Path:
    candidate = override or config.get("mutation", {}).get("source_artifact")
    if not candidate:
        raise RuntimeError(
            "MUTATION_REPLAY_MISMATCH: no frozen engineering mutation was supplied; "
            "refusing to retrain an unverified substitute"
        )
    path = Path(candidate)
    if not path.is_dir():
        raise RuntimeError(f"MUTATION_REPLAY_MISMATCH: source artifact does not exist: {path}")
    return path


def export(config: dict, source: Path, output: Path) -> dict:
    from minicells.hybrid.mutation import CellMutation

    for name in REQUIRED_ARTIFACTS:
        if not (source / name).is_file():
            raise RuntimeError(f"MUTATION_REPLAY_MISMATCH: source is missing {name}")
    mutation = CellMutation.from_pretrained(source)
    foundation = config["foundation"]
    if mutation.base_model and mutation.base_model != foundation["repo_id"]:
        raise RuntimeError("MUTATION_REPLAY_MISMATCH: source base model differs from release config")
    if mutation.base_revision and mutation.base_revision != foundation["revision"]:
        raise RuntimeError("MUTATION_REPLAY_MISMATCH: source base revision differs from release config")
    layer = int(config["mutation"]["layer"])
    if mutation.placements and all(item.layer != layer for item in mutation.placements):
        raise RuntimeError("MUTATION_REPLAY_MISMATCH: source placement differs from release config")

    output.mkdir(parents=True, exist_ok=True)
    existing_manifest = output / "manifest.json"
    if existing_manifest.is_file():
        existing = _json(existing_manifest)
        if existing.get("mutation_sha256") != _sha256(source / "mutation.safetensors"):
            raise RuntimeError("ARTIFACT_VALIDATION_FAILED: refusing to overwrite a conflicting output")
    mutation.save_pretrained(output)
    provenance = {
        "schema_version": 1,
        "release_version": config["package"]["version"],
        "release_tag": config["package"]["tag"],
        "release_commit": config["source"]["commit"],
        "experiment": config["mutation"]["experiment"],
        "engineering_seed": int(config["mutation"]["engineering_seed"]),
        "source_experiment_commit": _json(Path(config["evidence"]["run_identity"]))["source"]["source_commit"],
        "dataset_identity": None,
        "dataset_hash": None,
        "training_steps": None,
        "optimizer_configuration": None,
        "trainable_parameter_count": int(sum(value.numel() for value in mutation.tensors.values())),
        "formal_execution_started": False,
        "formal_status": "NOT_STARTED",
        "artifact_origin": config["mutation"].get("artifact_origin", "exact_saved_delta"),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evaluation = {"status": "Engineering Evidence", **config["expected_metrics"], "locality_status": "unresolved"}
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "status": "EXPORTED",
        "artifact_origin": provenance["artifact_origin"],
        "release": config["package"],
        "release_commit": config["source"]["commit"],
        "engineering_seed": provenance["engineering_seed"],
        "mutation_digest": mutation.digest(),
        "artifact_sha256": _sha256(output / "mutation.safetensors"),
        "formal_execution_started": False,
    }
    (output / "EXPORT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-mutation", type=Path)
    parser.add_argument("--device", default="cpu", help="reserved for deterministic replay implementations")
    parser.add_argument("--work-dir", type=Path, help="reserved replay workspace")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-replay", action="store_true", help="require an exact saved engineering delta")
    args = parser.parse_args()
    config = _json(args.release_config)
    expected_commit = config["source"]["commit"]
    actual_commit = _head()
    if expected_commit.startswith("<") or expected_commit != actual_commit:
        raise SystemExit(
            "RELEASE_IDENTITY_MISMATCH: release config must contain the exact immutable checkout commit"
        )
    try:
        _formal_guard(config)
        report = export(config, _source_path(config, args.source_mutation), args.output.resolve())
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
