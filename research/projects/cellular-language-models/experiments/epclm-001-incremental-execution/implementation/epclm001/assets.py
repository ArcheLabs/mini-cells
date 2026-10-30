"""Read-only asset discovery: never turn random initialization into recovered weights."""

from pathlib import Path

from .common import BASE, NATIVE, ROOT, digest, native, source

MODELS = {
    "E0B_T1": "T1-modern-transformer",
    "E0B_X3": "X3-n1-swiglu-rope",
    "E0B_X4": "X4-n1-swiglu-rope-rmsnorm",
}


def inventory(asset_roots=()):
    # Missing means absent from these searched locations, not globally nonexistent.
    roots = [
        BASE.parent / "native-clm-v0",
        Path("/kaggle/input"),
        Path("/kaggle/working/native-clm-output"),
    ]
    roots += [Path(p) for p in asset_roots]
    files = sorted(
        {p for root in roots if root.exists() for ext in ("*.pt", "*.pth") for p in root.rglob(ext)}
    )
    records = []
    for baseline, name in MODELS.items():
        candidates = [p for p in files if name in p.parts or name in p.name]
        row = {
            "baseline_id": baseline,
            "model_name": name,
            "architecture_source_sha": source(),
            "checkpoint_uri": None,
            "checkpoint_sha256": None,
            "tokenizer_sha256": None,
            "dataset_manifest_sha256": None,
            "evaluation_source_sha": source(),
            "training_seed": None,
            "evidence_status": "UNKNOWN",
            "parameter_count": None,
            "precision": None,
            "loadable": False,
            "status": "ASSET_UNAVAILABLE",
            "reason": "No exact named checkpoint in searched roots",
            "candidate_paths": [str(p) for p in candidates],
        }
        # Multiple candidates must not be selected by quality or latest-file timestamp.
        if len(candidates) == 1:
            p = candidates[0]
            row.update(
                checkpoint_uri=str(p),
                checkpoint_sha256=digest(p),
                status="IDENTITY_REVIEW_REQUIRED",
                reason="Checkpoint found; binding not yet verified",
            )
            try:
                model, metadata = load_native(p, baseline)
                row.update(
                    loadable=True,
                    parameter_count=sum(v.numel() for v in model.parameters()),
                    precision=str(next(model.parameters()).dtype),
                    training_seed=metadata.get("seed"),
                    checkpoint_keys=sorted(metadata),
                )
                nearby = [p.parent, p.parent.parent, p.parent.parent.parent]
                tokenizers = {f for d in nearby for f in d.glob("tokenizer.json")}
                manifests = {f for d in nearby for f in d.glob("*manifest*.json")}
                if len(tokenizers) == 1:
                    row["tokenizer_sha256"] = digest(next(iter(tokenizers)))
                if len(manifests) == 1:
                    row["dataset_manifest_sha256"] = digest(next(iter(manifests)))
            except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as e:
                row.update(status="LOAD_FAILED", reason=type(e).__name__)
        elif candidates:
            row.update(
                status="AMBIGUOUS_CHECKPOINTS",
                reason="Multiple candidates; no result-based selection",
            )
        records.append(row)
    records.append(
        {
            "baseline_id": "E0A_30M",
            "status": "DEFERRED / CONDITIONAL SCALE VALIDATION",
            "execution_allowed": False,
            "loadable": None,
            "reason": "Historical evidence only; not inspected or loaded in round 1",
            "historical_results": "research/projects/cellular-computing/experiments/007-minicells-30m/artifacts/RESULTS.md",
        }
    )
    return {
        "experiment_id": "EPCLM-001",
        "searched_roots": [str(p) for p in roots],
        "assets": records,
        "source_files": {str(p.relative_to(ROOT)): digest(p) for p in sorted(NATIVE.glob("*.py"))},
        "missing_dataset_reason": "No bound evaluation corpus supplied; quality comparison remains blocked",
    }


def load_native(path, baseline):
    if baseline not in MODELS:
        raise ValueError("BASELINE_OUTSIDE_ROUND1_SCOPE")
    import torch

    base, _, phase_c = native()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or "model_state_dict" not in payload
        or payload.get("model") != MODELS[baseline]
    ):
        raise ValueError("CHECKPOINT_SCHEMA_OR_MODEL_ID_MISMATCH")
    if int(payload.get("step", 0)) <= 0:
        raise ValueError("TRAINED_CHECKPOINT_REQUIRED")
    if baseline in ("E0B_X3", "E0B_X4"):
        model = phase_c.build_phase_c(MODELS[baseline])
    else:
        model = base.build_model(MODELS[baseline])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, {
        k: v
        for k, v in payload.items()
        if k != "model_state_dict" and isinstance(v, (str, int, float, bool, type(None)))
    }
