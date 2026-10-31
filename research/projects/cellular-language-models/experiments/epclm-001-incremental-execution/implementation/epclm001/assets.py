"""Read-only asset discovery: never turn random initialization into recovered weights."""

import hashlib
import json
import pickle
import pickletools
import zipfile
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
        ROOT / ".native-clm-work",
        Path("/shared/kaggle/working/native-clm"),
        Path("/kaggle/working/native-clm"),
        Path("/kaggle/input"),
        Path("/kaggle/working/native-clm-output"),
    ]
    roots += [Path(p) for p in asset_roots]
    files = sorted(
        {
            p
            for root in roots
            if root.exists()
            for ext in ("*.pt", "*.pth", "*.ckpt", "*.bin")
            for p in root.rglob(ext)
        }
    )
    headers = {p: checkpoint_models(p) for p in files}
    records = []
    for baseline, name in MODELS.items():
        candidates = [p for p in files if name in headers[p]]
        row = {
            "baseline_id": baseline,
            "model_name": name,
            "architecture_source_sha": None,
            "architecture_source_status": "UNKNOWN_TRAINING_SOURCE",
            "adapter_source_sha": source(),
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
            "binding_complete": False,
            "status": "ASSET_UNAVAILABLE",
            "reason": "No eligible checkpoint header in searched roots",
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
                    precision_scope="CPU loader dtype; not an assertion of historical autocast",
                    checkpoint_tensor_dtypes=metadata.get("checkpoint_tensor_dtypes"),
                    training_seed=metadata.get("seed"),
                    checkpoint_keys=sorted(metadata),
                )
                row["training_step"] = metadata.get("step")
                row["consumed_tokens"] = metadata.get("consumed_tokens")
                row["training_profile"] = metadata.get("profile")
                recorded_source = metadata.get("training_source_sha", metadata.get("source_sha"))
                if (
                    isinstance(recorded_source, str)
                    and len(recorded_source) == 40
                    and all(c in "0123456789abcdef" for c in recorded_source)
                ):
                    row["architecture_source_sha"] = recorded_source
                    row["architecture_source_status"] = (
                        "CHECKPOINT_RECORDED_NOT_EXTERNALLY_CERTIFIED"
                    )
                binding_path = p.with_name(p.name + ".identity.json")
                if binding_path.is_file():
                    binding = json.loads(binding_path.read_text())
                    value = binding.get("training_source_sha")
                    if (
                        binding.get("checkpoint_sha256") == row["checkpoint_sha256"]
                        and isinstance(value, str)
                        and len(value) == 40
                        and all(c in "0123456789abcdef" for c in value)
                    ):
                        row["architecture_source_sha"] = value
                        row["architecture_source_status"] = (
                            "CHECKPOINT_HASH_BOUND_SIDECAR_DECLARATION"
                        )
                # Match by saved hashes, including a sibling cache tree, not filename proximity.
                tokenizers = {
                    f for root in roots if root.exists() for f in root.rglob("tokenizer.json")
                }
                manifests = {
                    f for root in roots if root.exists() for f in root.rglob("corpus-manifest.json")
                }
                bound = []
                for candidate in manifests:
                    try:
                        content = json.loads(candidate.read_text())
                    except (ValueError, OSError):
                        continue
                    if not isinstance(content, dict):
                        continue
                    corpus_sha = hashlib.sha256(
                        json.dumps(content, sort_keys=True).encode()
                    ).hexdigest()
                    if corpus_sha == metadata.get("corpus_manifest_sha256"):
                        bound.append((candidate, content, corpus_sha))
                if len(bound) == 1:
                    manifest_path, content, corpus_sha = bound[0]
                    row["dataset_manifest_sha256"] = corpus_sha
                    row["dataset_manifest_file_sha256"] = digest(manifest_path)
                    row["dataset_revision"] = content.get("dataset_revision")
                    matching = [
                        t for t in tokenizers if digest(t) == content.get("tokenizer_sha256")
                    ]
                    if matching:
                        row["tokenizer_sha256"] = content["tokenizer_sha256"]
                row["binding_complete"] = all(
                    row.get(k) is not None
                    for k in (
                        "architecture_source_sha",
                        "training_seed",
                        "training_step",
                        "tokenizer_sha256",
                        "dataset_manifest_sha256",
                    )
                )
                if row["binding_complete"]:
                    row.update(
                        status="IDENTITY_BOUND_REVIEW_REQUIRED",
                        reason="Hashes bound; external provenance/evidence review still required",
                    )
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                StopIteration,
                pickle.UnpicklingError,
            ) as e:
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
        "unclassified_checkpoint_paths": [str(p) for p in files if not headers[p]],
        "checkpoint_discovery": "Bounded zip metadata inspection; no filename-based model inference; unknown/legacy archives remain unclassified",
        "root_availability": {str(p): p.exists() for p in roots},
        "source_files": {str(p.relative_to(ROOT)): digest(p) for p in sorted(NATIVE.glob("*.py"))},
        "missing_dataset_reason": "No bound evaluation corpus supplied; quality comparison remains blocked",
    }


def checkpoint_models(path):
    """Inspect Torch zip metadata without unpickling tensors or loading 30M models."""
    try:
        with zipfile.ZipFile(path) as archive:
            entries = [i for i in archive.infolist() if i.filename.endswith("/data.pkl")]
            if len(entries) != 1 or entries[0].file_size > 2_000_000:
                return set()
            strings = {
                arg
                for op, arg, _ in pickletools.genops(archive.read(entries[0]))
                if op.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE")
            }
        return strings & set(MODELS.values())
    except (OSError, ValueError, zipfile.BadZipFile):
        return set()


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
    metadata = {
        k: v
        for k, v in payload.items()
        if k != "model_state_dict" and isinstance(v, (str, int, float, bool, type(None)))
    }

    metadata["checkpoint_tensor_dtypes"] = sorted(
        {str(v.dtype) for v in payload["model_state_dict"].values()}
    )
    return model, metadata
