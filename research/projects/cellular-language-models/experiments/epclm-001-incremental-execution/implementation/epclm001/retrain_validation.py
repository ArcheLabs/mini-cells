"""Strict completed-state verification and inference export; no optimization."""

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from .common import BASE, digest, model_hash, native, tensor_hash, write


def verify(root, config, model_id, corpus, device="cuda:0"):
    import torch

    base, _, phase_c = native()
    name = config["models"][model_id]
    directory = Path(root) / "models" / name
    path = directory / "resume.pt"
    x = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "model": name,
        "seed": 91002,
        "profile": "baseline",
        "step": 10000,
        "consumed_tokens": 10000000,
        "asset_generation": "CONTROLLED_RETRAIN_001",
        "corpus_manifest_sha256": hashlib.sha256(
            json.dumps(corpus.manifest, sort_keys=True).encode()
        ).hexdigest(),
    }
    for k, v in expected.items():
        if x.get(k) != v:
            raise RuntimeError("COMPLETED_CHECKPOINT_IDENTITY_MISMATCH_" + k)
    from .retrain_accounting import completed_budget

    accounting = completed_budget(x)
    print(
        f"[EPCLM VERIFY] {model_id}: {accounting['training_steps']} loop steps = {accounting['successful_optimizer_updates']} successful + {accounting['skipped_optimizer_updates']} skipped; no compensation",
        flush=True,
    )
    for key in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "batch_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    ):
        if key not in x:
            raise RuntimeError("RESUME_STATE_MISSING_" + key)
    model = base.build_model(name) if model_id == "T1" else phase_c.build_phase_c(name)
    model.load_state_dict(x["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != config["parameter_counts"][model_id]:
        raise RuntimeError("PARAMETER_COUNT_MISMATCH")
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("NONFINITE_MODEL")
    rows = x["metrics"]
    if [int(r["consumed_tokens"]) for r in rows] != config["checkpoints"]:
        raise RuntimeError("TRAINING_CURVE_INCOMPLETE")
    if not (directory / "checkpoints.csv").exists():
        with (directory / "checkpoints.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if not (directory / "summary.json").exists():
        write(
            directory / "summary.json",
            {
                "format": "epclm001.restored-completed-resume-summary.v1",
                "model": name,
                "seed": 91002,
                "parameters": config["parameter_counts"][model_id],
                "final": rows[-1],
                "checkpoints": rows,
                "metadata_origin": "Completed controlled resume.pt; no new training",
            },
        )
    inference = {
        k: x[k]
        for k in (
            "model",
            "seed",
            "step",
            "consumed_tokens",
            "training_source_sha",
            "corpus_manifest_sha256",
            "asset_generation",
        )
    }
    inference.update(accounting)
    inference.update(
        format="epclm001.inference.v1",
        model_state_dict=x["model_state_dict"],
        parameter_count=config["parameter_counts"][model_id],
        tokenizer_sha256=config["tokenizer_sha256"],
    )
    base._atomic_save(inference, directory / "inference.pt")
    reloaded = base.build_model(name) if model_id == "T1" else phase_c.build_phase_c(name)
    reloaded.load_state_dict(
        torch.load(directory / "inference.pt", weights_only=True, map_location="cpu")[
            "model_state_dict"
        ]
    )
    reloaded.eval().requires_grad_(False)
    with torch.inference_mode():
        probe = torch.tensor([[2, 11, 17, 23, 3]], dtype=torch.long)
        if not torch.equal(model(probe), reloaded(probe)):
            raise RuntimeError("EXPORT_LOGIT_MISMATCH")
    from .incremental import IncrementalLM

    adapter = IncrementalLM(reloaded)
    cache = adapter.init_cache(1, "cpu", torch.float32, max_context=128)
    logits, _ = adapter.prefill(probe, cache)
    torch.testing.assert_close(logits, reloaded(probe)[:, -1], atol=2e-5, rtol=2e-4)
    shutil.copyfile(corpus.tokenizer_path, directory / "tokenizer.json")
    write(directory / "corpus-manifest.json", corpus.manifest)
    from .validation import parity

    fixed = json.loads((BASE / "validation_inputs.json").read_text())
    parity_result = parity(
        reloaded, torch.tensor(fixed["tokens"], dtype=torch.long), atol=2e-5, rtol=2e-4
    )
    write(directory / "REAL_WEIGHT_PARITY.json", parity_result)
    if not parity_result["passed"]:
        raise RuntimeError("REAL_WEIGHT_PARITY_FAILED")
    state_sha = model_hash(model)
    model.to(device)
    starts = base.fixed_validation_starts(
        len(base.open_memmap(corpus.validation_path)), seed=101002
    )
    validation = base.open_memmap(corpus.validation_path)
    first = base.evaluate(model, validation, starts, torch.device(device), torch.float16)
    second = base.evaluate(model, validation, starts, torch.device(device), torch.float16)
    if (
        not all(math.isfinite(first[k]) for k in ("validation_nll", "validation_ppl"))
        or abs(first["validation_nll"] - second["validation_nll"]) > 1e-5
    ):
        raise RuntimeError("QUALITY_INVALID_OR_NONREPRODUCIBLE")
    quality = {
        "status": "QUALITY_VALID",
        "training_accounting": accounting,
        "first": first,
        "second": second,
        "historical_ppl": config["historical_ppl"][model_id],
        "delta_ppl": first["validation_ppl"] - config["historical_ppl"][model_id],
        "interpretation": "Development asset validation, not new scientific confirmation; historical PPL is diagnostic only",
    }
    write(directory / "QUALITY.json", quality)
    identity = {
        "asset_generation": "CONTROLLED_RETRAIN_001",
        "model": name,
        "seed": 91002,
        "step": 10000,
        "training_source_sha": x["training_source_sha"],
        "resume_sha256": digest(path),
        "inference_sha256": digest(directory / "inference.pt"),
        "state_sha256": state_sha,
        "tokenizer_sha256": config["tokenizer_sha256"],
        "dataset_manifest_sha256": x["corpus_manifest_sha256"],
        "export_probe_sha256": tensor_hash(probe),
        "quality_sha256": digest(directory / "QUALITY.json"),
        "training_accounting": accounting,
    }
    write(directory / "IDENTITY.json", identity)
    return quality
