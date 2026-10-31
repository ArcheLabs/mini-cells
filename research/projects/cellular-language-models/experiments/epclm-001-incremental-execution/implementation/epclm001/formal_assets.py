"""Pinned v0.4 inference assets and held-out validation inputs; no training."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .assets import load_native
from .common import ROOT, digest, model_hash, native, sha, write


def prepare(directory, config):
    from huggingface_hub import hf_hub_download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base, _, _ = native()
    records = {}
    corpus = None
    tokenizer_path = None
    for key, expected in config["assets"].items():
        prefix = f"epclm001/controlled-retrain-001/{config['asset_run_id']}/{key}/"
        files = {
            name: Path(
                hf_hub_download(
                    config["repo"],
                    prefix + name,
                    revision=expected["revision"],
                    local_dir=str(directory / key),
                )
            )
            for name in (
                "inference.pt",
                "IDENTITY.json",
                "QUALITY.json",
                "tokenizer.json",
                "corpus-manifest.json",
            )
        }
        if (
            digest(files["inference.pt"]) != expected["inference_sha256"]
            or digest(files["tokenizer.json"]) != config["tokenizer_sha256"]
        ):
            raise RuntimeError("PINNED_ASSET_OR_TOKENIZER_HASH_MISMATCH_" + key)
        identity = json.loads(files["IDENTITY.json"].read_text())
        quality = json.loads(files["QUALITY.json"].read_text())
        manifest = json.loads(files["corpus-manifest.json"].read_text())
        corpus_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if (
            digest(files["QUALITY.json"]) != identity.get("quality_sha256")
            or corpus_hash != identity["dataset_manifest_sha256"]
        ):
            raise RuntimeError("QUALITY_OR_CORPUS_BINDING_MISMATCH_" + key)
        model, metadata = load_native(files["inference.pt"], "E0B_" + key)
        if (
            metadata.get("asset_generation") != "CONTROLLED_RETRAIN_001"
            or metadata.get("seed") != 91002
            or metadata.get("step") != 10000
            or metadata.get("consumed_tokens") != 10000000
        ):
            raise RuntimeError("TRAINED_ASSET_CONTRACT_MISMATCH_" + key)
        if (
            metadata.get("training_source_sha") != identity["training_source_sha"]
            or identity.get("inference_sha256") != expected["inference_sha256"]
            or identity.get("tokenizer_sha256") != config["tokenizer_sha256"]
            or metadata.get("tokenizer_sha256") != config["tokenizer_sha256"]
            or model_hash(model) != identity.get("state_sha256")
            or metadata.get("corpus_manifest_sha256") != corpus_hash
            or sum(p.numel() for p in model.parameters()) != expected["parameters"]
        ):
            raise RuntimeError("ASSET_SOURCE_OR_PARAMETERS_MISMATCH_" + key)
        evaluator = "research/projects/cellular-language-models/experiments/native-clm-v0/implementation/native_clm_runtime.py"
        training_source = identity["training_source_sha"]
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", training_source],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        historical = subprocess.check_output(
            ["git", "show", training_source + ":" + evaluator], cwd=ROOT
        )
        if hashlib.sha256(historical).hexdigest() != digest(ROOT / evaluator):
            raise RuntimeError("QUALITY_EVALUATOR_SOURCE_CHANGED_" + key)
        if (
            abs(quality["first"]["validation_ppl"] - expected["declared_ppl"]) > 1e-6
            or quality["status"] != "QUALITY_VALID"
        ):
            raise RuntimeError("DECLARED_QUALITY_NOT_BOUND_" + key)
        if corpus is not None and corpus != manifest:
            raise RuntimeError("CROSS_MODEL_CORPUS_MISMATCH")
        corpus = manifest
        tokenizer_path = files["tokenizer.json"]
        records[key] = dict(
            **expected,
            files={n: str(p.relative_to(directory)) for n, p in files.items()},
            training_source_sha=training_source,
            tokenizer_sha256=config["tokenizer_sha256"],
            dataset_manifest_sha256=corpus_hash,
            quality=quality["first"],
            quality_reused=True,
            quality_evaluation_dtype="float16_autocast",
            quality_evaluator_sha256=digest(ROOT / evaluator),
            quality_file_sha256=digest(files["QUALITY.json"]),
            quality_scope="Hash-bound controlled development evaluation; not newly evaluated; benchmark dtype separately FP32",
        )
        del model
    validation = directory / "validation.u16"
    base.DATASET_REVISION = corpus["dataset_revision"]
    if not validation.exists():
        tokenizer = base.load_tokenizer(tokenizer_path)
        pending = directory / "validation.u16.pending"
        base._write_token_stream(
            tokenizer, split="validation", output_path=pending, target_tokens=1000000
        )
        pending.replace(validation)
    if (
        digest(validation) != config["validation_raw_sha256"]
        or validation.stat().st_size != 2000000
    ):
        raise RuntimeError("VALIDATION_INPUT_STREAM_HASH_MISMATCH")
    stream = np.memmap(validation, mode="r", dtype=np.uint16)
    offsets = sorted(set(config["correctness_offsets"] + config["holdout_offsets"]))
    inputs = {
        str(start): np.asarray(stream[start : start + 128], dtype=np.int64).tolist()
        for start in offsets
    }
    write(
        directory / "INPUTS.json",
        {
            "source_split": "validation_not_train",
            "validation_sha256": digest(validation),
            "slices": inputs,
            "inputs_sha256": sha(inputs),
        },
    )
    return {
        "status": "PINNED_ASSETS_VERIFIED",
        "models": records,
        "inputs": inputs,
        "inputs_sha256": sha(inputs),
        "validation_stream_sha256": digest(validation),
    }


def load_model(assets, key, device, directory):
    import torch

    row = assets["models"][key]
    path = Path(directory) / row["files"]["inference.pt"]
    if digest(path) != row["inference_sha256"]:
        raise RuntimeError("WEIGHTS_CHANGED_AFTER_PREFLIGHT")
    model, _ = load_native(path, "E0B_" + key)
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
