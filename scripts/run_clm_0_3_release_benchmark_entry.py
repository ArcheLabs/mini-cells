#!/usr/bin/env python3
"""Formal entrypoint for the CLM-0.3 public release benchmark.

This wrapper adds two release-only integrity checks without modifying the
historical Experiment-006 corpus helper:

1. the first 15M tokens of the newly materialized 18M stream must exactly
   reproduce the Experiment-006 training stream SHA-256;
2. the 200K validation stream must exactly reproduce Experiment 006.

It also routes GPU workers through the inference-cleanup entrypoint.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE = HERE / "run_clm_0_3_release_benchmark.py"
WORKER_ENTRY = Path("scripts/run_clm_0_3_release_bridge_worker_entry.py")
SOURCE_006_CORPUS = ROOT / "artifacts/experiments/006-consumer-language-scaling/corpus-manifest.json"

spec = importlib.util.spec_from_file_location("clm_0_3_release_benchmark_core", CORE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load release benchmark runner: {CORE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.WORKER = WORKER_ENTRY

_original_prepare = module.prepare_scaling_corpus
_equivalence: dict[str, object] | None = None


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validated_prepare(*args, **kwargs):
    global _equivalence
    train, validation, tokenizer_path, manifest = _original_prepare(*args, **kwargs)
    source = json.loads(SOURCE_006_CORPUS.read_text(encoding="utf-8"))
    source_train_tokens = int(source["train_stream_tokens"])
    source_validation_tokens = int(source["validation_stream_tokens"])
    if source_train_tokens != 15_000_000 or source_validation_tokens != 200_000:
        raise RuntimeError("Experiment-006 corpus dimensions changed unexpectedly")
    if int(train.numel()) < source_train_tokens:
        raise RuntimeError("release corpus is shorter than the Experiment-006 training prefix")
    if int(validation.numel()) != source_validation_tokens:
        raise RuntimeError("release validation stream does not match Experiment-006 length")

    observed_train = _tensor_sha256(train[:source_train_tokens])
    observed_validation = _tensor_sha256(validation)
    expected_train = str(source["train_token_sha256"])
    expected_validation = str(source["validation_token_sha256"])
    tokenizer_sha = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    expected_tokenizer = str(source["tokenizer_sha256"])
    if observed_train != expected_train:
        raise RuntimeError(
            "newly materialized release corpus does not reproduce the exact Experiment-006 "
            f"15M training prefix: {observed_train} != {expected_train}"
        )
    if observed_validation != expected_validation:
        raise RuntimeError(
            "release validation stream does not reproduce Experiment 006: "
            f"{observed_validation} != {expected_validation}"
        )
    if tokenizer_sha != expected_tokenizer:
        raise RuntimeError(
            "release tokenizer does not reproduce Experiment 006: "
            f"{tokenizer_sha} != {expected_tokenizer}"
        )
    _equivalence = {
        "status": "CLM_RELEASE_SOURCE_006_CORPUS_EQUIVALENCE",
        "source_train_prefix_tokens": source_train_tokens,
        "source_train_prefix_sha256": observed_train,
        "source_validation_tokens": source_validation_tokens,
        "source_validation_sha256": observed_validation,
        "tokenizer_sha256": tokenizer_sha,
        "release_materialized_train_tokens": int(train.numel()),
        "unseen_suffix_tokens_available": int(train.numel()) - source_train_tokens,
    }
    print(json.dumps(_equivalence, indent=2, sort_keys=True), flush=True)
    return train, validation, tokenizer_path, manifest


module.prepare_scaling_corpus = _validated_prepare


def _output_root_from_argv() -> Path:
    default = ROOT / "results/clm-0.3-release-benchmark"
    try:
        index = sys.argv.index("--output-root")
    except ValueError:
        return default.resolve()
    if index + 1 >= len(sys.argv):
        return default.resolve()
    path = Path(sys.argv[index + 1])
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def main() -> int:
    code = module.main()
    if code != 0 or _equivalence is None or "--execute" not in sys.argv:
        return code
    output_root = _output_root_from_argv()
    output_root.mkdir(parents=True, exist_ok=True)
    proof_path = output_root / "source-corpus-equivalence.json"
    proof_path.write_text(
        json.dumps(_equivalence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("bridge-summary.json", "decision.json"):
        path = output_root / name
        if not path.is_file():
            raise FileNotFoundError(f"formal release output missing after aggregation: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_corpus_equivalence"] = _equivalence
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
