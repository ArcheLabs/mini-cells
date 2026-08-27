from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.clm_conditionality_002 import (  # noqa: E402
    Conditionality002Evidence,
    evaluate_conditionality_evidence,
    make_conditionality_002_decision,
)
from minicells.clm_release import build_release_model, save_release_bundle  # noqa: E402
from minicells.language_models import TextNCALM, count_parameters  # noqa: E402
from minicells.language_scaling import prepare_scaling_corpus  # noqa: E402

OUT = ROOT / "results" / "clm-0.1-release"
BUNDLE = OUT / "bundle" / "minicells-clm-0.1"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_006 = ROOT / "artifacts" / "experiments" / "006-consumer-language-scaling"
WORKER = ROOT / "scripts" / "run_clm_0_1_release_worker.py"
RELEASE_REPLICATE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and build MiniCells CLM-0.1.")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def complete(replicate: int) -> bool:
    path = OUT / f"r{replicate}-release-worker.json"
    return path.is_file() and json.loads(path.read_text()).get("complete") is True


def command(replicate: int, cache: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--replicate", str(replicate),
        "--cache-dir", str(cache),
        "--output-dir", str(OUT),
        "--checkpoint", str(SOURCE_006 / "minicells-v2-10m.pt"),
        "--model-config", str(SOURCE_006 / "model-configs.json"),
    ]


def run_workers(cache: Path) -> int:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("CUDA is required")
    used = min(2, gpu_count)
    queue = [r for r in range(3) if not complete(r)]
    while queue:
        active = []
        for gpu in range(min(used, len(queue))):
            replicate = queue.pop(0)
            log = OUT / f"r{replicate}.log"
            handle = log.open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command(replicate, cache), cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            active.append((replicate, process, handle, log))
        failures = []
        for replicate, process, handle, log in active:
            code = process.wait()
            handle.close()
            print(log.read_text())
            if code:
                failures.append(f"r{replicate} exited {code}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return used


def model_card(decision: dict[str, object], release_worker: dict[str, object], params: dict[str, int]) -> str:
    evidence = next(row for row in decision["evidence"] if int(row["replicate"]) == RELEASE_REPLICATE)
    return f"""# MiniCells CLM-0.1 Research Preview

MiniCells CLM-0.1 is a small recurrent cellular language model research release built from the
Experiment 006 TextNCA checkpoint by function-preserving MoE-style upcycling.

## Architecture

- Base: TextNCA, 3 recurrent NCA stages, 4 iterations per stage.
- Routing: strictly local top-1 cosine-prototype routing.
- Experts: 4 full-width inherited FFN experts per stage.
- Total expert capacity: 4x the original FFN expert capacity.
- Active expert capacity per local update: 1x the original FFN.
- Cell activation: fixed at 1.0 in this release.

## Validation

Conditionality Validation 002: `{decision['diagnosis']}`.

Release-candidate replicate: `{RELEASE_REPLICATE}`.

- PPL vs matched dense continuation: `{float(evidence['quality_ratio_to_dense_continued']):.6f}`.
- Aligned route disagreement: `{float(evidence['aligned_route_disagreement']):.6f}`.
- Normalized Dynamic advantage vs Static: `{float(evidence['static_advantage']):.6f}`.
- Normalized Dynamic advantage vs Shuffled: `{float(evidence['shuffled_advantage']):.6f}`.
- Usage entropy: `{float(evidence['usage_entropy']):.6f}`.

## Parameters

- Dense TextNCA parameters: `{params['dense_total']:,}`.
- CLM total parameters: `{params['clm_total']:,}`.
- Active routed expert parameters: `{params['active_expert']:,}`.
- Router parameters: `{params['router']:,}`.

## Scope and limitations

This is a research preview, not a general-purpose chat model. It was trained on TinyStories using
a 10M-token TextNCA base plus 1M continuation tokens. The release demonstrates function-preserving
capacity expansion and causally useful local conditional routing. It does **not** yet demonstrate
active FLOPs below the original dense TextNCA, autonomous capacity growth, online self-learning,
phenotype, multimodality, or 100M+ token scaling.

The sparse-dispatch implementation is a correctness/reference backend; wall-clock speedups are not
claimed. Benchmark telemetry is included in `benchmark.json`.
"""


def main() -> int:
    args = parse_args()
    if args.fresh and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, tokenizer_path, corpus_manifest = prepare_scaling_corpus(
        ROOT, source_005_dir=SOURCE_005
    )
    cache = ROOT / "results" / "consumer-language-scaling-v1" / "cache"
    used_gpus = run_workers(cache)
    workers = [json.loads((OUT / f"r{r}-release-worker.json").read_text()) for r in range(3)]

    evidence: list[Conditionality002Evidence] = []
    for worker in workers:
        evidence.append(evaluate_conditionality_evidence(
            replicate=int(worker["replicate"]),
            dense_ppl=float(worker["dense_ppl"]),
            dense_nll=float(worker["dense_nll"]),
            dynamic={
                "ppl": worker["dynamic"]["ppl"],
                "nll": worker["dynamic"]["nll"],
                "usage_entropy": worker["dynamic"]["usage_entropy"],
            },
            static=worker["static"],
            shuffled=worker["shuffled"],
            aligned_disagreement=float(worker["aligned_route_disagreement"]),
        ))
    conditionality = make_conditionality_002_decision(evidence)
    conditionality["provenance"] = {
        "base_checkpoint": "Experiment 006 minicells-v2-10m.pt",
        "upcycling_study_001": "CLM_UPCYCLING_QUALITY_SIGNAL",
        "release_candidate": "copy_geometry replicate 2",
        "training_tokens_after_base": 1_000_000,
    }
    (OUT / "conditionality-002-decision.json").write_text(
        json.dumps(conditionality, indent=2, sort_keys=True) + "\n"
    )
    if conditionality["status"] != "PASS":
        raise RuntimeError("CLM-0.1 release gate failed: Conditionality Validation 002 did not pass")

    release_worker = workers[RELEASE_REPLICATE]
    checkpoint = torch.load(
        OUT / f"r{RELEASE_REPLICATE}-geometry-release.pt",
        map_location="cpu", weights_only=False,
    )
    model = build_release_model()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.set_execution_backend("sparse_dispatch")

    dense_model = TextNCALM(
        vocab_size=2048, max_context=128, dim=128, heads=4, ffn_dim=512,
        windows=(8, 32, 128), iterations=(4, 4, 4), carry_bias=2.0,
        rms_norm=False, tie_embeddings=True, stage_supervision=False,
    )
    total_expert = sum(
        p.numel()
        for stage in model.stages
        for expert in stage.program_bank.experts
        for p in expert.parameters()
    )
    router_params = sum(
        p.numel() for stage in model.stages for p in stage.program_bank.router.parameters()
    )
    params = {
        "dense_total": count_parameters(dense_model),
        "clm_total": count_parameters(model),
        "active_expert": total_expert // model.config.num_experts,
        "router": router_params,
    }
    benchmark = {
        "format": "minicells.clm-0.1.benchmark.v1",
        "release_replicate": RELEASE_REPLICATE,
        "telemetry": release_worker["benchmark"],
        "parameters": params,
        "claim_boundary": "No wall-clock speedup or sub-dense active-FLOP claim is made for CLM-0.1.",
    }
    (OUT / "benchmark.json").write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")

    provenance = {
        "release": "clm-0.1",
        "source_checkpoint": "Experiment 006 minicells-v2-10m.pt",
        "source_training_tokens": 10_000_000,
        "continuation_training_tokens": 1_000_000,
        "upcycling_method": "copy_geometry",
        "replicate": RELEASE_REPLICATE,
        "reproduction_expected_ppl": release_worker["expected_geometry_ppl"],
        "reproduction_observed_ppl": release_worker["observed_geometry_ppl"],
        "conditionality_002": conditionality["diagnosis"],
    }
    metrics = {
        "validation_ppl": release_worker["dynamic"]["ppl"],
        "dense_continued_ppl": release_worker["dense_ppl"],
        "aligned_route_disagreement": release_worker["aligned_route_disagreement"],
        "usage_entropy": release_worker["dynamic"]["usage_entropy"],
    }
    save_release_bundle(model, tokenizer_path, BUNDLE, provenance=provenance, metrics=metrics)
    (BUNDLE / "MODEL_CARD.md").write_text(model_card(conditionality, release_worker, params))
    shutil.copy2(OUT / "benchmark.json", BUNDLE / "benchmark.json")
    shutil.copy2(OUT / "conditionality-002-decision.json", BUNDLE / "conditionality-002-decision.json")

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus_used": used_gpus,
        "corpus_manifest": corpus_manifest,
    }
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")

    release_decision = {
        "format": "minicells.clm-0.1.release.v1",
        "release": "MiniCells CLM-0.1 Research Preview",
        "status": "PASS",
        "diagnosis": "CLM_0_1_RELEASE_READY",
        "release_candidate": "copy_geometry replicate 2",
        "gates": {
            "reproduction": all(bool(row["reproduction_pass"]) for row in workers),
            "conditionality_002": conditionality["status"],
            "bundle_created": True,
        },
        "metrics": metrics,
        "parameters": params,
        "limitations": [
            "TinyStories-only research model",
            "active FFN compute is not below the original dense TextNCA",
            "no autonomous growth or online self-learning in CLM-0.1",
            "no phenotype, multimodality, or 100M+ token scaling claim",
        ],
    }
    (OUT / "decision.json").write_text(json.dumps(release_decision, indent=2, sort_keys=True) + "\n")

    for name in (
        "model.pt", "tokenizer.json", "config.json", "MODEL_CARD.md",
        "benchmark.json", "conditionality-002-decision.json",
    ):
        shutil.copy2(BUNDLE / name, OUT / name)

    print(json.dumps(release_decision, indent=2, sort_keys=True))
    print(f"Release bundle: {BUNDLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
