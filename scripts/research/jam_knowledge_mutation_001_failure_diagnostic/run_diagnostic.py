from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

import segmentation as seg
from validate_sources import validate_sources

from minicells.moe_multicoordinate import (
    apply_mutation_set_,
    capture_coordinate_set,
    load_mutation_set,
    restore_coordinate_set_,
)

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = (
    ROOT
    / "research"
    / "validations"
    / "jam-knowledge-mutation-001-failure-diagnostic"
    / "diagnostic_plan.json"
)
UPSTREAM_PROTOCOL_PATH = (
    ROOT / "research" / "validations" / "jam-knowledge-mutation-001" / "protocol.json"
)
UPSTREAM_ARTIFACTS = ROOT / "artifacts" / "experiments" / "jam-knowledge-mutation-001"
DATASET_ROOT = ROOT / "research" / "datasets" / "jam-knowledge-v0.1"
DATASET_BUILDER = ROOT / "scripts" / "research" / "jam_knowledge_v0_1" / "build_dataset.py"
DEFAULT_RESULT_ROOT = ROOT / "results" / "jam-knowledge-mutation-001-failure-diagnostic"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _progress(message: str) -> None:
    print(f"[jam001diag] {message}", flush=True)


def _require_dependencies():
    try:
        import huggingface_hub
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "JAM001 failure diagnostic requires transformers and huggingface_hub"
        ) from exc
    try:
        huggingface_hub.utils.disable_progress_bars()
    except Exception:
        pass
    try:
        transformers.logging.disable_progress_bar()
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    return huggingface_hub, transformers, snapshot_download, AutoModelForCausalLM, AutoTokenizer


def _build_misconception_rows(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subprocess.run(
        [sys.executable, str(DATASET_BUILDER)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    formal = _load_jsonl(DATASET_ROOT / "generated" / "evaluation" / "misconceptions.jsonl")
    expected_count = int(_load_json(UPSTREAM_PROTOCOL_PATH)["evaluation"]["heldout_families"]["misconceptions"])
    if len(formal) != expected_count:
        raise RuntimeError(f"misconception row count mismatch: {len(formal)} != {expected_count}")

    formal_prefix = str(plan["segmentation"]["formal_answer_prefix"])
    training_prefix = str(plan["segmentation"]["training_style_answer_prefix"])
    separator = str(plan["segmentation"]["separator"])
    training_style: list[dict[str, Any]] = []
    for row in formal:
        expected = formal_prefix + separator
        answer = str(row["answer"])
        if not answer.startswith(expected):
            raise RuntimeError(f"unexpected formal misconception answer for row {row['id']}")
        content = answer[len(expected) :]
        training_style.append(
            {
                **row,
                "id": f"{row['id']}.training-style",
                "answer": training_prefix + separator + content,
                "diagnostic_counterfactual": True,
            }
        )
    return formal, training_style


def _model(
    source_dir: Path,
    AutoModelForCausalLM,
    device: str,
):
    model = AutoModelForCausalLM.from_pretrained(
        source_dir,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model


def _reproduction(
    measured: dict[str, dict[str, float]],
    upstream: dict[str, Any],
    *,
    mutated: bool,
    tolerance: float,
) -> dict[str, Any]:
    key = "mutated_heldout" if mutated else "base_heldout"
    expected = float(upstream["evaluation"][key]["misconceptions"]["mean_reference_nll"])
    observed = float(measured["full"]["mean_reference_nll"])
    error = abs(observed - expected)
    return {
        "expected_mean_reference_nll": expected,
        "observed_mean_reference_nll": observed,
        "absolute_error": error,
        "within_tolerance": error <= tolerance,
    }


def _finalize_row_metrics(rows: list[dict[str, dict[str, float]]]) -> list[dict[str, dict[str, float]]]:
    return [seg.finalize_sums(row) for row in rows]


def _mean_field(cases: list[dict[str, Any]], field: str) -> float:
    return mean(float(case["decomposition"][field]) for case in cases)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_validation = validate_sources(require_git_identity=True)
    plan = _load_json(PLAN_PATH)
    upstream_protocol = _load_json(UPSTREAM_PROTOCOL_PATH)
    threshold = float(
        plan["upstream"]["formal_misconception_reference_nll_gain_threshold"]
    )
    tolerance = float(plan["metrics"]["reproduction_absolute_tolerance"])
    if plan["status"] != "POST_HOC_DIAGNOSTIC_FROZEN_GPU_PENDING":
        raise RuntimeError("unexpected diagnostic plan status")

    (
        huggingface_hub,
        transformers,
        snapshot_download,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = _require_dependencies()

    formal_rows, training_style_rows = _build_misconception_rows(plan)
    task = upstream_protocol["sequence_task"]
    training = upstream_protocol["training"]
    base = upstream_protocol["base"]
    formal_prefix = str(plan["segmentation"]["formal_answer_prefix"])
    training_prefix = str(plan["segmentation"]["training_style_answer_prefix"])
    separator = str(plan["segmentation"]["separator"])
    batch_size = int(training["batch_size"])

    _progress("loading frozen Granite base and exact JAM001 misconception rows")
    source_dir = Path(
        snapshot_download(repo_id=base["model_id"], revision=base["revision"])
    ).resolve()
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("pinned Granite tokenizer must be fast for exact offset diagnostic")

    model = _model(source_dir, AutoModelForCausalLM, args.device)
    parameter_map = dict(model.named_parameters())

    _progress("evaluating frozen-base formal and training-style formulations")
    base_formal, base_formal_rows_raw = seg.evaluate_segmented_rows(
        model,
        tokenizer,
        formal_rows,
        prompt_template=task["prompt_template"],
        max_length=int(task["max_sequence_tokens"]),
        device=args.device,
        batch_size=batch_size,
        prefix=formal_prefix,
        separator=separator,
    )
    base_training_style, _ = seg.evaluate_segmented_rows(
        model,
        tokenizer,
        training_style_rows,
        prompt_template=task["prompt_template"],
        max_length=int(task["max_sequence_tokens"]),
        device=args.device,
        batch_size=batch_size,
        prefix=training_prefix,
        separator=separator,
    )
    base_formal_rows = _finalize_row_metrics(base_formal_rows_raw)

    result_root = (args.result_dir or DEFAULT_RESULT_ROOT).resolve()
    shutil.rmtree(result_root, ignore_errors=True)
    result_root.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    per_row_output: list[dict[str, Any]] = []

    for seed in [int(value) for value in plan["upstream"]["formal_seeds"]]:
        for capacity in [int(value) for value in plan["upstream"]["capacities"]]:
            _progress(f"seed={seed} capacity={capacity} applying frozen mutation artifact")
            capacity_root = UPSTREAM_ARTIFACTS / f"seed-{seed}" / f"capacity-{capacity}"
            upstream_result = _load_json(capacity_root / "result.json")
            mutation_root = capacity_root / "mutation"
            mutation_manifest = load_mutation_set(mutation_root)
            targets = [row["target"] for row in mutation_manifest["coordinates"]]
            originals = capture_coordinate_set(parameter_map, targets)

            apply_mutation_set_(parameter_map, mutation_root)
            mutated_formal, mutated_formal_rows_raw = seg.evaluate_segmented_rows(
                model,
                tokenizer,
                formal_rows,
                prompt_template=task["prompt_template"],
                max_length=int(task["max_sequence_tokens"]),
                device=args.device,
                batch_size=batch_size,
                prefix=formal_prefix,
                separator=separator,
            )
            mutated_training_style, _ = seg.evaluate_segmented_rows(
                model,
                tokenizer,
                training_style_rows,
                prompt_template=task["prompt_template"],
                max_length=int(task["max_sequence_tokens"]),
                device=args.device,
                batch_size=batch_size,
                prefix=training_prefix,
                separator=separator,
            )
            restore_coordinate_set_(parameter_map, targets, originals)

            mutated_formal_rows = _finalize_row_metrics(mutated_formal_rows_raw)
            base_repro = _reproduction(
                base_formal,
                upstream_result,
                mutated=False,
                tolerance=tolerance,
            )
            mutated_repro = _reproduction(
                mutated_formal,
                upstream_result,
                mutated=True,
                tolerance=tolerance,
            )
            if not base_repro["within_tolerance"] or not mutated_repro["within_tolerance"]:
                raise RuntimeError(
                    f"original misconception metric reproduction failed seed={seed} "
                    f"capacity={capacity}: base={base_repro} mutated={mutated_repro}"
                )

            decomposition = seg.gain_decomposition(
                base_formal,
                mutated_formal,
                original_threshold=threshold,
            )
            if decomposition["decomposition_absolute_error"] > 1e-6:
                raise RuntimeError("formal answer gain decomposition is not exact enough")
            training_style_decomposition = seg.gain_decomposition(
                base_training_style,
                mutated_training_style,
                original_threshold=threshold,
            )

            case = {
                "seed": seed,
                "capacity": capacity,
                "mutation_identity_sha256": mutation_manifest["identity_sha256"],
                "upstream_formal_status": upstream_result["status"],
                "upstream_formal_misconception_reference_nll_gain": float(
                    upstream_result["metrics"]["misconception_reference_nll_gain"]
                ),
                "formal_answer": {
                    "base": base_formal,
                    "mutated": mutated_formal,
                },
                "decomposition": decomposition,
                "training_style_counterfactual": {
                    "base": base_training_style,
                    "mutated": mutated_training_style,
                    "decomposition": training_style_decomposition,
                    "full_gain_difference_vs_formal_answer": (
                        training_style_decomposition["full_reference_nll_gain"]
                        - decomposition["full_reference_nll_gain"]
                    ),
                },
                "reproduction": {
                    "base": base_repro,
                    "mutated": mutated_repro,
                },
            }
            cases.append(case)

            for index, row in enumerate(formal_rows):
                row_decomposition = seg.gain_decomposition(
                    base_formal_rows[index],
                    mutated_formal_rows[index],
                    original_threshold=threshold,
                )
                per_row_output.append(
                    {
                        "seed": seed,
                        "capacity": capacity,
                        "row_id": row["id"],
                        "concept_ids": row.get("concept_ids", []),
                        "decomposition": row_decomposition,
                    }
                )

            _progress(
                "seed={} capacity={} full_gain={:.4f} content+eos_gain={:.4f} "
                "prefix_gain={:.4f} training_style_full_gain={:.4f}".format(
                    seed,
                    capacity,
                    decomposition["full_reference_nll_gain"],
                    decomposition["content_plus_eos_reference_nll_gain"],
                    decomposition["prefix_reference_nll_gain"],
                    training_style_decomposition["full_reference_nll_gain"],
                )
            )

    capacity_summaries: dict[str, Any] = {}
    for capacity in [int(value) for value in plan["upstream"]["capacities"]]:
        subset = [case for case in cases if int(case["capacity"]) == capacity]
        capacity_summaries[str(capacity)] = {
            "seeds": [int(case["seed"]) for case in subset],
            "mean_full_reference_nll_gain": _mean_field(
                subset, "full_reference_nll_gain"
            ),
            "mean_prefix_reference_nll_gain": _mean_field(
                subset, "prefix_reference_nll_gain"
            ),
            "mean_canonical_content_reference_nll_gain": _mean_field(
                subset, "canonical_content_reference_nll_gain"
            ),
            "mean_content_plus_eos_reference_nll_gain": _mean_field(
                subset, "content_plus_eos_reference_nll_gain"
            ),
            "mean_prefix_dilution_vs_content_plus_eos": _mean_field(
                subset, "prefix_dilution_vs_content_plus_eos"
            ),
            "mean_training_style_full_reference_nll_gain": mean(
                float(
                    case["training_style_counterfactual"]["decomposition"][
                        "full_reference_nll_gain"
                    ]
                )
                for case in subset
            ),
        }

    focus_capacity = int(plan["interpretation"]["focus_capacity"])
    focus_cases = [case for case in cases if int(case["capacity"]) == focus_capacity]
    classification = seg.classify_capacity_four(
        focus_cases,
        original_threshold=threshold,
    )

    diagnostic = {
        "experiment": "JAM_KNOWLEDGE_MUTATION_001_FAILURE_DIAGNOSTIC",
        "status": "POST_HOC_DIAGNOSTIC_COMPLETE",
        "upstream_formal_decision_unchanged": plan["upstream"]["formal_decision"],
        "changes_upstream_formal_decision": False,
        "diagnostic_plan_sha256": _sha256(PLAN_PATH),
        "source_validation": source_validation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "device": args.device,
            "cuda_device_name": (
                torch.cuda.get_device_name(torch.device(args.device))
                if str(args.device).startswith("cuda") and torch.cuda.is_available()
                else None
            ),
            "dtype": "torch.float32",
        },
        "misconception_rows": len(formal_rows),
        "formal_threshold_reference": threshold,
        "base_formal_answer_metrics": base_formal,
        "base_training_style_counterfactual_metrics": base_training_style,
        "cases": cases,
        "capacity_summaries": capacity_summaries,
        "focus_capacity": focus_capacity,
        "interpretation": {
            "classification": classification,
            "classification_is_post_hoc_not_formal": True,
            "capacity4_full_gains": [
                float(case["decomposition"]["full_reference_nll_gain"])
                for case in focus_cases
            ],
            "capacity4_content_plus_eos_gains": [
                float(case["decomposition"]["content_plus_eos_reference_nll_gain"])
                for case in focus_cases
            ],
            "capacity4_prefix_gains": [
                float(case["decomposition"]["prefix_reference_nll_gain"])
                for case in focus_cases
            ],
            "capacity4_training_style_full_gains": [
                float(
                    case["training_style_counterfactual"]["decomposition"][
                        "full_reference_nll_gain"
                    ]
                )
                for case in focus_cases
            ],
        },
        "not_claimed": [
            "JAM Knowledge Mutation 001 formal support",
            "replacement of the frozen JAM001 formal decision",
            "a new post-hoc PASS/FAIL gate",
            "proof that four coordinates are sufficient for all JAM knowledge",
            "proof that answer formulation is irrelevant outside this diagnostic",
        ],
    }
    _write_json(result_root / "diagnostic.json", diagnostic)
    _write_jsonl(result_root / "per_row.jsonl", per_row_output)
    _progress(
        f"complete classification={classification}; upstream formal decision remains "
        f"{plan['upstream']['formal_decision']}"
    )
    return diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    diagnostic = run(parse_args())
    print(
        json.dumps(
            {
                "status": diagnostic["status"],
                "classification": diagnostic["interpretation"]["classification"],
                "upstream_formal_decision_unchanged": diagnostic[
                    "upstream_formal_decision_unchanged"
                ],
                "diagnostic_plan_sha256": diagnostic["diagnostic_plan_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
