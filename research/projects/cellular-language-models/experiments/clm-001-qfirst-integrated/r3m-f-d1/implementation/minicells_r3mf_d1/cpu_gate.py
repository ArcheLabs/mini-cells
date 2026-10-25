"""Small CPU-only sampler, identity, isolation, and budget proof generator."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from .data import build_update, source_budget, split_independent, target_masks
from .identity import canonical, sha256_bytes

BASE = Path(__file__).resolve().parents[2]
REPO = BASE.parents[5]
AUDIT = (
    REPO
    / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-f/audits/tiny-failure-001"
)
TOKENIZER = (
    REPO
    / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m/registration/artifact-recovery-001/independent-review-002/tokenizer.json"
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _generator(state):
    import numpy as np

    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


def _causal_pairs(lengths):
    return sum(n * (n + 1) // 2 for n in lengths)


def run(output=None):
    config = _read(BASE / "config/frozen.json")
    config_sha = sha256_bytes(canonical(config).encode("utf-8"))
    frozen_sha_path = BASE / "config/frozen.sha256"
    if frozen_sha_path.read_text(encoding="utf-8").strip() != config_sha:
        raise RuntimeError("D1_FROZEN_CONFIG_SHA_MISMATCH")
    input_rows_path = REPO / config["input_records"]["path"]
    distribution_path = REPO / config["sampling_start"]["evidence_path"]
    if _sha_file(input_rows_path) != config["input_records"]["sha256"]:
        raise RuntimeError("D1_TINY_RECORDS_SHA_MISMATCH")
    if _sha_file(distribution_path) != config["sampling_start"]["evidence_sha256"]:
        raise RuntimeError("D1_RNG_EVIDENCE_SHA_MISMATCH")
    if _sha_file(TOKENIZER) != config["invariants"]["tokenizer_sha256"]:
        raise RuntimeError("D1_TOKENIZER_SHA_MISMATCH")
    audit = _read(distribution_path)
    rng_state = config["sampling_start"]["data_rng"]
    if audit["initial_rng_evidence"]["data_rng"] != rng_state:
        raise RuntimeError("D1_START_RNG_CHANGED")
    if (
        audit["initial_rng_evidence"]["checkpoint_sha256"]
        != config["anchor_checkpoint"]["checkpoint_sha256"]
    ):
        raise RuntimeError("D1_ANCHOR_CHECKPOINT_CHANGED")
    records = _read(input_rows_path)
    if len(records) != 8 or tuple(row["group"] for row in records) != tuple(config["tiny_facts"]):
        raise RuntimeError("D1_FACT_SET_OR_ORDER_CHANGED")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    for record in records:
        prompt_ids = tokenizer.encode(record["prompt"]).ids
        full_ids = tokenizer.encode(record["text"]).ids
        if prompt_ids != record["prompt_ids"] or full_ids != record["ids"]:
            raise RuntimeError("D1_TOKENIZER_PROMPT_OR_ANSWER_ALIGNMENT_CHANGED")
        if prompt_ids[-1] != config["prompt_final_token_id"] or tuple(
            full_ids[len(prompt_ids) :]
        ) != tuple(record["answer_tokens"]):
            raise RuntimeError("D1_ANSWER_BOUNDARY_OR_FINAL_PROMPT_TOKEN_CHANGED")

    rng = _generator(rng_state)
    plan = []
    answer_targets = Counter()
    eos_targets = Counter()
    complete_answers = Counter()
    complete_answer_eos = Counter()
    totals = Counter()
    independent_lengths = []
    packed_lengths = []
    qa_fragments = 0
    for relative_step in range(config["steps"]):
        rows = build_update(
            rng,
            records,
            rows=config["source_rows_per_step"],
            sequence_tokens=config["source_sequence_tokens"],
            eos_id=config["eos_token_id"],
        )
        per_step = []
        step_budget = Counter()
        for row in rows:
            masks = target_masks(row)
            for kind, fact in zip(masks["label_kinds"], masks["label_facts"]):
                if kind == "answer":
                    answer_targets[fact] += 1
                elif kind == "eos":
                    eos_targets[fact] += 1
            for span in row.qa_spans:
                if span["answer_complete"]:
                    complete_answers[span["fact"]] += 1
                if span["eos_complete"]:
                    complete_answer_eos[span["fact"]] += 1
            fragments = split_independent(row)
            qa_fragments += len(fragments)
            packed_lengths.append(len(row.inputs))
            independent_lengths.extend(len(fragment.input_tokens) for fragment in fragments)
            row_budget = source_budget((row,))
            for key, value in row_budget.items():
                if isinstance(value, int):
                    totals[key] += value
                    step_budget[key] += value
            per_step.append(
                {
                    "source_microbatch": row.microbatch,
                    "sampled_record_indices": list(row.sampled_indices),
                    "source_token_sha256": sha256_bytes(canonical(row.tokens).encode("utf-8")),
                    "source_budget": row_budget,
                    "qa_fragments": [
                        {
                            "source_start": fragment.source_start,
                            "length": len(fragment.tokens),
                            "input_positions": len(fragment.input_tokens),
                            "fact": fragment.fact,
                        }
                        for fragment in fragments
                    ],
                }
            )
        plan.append(
            {
                "relative_step": relative_step + 1,
                "rows": per_step,
                "step_budget": dict(step_budget),
                "arm_denominators": {
                    "B_answer_eos": step_budget["answer_target_positions"]
                    + step_budget["eos_target_positions"],
                    "C_uniform": step_budget["independent_target_positions"],
                    "D_answer_eos": step_budget["answer_target_positions"]
                    + step_budget["eos_target_positions"],
                },
                "rng_state_after": rng.bit_generator.state,
            }
        )

    expected = {
        "packed_target_positions": 102400,
        "independent_target_positions": 97600,
        "dropped_cross_qa_targets": 4800,
        "answer_target_positions": 6661,
        "eos_target_positions": 4805,
        "complete_answer_instances": 4849,
        "complete_answer_plus_eos_instances": 4805,
    }
    for key, value in expected.items():
        if totals[key] != value:
            raise RuntimeError("D1_EXPECTED_BUDGET_MISMATCH_%s_%s_%s" % (key, totals[key], value))
    replay_end_matches_prior = rng.bit_generator.state == audit["final_replayed_rng"]
    if not replay_end_matches_prior:
        raise RuntimeError("D1_PRIOR_CPU_REPLAY_END_STATE_MISMATCH")
    for fact in config["tiny_facts"]:
        if answer_targets[fact] == 0:
            raise RuntimeError("D1_FACT_WITHOUT_ANSWER_TARGET_%s" % fact)

    mac_per_ffn_call = config["budget_accounting"]["cell_ffn_mac_per_hard_call"]
    mac_per_head_position = config["budget_accounting"]["tied_lm_head_mac_per_position"]
    packed_positions = totals["packed_target_positions"]
    independent_positions = totals["independent_target_positions"]
    common = {
        "logical_updates": config["steps"],
        "source_rows": config["steps"] * config["source_rows_per_step"],
        "source_slots": totals["source_slots"],
        "answer_target_positions": totals["answer_target_positions"],
        "eos_target_positions": totals["eos_target_positions"],
        "answer_eos_target_positions": totals["answer_target_positions"]
        + totals["eos_target_positions"],
        "complete_answer_instances": totals["complete_answer_instances"],
        "complete_answer_plus_eos_instances": totals["complete_answer_plus_eos_instances"],
        "answer_exposures_by_fact": dict(sorted(answer_targets.items())),
        "complete_answer_exposures_by_fact": dict(sorted(complete_answers.items())),
        "complete_answer_plus_eos_by_fact": dict(sorted(complete_answer_eos.items())),
        "sampled_qa_instances": sum(
            len(row["sampled_record_indices"]) for step in plan for row in step["rows"]
        ),
        "truncated_qa_instances": totals["truncated_qa_instances"],
        "plan_sha256": sha256_bytes(canonical(plan).encode("utf-8")),
        "sampling_version_caveat": config["sampling_start"]["cross_version_note"],
    }
    budgets = {
        "A": {
            "logical_updates": config["steps"],
            "source_rows": config["steps"] * config["source_rows_per_step"],
            "published_target_positions": 102400,
            "answer_target_positions": None,
            "eos_target_positions": None,
            "reconstructed_answer_target_positions": totals["answer_target_positions"],
            "reconstructed_eos_target_positions": totals["eos_target_positions"],
            "reconstructed_answer_exposures_by_fact": dict(sorted(answer_targets.items())),
            "exposure_provenance": "Start-state CPU replay only; original run endpoint RNG was not independently attested, so per-fact counts are not asserted as certified production exposure.",
            "sample_plan_reference_sha256": common["plan_sha256"],
            "run_status": "REUSED_PUBLISHED_RESULT_NO_TRAINING",
            "model_input_positions": 102400,
            "cell_hard_calls": 102400 * 8,
            "cell_ffn_mac": 102400 * 8 * mac_per_ffn_call,
            "tied_lm_head_mac": 102400 * mac_per_head_position,
            "causal_input_memory_pairs": _causal_pairs([64] * 1600),
            "source": "Published TINY_OVERFIT_REPORT.json; historical execution was not repeated",
        },
        "B": {
            **common,
            "model_input_positions": packed_positions,
            "selected_loss_targets": totals["answer_target_positions"]
            + totals["eos_target_positions"],
            "gradient_denominator": "SUM_SELECTED_ANSWER_AND_EOS_TARGET_COUNT_PER_UPDATE",
            "cell_hard_calls": packed_positions * 8,
            "cell_ffn_mac": packed_positions * 8 * mac_per_ffn_call,
            "tied_lm_head_mac": packed_positions * mac_per_head_position,
            "causal_input_memory_pairs": _causal_pairs([64] * (config["steps"] * 8)),
            "run_status": "NOT_RUN_REVIEW_REQUIRED",
        },
        "C": {
            **common,
            "model_input_positions": independent_positions,
            "selected_loss_targets": totals["independent_target_positions"],
            "gradient_denominator": "SUM_INDEPENDENT_TARGET_COUNT_PER_UPDATE",
            "cell_hard_calls": independent_positions * 8,
            "cell_ffn_mac": independent_positions * 8 * mac_per_ffn_call,
            "tied_lm_head_mac": independent_positions * mac_per_head_position,
            "causal_input_memory_pairs": _causal_pairs(independent_lengths),
            "run_status": "NOT_RUN_REVIEW_REQUIRED",
        },
        "D": {
            **common,
            "model_input_positions": independent_positions,
            "selected_loss_targets": totals["answer_target_positions"]
            + totals["eos_target_positions"],
            "gradient_denominator": "SUM_SELECTED_ANSWER_AND_EOS_TARGET_COUNT_PER_UPDATE",
            "cell_hard_calls": independent_positions * 8,
            "cell_ffn_mac": independent_positions * 8 * mac_per_ffn_call,
            "tied_lm_head_mac": independent_positions * mac_per_head_position,
            "causal_input_memory_pairs": _causal_pairs(independent_lengths),
            "run_status": "NOT_RUN_REVIEW_REQUIRED",
        },
    }
    for arm in ("B", "C", "D"):
        budgets[arm]["estimated_forward_backward_flops"] = None
        budgets[arm]["flops_note"] = (
            "Only explicit FFN/LM-head MAC counts and causal memory pair counts are provided; optimizer/backward/normalization/router/runtime FLOPs are not measured."
        )

    source_commit = _source_commit()
    data_sha = _sha_file(input_rows_path)
    original_report_path = REPO / config["inputs"]["original-TINY_OVERFIT_REPORT.json"]["path"]
    if (
        _sha_file(original_report_path)
        != config["inputs"]["original-TINY_OVERFIT_REPORT.json"]["sha256"]
    ):
        raise RuntimeError("D1_ORIGINAL_TINY_REPORT_SHA_MISMATCH")
    original_report = _read(original_report_path)
    if (
        original_report["status"] != "TINY_OVERFIT_FAILED"
        or original_report["optimizer_updates"] != 200
        or len(original_report["initial"]["predictions"]) != 8
        or len(original_report["final"]["predictions"]) != 8
    ):
        raise RuntimeError("D1_ORIGINAL_TINY_RESULT_CHANGED")
    a_predictions = {
        "schema": "minicells.r3mf_d1.reused_arm_a_predictions.v1",
        "source_result_commit": config["original_result_commit"],
        "original_status": original_report["status"],
        "run_id": original_report["identity"]["run_id"],
        "optimizer_updates": original_report["optimizer_updates"],
        "initial": original_report["initial"]["predictions"],
        "final": original_report["final"]["predictions"],
        "reused_without_model_load_or_retraining": True,
    }
    code_sha = _code_sha()
    run_id = sha256_bytes(
        (config_sha + data_sha + common["plan_sha256"] + source_commit + code_sha).encode("utf-8")
    )[:32]
    manifest = {
        "schema": "minicells.r3mf_d1.run_manifest.v1",
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "source_sha": source_commit,
        "source_code_sha256": code_sha,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "sample_plan_sha256": common["plan_sha256"],
        "seed": config["seed"],
        "anchor_step": config["anchor_step"],
        "run_status": "CPU_GATE_PASS_GPU_NOT_STARTED",
        "trial_status": "D1_ENGINEERING_ONLY_INDEPENDENT_REVIEW_REQUIRED",
        "exit_code": 0,
        "arm_status": {
            "A": "REUSED_PUBLISHED_TINY_OVERFIT_FAILED",
            "B": "NOT_RUN",
            "C": "NOT_RUN",
            "D": "NOT_RUN",
        },
        "budgets": budgets,
        "original_tiny_overfit_status": "TINY_OVERFIT_FAILED_UNMODIFIED",
        "optimizer_updates_this_run": 0,
        "gpu_used": False,
        "full_r3mf_pilot_authorized": False,
        "main_or_confirmation_accessed": False,
        "publication_status": "LOCAL_ENGINEERING_EVIDENCE_NOT_PUSHED",
        "remote_commit_sha": None,
        "evaluation_times": config["evaluation_steps"],
        "artifacts": {},
    }
    plan_payload = {
        "schema": config["sampling_plan"]["schema"],
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "anchor_checkpoint_sha256": config["anchor_checkpoint"]["checkpoint_sha256"],
        "rng_start": rng_state,
        "numpy_version": __import__("numpy").__version__,
        "relative_updates": plan,
    }
    proof = {
        "schema": "minicells.r3mf_d1.cpu_gate.v1",
        "status": "CPU_GATE_PASS",
        "torch_imported": "torch" in sys.modules,
        "model_loaded": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "tokenizer_sha256": _sha_file(TOKENIZER),
        "rng_start_checkpoint_sha256": config["sampling_start"]["checkpoint_sha256"],
        "rng_replay_numpy_version": __import__("numpy").__version__,
        "rng_producer_numpy_version": audit["production_numpy_version"],
        "rng_endpoint_certified_against_original_trial": audit[
            "actual_final_rng_independently_compared"
        ],
        "budget_totals": dict(totals),
        "arm_budgets": budgets,
        "not_a_pure_equal_budget_factorial": True,
        "cpu_tests": "tests/research/r3mf_d1/test_cpu_gate.py; result recorded separately in CPU_GATE_TESTS.json",
        "replay_end_matches_prior_cpu_reconstruction": replay_end_matches_prior,
        "limitations": [
            "Original Tiny result A is reused without retraining; only B/C/D are prospective.",
            "The full original trial endpoint RNG was not independently attested; this CPU gate freezes a shared prospective sample plan from the verified Step300 state.",
            "Independent QA context removes EOS-to-next-prompt targets, changes position indices, and uses fewer model input positions and causal memory pairs.",
            "FLOPs are analytic MAC counters only, not measured device FLOPs or wall time.",
        ],
    }
    if "torch" in sys.modules:
        raise RuntimeError("D1_CPU_GATE_MUST_NOT_IMPORT_TORCH")
    outputs = {
        "SAMPLE_PLAN.json": plan_payload,
        "BUDGET_PROOF.json": proof,
        "A_REUSED_PREDICTIONS.json": a_predictions,
        "RUN_MANIFEST.json": manifest,
    }
    if output:
        target = Path(output) / run_id
        target.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            if name == "RUN_MANIFEST.json":
                continue
            path = target / name
            serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
            if path.exists():
                if path.read_text(encoding="utf-8") != serialized:
                    raise RuntimeError("D1_EXISTING_PREFLIGHT_ARTIFACT_MISMATCH_" + name)
            else:
                path.write_text(serialized, encoding="utf-8")
            manifest["artifacts"][name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            }
        outputs["RUN_MANIFEST.json"] = manifest
        manifest_path = target / "RUN_MANIFEST.json"
        serialized_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if manifest_path.exists():
            if manifest_path.read_text(encoding="utf-8") != serialized_manifest:
                raise RuntimeError("D1_EXISTING_PREFLIGHT_MANIFEST_MISMATCH")
        else:
            manifest_path.write_text(serialized_manifest, encoding="utf-8")
    return {"proof": proof, "manifest": manifest, "sample_plan": plan_payload}


def _source_commit():
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "UNKNOWN_SOURCE_COMMIT"


def _code_sha():
    directory = BASE / "implementation/minicells_r3mf_d1"
    files = sorted(directory.rglob("*.py")) + [BASE / "run.py"]
    material = []
    for path in files:
        material.append((path.relative_to(REPO).as_posix(), _sha_file(path)))
    return sha256_bytes(canonical(material).encode("utf-8"))
