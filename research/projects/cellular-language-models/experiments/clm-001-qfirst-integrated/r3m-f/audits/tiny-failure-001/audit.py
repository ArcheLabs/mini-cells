"""Pinned-source CPU token audit. No Torch import, model loading, forward or optimizer."""

import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from tokenizers import Tokenizer

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
UNIT = "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/"
SOURCE = "bb43297a39a4ccca14dcaeb5863f21d42d9aad05"
RESULT = "ab5c734537776874ea5a57faa68a548f7c52b01b"
PREFIX = (
    UNIT
    + "r3m-f/reports/published/f62d85e3ef78da73f7919d178d7496280ed1f5a8b879adc5f1c90004909f9f22/"
)
DEST = Path(__file__).resolve().parent


def blob(ref, path):
    return subprocess.check_output(["git", "show", ref + ":" + path], cwd=ROOT)


def sha(x):
    return hashlib.sha256(x).hexdigest()


def canonical(x):
    return sha(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def write(name, value):
    (DEST / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def functions(path, namespace):
    text = blob(SOURCE, path).decode()
    tree = ast.parse(text)
    tree.body = [x for x in tree.body if isinstance(x, ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(tree), path, "exec"), namespace)
    return namespace


def main():
    config = json.loads(blob(SOURCE, UNIT + "r3m-f/config/exploratory.frozen.json"))
    manifest_bytes = blob(RESULT, PREFIX + "RUN_MANIFEST.json")
    manifest = json.loads(manifest_bytes)
    evidence = {}
    input_hashes = {}
    for name in (
        "TINY_OVERFIT_REPORT.json",
        "MODEL_LINEAGE.json",
        "COMPUTE_COST.json",
        "DATA_AUDIT.json",
    ):
        raw = blob(RESULT, PREFIX + name)
        meta = manifest["artifact_files"][name]
        assert len(raw) == meta["bytes"] and sha(raw) == meta["sha256"]
        evidence[name] = json.loads(raw)
        input_hashes[name] = meta
        if name != "DATA_AUDIT.json":
            (DEST / ("original-" + name)).write_bytes(raw)
    (DEST / "original-RUN_MANIFEST.json").write_bytes(manifest_bytes)
    audit = evidence["DATA_AUDIT.json"]["manifest"]
    assert canonical(config) == audit["config_sha256"]
    tok_path = UNIT + "r3m/registration/artifact-recovery-001/independent-review-002/tokenizer.json"
    tok_bytes = blob(SOURCE, tok_path)
    assert sha(tok_bytes) == config["model"]["tokenizer_sha256"]
    tokenizer = Tokenizer.from_str(tok_bytes.decode())
    generator = functions(
        UNIT + "r3m-f/implementation/minicells_r3mf/arithmetic_tasks.py",
        dict(Counter=Counter, canonical=canonical),
    )
    generated = generator["generate"](config, tokenizer)
    for role in ("train", "dev"):
        assert (
            canonical(generated["a1"][role]) == audit["task_audit"]["facts"]["a1"][role + "_hash"]
        )
    records = generated["tiny"]
    assert len(records) == 8
    boundary_path = (
        UNIT
        + "r3m/registration/sampling-trace-001/boundary-consistency-review-001/CHECKPOINT_BOUNDARY_EVIDENCE.json"
    )
    boundary = json.loads(blob(SOURCE, boundary_path))
    job = next(j for j in boundary["jobs"] if j["seed"] == 151001301 and j["arm"] == "B-H")
    initial = next(b for b in job["boundaries"] if b["step"] == 300)
    receipt = evidence["MODEL_LINEAGE.json"]["anchors"][0]["evidence"]["receipt"]
    assert initial["checkpoint_sha256"] == receipt["sha256"]
    assert initial["original_state_fingerprint"] == receipt["state_fingerprint"]
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(initial["data_rng"])
    reference_rng = np.random.default_rng()
    reference_rng.bit_generator.state = copy.deepcopy(initial["data_rng"])
    # Execute the EXACT frozen batch function using CPU arrays instead of Torch tensor allocation.
    # Only the final array materialization/device lookup is substituted; sampling and slicing are original.
    array_api = SimpleNamespace(
        long=np.int64, tensor=lambda values, device, dtype: np.asarray(values, dtype=dtype)
    )
    ns = functions(UNIT + "r3m-e/implementation/minicells_r3me/data.py", dict(torch=array_api))
    device_only = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    categories = Counter()
    overlays = Counter()
    facts = defaultdict(Counter)
    steps = []
    examples = []
    positions = Counter()
    trace = hashlib.sha256()
    tokens_hash = hashlib.sha256()
    for step in range(200):
        local = Counter()
        constructed = []
        reference = ns["arithmetic_batches"](reference_rng, records, config, device_only)
        for batch in range(8):
            ids = []
            labels = []
            instances = []
            while len(ids) < 65:
                index = int(rng.integers(len(records)))
                r = records[index]
                start = len(ids)
                trace.update(("%s:%s:%s\n" % (step, batch, index)).encode())
                seq = r["ids"] + [1]
                ids.extend(seq)
                labels.extend(
                    [
                        (
                            r["group"],
                            "answer" if i in r["answer_positions"] else "prompt",
                            len(instances),
                        )
                        for i in range(len(r["ids"]))
                    ]
                    + [(r["group"], "eos", len(instances))]
                )
                instances.append((start, len(ids), r))
            fixed = ids[:65]
            constructed.append(fixed)
            assert reference[batch][0].tolist() == [fixed[:-1]]
            assert reference[batch][1].tolist() == [fixed[1:]]
            for p in range(1, 65):
                group, category, instance = labels[p]
                categories[category] += 1
                local[category] += 1
                facts[group][category + "_target_tokens"] += 1
                if fixed[p - 1] == 1 and category == "prompt":
                    overlays["eos_to_next_prompt_edges"] += 1
                if instance > 0:
                    overlays["targets_with_prior_qa_context"] += 1
                    if category == "answer":
                        overlays["answer_targets_with_prior_qa_context"] += 1
                if category == "answer":
                    positions[p - 1] += 1
            for start, end, r in instances:
                facts[r["group"]]["sampled_qa"] += 1
                aps = [start + i for i in r["answer_positions"]]
                complete = all(1 <= i < 65 for i in aps)
                facts[r["group"]]["complete_answer_supervisions"] += int(complete)
                facts[r["group"]]["complete_answer_and_eos_supervisions"] += int(
                    complete and end - 1 < 65
                )
                facts[r["group"]]["complete_qa_retained"] += int(end <= 65)
                overlays[
                    "complete_answer_with_clean_prefix"
                    if start == 0
                    else "complete_answer_with_prior_qa"
                ] += int(complete)
                if end > 65:
                    overlays["truncated_qa"] += 1
                    overlays["discarded_tail_tokens"] += end - 65
                    overlays["truncated_qa_with_partial_answer"] += int(
                        any(i < 65 for i in aps) and not complete
                    )
                    overlays["truncated_qa_with_no_answer"] += int(all(i >= 65 for i in aps))
                    overlays["truncated_qa_after_complete_answer_before_eos"] += int(complete)
                    if len(examples) < 8:
                        examples.append(
                            dict(
                                step=step + 1,
                                batch=batch,
                                group=r["group"],
                                start=start,
                                end=end,
                                cut=65,
                                retained_text=tokenizer.decode(
                                    ids[start:65], skip_special_tokens=False
                                ),
                            )
                        )
            tokens_hash.update(np.asarray(fixed, dtype="<i8").tobytes())
        assert rng.bit_generator.state == reference_rng.bit_generator.state
        steps.append(dict(step=step + 1, target_tokens=512, **local))
    assert (
        sum(categories.values()) == 102400
        and sum(sum(s[k] for k in ("prompt", "answer", "eos")) for s in steps) == 102400
    )
    assert "torch" not in sys.modules, "Audit must not import Torch or a model"
    # Record static token boundaries against actual saved prediction IDs; no model scoring.
    final = evidence["TINY_OVERFIT_REPORT.json"]["final"]
    initial_metrics = evidence["TINY_OVERFIT_REPORT.json"]["initial"]
    predictions = []
    for row, p0, p1 in zip(records, initial_metrics["predictions"], final["predictions"]):
        assert row["group"] == p0["group"] == p1["group"]
        p = tokenizer.encode(row["prompt"])
        full = tokenizer.encode(row["text"])
        assert p.ids == row["prompt_ids"] and full.ids[: len(p.ids)] == p.ids
        assert row["answer_positions"][0] == len(p.ids)
        assert row["answer_tokens"] == full.ids[len(p.ids) :]
        assert all(
            row["text"][lo:hi].isdigit()
            for i, (lo, hi) in enumerate(full.offsets)
            if i in row["answer_positions"]
        )
        for saved in (p0, p1):
            assert (
                tokenizer.decode(saved["generated_ids"], skip_special_tokens=False).strip()
                == saved["prediction"]
            )
        predictions.append(
            dict(
                group=row["group"],
                prompt=row["prompt"],
                prompt_last_token_id=p.ids[-1],
                prompt_last_token=tokenizer.id_to_token(p.ids[-1]),
                prompt_length=len(p.ids),
                first_answer_token_position=len(p.ids),
                first_answer_logit_position=len(p.ids) - 1,
                answer_tokens=row["answer_tokens"],
                answer=row["answer"],
                initial_prediction=p0["prediction"],
                final_prediction=p1["prediction"],
                final_eos=p1["eos_reached"],
                final_first_token_correct=p1["first_token_correct"],
                text_only_match=p1["text_only_match"],
                full_em=p1["exact_match"],
            )
        )
    result = dict(
        schema="minicells.r3mf.readonly-failure-audit.v1",
        source_sha=SOURCE,
        result_commit=RESULT,
        identity=manifest["identity"],
        frozen_config_sha256=canonical(config),
        scope="CPU_BATCH_RECONSTRUCTION_ONLY_NO_MODEL_LOAD_NO_TORCH",
        optimizer_updates=0,
        gpu_used=False,
        input_evidence_hashes=input_hashes,
        tokenizer_sha256=sha(tok_bytes),
        boundary_evidence_sha256=sha(blob(SOURCE, boundary_path)),
        initial_rng_evidence=initial,
        parent_receipt=receipt,
        replay_numpy_version=np.__version__,
        production_numpy_version=manifest["runtime_environment"]["numpy"],
        final_replayed_rng=rng.bit_generator.state,
        actual_final_rng_independently_compared=False,
        replay_status="PINNED_SOURCE_AND_MATCHED_PARENT_START_RECONSTRUCTION_NOT_ENDPOINT_CERTIFIED",
        reference_sampler_match_all_1600_rows=True,
        trace_sha256=trace.hexdigest(),
        sampled_65_token_rows_sha256=tokens_hash.hexdigest(),
        total_input_tokens=102400,
        total_target_tokens=102400,
        rows=1600,
        primary_disjoint_categories={
            k: dict(tokens=v, fraction=v / 102400) for k, v in categories.items()
        },
        overlays={
            k: dict(
                count=v,
                fraction_of_target_tokens=v / 102400
                if k
                in (
                    "eos_to_next_prompt_edges",
                    "targets_with_prior_qa_context",
                    "answer_targets_with_prior_qa_context",
                )
                else None,
            )
            for k, v in overlays.items()
        },
        per_fact=dict(facts),
        answer_target_input_position_histogram=dict(positions),
        truncation_examples=examples,
        boundary_validation=predictions,
        initial_metrics={
            k: initial_metrics[k]
            for k in ("answer_nll", "full_nll", "first_token_accuracy", "exact_match")
        },
        final_metrics={
            k: final[k] for k in ("answer_nll", "full_nll", "first_token_accuracy", "exact_match")
        },
        original_trial_status="TINY_OVERFIT_FAILED",
        full_pilot_authorized=False,
        evidence_limits=[
            "No actual F terminal/intermediate raw data_rng state was opened or compared. Counts reconstruct from matched parent start and frozen code; they are not newly endpoint-attested runtime observations.",
            "Replay NumPy differs from production; sampler-reference comparison shares the local NumPy implementation and is not independent cross-version certification.",
            "No model weights, logits, activations or new counterfactual predictions were computed. Causal bottleneck attribution remains limited.",
        ],
    )
    write("TOKEN_DISTRIBUTION_PROOF.json", result)
    write("STEP_TOKEN_COUNTS.json", steps)
    write("RECONSTRUCTED_TINY_RECORDS.json", records)
    from summarize import summarize

    summarize(DEST)
    print(
        json.dumps(
            dict(
                categories=result["primary_disjoint_categories"],
                overlays=result["overlays"],
                per_fact=result["per_fact"],
                metrics=result["final_metrics"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
