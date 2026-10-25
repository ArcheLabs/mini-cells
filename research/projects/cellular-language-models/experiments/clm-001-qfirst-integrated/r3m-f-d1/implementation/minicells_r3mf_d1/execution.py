"""Bounded, resumable B/C/D diagnostic execution; called only after explicit opt-in."""

import copy
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

from .identity import canonical, sha256_bytes
from .sample_plan import replay_planned_update, verify_plan_identity, verify_start_rng
from .training import advance, torch_batches


BASE = Path(__file__).resolve().parents[2]
REPO = BASE.parents[5]
RECORDS = REPO / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-f/audits/tiny-failure-001/RECONSTRUCTED_TINY_RECORDS.json"
TOKENIZER = REPO / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m/registration/artifact-recovery-001/independent-review-002/tokenizer.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(torch, path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _capture_rng(torch, np):
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
    }


def _restore_rng(torch, np, state):
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])


def _restore_d1_trainer(runtime, state, seed, device):
    from minicells_r3h.trainer import Trainer
    from minicells_r3e.common import fingerprint

    Model = __import__("minicells_r3mf_d1.routing", fromlist=["make_model_class"]).make_model_class()
    model = Model(runtime, seed, "B-H").to(device)
    model.load_state_dict(state["model"], strict=True)
    model.temperature = state["r3h"]["temperature"]
    trainer = Trainer(model, runtime, "B-H")
    trainer.parent_identity = copy.deepcopy(state["r3h"]["parent_identity"])
    trainer.restore(copy.deepcopy(state))
    if fingerprint(trainer.cell_opt.state_dict()) != fingerprint(state["cell_optimizer"]):
        raise RuntimeError("D1_CELL_OPTIMIZER_RESTORE_MISMATCH")
    if any(
        parameter.requires_grad != name.startswith("cells.")
        for name, parameter in trainer.model.named_parameters()
    ):
        raise RuntimeError("D1_PARAMETER_FREEZE_MASK_CHANGED")
    optimized = {
        id(parameter)
        for group in trainer.cell_opt.param_groups
        for parameter in group["params"]
    }
    if optimized != {id(parameter) for parameter in trainer.model.cells.parameters()}:
        raise RuntimeError("D1_CELL_OPTIMIZER_PARAMETER_SET_CHANGED")
    trainer.verify_frozen()
    if trainer.step < 300 or trainer.data_cursor != trainer.step * runtime["training"]["tokens_per_step"]:
        raise RuntimeError("D1_PARENT_CURSOR_INVALID")
    return trainer


def _cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return type(value)((key, _cpu_tree(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _checkpoint_metadata(state, arm_identity, relative_step, anchor_step):
    from minicells_r3e.common import fingerprint

    return {
        "schema": "minicells.r3mf_d1.checkpoint.v2",
        "identity": dict(arm_identity, max_steps=200),
        "relative_step": relative_step,
        "step": anchor_step + relative_step,
        "trainer_state_sha256": fingerprint(state),
        "optimizer_state_sha256": fingerprint(state["cell_optimizer"]),
        "data_rng_state": copy.deepcopy(state["data_rng"]),
        "route_rng_state_sha256": fingerprint(state["route_rng"]),
        "frozen_parameter_sha256": state["frozen_sha256"],
    }


def _validate_checkpoint(checkpoint, expected_identity, plan, records, anchor_step):
    from .identity import validate_resume

    if checkpoint.get("schema") != "minicells.r3mf_d1.checkpoint.v2":
        raise ValueError("D1_CHECKPOINT_SCHEMA_INVALID")
    validate_resume(checkpoint["metadata"], dict(expected_identity, max_steps=200))
    relative = checkpoint["metadata"]["relative_step"]
    state = checkpoint["trainer_state"]
    if checkpoint["metadata"] != _checkpoint_metadata(
        state, expected_identity, relative, anchor_step
    ):
        raise ValueError("D1_CHECKPOINT_STATE_HASH_MISMATCH")
    if relative:
        expected_rng = plan["relative_updates"][relative - 1]["rng_state_after"]
    else:
        expected_rng = plan["rng_start"]
    if canonical(state["data_rng"]) != canonical(expected_rng):
        raise ValueError("D1_CHECKPOINT_DATA_RNG_MISMATCH")
    if state["step"] != anchor_step + relative:
        raise ValueError("D1_CHECKPOINT_GLOBAL_STEP_MISMATCH")
    if state["data_cursor"] != checkpoint["anchor_data_cursor"] + relative * 512:
        raise ValueError("D1_CHECKPOINT_DATA_CURSOR_MISMATCH")
    return True


def _evaluate(trainer, records, tokenizer, device, step):
    import torch
    import torch.nn.functional as F
    from .evaluation import generate_tokens, summarize_predictions

    model = trainer.model
    rng = _capture_rng(torch, __import__("numpy"))
    route_rng_state = trainer.route_rng.get_state()
    work_before = copy.deepcopy(model.work)
    active_step = getattr(model, "active_step", None)
    training_context = getattr(model, "training_context", None)
    model.eval()
    model.active_step = None
    model.training_context = None
    model.observer = None
    rows = []
    try:
        with torch.no_grad():
            for record in records:
                prompt_ids = tuple(record["prompt_ids"])

                def next_token(prefix):
                    tokens = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
                    output = model(tokens, epsilon=0, route_rng=trainer.route_rng)
                    return int(output.logits[0, -1].argmax(-1))

                generated, eos_generated = generate_tokens(
                    next_token, prompt_ids, max_tokens=8, eos_id=1
                )
                full_ids = tuple(record["ids"]) + (1,)
                x = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
                logits = model(x, epsilon=0, route_rng=trainer.route_rng).logits[0].float()
                nll = F.cross_entropy(logits, y, reduction="none")
                prompt_length = len(prompt_ids)
                answer_values = nll[prompt_length - 1 : len(record["ids"]) - 1]
                eos_value = nll[len(record["ids"]) - 1]
                rows.append(
                    {
                        "fact": record["group"],
                        "expected_answer": record["answer"],
                        "expected_answer_tokens": list(record["answer_tokens"]),
                        "prompt_ids": list(prompt_ids),
                        "generated_answer_tokens": list(generated),
                        "generated_text": tokenizer.decode(list(generated)),
                        "eos_generated": eos_generated,
                        "answer_nll": float(answer_values.mean()),
                        "eos_nll": float(eos_value),
                    }
                )
    finally:
        model.active_step = active_step
        model.training_context = training_context
        model.train()
        model.work = work_before
        _restore_rng(torch, __import__("numpy"), rng)
        trainer.route_rng.set_state(route_rng_state)
    summary = summarize_predictions(rows)
    detailed_predictions = []
    for summarized, raw in zip(summary["predictions"], rows):
        detailed_predictions.append(
            dict(
                summarized,
                expected_answer=raw["expected_answer"],
                generated_text=raw["generated_text"],
                prompt_ids=raw["prompt_ids"],
            )
        )
    return {
        "relative_step": step,
        "global_step": trainer.step,
        "predictions": detailed_predictions,
        "first_token_accuracy": summary["first_token_accuracy"],
        "text_only_exact_match": summary["text_only_exact_match"],
        "strict_exact_match": summary["strict_exact_match"],
        "eos_termination_rate": summary["eos_termination_rate"],
        "answer_nll": summary["answer_nll"],
        "eos_nll": summary["eos_nll"],
        "exact_match_requires_eos": True,
    }


def _artifact(manifest, root, path):
    path = Path(path)
    relative = path.relative_to(root).as_posix()
    manifest["artifacts"][relative] = {"bytes": path.stat().st_size, "sha256": _file_sha(path)}


def _write_manifest(path, manifest):
    _atomic_json(path, manifest)


def observe_gpu_usage(manifest, model=None, torch_module=None):
    """Observe existing CUDA tensors/allocator records without allocating or syncing GPU work."""
    if torch_module is None:
        import torch as torch_module

    evidence = list(manifest.get("gpu_usage_evidence", []))
    if model is not None:
        devices = {str(parameter.device) for parameter in model.parameters() if parameter.device.type == "cuda"}
        for device in sorted(devices):
            row = {"kind": "MODEL_PARAMETERS_ON_CUDA", "device": device}
            if row not in evidence:
                evidence.append(row)
    observation_failed = False
    try:
        # Querying allocator records only after initialization also handles failed model loads.
        if torch_module.cuda.is_initialized():
            for device in range(torch_module.cuda.device_count()):
                peak = int(torch_module.cuda.max_memory_allocated(device))
                if peak > 0:
                    row = {"kind": "CUDA_ALLOCATOR_PEAK", "device": device, "bytes": peak}
                    if row not in evidence:
                        evidence.append(row)
    except Exception:
        observation_failed = True
    manifest["gpu_usage_evidence"] = evidence
    if evidence or manifest.get("gpu_used") is True:
        manifest["gpu_used"] = True
        manifest["gpu_usage_status"] = "ACTUAL_USE_OBSERVED"
    elif observation_failed:
        manifest["gpu_used"] = None
        manifest["gpu_usage_status"] = "UNKNOWN_OBSERVATION_FAILED"
    else:
        manifest["gpu_used"] = False
        manifest["gpu_usage_status"] = "NO_CUDA_ALLOCATION_OBSERVED"
    return manifest["gpu_used"]


def _checkpoint(trainer, arm_identity, relative_step, config, arm_dir, manifest, run_root, result):
    import torch
    from minicells_r3e.common import fingerprint

    state = copy.deepcopy(trainer.state())
    metadata = _checkpoint_metadata(state, arm_identity, relative_step, config["anchor_step"])
    payload = {
        "schema": "minicells.r3mf_d1.checkpoint.v2",
        "metadata": metadata,
        "anchor_data_cursor": config["sampling_start"]["data_cursor"],
        "arm_result": copy.deepcopy(result),
        "trainer_state": state,
        "state_fingerprint": fingerprint(state),
    }
    path = Path(arm_dir) / ("checkpoint-step-%06d.pt" % relative_step)
    _atomic_torch_save(torch, path, payload)
    _artifact(manifest, run_root, path)
    return path, manifest["artifacts"][path.relative_to(run_root).as_posix()]["sha256"]


def _save_arm_result(arm_dir, result, manifest, run_root):
    path = Path(arm_dir) / "ARM_RESULT.json"
    _atomic_json(path, result)
    _artifact(manifest, run_root, path)


def _resume_checkpoint(arm_dir, expected_identity, plan, records, config, device, manifest, run_root, runtime=None):
    import torch
    from minicells_r3e.common import fingerprint

    candidates = sorted(Path(arm_dir).glob("checkpoint-step-*.pt"))
    if not candidates:
        return None
    path = candidates[-1]
    relative_path = path.relative_to(run_root).as_posix()
    actual_file = {"bytes": path.stat().st_size, "sha256": _file_sha(path)}
    retained_file = manifest.get("artifacts", {}).get(relative_path)
    if retained_file is not None and retained_file != actual_file:
        raise ValueError("D1_CHECKPOINT_ARTIFACT_MANIFEST_MISMATCH")
    manifest.setdefault("artifacts", {})[relative_path] = actual_file
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(payload, expected_identity, plan, records, config["anchor_step"])
    if fingerprint(payload["trainer_state"]) != payload["state_fingerprint"]:
        raise ValueError("D1_CHECKPOINT_FINGERPRINT_MISMATCH")
    trainer = _restore_d1_trainer(
        runtime if runtime is not None else _runtime(config), payload["trainer_state"], config["seed"], device
    )
    result_path = Path(arm_dir) / "ARM_RESULT.json"
    existing = _read(result_path) if result_path.exists() else None
    result = _reconcile_result(payload, existing, expected_identity, config, arm_dir)
    return trainer, payload["metadata"]["relative_step"], result


def _reconcile_result(payload, existing, identity, config, arm_dir):
    """Checkpoint state owns completed updates; result files may only lag behind it."""
    anchor = config["anchor_step"]
    relative = payload["metadata"]["relative_step"]
    curve = [copy.deepcopy(row) for row in payload["trainer_state"]["curve"] if row["step"] > anchor]
    if [row["step"] for row in curve] != list(range(anchor + 1, anchor + relative + 1)):
        raise ValueError("D1_CHECKPOINT_CURVE_NOT_CONTIGUOUS")
    snapshot = payload["arm_result"]
    if snapshot.get("identity") != identity or snapshot.get("curve") != curve:
        raise ValueError("D1_CHECKPOINT_RESULT_STATE_MISMATCH")
    result = copy.deepcopy(snapshot)
    result["status"] = "RUNNING"
    evaluations = {}
    for candidate in (snapshot, existing):
        if candidate is None:
            continue
        if candidate.get("identity") != identity:
            raise ValueError("D1_ARM_RESULT_IDENTITY_MISMATCH")
        reported_curve = candidate.get("curve", [])
        if reported_curve != curve[:len(reported_curve)]:
            raise ValueError("D1_ARM_RESULT_CURVE_CONFLICT")
        for row in candidate.get("evaluations", []):
            step = row["relative_step"]
            if step not in config["evaluation_steps"] or step > relative or row["global_step"] != anchor + step:
                raise ValueError("D1_ARM_RESULT_EVALUATION_BOUNDARY_INVALID")
            checkpoint_path = Path(arm_dir) / ("checkpoint-step-%06d.pt" % step)
            if row.get("checkpoint_sha256") != _file_sha(checkpoint_path):
                raise ValueError("D1_ARM_RESULT_EVALUATION_CHECKPOINT_MISMATCH")
            if step in evaluations and evaluations[step] != row:
                raise ValueError("D1_ARM_RESULT_EVALUATION_CONFLICT")
            evaluations[step] = copy.deepcopy(row)
    result["curve"] = curve
    result["evaluations"] = [evaluations[step] for step in sorted(evaluations)]
    earlier = [step for step in config["evaluation_steps"] if step < relative]
    if any(step not in evaluations for step in earlier):
        raise ValueError("D1_CHECKPOINT_PRIOR_EVALUATION_MISSING")
    return result


def _runtime(config):
    from minicells_r3m.anchor import runtime

    return runtime(config)


def _runtime_secret():
    from minicells_r3e.runner import secrets

    secrets()


def _load_parent_state(config, output, device):
    import torch
    from minicells_artifacts.hf_storage import HubBackend
    from minicells_r3m.anchor import load
    from minicells_r3e.common import fingerprint

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("D1_EXACT_R3H_RESTORE_REQUIRES_TWO_VISIBLE_CUDA_DEVICES")
    _runtime_secret()
    reference = config["baseline"]["hf_reference"]
    backend = HubBackend(reference["hf_repo"])
    info = backend.info(reference["hf_revision"])
    if info.sha != reference["hf_revision"]:
        raise RuntimeError("D1_PINNED_HF_REVISION_CHANGED")
    anchor_config = dict(config, tokens_per_step=config["source_targets_per_step"])
    anchor, receipt = load(
        anchor_config, backend, Path(output) / "parent-cache", config["seed"], device, step=300
    )
    if receipt["receipt"]["sha256"] != config["anchor_checkpoint"]["checkpoint_sha256"]:
        raise RuntimeError("D1_ANCHOR_CHECKPOINT_SHA_MISMATCH")
    raw_state = copy.deepcopy(anchor.state())
    if (
        receipt["receipt"].get("state_fingerprint")
        != config["anchor_checkpoint"]["state_fingerprint"]
        or fingerprint(raw_state) != config["anchor_checkpoint"]["state_fingerprint"]
    ):
        raise RuntimeError("D1_ANCHOR_MODEL_STATE_FINGERPRINT_MISMATCH")
    state = _cpu_tree(raw_state)
    del raw_state, anchor
    if state["step"] != config["anchor_step"] or state["data_cursor"] != config["sampling_start"]["data_cursor"]:
        raise RuntimeError("D1_ANCHOR_STEP_OR_CURSOR_MISMATCH")
    if canonical(state["data_rng"]) != canonical(config["sampling_start"]["data_rng"]):
        raise RuntimeError("D1_ANCHOR_DATA_RNG_MISMATCH")
    return state, receipt


def run_bcd(config, preflight_dir, output_root, source_sha, device="cuda:0"):
    """Run only B/C/D with a fixed 200-update ceiling and exact resume checks."""
    from tokenizers import Tokenizer

    from .source_identity import verify_source_tree, code_sha

    if verify_source_tree(REPO) != source_sha:
        raise RuntimeError("D1_SOURCE_COMMIT_MISMATCH")
    source_code_sha = code_sha(BASE, REPO)
    config = copy.deepcopy(config)
    config_sha = sha256_bytes(canonical(config).encode("utf-8"))
    if config_sha != Path(BASE / "config/frozen.sha256").read_text().strip():
        raise RuntimeError("D1_FROZEN_CONFIG_SHA_MISMATCH")
    data_sha = _file_sha(RECORDS)
    if data_sha != config["input_records"]["sha256"]:
        raise RuntimeError("D1_TINY_DATA_SHA_MISMATCH")
    if _file_sha(TOKENIZER) != config["invariants"]["tokenizer_sha256"]:
        raise RuntimeError("D1_TOKENIZER_SHA_MISMATCH")
    gate_manifest = _read(Path(preflight_dir) / "RUN_MANIFEST.json")
    plan_path = Path(preflight_dir) / "SAMPLE_PLAN.json"
    proof_path = Path(preflight_dir) / "BUDGET_PROOF.json"
    plan = _read(plan_path)
    proof = _read(proof_path)
    if (
        gate_manifest.get("experiment_id") != config["experiment_id"]
        or gate_manifest.get("source_sha") != source_sha
        or gate_manifest.get("config_sha256") != config_sha
        or gate_manifest.get("data_sha256") != data_sha
        or gate_manifest.get("source_code_sha256") != source_code_sha
    ):
        raise RuntimeError("D1_PREFLIGHT_MANIFEST_IDENTITY_MISMATCH")
    verify_plan_identity(
        plan,
        config_sha,
        data_sha,
        config["anchor_checkpoint"]["checkpoint_sha256"],
    )
    if proof.get("status") != "CPU_GATE_PASS" or proof.get("gpu_used") or proof.get("optimizer_updates"):
        raise RuntimeError("D1_CPU_GATE_PROOF_INVALID")
    for name, path in (
        ("SAMPLE_PLAN.json", plan_path),
        ("BUDGET_PROOF.json", proof_path),
        ("A_REUSED_PREDICTIONS.json", Path(preflight_dir) / "A_REUSED_PREDICTIONS.json"),
    ):
        expected = gate_manifest["artifacts"].get(name)
        if expected is None or expected["sha256"] != _file_sha(path):
            raise RuntimeError("D1_PREFLIGHT_ARTIFACT_HASH_MISMATCH_" + name)

    records = _read(RECORDS)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    plan_sha = _file_sha(plan_path)
    run_key = sha256_bytes(
        canonical(
            {
                "experiment_id": config["experiment_id"],
                "source_sha": source_sha,
                "source_code_sha256": source_code_sha,
                "config_sha256": config_sha,
                "data_sha256": data_sha,
                "sample_plan_sha256": plan_sha,
                "seed": config["seed"],
                "anchor_sha256": config["anchor_checkpoint"]["checkpoint_sha256"],
            }
        ).encode("utf-8")
    )[:32]
    run_root = Path(output_root) / "runs" / run_key
    if run_root.exists() and any(run_root.iterdir()) and not (run_root / "RUN_MANIFEST.json").exists():
        raise RuntimeError("D1_OUTPUT_DIRECTORY_HAS_UNCLAIMED_CONTENTS")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "RUN_MANIFEST.json"
    identity_base = {
        "experiment_id": config["experiment_id"],
        "source_sha": source_sha,
        "source_code_sha256": source_code_sha,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "sample_plan_sha256": plan_sha,
        "seed": config["seed"],
        "anchor_step": config["anchor_step"],
        "anchor_checkpoint_sha256": config["anchor_checkpoint"]["checkpoint_sha256"],
        "max_steps": 200,
    }
    if manifest_path.exists():
        manifest = _read(manifest_path)
        for key, value in identity_base.items():
            if manifest.get(key) != value:
                raise RuntimeError("D1_EXISTING_RUN_IDENTITY_MISMATCH_" + key)
    else:
        manifest = {
            "schema": "minicells.r3mf_d1.run_manifest.v1",
            "run_id": run_key,
            **identity_base,
            "run_status": "PREFLIGHT_PASSED_TRAINING_NOT_STARTED",
            "trial_status": "BOUNDED_DEVELOPMENT_DIAGNOSIS",
            "explicit_run_authorization": True,
            "arms": {arm: {"status": "NOT_STARTED", "updates": 0} for arm in ("B", "C", "D")},
            "arm_a": "REUSED_PUBLISHED_TINY_OVERFIT_FAILED_WITHOUT_RETRAINING",
            "budgets": proof["arm_budgets"],
            "logical_optimizer_updates_total_max": 600,
            "logical_optimizer_updates": 0,
            "gpu_requested": str(device).split(":")[0] == "cuda",
            "gpu_used": False,
            "gpu_usage_status": "NOT_YET_OBSERVED",
            "gpu_usage_evidence": [],
            "update_budget_semantics": "AT_MOST_200_LOGICAL_UPDATES_PER_ARM",
            "cumulative_optimizer_calls_across_retries": None,
            "cumulative_gpu_seconds_across_retries": None,
            "retry_cost_accounting": "CUMULATIVE_FAILED_ATTEMPT_WORK_NOT_MEASURED",
            "cumulative_retry_hard_cap": False,
            "main_or_confirmation_accessed": False,
            "publication_status": "KAGGLE_OUTPUT_ONLY",
            "remote_commit_sha": None,
            "artifacts": {},
        }
    manifest["run_status"] = "RUNNING"
    for name, source in (
        ("SAMPLE_PLAN.json", plan_path),
        ("BUDGET_PROOF.json", proof_path),
        ("A_REUSED_PREDICTIONS.json", Path(preflight_dir) / "A_REUSED_PREDICTIONS.json"),
        ("CPU_GATE_RUN_MANIFEST.json", Path(preflight_dir) / "RUN_MANIFEST.json"),
    ):
        target = run_root / name
        if target.exists():
            if _file_sha(target) != _file_sha(source):
                raise RuntimeError("D1_EXISTING_ARTIFACT_MISMATCH_" + name)
        else:
            target.write_bytes(Path(source).read_bytes())
        _artifact(manifest, run_root, target)
    _write_manifest(manifest_path, manifest)

    return _execute_run(
        config, plan, proof, records, tokenizer, _runtime(config), run_root,
        manifest, identity_base, device,
    )


def _execute_run(config, plan, proof, records, tokenizer, runtime, run_root,
                 manifest, identity_base, device):
    """Shared production control flow; tests inject synthetic CPU parent state only."""
    import torch
    from minicells_r3e.common import fingerprint

    run_root = Path(run_root)
    manifest_path = run_root / "RUN_MANIFEST.json"
    run_key = manifest["run_id"]
    schedule_state = None
    parent_state = None
    anchor_receipt = None
    optimizer_start = None
    for arm in ("B", "C", "D"):
        arm_dir = run_root / ("arm-" + arm)
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_identity = dict(identity_base, arm=arm)
        resume = _resume_checkpoint(arm_dir, arm_identity, plan, records, config, device, manifest, run_root, runtime)
        if resume is not None:
            trainer, relative_step, result = resume
            schedule_state = result.get("schedule_state")
            if schedule_state is None:
                anchor_file = arm_dir / "schedule.json"
                schedule_state = _read(anchor_file)
            manifest["arms"][arm] = {"status": "RUNNING", "updates": relative_step}
            manifest["logical_optimizer_updates"] = sum(
                row.get("updates", 0) for row in manifest["arms"].values()
            )
            _write_manifest(manifest_path, manifest)
        else:
            if parent_state is None:
                parent_state, anchor_receipt = _load_parent_state(config, run_root, device)
                schedule_state = {
                    "anchor_lr": float(parent_state["cell_optimizer"]["param_groups"][0]["lr"]),
                    "anchor_epsilon": float(parent_state["step"] / max(1, runtime["training"]["steps"] - 1)
                    * (runtime["training"]["epsilon_final"] - runtime["training"]["epsilon_initial"])
                    + runtime["training"]["epsilon_initial"]),
                }
                optimizer_start = fingerprint(parent_state["cell_optimizer"])
                manifest["anchor_receipt"] = anchor_receipt
                manifest["anchor_state_fingerprint"] = fingerprint(parent_state)
                manifest["optimizer_state_at_anchor_sha256"] = optimizer_start
            trainer = _restore_d1_trainer(runtime, parent_state, config["seed"], device)
            if fingerprint(trainer.cell_opt.state_dict()) != optimizer_start:
                raise RuntimeError("D1_CELL_OPTIMIZER_RESET_OR_CHANGED_AT_ARM_START")
            verify_start_rng(trainer.data_rng, plan["rng_start"])
            relative_step = 0
            result = {
                "schema": "minicells.r3mf_d1.arm_result.v1",
                "arm": arm,
                "identity": arm_identity,
                "status": "RUNNING",
                "context": config["arms"][arm]["context"],
                "objective": config["arms"][arm]["loss"],
                "evaluations": [],
                "curve": [],
                "schedule_state": schedule_state,
            }
            _atomic_json(arm_dir / "schedule.json", schedule_state)
            manifest["arms"][arm] = {"status": "RUNNING", "updates": 0}
            _write_manifest(manifest_path, manifest)

        observe_gpu_usage(manifest, trainer.model)
        _write_manifest(manifest_path, manifest)
        if relative_step not in config["evaluation_steps"] and relative_step % 25 != 0:
            raise RuntimeError("D1_RESUME_NOT_AT_DURABLE_BOUNDARY")
        if relative_step == 0:
            if not result["evaluations"]:
                path, checkpoint_sha = _checkpoint(
                    trainer, arm_identity, 0, config, arm_dir, manifest, run_root, result
                )
                result["evaluations"].append(
                    dict(_evaluate(trainer, records, tokenizer, device, 0), checkpoint_sha256=checkpoint_sha)
                )
                _save_arm_result(arm_dir, result, manifest, run_root)
                _write_manifest(manifest_path, manifest)
        elif (
            relative_step in config["evaluation_steps"]
            and (not result["evaluations"] or result["evaluations"][-1]["relative_step"] != relative_step)
        ):
            checkpoint_path = arm_dir / ("checkpoint-step-%06d.pt" % relative_step)
            checkpoint_sha = _file_sha(checkpoint_path)
            result["evaluations"].append(
                dict(_evaluate(trainer, records, tokenizer, device, relative_step), checkpoint_sha256=checkpoint_sha)
            )
            _save_arm_result(arm_dir, result, manifest, run_root)
            _write_manifest(manifest_path, manifest)

        for next_relative in range(relative_step + 1, config["steps"] + 1):
            planned = plan["relative_updates"][next_relative - 1]
            rows = replay_planned_update(trainer.data_rng, records, planned)
            batches = torch_batches(
                rows,
                config["arms"][arm]["context"],
                device,
            )
            update_row = advance(
                trainer,
                batches,
                objective=config["arms"][arm]["loss"],
                context=config["arms"][arm]["context"],
                d1_config=config,
                schedule=__import__("minicells_r3m.protocol", fromlist=["schedule"]).schedule,
                schedule_state=schedule_state,
            )
            result["curve"].append(update_row)
            relative_step = next_relative
            manifest["arms"][arm]["updates"] = relative_step
            manifest["logical_optimizer_updates"] = sum(
                row.get("updates", 0) for row in manifest["arms"].values()
            )
            should_checkpoint = relative_step % 25 == 0 or relative_step in config["evaluation_steps"]
            if should_checkpoint:
                metrics = []
                if relative_step in config["evaluation_steps"]:
                    checkpoint_path, checkpoint_sha = _checkpoint(
                        trainer, arm_identity, relative_step, config, arm_dir, manifest, run_root, result
                    )
                    metrics = _evaluate(trainer, records, tokenizer, device, relative_step)
                    result["evaluations"].append(dict(metrics, checkpoint_sha256=checkpoint_sha))
                else:
                    _checkpoint(
                        trainer, arm_identity, relative_step, config, arm_dir, manifest, run_root, result
                    )
                _save_arm_result(arm_dir, result, manifest, run_root)
                _write_manifest(manifest_path, manifest)
                print(
                    json.dumps(
                        {"arm": arm, "step": relative_step, "ce": update_row["loss"], "evaluation": bool(metrics)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        result["status"] = "COMPLETE"
        result["logical_optimizer_updates"] = relative_step
        result["budget"] = proof["arm_budgets"][arm]
        result["cell_optimizer_end_sha256"] = fingerprint(trainer.cell_opt.state_dict())
        trainer.verify_frozen()
        if fingerprint(trainer.model.connection.state_dict()) != trainer.connection_sha:
            raise RuntimeError("D1_CONNECTION_FROZEN_CHECK_FAILED")
        _save_arm_result(arm_dir, result, manifest, run_root)
        manifest["arms"][arm] = {"status": "COMPLETE", "updates": relative_step}
        _write_manifest(manifest_path, manifest)
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["run_status"] = "COMPLETE"
    manifest["exit_code"] = 0
    manifest["logical_optimizer_updates"] = sum(row["updates"] for row in manifest["arms"].values())
    manifest["completed_at_unix"] = time.time()
    results = {
        "schema": "minicells.r3mf_d1.results.v1",
        "run_id": run_key,
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE",
        "A_reused": _read(run_root / "A_REUSED_PREDICTIONS.json"),
        "arms": {
            arm: _read(run_root / ("arm-" + arm) / "ARM_RESULT.json")
            for arm in ("B", "C", "D")
        },
        "pure_equal_budget_factorial": False,
        "update_budget_semantics": "AT_MOST_200_LOGICAL_UPDATES_PER_ARM",
        "cumulative_optimizer_calls_across_retries": None,
        "cumulative_gpu_seconds_across_retries": None,
    }
    results_path = run_root / "RESULTS.json"
    _atomic_json(results_path, results)
    _artifact(manifest, run_root, results_path)
    _write_manifest(manifest_path, manifest)
    return {"status": "COMPLETE", "run_id": run_key, "output": str(run_root), "manifest": manifest}
