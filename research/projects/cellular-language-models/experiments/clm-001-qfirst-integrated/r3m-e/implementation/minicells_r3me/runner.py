"""One-launch R3m-E Gate 0/1 preflight; training remains separately authorized."""

import argparse
import copy
import json
import os
from pathlib import Path
import time
import uuid

import torch

from . import data, evaluation, protocol, reports
from minicells_r3m import anchor
from .common import BASE, atomic, canonical, digest, fingerprint, read, source_sha


def _token():
    if os.environ.get("HF_TOKEN", "").strip():
        return
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception as exc:
        raise RuntimeError("HF_TOKEN_REQUIRED_TO_VERIFY_PINNED_R3H_CHECKPOINT") from exc
    if not os.environ["HF_TOKEN"]:
        raise RuntimeError("HF_TOKEN_REQUIRED_TO_VERIFY_PINNED_R3H_CHECKPOINT")


def run_identity(c, sha):
    return protocol.identity(c, sha)


def cpu_gate(c):
    protocol.validate(c)
    if len(c["seeds"]) * len(c["arms"]) != 12:
        raise ValueError("R3ME_MATRIX_SIZE_CHANGED")
    expected_updates = len(c["seeds"]) * len(c["arms"]) * c["continuation_steps"]
    if c["training"]["logical_updates"] != expected_updates:
        raise ValueError("R3ME_LOGICAL_UPDATE_ACCOUNTING_INVALID")
    if c["training"]["logical_tokens"] != expected_updates * c["tokens_per_step"]:
        raise ValueError("R3ME_LOGICAL_TOKEN_ACCOUNTING_INVALID")
    task_counts = {
        str(lo): sum(protocol.task_for(i) == "arithmetic" for i in range(lo, hi))
        for lo, hi in ((0, 100), (100, 200), (200, 400), (400, 600))
    }
    expected = {"0": 10, "100": 30, "200": 100, "400": 100}
    if task_counts != expected:
        raise ValueError("R3ME_FIXED_TASK_SCHEDULE_INVALID")
    return {
        "status": "CPU_GATE_PASS",
        "experiment_id": c["experiment_id"],
        "arms": list(c["arms"]),
        "seeds": list(c["seeds"]),
        "task_counts_per_phase": task_counts,
        "logical_updates": c["training"]["logical_updates"],
        "logical_tokens": c["training"]["logical_tokens"],
        "confirmation": "SEALED",
        "formal_confirmation_authorized": False,
        "full_gpu_training_authorized": False,
    }


def _arm_runtime_name(arm):
    return {
        "E0-STORY-STABLE": "STABLE",
        "E1-MIX-STABLE": "STABLE",
        "E2-MIX-REORG": "P1",
        "E3-MIX-SHAM": "P0",
    }[arm]


def _job_root(output, seed, arm, run_id):
    return Path(output) / "private-checkpoints" / run_id / "jobs" / str(seed) / arm


def analyze_job(c, output, runtime, train_assets, research, tokenizer, identity, seed, arm, device):
    from minicells_r3j.storage import Store
    from minicells_r3m.training import restore as restore_trainer

    root = _job_root(output, seed, arm, identity["run_id"])
    evaluation_file = root / "EVALUATION.json"
    if evaluation_file.is_file():
        cached = read(evaluation_file)
        if cached.get("identity") != identity or cached.get("seed") != seed or cached.get("arm") != arm:
            raise RuntimeError("R3ME_CACHED_EVALUATION_IDENTITY_MISMATCH")
        return cached
    arm_identity = dict(identity, seed=seed, arm=arm)
    evaluations = {}
    models = {}
    for relative in c["barriers"]:
        store = Store(root, arm_identity)
        receipt = read(store.receipt_file(relative))
        payload = store.verified(receipt)
        trainer = restore_trainer(runtime, payload["trainer"], seed, device)
        metrics = evaluation.measure(trainer.model, research, tokenizer, c)
        evaluations[str(relative)] = metrics
        if relative in c["evaluation"]["cell_connection_grid_relative_steps"]:
            models[relative] = trainer.model
    final_payload = Store(root, arm_identity).verified(read(Store(root, arm_identity).receipt_file(600)))
    final_tr = restore_trainer(runtime, final_payload["trainer"], seed, device)
    specialization = evaluation.specialization(final_tr.model, research, tokenizer, c)
    gradient = evaluation.gradient_conflict(final_tr, train_assets, research, seed, c)
    four_grid = {}
    if arm == "E1-MIX-STABLE":
        reorg_root = _job_root(output, seed, "E2-MIX-REORG", identity["run_id"])
        reorg_id = dict(identity, seed=seed, arm="E2-MIX-REORG")
        for relative in c["evaluation"]["cell_connection_grid_relative_steps"]:
            stable_model = models.get(relative)
            if stable_model is None:
                continue
            reorg_store = Store(reorg_root, reorg_id)
            reorg_payload = reorg_store.verified(read(reorg_store.receipt_file(relative)))
            reorg_tr = restore_trainer(runtime, reorg_payload["trainer"], seed, device)
            grid = evaluation.four_grid(stable_model, reorg_tr.model, research, tokenizer, c)
            four_grid[str(relative)] = {"".join(k): v for k, v in grid.items()}
    curve = []
    training_exposure = {"story": [], "arithmetic": []}
    for source_row in final_payload["aux"]["curve"]:
        row = dict(source_row)
        relative = int(row.get("relative_step", row.get("step", c["anchor_step"] + 1)))
        relative = relative - c["anchor_step"] if relative > c["continuation_steps"] else relative
        task = "story" if arm == "E0-STORY-STABLE" else protocol.task_for(max(0, relative - 1))
        row["task"] = task
        row["task_label_scope"] = "POSTHOC_FIXED_SCHEDULE_LABEL_NOT_ROUTER_OR_PROPOSAL_INPUT"
        curve.append(row)
        counts = row.get("cell_counts", [])
        if counts:
            if not training_exposure[task]:
                training_exposure[task] = [0] * len(counts)
            training_exposure[task] = [a + int(b) for a, b in zip(training_exposure[task], counts)]
    result = {
        "seed": seed,
        "arm": arm,
        "identity": identity,
        "status": "COMPLETE",
        "evaluations": evaluations,
        "curve": curve,
        "training_exposure": training_exposure,
        "transactions": final_payload["aux"]["transactions"],
        "cell_exposure": {key: value["teacher_forced_cell_usage"] for key, value in evaluations.items()},
        "specialization": specialization,
        "gradient_conflict": gradient,
        "four_grid": four_grid,
        "cost": {
            "main_updates": final_payload["aux"]["main_updates"],
            "main_tokens": final_payload["aux"]["main_updates"] * c["tokens_per_step"],
            "shadow_updates": final_payload["aux"]["shadow_updates"],
            "discarded_shadow_updates": final_payload["aux"]["discarded_shadow_updates"],
            "proposal_optimizer_updates": final_payload["aux"]["proposal_optimizer_updates"],
            "trainer_cost": final_tr.cost,
            "evaluation_seconds": sum(v["seconds"] for v in evaluations.values()),
            "intervention_seconds": sum(v.get("seconds", 0.0) for v in specialization),
            "strict_equal_total_compute": False,
        },
        "checkpoint_receipts": [
            read(Store(root, arm_identity).receipt_file(t))
            for t in c["barriers"]
        ],
        "confirmation": "SEALED",
    }
    atomic(root / "EVALUATION.json", result)
    return result


def run_study(c, output, assets_dir, device="cuda:0"):
    """Run the explicitly released exploratory matrix; resume from immutable HF receipts."""
    if not c["authorization"]["full_training"] or c["formal_confirmation"] != "NOT_AUTHORIZED":
        raise RuntimeError("R3ME_EXPLORATORY_GPU_TRAINING_NOT_AUTHORIZED_IN_FROZEN_CONFIG")
    from minicells_artifacts.hf_storage import HubBackend
    from minicells_r3m import engine as r3m_engine
    from minicells_r3m import anchor as r3m_anchor
    from minicells_r3m.storage import restore as restore_private
    from .storage import Publisher as R3mPublisher
    from minicells_r3e.runner import secrets
    from tokenizers import Tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("R3ME_EXPLORATORY_TRAINING_REQUIRES_CUDA")
    secrets()
    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    backend.check_write()
    runtime = anchor.runtime(c)
    train_assets, research, input_manifest, tokenizer_path = data.prepare(
        c, runtime, backend, assets_dir
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ident = run_identity(c, source_sha())
    anchor_evidence = []
    private = Path(output) / "private-checkpoints" / ident["run_id"]
    private.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex
    revision = backend.info().sha
    pointer_path = "r3m-e/" + ident["run_id"] + "/latest.json"
    prior = backend.optional(pointer_path, revision)
    prior = dict(prior, hf_repo=backend.repo, hf_revision=revision) if prior else None
    if prior:
        restore_private(backend, prior, private, output, ident, session, c["barriers"])
    publisher = R3mPublisher(backend, private, Path(output), ident, session, prior)

    def persist(force=True):
        publisher.publish(force=force)

    failures = []
    runtime_for_arm = copy.deepcopy(runtime)
    for seed in c["seeds"]:
        for logical_arm in c["arms"]:
            root = _job_root(output, seed, logical_arm, ident["run_id"])
            aid = dict(ident, seed=seed, arm=logical_arm)
            try:
                base, evidence = r3m_anchor.load(
                    c, backend, Path(output) / "parent-r3h", seed, device, step=c["anchor_step"]
                )
                if seed not in {item["seed"] for item in anchor_evidence}:
                    anchor_evidence.append({"seed": seed, **evidence})
                atomic(root / "ANCHOR.json", evidence)
                if not (root / "COMPLETE.json").exists():
                    r3m_engine.run_arm(
                        c,
                        copy.deepcopy(runtime_for_arm),
                        base,
                        aid,
                        root,
                        train_assets,
                        research,
                        "N" if logical_arm == "E0-STORY-STABLE" else "C",
                        _arm_runtime_name(logical_arm),
                        device,
                        persist,
                    )
                persist()
            except Exception as exc:
                failures.append({"seed": seed, "arm": logical_arm, "type": type(exc).__name__, "message": str(exc)[:180]})
                atomic(Path(output) / "FAILURES.json", failures)
                try:
                    persist()
                except Exception:
                    pass
                break
        if failures:
            break
    input_identity = {
        "identity": ident,
        "input_manifest": input_manifest,
        "anchors": anchor_evidence,
        "confirmation": "SEALED",
        "r3m_main_candidates_accessed": False,
    }
    atomic(Path(output) / "INPUT_IDENTITY.json", input_identity)
    if not failures:
        for seed in c["seeds"]:
            for logical_arm in c["arms"]:
                try:
                    analyze_job(
                        c, output, runtime, train_assets, research, tokenizer,
                        ident, seed, logical_arm, device,
                    )
                except Exception as exc:
                    failures.append({"seed": seed, "arm": logical_arm, "phase": "analysis", "type": type(exc).__name__, "message": str(exc)[:180]})
                    break
            if failures:
                break
    result = reports.write(c, output, ident, input_identity, failures)
    result["hf_reference"] = publisher.prior
    atomic(Path(output) / "reports" / "RUN_MANIFEST.json", result)
    return result


def preflight(c, output, assets_dir):
    """Verify pinned config, HF identity, immutable parent checkpoint and data assets."""
    from minicells_artifacts.hf_storage import HubBackend

    _token()
    runtime = anchor.runtime(c)
    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    folder = Path(assets_dir)
    historical, research, manifest, tokenizer_path = data.prepare(c, runtime, backend, folder)
    from .parent import load_for_cpu_inspection

    anchors = []
    for seed in c["seeds"]:
        trainer, proof = load_for_cpu_inspection(c, backend, Path(output) / "parent-r3h", seed)
        anchors.append(
            {
                "seed": seed,
                "step": trainer.step,
                "data_cursor": trainer.data_cursor,
                "model_sha256": fingerprint(trainer.model.state_dict()),
                "cell_optimizer_sha256": fingerprint(trainer.cell_opt.state_dict()),
                "connection_sha256": fingerprint(trainer.model.connection.state_dict()),
                "anchor_receipt_sha256": proof["receipt"].get("sha256"),
                "temperature": trainer.model.temperature,
                "routing_mode": trainer.mode,
                "cpu_inspection": proof["cpu_inspection"],
            }
        )
        del trainer
    identity = run_identity(c, source_sha())
    result = {
        "status": "R3ME_ENGINEERING_PREFLIGHT_PASS",
        "identity": identity,
        "config_sha256": canonical(c),
        "input_manifest": manifest,
        "tokenizer_sha256": digest(tokenizer_path),
        "anchors": anchors,
        "history_overlap_disclosure": c["known_limitations"],
        "confirmation": "SEALED",
        "r3m_main_candidates_accessed": False,
        "gpu_training_started": False,
    }
    atomic(Path(output) / "INPUT_IDENTITY.json", result)
    return historical, research, manifest, result


def smoke(c, output, assets_dir, device="cuda:0"):
    """Single disposable optimizer update from a verified real checkpoint; no study cursor."""
    from minicells_artifacts.hf_storage import HubBackend
    from minicells_r3m.training import advance, initial_aux

    if not torch.cuda.is_available() or not device.startswith("cuda:"):
        raise RuntimeError("R3ME_REAL_CHECKPOINT_SMOKE_REQUIRES_CUDA")
    _token()
    runtime = anchor.runtime(c)
    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    historical, research, manifest, _ = preflight(c, output, assets_dir)
    rows = []
    seed = c["seeds"][0]
    trainer, proof = anchor.load(c, backend, Path(output) / "parent-r3h", seed, device)
    before = trainer.state()
    for logical_arm, task in (("E0-STORY-STABLE", "story"), ("E1-MIX-STABLE", "arithmetic")):
        disposable = copy.deepcopy(trainer)
        batch, actual_task = data.batches(disposable, historical, research, 0, logical_arm, c)
        if actual_task != task:
            raise RuntimeError("R3ME_SMOKE_TASK_ROUTING_CHANGED")
        aux = initial_aux(disposable, c, dict(seed=seed, arm=logical_arm))
        row = advance(disposable, batch, c, aux["schedule"])
        if not torch.isfinite(torch.tensor(row["ce"])) or not all(
            bool(torch.isfinite(p.grad).all())
            for p in disposable.model.parameters()
            if p.grad is not None
        ):
            raise RuntimeError("R3ME_SMOKE_NONFINITE_GRADIENT")
        rows.append(
            {
                "arm": logical_arm,
                "task": task,
                "ce": row["ce"],
                "step_after_disposable_update": disposable.step,
                "cell_optimizer_sha256": fingerprint(disposable.cell_opt.state_dict()),
                "connection_frozen": True,
                "production_cursor_advanced": False,
            }
        )
    if fingerprint(trainer.state()) != fingerprint(before):
        raise RuntimeError("R3ME_SMOKE_MUTATED_ANCHOR")
    result = {
        "status": "REAL_CHECKPOINT_SMOKE_PASS",
        "identity": run_identity(c, source_sha()),
        "anchor_seed": seed,
        "anchor_step": trainer.step,
        "anchor_receipt": proof["receipt"],
        "data_manifest_sha256": canonical(manifest),
        "updates": rows,
        "scientific_updates": 0,
        "confirmation": "SEALED",
    }
    atomic(Path(output) / "REAL_CHECKPOINT_SMOKE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu-gate", "auto", "smoke", "gate2", "train"), default="auto")
    parser.add_argument("--config", type=Path, default=BASE / "config/exploratory.frozen.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    c = read(args.config)
    protocol.validate(c)
    args.output.mkdir(parents=True, exist_ok=True)
    result = cpu_gate(c)
    atomic(args.output / "CPU_GATE.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.mode == "cpu-gate":
        return
    assets_dir = args.assets or (args.output / "assets")
    if args.mode == "gate2":
        from .gate2 import run as run_gate2

        result = run_gate2(c, args.output, assets_dir, args.device)
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    if args.mode == "train":
        result = run_study(c, args.output, assets_dir, args.device)
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    if args.mode == "smoke":
        result = smoke(c, args.output, assets_dir, args.device)
    else:
        _, _, _, result = preflight(c, args.output, assets_dir)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.mode == "auto":
        print(
            json.dumps(
                {
                    "status": "ENGINEERING_READY_GPU_GATES_PENDING",
                    "next": "Review identity, then explicitly authorize the real-checkpoint smoke. Full exploratory training remains disabled.",
                    "scientific_decision": False,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
