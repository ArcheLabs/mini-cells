"""One bounded engineering fixture; emits every protocol artifact without scientific claims."""

import random
import time
from pathlib import Path
import numpy as np
import torch
from minicells_r3e.common import fingerprint
from .common import BASE, ARMS, atomic, digest, read, run_identity
from .data import TaskBank, unit_data, feature_tensor
from .training import Trainer, snapshot_groups, group_delta
from .gates import status

ARTIFACTS = (
    "JCT001_PROTOCOL.md",
    "CONFIG.json",
    "DATA_REGISTRY.json",
    "CHECKPOINT_RECEIPTS.json",
    "RUN_MANIFEST.json",
    "ARM_RESULT.json",
    "ROUTING_EVOLUTION.json",
    "CELL_CONNECTION_ATTRIBUTION.json",
    "COMPUTE_COST.json",
    "BUDGET_PROOF.json",
    "SEED_SUMMARY.json",
    "SCIENTIFIC-FINDINGS.md",
    "GATE_STATUS.json",
)


def engineering(c, output, publish_github=False):
    torch.set_num_threads(1)
    bank = TaskBank(c["fixture"])
    registry = bank.registry()
    identity = run_identity(c, registry)
    root = Path(output) / identity["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "RUN_MANIFEST.json"
    if manifest_path.exists():
        existing = read(manifest_path)
        if existing["identity"] != identity:
            raise RuntimeError("JCT_EXISTING_RUN_IDENTITY_CONFLICT")
        if existing["run_status"] == "CPU_ENGINEERING_PASS_SCIENTIFIC_BLOCKED":
            for name, row in existing["artifact_files"].items():
                if (
                    digest(root / name) != row["sha256"]
                    or (root / name).stat().st_size != row["bytes"]
                ):
                    raise RuntimeError("JCT_COMPLETED_ARTIFACT_CHANGED")
            receipt = None
            if publish_github:
                from .publication import publish

                receipt = publish(root)
            return dict(
                status=existing["run_status"],
                output=str(root),
                run_id=identity["run_id"],
                scientific_optimizer_updates=0,
                reused_completed_engineering=True,
                publication=receipt,
            )
    manifest = dict(
        schema="minicells.jct001.run.v1",
        identity=identity,
        run_id=identity["run_id"],
        experiment_id="JCT-001",
        source_sha=identity["source_sha"],
        config_sha256=identity["config_sha256"],
        scope=identity["scope"],
        run_status="ENGINEERING_STARTED",
        scientific_decision=False,
        scientific_optimizer_updates=0,
        gpu_requested=False,
        gpu_used=False,
        confirmation_materialized=False,
        publication_status="NOT_REQUESTED",
        artifact_files={},
    )
    atomic(manifest_path, manifest)
    if publish_github:
        from .publication import preflight

        preflight(root)
        manifest = read(manifest_path)
    started = time.perf_counter()
    fixture = c["fixture"]
    seed = fixture["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    base_trainer = Trainer(c, "BASE", identity, root / "private/base-ledger.jsonl")
    base_state = base_trainer.state()
    base_receipt = base_trainer.save(root / "private/base.pt")
    base_receipt["file"] = "private/base.pt"
    data = unit_data(fixture)
    data["reference_features"] = feature_tensor(bank.records("calibration")[:4])
    arms = {}
    receipts = {"base": base_receipt}
    routing = {}
    cost = {}
    restored = []
    try:
        for arm in ARMS:
            directory = root / "private" / arm
            directory.mkdir(parents=True, exist_ok=True)
            trainer = Trainer(c, arm, identity, directory / "attempts.jsonl", base_state)
            checkpoint = directory / "checkpoint.pt"
            if checkpoint.exists():
                trainer.restore(checkpoint)
            else:
                trainer.save(checkpoint)
            before = snapshot_groups(trainer.model)
            with trainer.journal.phase("audit_evaluation"):
                base_trainer.model.meter = trainer.journal.meter
                initial = base_trainer.model(data["reference_features"]).actions.clone()
            if trainer.step < fixture["logical_updates"]:
                trainer.advance(data, checkpoint)
            with trainer.journal.phase("audit_evaluation"):
                final = trainer.model(data["reference_features"]).actions.clone()
            receipt = trainer.save(checkpoint)
            receipt["file"] = str(checkpoint.relative_to(root))
            receipts[arm] = receipt
            restored_trainer = Trainer(c, arm, identity, directory / "attempts.jsonl", base_state)
            restored_trainer.restore(checkpoint)
            exact = fingerprint(trainer.state()) == fingerprint(restored_trainer.state())
            restored.append(exact)
            if not exact:
                raise RuntimeError("JCT_CPU_RESTORE_NOT_EXACT")
            row = trainer.curve[-1]
            arms[arm] = dict(
                identity=dict(identity, arm=arm),
                status="CPU_FIXTURE_ONLY",
                logical_updates=trainer.step,
                treatment=c["arms"][arm],
                curve=trainer.curve,
                initial_shared_state_sha256=trainer.parent_base_sha,
                actual_parameter_counts=dict(
                    total=sum(p.numel() for p in trainer.model.parameters()),
                    cells=sum(p.numel() for p in trainer.model.cells.parameters()),
                    connection=sum(p.numel() for p in trainer.model.connection.parameters()),
                ),
                parameter_delta_l2=row["parameter_delta_l2"],
                data_rng_sha256=fingerprint(trainer.data_rng.bit_generator.state),
                main_route_rng_sha256=fingerprint(trainer.route_rng.get_state()),
                local_restore_exact=exact,
                connection_optimizer_lifecycle=fixture["connection_optimizer_lifecycle"],
                connection_optimizer_state_sha256=fingerprint(trainer.connection_optimizer_state),
                connection_candidate_admitted=row["connection_admitted"],
                scientific_decision=False,
                final_parameter_fingerprints={
                    group: fingerprint(getattr(trainer.model, group).state_dict())
                    for group in ("cells", "connection")
                },
            )
            if trainer.step == 1 and any(
                not c["arms"][arm][group] and group_delta(before, trainer.model, group) != 0
                for group in ("cells", "connection")
            ):
                raise RuntimeError("JCT_FROZEN_GROUP_MOVED")
            routing[arm] = dict(
                hard_action_change=float((initial != final).float().mean()),
                hard_path_change=float((initial != final).any(-1).float().mean()),
                final_cell_utilization=torch.bincount(final.flatten(), minlength=6).tolist(),
                scope="CALIBRATION_FEATURE_ONLY_ENGINEERING_ROUTE_AUDIT",
            )
            cost[arm] = trainer.journal.summary()
        for arm in ARMS:
            for group in ("cells", "connection"):
                moved = arms[arm]["parameter_delta_l2"][group] > 0
                if moved != c["arms"][arm][group]:
                    raise RuntimeError("JCT_UNIT_REQUIRED_UPDATE_NOT_OBSERVED:" + arm + ":" + group)
        shared = all(
            arms[a]["data_rng_sha256"] == arms["A"]["data_rng_sha256"]
            and arms[a]["main_route_rng_sha256"] == arms["A"]["main_route_rng_sha256"]
            for a in ARMS
        )
        if not shared or len({arms[a]["curve"][0]["input_batch_sha256"] for a in ARMS}) != 1:
            raise RuntimeError("JCT_MAIN_INPUT_OR_RANDOM_STREAMS_NOT_MATCHED")
        gates = status(
            c,
            local_restore=all(restored),
            update_audit=True,
            cpu_cost=all(v["accounting_status"] == "RECONCILED_CPU_FIXTURE" for v in cost.values()),
        )
        registry["historical_global_ledger_reference"] = {
            p: h
            for p, h in c["source_references"].items()
            if "GLOBAL_DATASET" in p or "content-isolation" in p
        }
        registry["real_model_registration_status"] = "BLOCKED_NO_NEW_STORY_RANGE"
        artifacts = {
            "CONFIG.json": dict(
                configuration=c, fixture_config_frozen_before_inspection_commit="1181d832"
            ),
            "DATA_REGISTRY.json": registry,
            "CHECKPOINT_RECEIPTS.json": dict(
                receipts=receipts,
                all_local_cpu_restore_exact=all(restored),
                cross_session_verified=False,
                initialization="NEW_SYNTHETIC_FIXTURE_NOT_RELABELLED_R3H_CHECKPOINT",
            ),
            "ARM_RESULT.json": dict(
                arms=arms, scope="CPU_UNIT_UPDATE_AUDIT_NOT_SCIENTIFIC_TRAINING"
            ),
            "ROUTING_EVOLUTION.json": dict(
                arms=routing,
                main_random_streams_exactly_matched=shared,
                scientific_specialization_claim=False,
            ),
            "CELL_CONNECTION_ATTRIBUTION.json": dict(
                status="NOT_EVALUATED",
                reason="ONLY_READ_ONLY_INTERVENTION_IMPLEMENTATION_TESTED; NO_PILOT_RUN",
                scientific_decision=False,
            ),
            "COMPUTE_COST.json": dict(
                arms=cost,
                elapsed_engineering_cpu_seconds=time.perf_counter() - started,
                gpu_seconds=None,
                scientific_optimizer_updates=0,
            ),
            "BUDGET_PROOF.json": dict(
                scope="ACTUAL_CPU_UNIT_FIXTURE_ONLY_NOT_GPU_BUDGET_PROOF",
                planned_logical_updates_per_arm=1,
                actual_logical_updates=sum(v["logical_updates"] for v in cost.values()),
                actual_optimizer_calls=sum(v["completed_optimizer_calls"] for v in cost.values()),
                failed_attempts_counted_not_rolled_back=True,
                uncertain_calls=sum(v["uncertain_optimizer_calls"] for v in cost.values()),
                production_extra_counterfactual_work_included=True,
                main_data_and_routing_streams_matched=shared,
                scientific_budget_frozen=False,
            ),
            "SEED_SUMMARY.json": dict(
                status="NOT_ESTIMATED",
                fixture_seed=seed,
                scientific_training_seeds=[],
                reason="ONE_ENGINEERING_FIXTURE_IS_NOT_A_THREE_SEED_PILOT",
                primary_joint_vs_cell=None,
                interaction=None,
                confirmation_power_analysis=None,
            ),
            "GATE_STATUS.json": gates,
        }
        for name, value in artifacts.items():
            atomic(root / name, dict(identity=identity, **value))
        (root / "JCT001_PROTOCOL.md").write_text(
            "<!-- Run identity: "
            + __import__("json").dumps(identity, sort_keys=True)
            + " -->\n\n"
            + (BASE / "JCT001_PROTOCOL.md").read_text()
        )
        (root / "SCIENTIFIC-FINDINGS.md").write_text(
            "# JCT-001 CPU engineering only\n\nRun identity: `"
            + identity["run_id"]
            + "`; source `"
            + identity["source_sha"]
            + "`.\n\nThe four-arm update/freeze and local CPU state checks passed on a known-primitive unit fixture. This is not a Pilot, task-learnability result, production-model validation, or evidence of joint-training benefit.\n\nScientific execution remains BLOCKED: real data isolation, arithmetic learnability, independent cross-session recovery, reviewed task/Connection/Trust Region configuration, runtime GPU budget, scientific Seeds/endpoints and Confirmation power plan remain unresolved. No Story range was selected and no Confirmation examples were materialized.\n\nJoint Cell–Connection training: NOT YET ESTABLISHED.\n"
        )
        manifest["run_status"] = "CPU_ENGINEERING_PASS_SCIENTIFIC_BLOCKED"
        manifest["exit_code"] = 0
        manifest["cpu_fixture_optimizer_calls"] = sum(
            v["completed_optimizer_calls"] for v in cost.values()
        )
        for name in ARTIFACTS:
            if name != "RUN_MANIFEST.json":
                path = root / name
                manifest["artifact_files"][name] = dict(
                    bytes=path.stat().st_size, sha256=digest(path)
                )
        atomic(manifest_path, manifest)
    except Exception as error:
        manifest.update(
            run_status="ENGINEERING_FAILED_SCIENTIFIC_BLOCKED",
            exit_code=1,
            failure_type=type(error).__name__,
        )
        atomic(manifest_path, manifest)
        if publish_github:
            from .publication import publish

            try:
                publish(root)
            except Exception:
                pass
        raise
    receipt = None
    if publish_github:
        from .publication import publish

        receipt = publish(root)
    return dict(
        status=manifest["run_status"],
        output=str(root),
        run_id=identity["run_id"],
        scientific_optimizer_updates=0,
        gpu_used=False,
        publication=receipt,
    )
