"""Prospectively fixed, exploratory known-path GPU Pilot; no real-model release."""

import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind
from .common import BASE, atomic, configuration, read, digest, sha, run_identity
from .data import TaskBank, feature_tensor, unit_data, equivalent_paths, signature
from .gpu_training import DeviceTrainer
from .analysis import oracle_diagnostic
from . import gpu_publication


def pilot_configuration():
    root = BASE / "synthetic-pilot"
    p = read(root / "PILOT_CONFIG.json")
    if sha(p) != (root / "PILOT_CONFIG.sha256").read_text().strip():
        raise RuntimeError("JCT_PILOT_CONFIGURATION_CHANGED")
    if p["real_model_authorized"] or p["confirmation_authorized"]:
        raise RuntimeError("JCT_PILOT_SCOPE_CHANGED")
    c = configuration()
    if p["fixture"] != c["fixture"]:
        raise RuntimeError("JCT_PILOT_GENERATOR_OR_MECHANISM_CHANGED")
    return p, c


@torch.no_grad()
def evaluate(model, records, fixture, device, random_seed=None):
    x = feature_tensor(records, device)
    target = torch.tensor([r["target"] for r in records], device=device)
    if random_seed is None:
        out = model(x)
        logits = out.logits
        actions = out.actions
    else:
        rng = torch.Generator().manual_seed(random_seed)
        h = x
        previous = torch.full((len(x),), 6, dtype=torch.long, device=device)
        paths = []
        for stage in range(model.steps):
            action = torch.randint(6, (len(x),), generator=rng).to(device)
            paths.append(action)
            h, previous = model.transition(h, stage, previous, action)
        logits = model.head(h)
        actions = torch.stack(paths, -1)
    paths = actions.cpu().tolist()
    equivalent = []
    unique = []
    families = {}
    for record, path in zip(records, paths):
        equivalent.append(
            signature(path, fixture["primitive_affines"]) == (record["a"], record["b"])
        )
        valid = equivalent_paths(record, fixture)
        if len(valid) == 1:
            unique.append(tuple(path) == valid[0])
        families.setdefault(record["canonical_id"], []).append(tuple(path))
    pairs = [
        a == b
        for routes in families.values()
        for i, a in enumerate(routes)
        for b in routes[i + 1 :]
    ]
    return dict(
        accuracy=float((logits.argmax(-1) == target).float().mean()),
        loss=float(torch.nn.functional.cross_entropy(logits, target)),
        path_equivalence_rate=sum(equivalent) / len(equivalent),
        unique_correct_path_rate=sum(unique) / len(unique) if unique else None,
        cell_utilization=torch.bincount(actions.flatten(), minlength=6).cpu().tolist(),
        distinct_routes=len({tuple(p) for p in paths}),
        equivalent_input_route_stability=sum(pairs) / len(pairs) if pairs else None,
        route_stability_definition="SAME_CANONICAL_COMPOSITION_ACROSS_REGISTERED_OPERANDS",
        oracle_used=False,
        examples=len(records),
        canonical_compositions=len(families),
    )


def oracle_evaluate(model, records, device):
    # The original feature helper explicitly defaults to CPU, so bind the device;
    # a Torch default-device context alone would leave its feature rows on CPU.
    diagnostic = bind(
        oracle_diagnostic.__wrapped__, feature_tensor=lambda rows: feature_tensor(rows, device)
    )
    with torch.no_grad(), torch.device(device):
        return diagnostic(model, records)


def initial_state(c, identity, root, device):
    seed = c["fixture"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed(seed)
    base = DeviceTrainer(c, "BASE", identity, root / "base-ledger.jsonl", device=device)
    return base, base.state()


def functional_preflight(c, identity, root, refs, device):
    base, state = initial_state(c, identity, root, device)
    trainer = DeviceTrainer(c, "C", identity, root / "unit-ledger.jsonl", state, device=device)
    data = unit_data(c["fixture"])
    data["reference_features"] = refs
    row = trainer.advance(data, root / "unit.pt")
    trainer.save(root / "unit.pt")
    clone = DeviceTrainer(c, "C", identity, root / "unit-ledger.jsonl", state, device=device)
    clone.restore(root / "unit.pt")
    exact = fingerprint(trainer.state()) == fingerprint(clone.state())
    with torch.no_grad():
        before = trainer.model(refs).logits
        after = clone.model(refs).logits
    close = torch.allclose(before, after, rtol=1e-5, atol=1e-7)
    if (
        not exact
        or not close
        or row["parameter_delta_l2"]["cells"] != 0
        or row["parameter_delta_l2"]["connection"] <= 0
    ):
        raise RuntimeError("JCT_GPU_FUNCTIONAL_UPDATE_RESTORE_FAILED")
    return dict(
        status="FUNCTIONAL_PREFLIGHT_PASS",
        identity=identity,
        config_sha256=sha(c),
        reference_sha256=fingerprint(refs),
        checkpoint_sha256=digest(root / "unit.pt"),
        local_full_state_exact=exact,
        next_forward_allclose=bool(close),
        actual_optimizer_calls=trainer.journal.summary()["completed_optimizer_calls"],
        scientific_updates=0,
        cross_session_verified=False,
        parameter_delta_l2=row["parameter_delta_l2"],
    )


def run(p, c, output, device="cuda:0", publish_github=True, test_mode=False):
    device = torch.device(device)
    if not test_mode and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("JCT_GPU_REQUIRED_ENABLE_KAGGLE_ACCELERATOR_THEN_RUN_ALL")
    if test_mode and publish_github:
        raise RuntimeError("JCT_CPU_TEST_CANNOT_PUBLISH_PILOT_RESULT")
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    bank = TaskBank(c["fixture"])
    registry = bank.registry()
    identity = run_identity(c, registry)
    identity.update(
        experiment_id="JCT-001-S1",
        scope="SYNTHETIC_KNOWN_PATH_PILOT" if not test_mode else "CPU_UNIT_TEST_ONLY",
        pilot_config_sha256=sha(p),
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )
    identity.pop("run_id")
    identity["run_id"] = sha(identity)[:32]
    root = Path(output) / identity["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "RUN_MANIFEST.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["identity"] != identity:
            raise RuntimeError("JCT_PILOT_RETAINED_IDENTITY_CHANGED")
        if manifest["run_status"] == "SYNTHETIC_PILOT_COMPLETE_EXPLORATORY_REVIEW_REQUIRED":
            for name, item in manifest["artifact_files"].items():
                if (
                    digest(root / name) != item["sha256"]
                    or (root / name).stat().st_size != item["bytes"]
                ):
                    raise RuntimeError("JCT_PILOT_RETAINED_REPORT_CHANGED")
            return dict(
                status=manifest["run_status"],
                output=str(root),
                reused_completed_run=True,
                publication=gpu_publication.publish(root) if publish_github else None,
            )
    manifest = dict(
        schema="minicells.jct001.synthetic_run.v1",
        identity=identity,
        run_id=identity["run_id"],
        experiment_id="JCT-001-S1",
        source_sha=identity["source_sha"],
        config_sha256=sha(p),
        scope=identity["scope"],
        run_status="PILOT_STARTED",
        started_at_unix=time.time(),
        confirmation_materialized=False,
        gpu_used=device.type == "cuda",
        scientific_optimizer_updates=0,
        artifact_files={},
        publication_status="NOT_REQUESTED",
        formal_protocol_pilot_pass=False,
    )
    atomic(manifest_path, manifest)
    if publish_github:
        gpu_publication.preflight(root)
        manifest = read(manifest_path)
    results = {}
    receipts = {}
    cost = {}
    preflight = {}
    started = time.perf_counter()
    refs = feature_tensor(bank.records("calibration"), device)
    data = dict(role="train", records=bank.records("train"), reference_features=refs)
    try:
        first = copy.deepcopy(c)
        first["fixture"]["seed"] = c["fixture"]["seed"]
        first["fixture"]["logical_updates"] = p["logical_updates_per_arm"]
        unit_start = time.perf_counter()
        unit_dir = root / "private/functional-preflight"
        if (unit_dir / "PREFLIGHT.json").exists():
            preflight = read(unit_dir / "PREFLIGHT.json")
            if (
                preflight.get("identity") != dict(identity, role="DISPOSABLE_FUNCTIONAL_PREFLIGHT")
                or preflight.get("config_sha256") != sha(first)
                or preflight.get("reference_sha256") != fingerprint(refs)
                or preflight.get("checkpoint_sha256") != digest(unit_dir / "unit.pt")
            ):
                raise RuntimeError("JCT_PILOT_FUNCTIONAL_RECEIPT_CHANGED")
        else:
            preflight = functional_preflight(
                first,
                dict(identity, role="DISPOSABLE_FUNCTIONAL_PREFLIGHT"),
                unit_dir,
                refs,
                device,
            )
            preflight["host_wall_seconds"] = time.perf_counter() - unit_start
            atomic(unit_dir / "PREFLIGHT.json", preflight)
        print("[JCT-001 GPU] preflight passed; starting registered synthetic Pilot", flush=True)
        for seed in p["seeds"]:
            cfg = copy.deepcopy(c)
            cfg["fixture"]["seed"] = seed
            cfg["fixture"]["logical_updates"] = p["logical_updates_per_arm"]
            branch_identity = dict(identity, seed=seed)
            directory = root / "private" / str(seed)
            base, state = initial_state(cfg, branch_identity, directory, device)
            receipts[str(seed)] = dict(base=base.save(directory / "base.pt"))
            results[str(seed)] = {}
            cost[str(seed)] = {}
            for arm in p["arms"]:
                tr = DeviceTrainer(
                    cfg,
                    arm,
                    branch_identity,
                    directory / (arm + "-ledger.jsonl"),
                    state,
                    device=device,
                )
                checkpoint = directory / (arm + ".pt")
                if checkpoint.exists():
                    tr.restore(checkpoint)
                else:
                    tr.save(checkpoint)
                # Fixed initial state, never a quality-based checkpoint choice.
                with tr.journal.phase("initial_evaluation"):
                    base.model.meter = tr.journal.meter
                    initial = evaluate(
                        base.model, bank.records("pilot_report"), cfg["fixture"], device
                    )
                starting_step = tr.step
                timing_path = directory / (arm + "-runtime.json")
                timing_records = read(timing_path) if timing_path.exists() else []
                timer = time.perf_counter()
                events = None
                if device.type == "cuda":
                    events = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    events[0].record()
                while tr.step < p["logical_updates_per_arm"]:
                    row = tr.advance(data, checkpoint)
                    if tr.step == 1 or tr.step % 25 == 0:
                        print(
                            json.dumps(
                                dict(
                                    seed=seed,
                                    arm=arm,
                                    step=tr.step,
                                    of=p["logical_updates_per_arm"],
                                    train_loss=row["loss"],
                                )
                            ),
                            flush=True,
                        )
                if events:
                    events[1].record()
                    events[1].synchronize()
                if tr.step > starting_step:
                    timing_records.append(
                        dict(
                            start_step=starting_step,
                            end_step=tr.step,
                            host_training_wall_seconds=time.perf_counter() - timer,
                            cuda_interval_seconds=events[0].elapsed_time(events[1]) / 1000
                            if events
                            else None,
                        )
                    )
                    atomic(timing_path, timing_records)
                covered = {
                    step
                    for row in timing_records
                    for step in range(row["start_step"], row["end_step"])
                }
                timing_complete = covered == set(range(tr.step))
                with tr.journal.phase("final_evaluation"):
                    final = evaluate(tr.model, bank.records("pilot_report"), cfg["fixture"], device)
                    randomized = evaluate(
                        tr.model,
                        bank.records("pilot_report"),
                        cfg["fixture"],
                        device,
                        seed + p["random_route_seed_offset"],
                    )
                    # Oracle only runs inside this isolated diagnostic; never supplied to training.
                    oracle = oracle_evaluate(tr.model, bank.records("pilot_report"), device)
                receipt = tr.save(checkpoint)
                receipt["file"] = str(checkpoint.relative_to(root))
                receipt["scope"] = "LOCAL_DEVICE_RESTORE_ONLY"
                clone = DeviceTrainer(
                    cfg,
                    arm,
                    branch_identity,
                    directory / (arm + "-ledger.jsonl"),
                    state,
                    device=device,
                )
                clone.restore(checkpoint)
                if fingerprint(clone.state()) != fingerprint(tr.state()):
                    raise RuntimeError("JCT_PILOT_FINAL_RESTORE_DIFFERED")
                receipts[str(seed)][arm] = receipt
                summary = tr.journal.summary()
                if (
                    summary["accounting_status"] != "RECONCILED_CPU_FIXTURE"
                    or tr.step != p["logical_updates_per_arm"]
                ):
                    raise RuntimeError("JCT_PILOT_ATTEMPT_BUDGET_NOT_RECONCILED")
                summary.update(
                    accounting_status="RECONCILED_ACTUAL_PILOT_ATTEMPTS",
                    gpu_used=device.type == "cuda",
                    host_wall_seconds=time.perf_counter() - timer,
                    cuda_event_training_interval_seconds=sum(
                        r["cuda_interval_seconds"] for r in timing_records
                    )
                    if events
                    and timing_complete
                    and all(r["cuda_interval_seconds"] is not None for r in timing_records)
                    else None,
                    cuda_timing_coverage="COMPLETE_RETAINED_INTERVALS"
                    if timing_complete
                    else "UNKNOWN_INTERRUPTED_INTERVALS_NOT_FABRICATED",
                    current_invocation_training_interval_seconds=events[0].elapsed_time(events[1])
                    / 1000
                    if events
                    else None,
                    claim_scope="ACTUAL_DEVICE_EXECUTION; CUDA_INTERVAL_INCLUDES_HOST_GAPS_NOT_ACTIVE_KERNEL_TIME",
                )
                summary["host_wall_seconds_by_phase"] = summary.pop("cpu_seconds_by_phase")
                cost[str(seed)][arm] = summary
                results[str(seed)][arm] = dict(
                    initial=initial,
                    final=final,
                    randomized=randomized,
                    oracle=oracle,
                    curve=tr.curve,
                    initial_shared_state_sha256=tr.parent_base_sha,
                    data_rng_sha256=fingerprint(tr.data_rng.bit_generator.state),
                    main_route_rng_sha256=fingerprint(tr.route_rng.get_state()),
                    cells_sha256=fingerprint(tr.model.cells.state_dict()),
                    connection_sha256=fingerprint(tr.model.connection.state_dict()),
                    local_restore_exact=True,
                )
            a = results[str(seed)]["A"]
            q = results[str(seed)]["C"]
            if (
                a["initial_shared_state_sha256"] != q["initial_shared_state_sha256"]
                or a["cells_sha256"] != q["cells_sha256"]
                or a["data_rng_sha256"] != q["data_rng_sha256"]
                or a["main_route_rng_sha256"] != q["main_route_rng_sha256"]
            ):
                raise RuntimeError("JCT_PILOT_BRANCH_EXPOSURE_OR_FROZEN_CELL_MISMATCH")
        seed_summary = []
        for seed, arms in results.items():
            a = arms["A"]["final"]
            q = arms["C"]["final"]
            r = arms["C"]["randomized"]
            criterion = q["accuracy"] > max(a["accuracy"], r["accuracy"]) and q["loss"] < min(
                a["loss"], r["loss"]
            )
            seed_summary.append(
                dict(
                    seed=int(seed),
                    accuracy_delta_C_minus_A=q["accuracy"] - a["accuracy"],
                    loss_delta_C_minus_A=q["loss"] - a["loss"],
                    provisional_routing_criterion=criterion,
                )
            )
        total_calls = sum(
            arm["completed_optimizer_calls"] for seed in cost.values() for arm in seed.values()
        )
        logical = sum(arm["logical_updates"] for seed in cost.values() for arm in seed.values())
        artifacts = {
            "CONFIG.json": dict(engineering_configuration=c),
            "PILOT_CONFIG.json": dict(configuration=p),
            "DATA_REGISTRY.json": dict(
                registry=registry, real_model_data_accessed=False, confirmation_materialized=False
            ),
            "CHECKPOINT_RECEIPTS.json": dict(
                receipts=receipts, cross_session_verified=False, functional_preflight=preflight
            ),
            "ARM_RESULT.json": dict(
                seeds=results,
                arms=["A", "C"],
                B_and_D="NOT_PART_OF_STAGE_1_ROUTING_ONLY",
                scientific_confirmation=False,
            ),
            "ROUTING_EVOLUTION.json": dict(
                seeds={
                    s: {
                        a: dict(initial=r["initial"], final=r["final"], randomized=r["randomized"])
                        for a, r in arms.items()
                    }
                    for s, arms in results.items()
                }
            ),
            "CELL_CONNECTION_ATTRIBUTION.json": dict(
                status="ROUTING_ONLY_CELL_WEIGHTS_FROZEN",
                joint_training_benefit="NOT_TESTED",
                interaction="NOT_TESTED",
            ),
            "COMPUTE_COST.json": dict(
                seeds=cost,
                functional_preflight=preflight,
                total_host_wall_seconds=time.perf_counter() - started,
                gpu_used=device.type == "cuda",
                observed_flops=None,
            ),
            "BUDGET_PROOF.json": dict(
                actual_optimizer_calls=total_calls,
                functional_preflight_actual_optimizer_calls=preflight["actual_optimizer_calls"],
                expected_scientific_optimizer_calls=len(p["seeds"])
                * p["logical_updates_per_arm"]
                * 2,
                logical_updates=logical,
                actual_source_examples_attempted=sum(
                    a["input_examples_attempted"] for seed in cost.values() for a in seed.values()
                ),
                extra_counterfactual_target_evaluations=sum(
                    a["counterfactual_target_evaluations"]
                    for seed in cost.values()
                    for a in seed.values()
                ),
                equal_compute_claim=False,
                training_tokens=None,
                unit="AFFINE_EXAMPLES_NOT_LANGUAGE_TOKENS",
            ),
            "SEED_SUMMARY.json": dict(
                per_seed=seed_summary,
                provisional_all_seeds_criterion=all(
                    r["provisional_routing_criterion"] for r in seed_summary
                ),
                formal_protocol_pilot_pass=False,
                decision="INDEPENDENT_REVIEW_REQUIRED",
                confidence_intervals=None,
                statistical_unit="REGISTERED_TRAINING_SEED",
            ),
            "GATE_STATUS.json": dict(
                synthetic_execution="USER_AUTHORIZED_EXPLORATORY",
                independent_scientific_review="PENDING",
                real_model="BLOCKED",
                confirmation="SEALED",
                cross_session_verified=False,
                scientific_joint_training_benefit="NOT_TESTED",
            ),
        }
        for name, value in artifacts.items():
            atomic(root / name, dict(identity=identity, **value))
        (root / "JCT001_PROTOCOL.md").write_text(
            (BASE / "JCT001_PROTOCOL.md").read_text()
            + "\n\n"
            + (BASE / "synthetic-pilot/AMENDMENT.md").read_text()
        )
        (root / "SCIENTIFIC-FINDINGS.md").write_text(
            "# JCT-001 Stage 1 exploratory result\n\n"
            + json.dumps(seed_summary, indent=2)
            + "\n\nOnly 2 held-out canonical compositions / 4 examples. Seeds do not create additional independent task families. Unequal actual compute is disclosed. This routing-only experiment does not test joint Cell–Connection benefit, release real-model training, establish Confirmation or automatically pass the formal protocol.\n"
        )
        manifest.update(
            run_status="SYNTHETIC_PILOT_COMPLETE_EXPLORATORY_REVIEW_REQUIRED",
            finished_at_unix=time.time(),
            exit_code=0,
            scientific_optimizer_updates=total_calls,
            functional_preflight_optimizer_calls=preflight["actual_optimizer_calls"],
        )
    except Exception as error:
        manifest.update(
            run_status="FAILED_RETAINED_RESUMABLE_IF_LEDGER_KNOWN",
            exit_code=1,
            failure_type=type(error).__name__,
            finished_at_unix=time.time(),
        )
        from .cost import Journal

        retained_attempts = {}
        for ledger in (root / "private").glob("*/*-ledger.jsonl"):
            if ledger.name not in ("A-ledger.jsonl", "C-ledger.jsonl"):
                continue
            header = json.loads(ledger.read_text().splitlines()[0])
            retained_attempts[str(ledger.relative_to(root))] = Journal(
                ledger, header["data"]
            ).summary()
        manifest["scientific_optimizer_updates"] = sum(
            s["completed_optimizer_calls"] for s in retained_attempts.values()
        )
        manifest["uncertain_optimizer_calls"] = sum(
            s["uncertain_optimizer_calls"] for s in retained_attempts.values()
        )
        atomic(
            root / "PARTIAL_RESULTS.json",
            dict(identity=identity, seeds=results, cost=cost, retained_attempts=retained_attempts),
        )
        for path in root.iterdir():
            if (
                path.is_file()
                and path.name in gpu_publication.ALLOWED
                and path.name != "RUN_MANIFEST.json"
            ):
                manifest["artifact_files"][path.name] = dict(
                    bytes=path.stat().st_size, sha256=digest(path)
                )
        atomic(manifest_path, manifest)
        if publish_github:
            try:
                gpu_publication.publish(root)
            except Exception as publication_error:
                print(
                    "[JCT-001 GPU] publication failed; Output retained: "
                    + type(publication_error).__name__,
                    flush=True,
                )
        raise
    for path in root.iterdir():
        if (
            path.is_file()
            and path.name in gpu_publication.ALLOWED
            and path.name != "RUN_MANIFEST.json"
        ):
            manifest["artifact_files"][path.name] = dict(
                bytes=path.stat().st_size, sha256=digest(path)
            )
    atomic(manifest_path, manifest)
    receipt = gpu_publication.publish(root) if publish_github else None
    return dict(
        status=manifest["run_status"],
        run_id=identity["run_id"],
        output=str(root),
        gpu_used=manifest["gpu_used"],
        scientific_optimizer_updates=manifest["scientific_optimizer_updates"],
        publication=receipt,
    )
