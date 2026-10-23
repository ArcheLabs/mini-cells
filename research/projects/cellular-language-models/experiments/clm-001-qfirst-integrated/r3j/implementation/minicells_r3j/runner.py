"""Historical audit -> fixed cases -> paired continuation -> terminal evaluation -> publication."""

import argparse
import copy
import inspect
import json
from pathlib import Path
import time
import uuid
import torch
from minicells_artifacts.hf_storage import HubBackend
from minicells_r3e.common import atomic, read, canonical, digest, source_sha, fingerprint
from minicells_r3e.runner import secrets
from minicells_r3h import runner as coordinator
from minicells_r3h.evaluation import assets
from minicells_r3e.storage import snapshot_lock
from .parent_restore import Parent, validate_pair
from .case_selection import select
from .continuation import trainer, advance
from .evaluation import evaluate
from .storage import Store, save, latest, Incremental, restore_segment
from .reports import report

BASE = Path(__file__).resolve().parents[2]


def runtime(c):
    parent = read(BASE.parent / "r3i/config/frozen.json")
    if canonical(parent) != c["parent"]["config_sha256"]:
        raise RuntimeError("R3I_CONFIG_IDENTITY_MISMATCH")
    return parent


def identity(c):
    x = dict(
        experiment_id=c["experiment_id"],
        source_sha=source_sha(),
        config_sha256=canonical(c),
        parent_r3i_run_id=c["parent"]["run_id"],
        parent_manifest_sha256=c["parent"]["hf_reference"]["manifest_sha256"],
    )
    return dict(x, run_id=canonical(x))


def case_root(output, case):
    return Path(output) / "private-checkpoints/cases" / case["case_id"]


def public_history(c):
    folder = (
        BASE.parent
        / "r3i/reports/published/4ef0eb15e8b3564748d917f19b003d6b5da2e816c6729721296974f28723f973"
    )
    manifest = read(folder / "RUN_MANIFEST.json")
    if any(manifest.get(k) != c["parent"][k] for k in ("source_sha", "config_sha256", "run_id")):
        raise RuntimeError("PUBLIC_R3I_RESULT_IDENTITY_MISMATCH")
    records = read(folder / "PROPOSAL_RESULTS.json")
    declared = {r["path"]: r for r in manifest["artifacts"]}
    if digest(folder / "PROPOSAL_RESULTS.json") != declared["PROPOSAL_RESULTS.json"]["sha256"]:
        raise RuntimeError("PUBLIC_REPORT_HASH_MISMATCH")
    return records


def prepare(c, output, backend):
    private = Path(output) / "private-checkpoints"
    if (private / "CASE_SELECTION_MANIFEST.json").exists() and (
        private / "HISTORICAL_PROPOSAL_AUDIT.json"
    ).exists():
        audit = read(private / "HISTORICAL_PROPOSAL_AUDIT.json")
        selection = read(private / "CASE_SELECTION_MANIFEST.json")
        if select(audit) != selection:
            raise RuntimeError("CACHED_CASE_SELECTION_CHANGED")
        atomic(Path(output) / "reports/HISTORICAL_PROPOSAL_AUDIT.json", audit)
        atomic(Path(output) / "reports/CASE_SELECTION_MANIFEST.json", selection)
        return selection
    parent = Parent(c, backend, Path(output) / "parent-r3i")
    records = []
    failures = []
    for event in public_history(c):
        seed, arm, step = event["seed"], event["arm"], event["at_step"]
        try:
            package, receipt = parent.checkpoint(seed, arm, step, 0, "candidate")
            # Load the K10 receipt metadata now, without its large paired tensor object.
            ident = dict(parent.expected, seed=seed, arm=arm, proposal_at_step=step)
            root = (
                Path("jobs")
                / ("seed-%s-arm-%s" % (seed, arm))
                / "transactions"
                / ("proposal-%04d" % step)
            )
            pair_receipt = read(parent.file(Store(root, ident).receipt_file(11).as_posix()))
            candidate_id = canonical(
                dict(
                    source_run_id=c["parent"]["run_id"],
                    seed=seed,
                    arm=arm,
                    proposal_at_step=step,
                    candidate_fingerprint=package["candidate_fingerprint"],
                    chosen_scale=event["chosen_scale"],
                    transaction_receipt_sha256=pair_receipt["sha256"],
                )
            )
            records.append(
                dict(
                    case_id=candidate_id,
                    seed=seed,
                    arm=arm,
                    proposal_at_step=step,
                    historical_accepted=event["accepted"],
                    trust_region_passed=event["trust_region_passed"],
                    chosen_scale=event["chosen_scale"],
                    candidate_fingerprint=package["candidate_fingerprint"],
                    transaction_receipt_sha256=pair_receipt["sha256"],
                    candidate_receipt_sha256=receipt["sha256"],
                    historical_event=event,
                    connection_version_before=package["official"]["connection_version"],
                    historical_optimizer_losses=package["losses"],
                    counterfactual_state_fingerprint=package["state_sample_fingerprint"],
                    original_value_prediction=None,
                    prediction_status="HISTORICAL_VALUE_PREDICTION_NOT_RECOVERABLE",
                    prediction_note="Archived centered optimization losses and conditional action costs are available; no predeclared comparable long-horizon scalar forecast was stored.",
                    counterfactual_cost_summary={
                        "mean_by_action": torch.stack(
                            [r["values"].detach().cpu() for r in package["sampled_states"]]
                        )
                        .double()
                        .mean(0)
                        .tolist(),
                        "state_count": len(package["sampled_states"]),
                        "source": "DIRECTLY_ARCHIVED_CONDITIONAL_ACTION_COSTS_AGGREGATED_BY_ACTION",
                    },
                )
            )
        except Exception as error:
            failures.append(dict(seed=seed, arm=arm, proposal_at_step=step, message=str(error)))
            records.append(
                dict(
                    case_id=canonical(
                        {
                            "source": c["parent"]["run_id"],
                            "seed": seed,
                            "arm": arm,
                            "step": step,
                            "unrecoverable": True,
                        }
                    ),
                    seed=seed,
                    arm=arm,
                    proposal_at_step=step,
                    historical_accepted=event["accepted"],
                    trust_region_passed=event["trust_region_passed"],
                    chosen_scale=event["chosen_scale"],
                    candidate_fingerprint=None,
                    transaction_receipt_sha256=None,
                    candidate_receipt_sha256=None,
                    historical_event=event,
                    connection_version_before=package["official"]["connection_version"],
                    status="PAIRED_BRANCH_NOT_RECOVERABLE",
                    candidate_id_status="UNVERIFIED_METADATA_ONLY",
                    error=str(error),
                )
            )
    selection = select(records)
    private = Path(output) / "private-checkpoints"
    path = private / "CASE_SELECTION_MANIFEST.json"
    if path.exists() and read(path) != selection:
        raise RuntimeError("IMMUTABLE_CASE_SELECTION_CONFLICT")
    atomic(path, selection)
    atomic(private / "HISTORICAL_PROPOSAL_AUDIT.json", records)
    atomic(Path(output) / "reports/HISTORICAL_PROPOSAL_AUDIT.json", records)
    atomic(Path(output) / "reports/CASE_SELECTION_MANIFEST.json", selection)
    parent.unchanged()
    return selection


def load_pair(c, output, case, device):
    parent = Parent(
        c, HubBackend(c["parent"]["hf_reference"]["hf_repo"]), Path(output) / "parent-r3i"
    )
    tx, receipt = parent.checkpoint(case["seed"], case["arm"], case["proposal_at_step"], 11)
    package, candidate_receipt = parent.checkpoint(
        case["seed"], case["arm"], case["proposal_at_step"], 0, "candidate"
    )
    if (
        receipt["sha256"] != case["transaction_receipt_sha256"]
        or candidate_receipt["sha256"] != case["candidate_receipt_sha256"]
    ):
        raise RuntimeError("CASE_RECEIPT_CHANGED")
    proof = validate_pair(c, tx, package, case["seed"], case["arm"], case["proposal_at_step"])
    committed, commit_receipt = parent.checkpoint(
        case["seed"], case["arm"], case["proposal_at_step"], 21
    )
    if (
        committed["phase"] != "COMMITTED"
        or committed["event"]["accepted"] != case["historical_accepted"]
        or committed["event"]["chosen_scale"] != case["chosen_scale"]
    ):
        raise RuntimeError("HISTORICAL_ACCEPTANCE_CHANGED")
    chosen = tx["new"] if case["historical_accepted"] else tx["old"]
    for key in ("model", "cell_optimizer", "data_cursor", "step", "data_rng", "route_rng"):
        if fingerprint(committed["selected"][key]) != fingerprint(chosen[key]):
            raise RuntimeError("HISTORICAL_SELECTED_BRANCH_MISMATCH")
    proof["committed_receipt_sha256"] = commit_receipt["sha256"]
    old = trainer(runtime(c), tx["old"], case["seed"], case["arm"], device)
    new = trainer(runtime(c), tx["new"], case["seed"], case["arm"], device)
    parent.unchanged()
    return old, new, proof


def pair_identity(c, case, output):
    return dict(
        identity(c),
        case_id=case["case_id"],
        case_selection_manifest_sha256=read(
            Path(output) / "private-checkpoints/CASE_SELECTION_MANIFEST.json"
        )["sha256"],
        historical_accepted=case["historical_accepted"],
        counterfactual_new_branch=True,
    )


def preflight_case(c, output, case, folder, device):
    root = case_root(output, case)
    ident = pair_identity(c, case, output)
    if (root / "READY.json").exists():
        state = latest(root, ident)
        if state is None:
            raise RuntimeError("IMPORTED_CHECKPOINT_MISSING")
        return
    restore_started = time.perf_counter()
    old, new, proof = load_pair(c, output, case, device)
    data = assets(
        runtime(c), HubBackend(c["parent"]["hf_reference"]["hf_repo"]), folder, ("calibration",)
    )
    result = evaluate(old, new, runtime(c), data, "calibration", c["evaluation_repeats"])
    expected = case["historical_event"]["attribution"]["values"]
    errors = {k: abs(result["values"][k] - expected[k]) for k in expected}
    if max(errors.values()) > c["historical_nll_atol"]:
        raise RuntimeError("HISTORICAL_K10_CALIBRATION_NOT_REPRODUCED")
    proof.update(
        calibration_errors=errors,
        restore_and_calibration_seconds=time.perf_counter() - restore_started,
        source_hf_revision=c["parent"]["hf_reference"]["hf_revision"],
        restore_scope="PINNED_HF_TO_NEW_R3J_EXECUTION",
        historical_writer_session=c["parent"]["hf_reference"]["writer_session_id"],
    )
    state = dict(
        phase="IMPORTED_HISTORICAL_K10",
        old=copy.deepcopy(old.state()),
        new=copy.deepcopy(new.state()),
        rows=[],
        horizon=10,
        new_updates_per_branch=0,
    )
    save(root, ident, state, 0)
    atomic(root / "READY.json", proof)
    print("[R3J_SNAPSHOT]", case["case_id"][:12], "IMPORTED", flush=True)


def continuation_case(c, output, case, folder, device):
    root = case_root(output, case)
    ident = pair_identity(c, case, output)
    if (root / "TRAINED.json").exists():
        return
    state = latest(root, ident)
    if state is None:
        raise RuntimeError("HISTORICAL_IMPORT_REQUIRED")
    old = trainer(runtime(c), state["old"], case["seed"], case["arm"], device)
    new = trainer(runtime(c), state["new"], case["seed"], case["arm"], device)
    data = assets(
        runtime(c), HubBackend(c["parent"]["hf_reference"]["hf_repo"]), folder, ("train",)
    )
    base = case["proposal_at_step"] + 10
    imported = Store(root, ident).verified(read(Store(root, ident).receipt_file(0)))
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    rows = list(state["rows"])
    started = time.perf_counter()

    def checkpoint(a, b, row):
        rows.append(row)
        sequence = a.step - base
        payload = dict(
            phase="CONTINUING",
            old=copy.deepcopy(a.state()),
            new=copy.deepcopy(b.state()),
            rows=copy.deepcopy(rows),
            horizon=a.step - case["proposal_at_step"],
            new_updates_per_branch=sequence,
        )
        private = Path(output) / "private-checkpoints"
        with snapshot_lock(private):
            save(root, ident, payload, sequence)
            # Keep horizon states and the newest complete paired update, not 90 duplicate full states.
            keep = {0, 40, 90, sequence}
            for file in (root / "receipts").glob("*.json"):
                previous = read(file)
                if previous["step"] not in keep:
                    file.unlink()
                    obj = root / previous["file"]
                    if obj.exists():
                        obj.unlink()

    for horizon in (50, 100):
        if old.step < case["proposal_at_step"] + horizon:
            advance(old, new, data, case["proposal_at_step"] + horizon, checkpoint)
            print("[R3J_SNAPSHOT]", case["case_id"][:12], "HORIZON", horizon, flush=True)
    if old.step != case["proposal_at_step"] + 100 or state["new_updates_per_branch"] > 90:
        raise RuntimeError("CONTINUATION_BUDGET_INVALID")
    atomic(
        root / "TRAINED.json",
        dict(
            status="COMPLETE",
            case_id=case["case_id"],
            case_selection_manifest_sha256=read(
                Path(output) / "private-checkpoints/CASE_SELECTION_MANIFEST.json"
            )["sha256"],
            updates_per_branch=90,
            new_tokens=2 * 90 * runtime(c)["training"]["tokens_per_step"],
            elapsed_this_session_seconds=time.perf_counter() - started,
            old_new_forward_work={
                branch: {
                    k: tr.model.work[k] - imported[branch]["work"].get(k, 0) for k in tr.model.work
                }
                for branch, tr in [("old", old), ("new", new)]
            },
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device)
            if str(device).startswith("cuda")
            else None,
            work_scope="EXECUTED_OPERATOR_ROWS_NOT_HARDWARE_FLOPS; BOTH_BRANCHES_90_NEW_UPDATES",
        ),
    )


def evaluate_case(c, output, case, folder, device):
    root = case_root(output, case)
    ident = pair_identity(c, case, output)
    backend = HubBackend(c["parent"]["hf_reference"]["hf_repo"])
    data = assets(runtime(c), backend, folder, ("development", "admission"))
    for horizon, sequence in ((10, 0), (50, 40), (100, 90)):
        file = root / ("EVALUATION-%s.json" % horizon)
        if file.exists():
            continue
        receipt = read(Store(root, ident).receipt_file(sequence))
        s = Store(root, ident).verified(receipt)
        old = trainer(runtime(c), s["old"], case["seed"], case["arm"], device)
        new = trainer(runtime(c), s["new"], case["seed"], case["arm"], device)
        result = evaluate(old, new, runtime(c), data, "development", c["evaluation_repeats"])
        from minicells_r3h.alignment import measure

        result["old_alignment"] = measure(old.model, runtime(c), data, old.epsilon())
        result["new_alignment"] = measure(new.model, runtime(c), data, new.epsilon())
        result.update(
            case_id=case["case_id"],
            case_selection_manifest_sha256=read(
                Path(output) / "private-checkpoints/CASE_SELECTION_MANIFEST.json"
            )["sha256"],
            horizon=horizon,
            training_path_rows=[
                r for r in s["rows"] if r["step"] <= case["proposal_at_step"] + horizon
            ],
            checkpoint=receipt,
        )
        atomic(file, result)
    print("[R3J_SNAPSHOT]", case["case_id"][:12], "EVALUATED", flush=True)


def worker(c, output, folder, index, count, phase):
    selection = read(Path(output) / "private-checkpoints/CASE_SELECTION_MANIFEST.json")
    for case in selection["cases"][index::count]:
        if phase != "prepare" and not (case_root(output, case) / "READY.json").exists():
            continue
        try:
            if phase == "prepare":
                preflight_case(c, output, case, folder, "cuda:%s" % index)
            elif phase == "train":
                continuation_case(c, output, case, folder, "cuda:%s" % index)
            else:
                evaluate_case(c, output, case, folder, "cuda:%s" % index)
        except Exception as error:
            atomic(
                case_root(output, case) / "FAILURE.json",
                dict(
                    status="ENGINEERING_FAILURE_NOT_SCIENTIFIC_NEGATIVE",
                    message=str(error),
                    case_id=case["case_id"],
                    case_selection_manifest_sha256=read(
                        Path(output) / "private-checkpoints/CASE_SELECTION_MANIFEST.json"
                    )["sha256"],
                ),
            )
            if phase == "prepare":
                continue
            raise


def automatic(c, path, output, folder, report_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ident = identity(c)
    execution = uuid.uuid4().hex
    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise RuntimeError("OUTPUT_IDENTITY_CHANGED")
    atomic(output / "identity.json", ident)
    atomic(output / "EXECUTION.json", {"execution_id": execution})
    secrets()
    backend = HubBackend(c["parent"]["hf_reference"]["hf_repo"])
    backend.check_write()
    revision = backend.info().sha
    pointer = backend.optional("r3j/" + ident["run_id"] + "/latest.json", revision)
    prior = {**pointer, "hf_repo": backend.repo, "hf_revision": revision} if pointer else None
    private = output / "private-checkpoints"
    if prior and not list(private.rglob("receipts/*.json")):
        restore_segment(backend, prior, private, ident, execution)
    selection = prepare(c, output, backend)
    publisher = Incremental(backend, private, output, ident, execution, prior)
    failures = []
    last = [0.0]

    def callback(force):
        if not force and time.monotonic() - last[0] < 180:
            return
        begin = time.perf_counter()
        print("[R3j] incremental checkpoint publication starting", flush=True)
        with snapshot_lock(private):
            publisher.publish(force)
        last[0] = time.monotonic()
        print(
            "[R3j] checkpoint publication done in %.1fs" % (time.perf_counter() - begin), flush=True
        )

    try:
        if not report_only:
            if torch.cuda.device_count() != 2:
                raise RuntimeError("TWO_GPUS_REQUIRED")
            g = dict(coordinator.workers.__globals__, BASE=BASE)
            exec(
                inspect.getsource(coordinator.workers).replace("[R3H_SNAPSHOT]", "[R3J_SNAPSHOT]"),
                g,
            )
            workers = g["workers"]
            workers(runtime(c), path, output, folder, "prepare", callback)
            valid = [
                case
                for case in selection["cases"]
                if (case_root(output, case) / "READY.json").exists()
            ]
            for phase in ("train", "evaluate"):
                if phase == "evaluate" and any(
                    not (case_root(output, case) / "TRAINED.json").exists() for case in valid
                ):
                    raise RuntimeError("ALL_CONTINUATION_TERMINAL_BARRIER_NOT_REACHED")
                workers(runtime(c), path, output, folder, phase, callback)
        publisher.publish(True)
    except Exception as error:
        failures.append(dict(type=type(error).__name__, message=str(error)))
    result = report(c, output, selection, failures, publisher.prior)
    try:
        from .publication import publish

        publication = publish(output)
    except Exception as error:
        publication = dict(status="LOCAL_RETAINED_PUBLICATION_RETRY_PENDING", message=str(error))
    atomic(output / "PUBLICATION.json", publication)
    print(
        json.dumps(dict(status=result["status"], publication=publication, output=str(output))),
        flush=True,
    )
    if failures:
        raise RuntimeError("R3J_PARTIAL_SEE_RETAINED_REPORTS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--report-existing", action="store_true")
    p.add_argument("--output", default="/kaggle/working/r3j-output")
    p.add_argument("--assets", default="/kaggle/working/r3j-assets")
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--phase", choices=["prepare", "train", "evaluate"])
    a = p.parse_args()
    c = read(a.config)
    if canonical(c) != (BASE / "config/releases.lock").read_text().strip():
        raise RuntimeError("FROZEN_CONFIG_CHANGED")
    if a.worker is not None:
        worker(c, a.output, a.assets, a.worker, a.workers, a.phase)
    elif a.run_all or a.report_existing:
        automatic(c, Path(a.config).resolve(), a.output, a.assets, a.report_existing)
    else:
        p.error("--run-all required")
