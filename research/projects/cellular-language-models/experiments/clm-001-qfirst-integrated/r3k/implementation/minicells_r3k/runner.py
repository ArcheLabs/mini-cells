"""All predictions persist before NEW36 continuation; all models freeze before holdout evaluation."""

import argparse
import copy
import inspect
import time
import uuid
from pathlib import Path
import torch
from minicells_artifacts.hf_storage import HubBackend
from minicells_r3e.common import read, atomic, canonical, fingerprint, source_sha
from minicells_r3e.runner import secrets
from minicells_r3h import runner as coordinator
from minicells_r3j import runner as previous
from minicells_r3j.storage import Store, save
from minicells_r3j.continuation import trainer
from .evaluation import evaluate
from minicells_r3e.storage import snapshot_lock
from .reuse import bind
from .parent_restore import R3jParent
from .storage import Incremental, restore_active
from .prediction_freeze import scores, freeze, require_persisted
from . import data as research

BASE = Path(__file__).resolve().parents[2]
runtime = previous.runtime
case_root = previous.case_root


def identity(c):
    row = dict(
        experiment_id=c["experiment_id"],
        source_sha=source_sha(),
        config_sha256=canonical(c),
        parent_r3i_run_id=c["parent"]["run_id"],
        parent_r3j_run_id=c["parent_r3j_run"],
    )
    return dict(row, run_id=canonical(row))


def pair_identity(c, case, output):
    registry = read(Path(output) / "private-checkpoints/CASE_SELECTION_MANIFEST.json")
    return dict(
        identity(c),
        case_id=case["case_id"],
        case_selection_manifest_sha256=registry["sha256"],
        historical_accepted=case["historical_accepted"],
        counterfactual_new_branch=True,
    )


continuation_case = bind(previous.continuation_case, identity=identity, pair_identity=pair_identity)


def registry(c, output, backend):
    output = Path(output)
    private = output / "private-checkpoints"
    path = private / "CASE_SELECTION_MANIFEST.json"
    if path.exists():
        saved = read(path)
        check = dict(saved)
        sha = check.pop("sha256")
        if canonical(check) != sha or len(saved["cases"]) != 48:
            raise RuntimeError("REGISTRY_CHANGED")
        return saved
    # Historical R3j prepare audits all 48 without reading its future Development values.
    previous.prepare(c, output / "r3i-audit", backend)
    candidates = read(output / "r3i-audit/private-checkpoints/HISTORICAL_PROPOSAL_AUDIT.json")
    parent = R3jParent(c, backend, output / "parent-r3j")
    prior = {x["case_id"] for x in parent.selection()["cases"]}
    cases = []
    for case in candidates:
        if case.get("transaction_receipt_sha256") is None:
            raise RuntimeError("HISTORICAL_CANDIDATE_NOT_RECOVERABLE")
        cases.append(dict(case, reused_r3j=case["case_id"] in prior))
    if (
        len(cases) != 48
        or sum(x["reused_r3j"] for x in cases) != 12
        or any(x["historical_accepted"] for x in cases if not x["reused_r3j"])
    ):
        raise RuntimeError("R3K_FIXED_POPULATION_CHANGED")
    row = dict(
        cases=cases,
        selection="ALL48_NO_OUTCOME_SELECTION",
        r3j_manifest_sha256=c["parent_r3j_manifest_sha256"],
    )
    row["sha256"] = canonical(row)
    atomic(path, row)
    return row


def predict_case(c, output, case, folder, device):
    root = case_root(output, case)
    private = Path(output) / "private-checkpoints"
    if (root / "PREDICTION.json").exists():
        return
    start = time.perf_counter()
    old, new, proof = previous.load_pair(c, output, case, device)
    restore_seconds = time.perf_counter() - start
    backend = HubBackend(c["parent"]["hf_reference"]["hf_repo"])
    probe = research.load(c, folder, "predictor_probe")
    k10 = evaluate(old, new, runtime(c), probe, "development", 2)
    parent = previous.Parent(c, backend, Path(output) / "parent-r3i")
    package, _ = parent.checkpoint(
        case["seed"], case["arm"], case["proposal_at_step"], 0, "candidate"
    )
    initial_old = trainer(runtime(c), package["official"], case["seed"], case["arm"], device)
    initial_new = trainer(runtime(c), package["official"], case["seed"], case["arm"], device)
    initial_new.model.connection.load_state_dict(new.model.connection.state_dict())
    direct = evaluate(initial_old, initial_new, runtime(c), probe, "development", 2)
    historical = case["historical_event"]["attribution"]["values"]
    result = dict(
        case_id=case["case_id"],
        seed=case["seed"],
        arm=case["arm"],
        proposal_at_step=case["proposal_at_step"],
        scores=scores(historical, direct["values"], k10["values"]),
        numerical_error_bound=4
        * max(
            direct["numerical_resolution"],
            k10["numerical_resolution"],
            case["historical_event"]["attribution"].get("numerical_resolution", 0),
        ),
        measured_repeat_error=max(direct["measured_repeat_error"], k10["measured_repeat_error"]),
        seconds=time.perf_counter() - start,
        restore_seconds=restore_seconds,
        k10_evaluation_seconds=k10["evaluation_seconds"],
        predictor_operator_rows={
            "direct": direct["executed_operator_rows"],
            "k10": k10["executed_operator_rows"],
        },
        direct_evaluation_seconds=direct["evaluation_seconds"],
        cell_parameter_difference_l2=k10["cell_parameter_difference_l2"],
        historical_mechanism={
            "policy": case["historical_event"].get("actual_training_policy_diagnostic"),
            "chosen_scale": case["chosen_scale"],
            "chosen_scan": [
                r
                for r in case["historical_event"].get("scan", {}).get("attempts", [])
                if r.get("scale") == case["chosen_scale"]
            ],
            "K10_training_path_rows": case["historical_event"].get("training_path_rows", []),
            "full_scale_scan_source": "HISTORICAL_CANDIDATES.json",
        },
        precision="FP32",
        historical_accepted=case["historical_accepted"],
        reused_r3j=case["reused_r3j"],
        data_identity=canonical(probe["manifest"]),
    )
    if case["reused_r3j"]:
        R3jParent(c, backend, Path(output) / "parent-r3j").verify_k10(case, old, new)
    state = dict(
        phase="IMPORTED_HISTORICAL_K10",
        old=copy.deepcopy(old.state()),
        new=copy.deepcopy(new.state()),
        rows=[],
        horizon=10,
        new_updates_per_branch=0,
    )
    with snapshot_lock(private):
        save(root, pair_identity(c, case, output), state, 0)
    atomic(root / "READY.json", proof)
    result["seconds"] = time.perf_counter() - start
    atomic(root / "PREDICTION.json", result)
    print("[R3K_SNAPSHOT]", case["case_id"][:12], "PREDICTED", flush=True)


def import_r3j_case(c, output, case, device):
    require_barrier(c, output)
    root = case_root(output, case)
    if (root / "TRAINED.json").exists():
        return
    private = Path(output) / "private-checkpoints"
    initial = Store(root, pair_identity(c, case, output)).verified(
        read(Store(root, pair_identity(c, case, output)).receipt_file(0))
    )
    old = trainer(runtime(c), initial["old"], case["seed"], case["arm"], device)
    new = trainer(runtime(c), initial["new"], case["seed"], case["arm"], device)
    backend = HubBackend(c["parent"]["hf_reference"]["hf_repo"])
    r3j = R3jParent(c, backend, Path(output) / "parent-r3j")
    r3j.verify_k10(case, old, new)
    for sequence in (40, 90):
        state, receipt = r3j.checkpoint(case, sequence)
        # Connections remain exactly the historical pair at all reused horizons.
        for branch, tr in [("old", old), ("new", new)]:
            expected = {k: v for k, v in tr.state()["model"].items() if k.startswith("connection.")}
            actual = {
                k: v for k, v in state[branch]["model"].items() if k.startswith("connection.")
            }
            if fingerprint(expected) != fingerprint(actual):
                raise RuntimeError("R3J_CONNECTION_CHANGED")
        with snapshot_lock(private):
            save(root, pair_identity(c, case, output), state, sequence)
    atomic(
        root / "TRAINED.json",
        dict(
            case_id=case["case_id"],
            updates_per_branch=0,
            new_tokens=0,
            status="REUSED_R3J_NO_TRAINING",
            source_reference=c["r3j_reference"],
        ),
    )


def evaluate_case(c, output, case, folder, device):
    private = Path(output) / "private-checkpoints"
    require_barrier(c, output)
    registry_rows = read(private / "CASE_SELECTION_MANIFEST.json")["cases"]
    if not all((case_root(output, x) / "TRAINED.json").exists() for x in registry_rows):
        raise RuntimeError("ALL48_TRAINING_BARRIER_REQUIRED")
    holdout = research.load(c, folder, "value_holdout", True)
    root = case_root(output, case)
    ident = pair_identity(c, case, output)
    for horizon, sequence in ((10, 0), (50, 40), (100, 90)):
        path = root / ("EVALUATION-%s.json" % horizon)
        if path.exists():
            continue
        receipt = read(Store(root, ident).receipt_file(sequence))
        state = Store(root, ident).verified(receipt)
        old = trainer(runtime(c), state["old"], case["seed"], case["arm"], device)
        new = trainer(runtime(c), state["new"], case["seed"], case["arm"], device)
        optimizers = fingerprint([old.cell_opt.state_dict(), new.cell_opt.state_dict()])
        result = evaluate(old, new, runtime(c), holdout, "development", 2)
        if optimizers != fingerprint([old.cell_opt.state_dict(), new.cell_opt.state_dict()]):
            raise RuntimeError("EVALUATION_CHANGED_OPTIMIZER")
        result.update(
            case_id=case["case_id"],
            horizon=horizon,
            role="R3K_VALUE_HOLDOUT",
            checkpoint=receipt,
            data_identity=canonical(holdout["manifest"]),
        )
        atomic(path, result)
    print("[R3K_SNAPSHOT]", case["case_id"][:12], "EVALUATED", flush=True)


def require_barrier(c, output):
    private = Path(output) / "private-checkpoints"
    row = read(private / "FROZEN_PREDICTIONS.json")
    proof = read(private / "PREDICTION_PERSISTENCE.json")
    if require_persisted(row, row) != proof["prediction_sha256"]:
        raise RuntimeError("PREDICTION_PERSISTENCE_CHANGED")
    return row


def worker(c, output, folder, index, count, phase):
    cases = read(Path(output) / "private-checkpoints/CASE_SELECTION_MANIFEST.json")["cases"]
    for case in cases[index::count]:
        if phase == "predict":
            predict_case(c, output, case, folder, "cuda:%s" % index)
        elif phase == "train":
            require_barrier(c, output)
            if case["reused_r3j"]:
                import_r3j_case(c, output, case, "cuda:%s" % index)
            else:
                continuation_case(c, output, case, folder / "historical", "cuda:%s" % index)
        else:
            evaluate_case(c, output, case, folder, "cuda:%s" % index)


def automatic(c, path, output, folder, report_only=False):
    output = Path(output)
    folder = Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    ident = identity(c)
    session = uuid.uuid4().hex
    private = output / "private-checkpoints"
    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise RuntimeError("OUTPUT_IDENTITY_CHANGED")
    atomic(output / "identity.json", ident)
    secrets()
    backend = HubBackend(c["parent"]["hf_reference"]["hf_repo"])
    backend.check_write()
    revision = backend.info().sha
    pointer = backend.optional("r3k/" + ident["run_id"] + "/latest.json", revision)
    prior = dict(pointer, hf_repo=backend.repo, hf_revision=revision) if pointer else None
    if prior:
        restore_active(backend, prior, private, output, ident, session)
    selection = registry(c, output, backend)
    publisher = Incremental(backend, private, output, ident, session, prior)
    last = [0.0]

    def callback(force):
        if not force and time.monotonic() - last[0] < 180:
            return
        print("[R3k] publishing incremental checkpoint", flush=True)
        with snapshot_lock(private):
            publisher.publish(force)
        last[0] = time.monotonic()

    failures = []
    try:
        if not report_only:
            if torch.cuda.device_count() != 2:
                raise RuntimeError("TWO_GPUS_REQUIRED")
            research.prepare(c, runtime(c), backend, folder)
            atomic(
                output / "DATA_BOUNDARY_PROOF.json",
                read(folder / "research/DATA_BOUNDARY_PROOF.json"),
            )
            # Coordinator invokes this module's CLI; no CPU test suite in Kaggle.
            namespace = dict(coordinator.workers.__globals__, BASE=BASE)
            exec(
                inspect.getsource(coordinator.workers).replace(
                    'line.startswith("[R3H_SNAPSHOT]")',
                    'line.startswith(("[R3K_SNAPSHOT]", "[R3J_SNAPSHOT]"))',
                ),
                namespace,
            )
            workers = namespace["workers"]
            workers(runtime(c), path, output, folder, "predict", callback)
            predictions = [
                read(case_root(output, x) / "PREDICTION.json") for x in selection["cases"]
            ]
            frozen = freeze(
                private / "FROZEN_PREDICTIONS.json",
                predictions,
                read(folder / "research/manifest.json"),
                ident,
                protocol=c,
            )
            callback(True)
            ref = publisher.prior
            from minicells_r3c_moe_bridge.development import verified_manifest

            manifest = verified_manifest(backend, ref, ident)
            row = manifest["files"]["FROZEN_PREDICTIONS.json"]
            remote = read(backend.get(row["object"], ref["hf_revision"]))
            sha = require_persisted(frozen, remote)
            if (private / "PREDICTION_PERSISTENCE.json").exists():
                require_barrier(c, output)
            else:
                atomic(
                    private / "PREDICTION_PERSISTENCE.json",
                    dict(prediction_sha256=sha, hf_reference=ref),
                )
            callback(True)
            workers(runtime(c), path, output, folder, "train", callback)
            workers(runtime(c), path, output, folder, "evaluate", callback)
        callback(True)
    except Exception as error:
        failures.append(
            dict(
                code=str(error) if isinstance(error, RuntimeError) else type(error).__name__,
                type=type(error).__name__,
            )
        )
        try:
            callback(True)
        except Exception as storage_error:
            failures.append(
                dict(code="CHECKPOINT_PUBLICATION_FAILED", type=type(storage_error).__name__)
            )
    from .reports import report

    result = report(c, output, selection, failures, publisher.prior)
    try:
        from .publication import publish

        result["publication"] = publish(output)
    except Exception as error:
        result["publication"] = dict(status="LOCAL_RETAINED", type=type(error).__name__)
    print(result, flush=True)
    if failures or result["status"] != "COMPLETED_EXPLORATORY":
        raise RuntimeError("R3K_PARTIAL_SEE_RETAINED_REPORTS")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--report-existing", action="store_true")
    p.add_argument("--worker", type=int)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--phase")
    a = p.parse_args()
    c = read(a.config)
    if canonical(c) != (BASE / "config/releases.lock").read_text().strip():
        raise RuntimeError("FROZEN_CONFIGURATION_CHANGED")
    if a.worker is not None:
        secrets()
        worker(c, a.output, a.assets, a.worker, a.workers, a.phase)
    else:
        automatic(c, Path(a.config).resolve(), a.output, a.assets, a.report_existing)


if __name__ == "__main__":
    main()
