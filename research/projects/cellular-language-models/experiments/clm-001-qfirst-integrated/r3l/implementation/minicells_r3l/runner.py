"""Fixed candidate generation, independent selection, frozen Connection continuation and reports."""

import argparse
import copy
import inspect
import time
import uuid
from pathlib import Path
import torch
from functools import lru_cache
from minicells_artifacts.hf_storage import HubBackend
from minicells_r3e.common import read, atomic, canonical, fingerprint, source_sha
from minicells_r3e.runner import secrets
from minicells_r3h import runner as coordinator
from minicells_r3h.evaluation import assets
from minicells_r3j.parent_restore import Parent
from minicells_r3j.runner import runtime, public_history
from minicells_r3j.storage import Store, save
from minicells_r3k.evaluation import evaluate
from minicells_r3k.reuse import bind
from minicells_r3k.storage import Incremental as PreviousPublisher
from minicells_r3j.storage import publish_segment
from minicells_r3f_v2.scale_scan import scan as original_scan
from minicells_r3e.storage import snapshot_lock
from .continuation import restore
from .candidate_directions import directions
from .selection import shortlist, finalists
from . import data
from .evaluation import measured
from .storage import restore_active, recover_latest

latest = recover_latest

BASE = Path(__file__).resolve().parents[2]


def locked_scan_save(root, *args, **kwargs):
    private = next(p for p in Path(root).parents if p.name == "private-checkpoints")
    with snapshot_lock(private):
        return save(root, *args, **kwargs)


scan = bind(original_scan, save=locked_scan_save, latest=recover_latest)


@lru_cache(maxsize=1)
def implementation_source():
    return source_sha()


def identity(c):
    row = dict(
        experiment_id=c["experiment_id"],
        source_sha=implementation_source(),
        config_sha256=canonical(c),
    )
    return dict(row, run_id=canonical(row))


def publish_segment_r3l(*args, **kwargs):
    kwargs["pointer_path"] = kwargs["pointer_path"].replace("r3j/", "r3l/", 1)
    return publish_segment(*args, **kwargs)


class Publisher(PreviousPublisher):
    publish = bind(PreviousPublisher.publish, publish_segment=publish_segment_r3l)


def root_for(output, seed, arm):
    return Path(output) / "private-checkpoints/anchors" / f"{seed}-{arm}"


def anchor(c, output, seed, arm, device):
    parent = Parent(
        c, HubBackend(c["parent"]["hf_reference"]["hf_repo"]), Path(output) / "parent-r3i"
    )
    package, receipt = parent.checkpoint(seed, arm, 300, 0, "candidate")
    state = package["official"]
    if (
        state["step"] != 300
        or state["data_cursor"] != 300 * 512
        or fingerprint(package["original_connection"]) != package["original_fingerprint"]
        or any(
            not torch.equal(state["model"]["connection." + k], v)
            for k, v in package["original_connection"].items()
        )
    ):
        raise RuntimeError("NOT_THE_PRE_PROPOSAL_300_OFFICIAL_STATE")
    tr = restore(runtime(c), state, seed, arm, device)
    return tr, package, receipt


def immut_json(path, row):
    if Path(path).exists() and read(path) != row:
        raise RuntimeError("IMMUTABLE_METADATA_CHANGED")
    atomic(path, row)


def candidate_state(original, direction, scale):
    result = copy.deepcopy(original)
    for key, value in original.items():
        result[key] = (
            value + scale * direction[key].to(value) if value.is_floating_point() else value.clone()
        )
    return result


def generate(c, output, seed, arm, folder, device):
    root = root_for(output, seed, arm)
    if (root / "CANDIDATES.json").exists():
        return
    tr, package, receipt = anchor(c, output, seed, arm, device)
    backend = HubBackend(c["parent"]["hf_reference"]["hf_repo"])
    admission = assets(runtime(c), backend, folder / "historical", ("admission",))
    original = copy.deepcopy(tr.model.connection.state_dict())
    rows = []
    ident = dict(identity(c), seed=seed, arm=arm, role="PRIVATE_DIAGNOSTIC_NOT_R3I_COMMIT")
    with snapshot_lock(Path(output) / "private-checkpoints"):
        save(root / "anchor", dict(ident, kind="historical_official_state"), tr.state(), 0)
    historical = next(
        x
        for x in public_history(c)
        if x["seed"] == seed and x["arm"] == arm and x["at_step"] == 300
    )
    definitions = [
        dict(
            family="BASELINE",
            index=0,
            seed=None,
            direction={k: torch.zeros_like(v) for k, v in original.items()},
        ),
        dict(family="HISTORICAL", index=0, seed=None, direction=package["parameter_delta"]),
        dict(
            family="REVERSE",
            index=0,
            seed=None,
            direction={k: -v for k, v in package["parameter_delta"].items()},
        ),
    ]
    definitions += directions(
        tr.model.connection, arm[0], c["direction_seed"] + seed, c["direction_rms"]
    )
    direction_manifest = [
        dict(
            family=x["family"],
            index=x["index"],
            seed=x["seed"],
            direction_hash=fingerprint(x["direction"]),
        )
        for x in definitions
    ]
    immut_json(root / "DIRECTIONS.json", direction_manifest)
    for item in definitions:
        key = canonical(
            dict(
                anchor=ident,
                **{k: item[k] for k in ("family", "index", "seed")},
                direction=fingerprint(item["direction"]),
            )
        )
        family = item["family"]
        proposal = copy.deepcopy(package)
        proposal["parameter_delta"] = item["direction"]
        proposal["candidate_connection"] = candidate_state(original, item["direction"], 1.0)
        if family == "BASELINE":
            connection = original
            trace = {"chosen_scale": 0.0, "attempts": []}
            valid = True
        else:
            trace = scan(
                tr, admission, proposal, root / "scan" / key, dict(ident, candidate_id=key)
            )
            scale = historical["chosen_scale"] if family == "HISTORICAL" else trace["chosen_scale"]
            chosen = next(
                (r for r in trace["attempts"] if r["scale"] == scale and r["passed"]), None
            )
            if family == "HISTORICAL" and (scale != trace["chosen_scale"] or chosen is None):
                raise RuntimeError("HISTORICAL_SCALE_NOT_REPRODUCED")
            valid = scale is not None and scale > 0 and chosen is not None
            connection = candidate_state(original, item["direction"], scale) if valid else original
        with snapshot_lock(Path(output) / "private-checkpoints"):
            scan_root = root / "scan" / key
            if (scan_root / "latest.json").exists():
                keep = read(scan_root / "latest.json")["step"]
                for path in (scan_root / "receipts").glob("*.json"):
                    old = read(path)
                    if old["step"] != keep:
                        path.unlink()
                        obj = scan_root / old["file"]
                        if obj.exists():
                            obj.unlink()
            payload = dict(
                connection=connection,
                direction=item["direction"],
                original=original,
                identity=ident,
            )
            save(
                root / "candidates" / key,
                dict(ident, candidate_id=key, kind="experimental_connection_candidate"),
                payload,
                0,
            )
        rows.append(
            dict(
                candidate_id=key,
                family=family,
                index=item["index"],
                seed=item["seed"],
                valid=valid,
                parameter_sha256=fingerprint(connection),
                direction_sha256=fingerprint(item["direction"]),
                scale=trace["chosen_scale"],
                trust=trace,
                anchor_receipt=receipt,
                historical_candidate_fingerprint=package.get("candidate_fingerprint")
                if family == "HISTORICAL"
                else None,
                status="VALID" if valid else "NO_FEASIBLE_NONZERO_UPDATE",
            )
        )
        print("[R3L_SNAPSHOT]", seed, arm, family, item["index"], "CANDIDATE_SAVED", flush=True)
    if len(rows) != 15:
        raise RuntimeError("FIXED_15_CANDIDATES_REQUIRED")
    atomic(root / "CANDIDATES.json", rows)
    print("[R3L_SNAPSHOT]", seed, arm, "CANDIDATES_FROZEN", flush=True)


def connection_for(c, output, seed, arm, key):
    root = root_for(output, seed, arm) / "candidates" / key
    ident = dict(
        identity(c),
        seed=seed,
        arm=arm,
        role="PRIVATE_DIAGNOSTIC_NOT_R3I_COMMIT",
        candidate_id=key,
        kind="experimental_connection_candidate",
    )
    return latest(root, ident)["connection"]


def selection(c, output, seed, arm, folder, device):
    root = root_for(output, seed, arm)
    if (root / "FINALISTS.json").exists():
        return
    rows = read(root / "CANDIDATES.json")
    tr, package, _ = anchor(c, output, seed, arm, device)
    probe = data.load(c, folder, "search")
    results = {}
    details = {}
    baseline_quality, baseline_paths = measured(copy.deepcopy(tr.model), runtime(c), probe)
    for row in rows:
        if row["family"] == "BASELINE" or not row["valid"]:
            continue
        m = copy.deepcopy(tr.model)
        m.connection.load_state_dict(connection_for(c, output, seed, arm, row["candidate_id"]))
        begin = time.perf_counter()
        details[row["candidate_id"]], _ = measured(m, runtime(c), probe, baseline_paths)
        details[row["candidate_id"]].update(seconds=time.perf_counter() - begin)
        details[row["candidate_id"]]["reorganization_status"] = (
            "EFFECTIVE_REORGANIZATION"
            if details[row["candidate_id"]].get("effective_reorganization")
            else "PARAMETER_CHANGE_WITHOUT_EFFECTIVE_REORGANIZATION"
        )
        results[row["candidate_id"]] = details[row["candidate_id"]]["nll"]
    atomic(root / "SEARCH.json", dict(candidates=details, baseline=baseline_quality))
    top = shortlist(rows, results)
    immut_json(root / "SHORTLIST.json", top)
    select_data = data.load(c, folder, "select", True)
    scores = {}
    details = {}
    for key in top:
        m = copy.deepcopy(tr.model)
        m.connection.load_state_dict(connection_for(c, output, seed, arm, key))
        details[key], _ = measured(m, runtime(c), select_data)
        scores[key] = details[key]["nll"]
    atomic(root / "SELECT.json", details)
    immut_json(root / "FINALISTS.json", finalists(rows, results, scores))
    print("[R3L_SNAPSHOT]", seed, arm, "FINALISTS_FROZEN", flush=True)


def branch_identity(c, seed, arm, sha):
    return dict(
        identity(c), seed=seed, arm=arm, parameter_sha256=sha, kind="counterfactual_training_branch"
    )


def train(c, output, seed, arm, folder, device):
    root = root_for(output, seed, arm)
    if (root / "TRAINED.json").exists():
        return
    final = read(root / "FINALISTS.json")
    rows = {r["candidate_id"]: r for r in read(root / "CANDIDATES.json")}
    base, package, _ = anchor(c, output, seed, arm, device)
    physical = {}
    for key in final["roles"].values():
        if key:
            physical.setdefault(rows[key]["parameter_sha256"], key)
    trainers = {}
    initial = {}
    for sha, key in physical.items():
        directory = root / "branches" / sha
        ident = branch_identity(c, seed, arm, sha)
        saved = latest(directory, ident)
        tr = restore(
            runtime(c),
            saved if saved else package["official"],
            seed,
            arm,
            device,
            None if saved else connection_for(c, output, seed, arm, key),
        )
        initial[sha] = tr.step
        if saved is None:
            save(directory, ident, tr.state(), 0)
        trainers[sha] = tr
    training = assets(
        runtime(c),
        HubBackend(c["parent"]["hf_reference"]["hf_repo"]),
        folder / "historical",
        ("train",),
    )
    from minicells_r3i.adaptation import observe_update, paired

    while min(t.step for t in trainers.values()) < 400:
        step = min(t.step for t in trainers.values())
        active = [(sha, tr) for sha, tr in trainers.items() if tr.step == step]
        inputs = []
        for sha, tr in active:
            row = observe_update(tr, training)
            inputs.append(row["inputs"])
            tr.verify_frozen()
            directory = root / "branches" / sha
            ident = branch_identity(c, seed, arm, sha)
            with snapshot_lock(Path(output) / "private-checkpoints"):
                save(directory, ident, tr.state(), tr.step - 300)
                for path in (directory / "receipts").glob("*.json"):
                    old = read(path)
                    if old["step"] not in {0, 10, 50, 100, tr.step - 300}:
                        path.unlink()
                        obj = directory / old["file"]
                        if obj.exists():
                            obj.unlink()
        if any(x != inputs[0] for x in inputs):
            raise RuntimeError("PAIR_TRAIN_TOKENS_CHANGED")
        if len(active) == len(trainers):
            for tr in list(trainers.values())[1:]:
                paired(next(iter(trainers.values())), tr)
        if step + 1 in (310, 350, 400):
            print("[R3L_SNAPSHOT]", seed, arm, "HORIZON", step + 1 - 300, flush=True)
    initial_work = {}
    for sha in trainers:
        directory = root / "branches" / sha
        ident = branch_identity(c, seed, arm, sha)
        initial_work[sha] = Store(directory, ident).verified(
            read(Store(directory, ident).receipt_file(0))
        )["work"]
    atomic(
        root / "TRAINED.json",
        dict(
            physical_branches=len(trainers),
            logical_roles=final["roles"],
            new_updates=100 * len(trainers),
            new_tokens=runtime(c)["training"]["tokens_per_step"] * 100 * len(trainers),
            completed_step=400,
            resumed_from=initial,
            operator_rows={
                sha: {k: v - initial_work[sha].get(k, 0) for k, v in tr.model.work.items()}
                for sha, tr in trainers.items()
            },
        ),
    )


def evaluation_barrier(c, output):
    if not all(
        (root_for(output, seed, arm) / "TRAINED.json").exists()
        for seed in c["seeds"]
        for arm in c["arms"]
    ):
        raise RuntimeError("ALL_ANCHOR_TRAINING_BARRIER_REQUIRED")


def evaluate_anchor(c, output, seed, arm, folder, device):
    evaluation_barrier(c, output)
    root = root_for(output, seed, arm)
    if (root / "EVALUATION.json").exists():
        return
    final = read(root / "FINALISTS.json")
    rows = {r["candidate_id"]: r for r in read(root / "CANDIDATES.json")}
    report = data.load(c, folder, "report", True)
    results = []
    for horizon in c["horizons"]:
        restored = {}
        receipts = {}
        for key in set(x for x in final["roles"].values() if x):
            sha = rows[key]["parameter_sha256"]
            directory = root / "branches" / sha
            ident = branch_identity(c, seed, arm, sha)
            receipt = read(Store(directory, ident).receipt_file(horizon))
            state = Store(directory, ident).verified(receipt)
            receipts[key] = receipt
            restored[key] = restore(runtime(c), state, seed, arm, device)
        old = restored[final["roles"]["BASELINE"]]
        for role, key in final["roles"].items():
            if key is None:
                continue
            optimizer_before = fingerprint(
                [old.cell_opt.state_dict(), restored[key].cell_opt.state_dict()]
            )
            result = evaluate(old, restored[key], runtime(c), report, "development", 2)
            if optimizer_before != fingerprint(
                [old.cell_opt.state_dict(), restored[key].cell_opt.state_dict()]
            ):
                raise RuntimeError("EVALUATION_CHANGED_OPTIMIZER")
            result.update(
                role=role,
                data_role="R3L_REPORT",
                data_identity=canonical(report.get("manifest", {"scope": "SYNTHETIC_ONLY"})),
                horizon=horizon,
                candidate_id=key,
                parameter_sha256=rows[key]["parameter_sha256"],
                checkpoint={
                    "baseline": receipts[final["roles"]["BASELINE"]],
                    "candidate": receipts[key],
                },
            )
            results.append(result)
    atomic(root / "EVALUATION.json", results)
    print("[R3L_SNAPSHOT]", seed, arm, "EVALUATED", flush=True)


def worker(c, output, folder, index, count, phase):
    jobs = [(s, a) for s in c["seeds"] for a in c["arms"]]
    for seed, arm in jobs[index::count]:
        if phase == "generate":
            generate(c, output, seed, arm, folder, "cuda:%s" % index)
        elif phase == "select":
            selection(c, output, seed, arm, folder, "cuda:%s" % index)
        elif phase == "train":
            train(c, output, seed, arm, folder, "cuda:%s" % index)
        elif phase == "evaluate":
            evaluate_anchor(c, output, seed, arm, folder, "cuda:%s" % index)
        elif phase == "oracle" and seed == c["oracle_seed"]:
            root = root_for(output, seed, arm)
            if (root / "ORACLE.json").exists():
                continue
            from .path_oracle import run

            tr, _, _ = anchor(c, output, seed, arm, "cuda:%s" % index)
            atomic(
                root / "ORACLE.json",
                run(
                    tr.model,
                    runtime(c),
                    data.load(c, folder, "path_diagnostic"),
                    c["oracle_tokens"],
                    c["beam_width"],
                ),
            )


def automatic(c, path, output, folder, report_only=False):
    output = Path(output)
    folder = Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    ident = identity(c)
    session = uuid.uuid4().hex
    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise RuntimeError("OUTPUT_IDENTITY_CHANGED")
    atomic(output / "identity.json", ident)
    secrets()
    backend = HubBackend(c["parent"]["hf_reference"]["hf_repo"])
    backend.check_write()
    revision = backend.info().sha
    pointer = backend.optional("r3l/" + ident["run_id"] + "/latest.json", revision)
    prior = dict(pointer, hf_repo=backend.repo, hf_revision=revision) if pointer else None
    private = output / "private-checkpoints"
    if prior:
        restore_active(backend, prior, private, output, ident, session)
    publisher = Publisher(backend, private, output, ident, session, prior)
    last = [0.0]
    failures = []

    def callback(force):
        if not force and time.monotonic() - last[0] < 180:
            return
        print("[R3l] incremental publication", flush=True)
        with snapshot_lock(private):
            publisher.publish(force)
        last[0] = time.monotonic()

    try:
        if not report_only:
            if torch.cuda.device_count() != 2:
                raise RuntimeError("TWO_GPUS_REQUIRED")
            from .input_verification import verify

            verify(c, output)
            data.prepare(c, runtime(c), backend, folder)
            atomic(
                output / "DATA_BOUNDARY_PROOF.json",
                read(folder / "research/DATA_BOUNDARY_PROOF.json"),
            )
            namespace = dict(coordinator.workers.__globals__, BASE=BASE)
            exec(
                inspect.getsource(coordinator.workers).replace("[R3H_SNAPSHOT]", "[R3L_SNAPSHOT]"),
                namespace,
            )
            for phase in ("generate", "select", "train", "evaluate", "oracle"):
                if phase == "oracle" and not c["allow_oracle"]:
                    continue
                if phase in ("train", "evaluate"):
                    leaf = "FINALISTS.json" if phase == "train" else "TRAINED.json"
                    if not all(
                        (root_for(output, s, a) / leaf).exists()
                        for s in c["seeds"]
                        for a in c["arms"]
                    ):
                        raise RuntimeError("PHASE_BARRIER_INCOMPLETE")
                namespace["workers"](runtime(c), path, output, folder, phase, callback)
        callback(True)
    except Exception as error:
        failures.append(dict(code=type(error).__name__, message=str(error)[:120]))
    from .reports import report

    result = report(c, output, failures, publisher.prior)
    try:
        from .publication import publish

        result["publication"] = publish(output)
    except Exception as error:
        result["publication"] = dict(status="LOCAL_RETAINED", type=type(error).__name__)
    print(result, flush=True)
    if failures or result["status"] != "COMPLETED_EXPLORATORY":
        raise RuntimeError("R3L_PARTIAL_SEE_RETAINED_REPORTS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--phase")
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--report-existing", action="store_true")
    a = p.parse_args()
    c = read(a.config)
    if canonical(c) != (BASE / "config/releases.lock").read_text().strip():
        raise RuntimeError("CONFIGURATION_CHANGED")
    if a.worker is not None:
        secrets()
        worker(c, a.output, a.assets, a.worker, a.workers, a.phase)
    else:
        automatic(c, Path(a.config).resolve(), a.output, a.assets, a.report_existing)
