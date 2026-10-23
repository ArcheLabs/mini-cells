"""Single-launch historical-control coadaptation matrix, with an all-training terminal barrier."""

import argparse
import copy
import json
import os
from pathlib import Path
import time
import uuid
import tempfile
import shutil
import inspect
import torch
from minicells_artifacts.hf_storage import HubBackend
from minicells_r3e.common import read, atomic, canonical, fingerprint, source_sha, seed_all
from minicells_r3e.runner import secrets
from minicells_r3e.storage import Store, save, restore_parents, parent_state
from minicells_r3c_moe_bridge.checkpoint import publish_segment, restore_segment
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3h import runner as original_runner
from minicells_r3h.routing import Model
from minicells_r3h.evaluation import assets, quality
from minicells_r3h.alignment import measure
from .baseline_reference import restore as restore_baselines, compare_step100
from .trainer import Trainer
from .adaptation import execute
from .reports import report

BASE = Path(__file__).resolve().parents[2]


def identity(c):
    value = dict(
        experiment_id=c["experiment_id"], source_sha=source_sha(), config_sha256=canonical(c)
    )
    return dict(value, run_id=canonical(value))


def directory(output, seed, arm):
    return Path(output) / "private-checkpoints/jobs" / ("seed-%s-arm-%s" % (seed, arm))


def baseline(c, output, seed, arm, step):
    oldarm = arm[0] + "-" + arm[-1]
    root = Path(output) / "baseline-r3h" / ("jobs/seed-%s-arm-%s" % (seed, oldarm))
    expected = c["baseline"]
    ident = {k: expected[k] for k in ("experiment_id", "source_sha", "config_sha256", "run_id")}
    ident.update(seed=seed, architecture=arm[0], training_router=arm[-1])
    store = Store(root, ident)
    receipt = read(store.receipt_file(step))
    return store.verified(receipt), receipt


def initial(c, output, seed, arm, device):
    seed_all(seed)
    state, receipt = baseline(c, output, seed, arm, 0)
    m = Model(c, seed, arm[0] + "-" + arm[-1]).to(device)
    parent = parent_state(c, Path(output) / "parent-r3c", seed)
    for name, value in parent["model"].items():
        if name not in state["model"] or not torch.equal(value, state["model"][name]):
            raise RuntimeError("R3H_INITIAL_NOT_EQUAL_R3C_PARENT:" + name)
    m.load_state_dict(state["model"], strict=True)
    m.temperature = state["r3h"]["temperature"]
    tr = Trainer(m, c, arm)
    tr.parent_identity = copy.deepcopy(state["r3h"]["parent_identity"])
    replacement = copy.deepcopy(state)
    replacement["optimizer_identity"] = tr.opt_identity
    replacement["r3i"] = {"connection_commits": []}
    tr.restore(replacement)
    if fingerprint(tr.model.state_dict()) != fingerprint(state["model"]) or fingerprint(
        tr.cell_opt.state_dict()
    ) != fingerprint(state["cell_optimizer"]):
        raise RuntimeError("HISTORICAL_INITIALIZATION_MISMATCH")
    return tr, receipt


def latest(store):
    rows = [read(p) for p in (store.root / "receipts").glob("*.json")]
    return max(rows, key=lambda r: r["step"]) if rows else None


def run_job(c, output, seed, arm, data, device):
    root = directory(output, seed, arm)
    ident = dict(identity(c), seed=seed, arm=arm)
    store = Store(root, ident)
    current = latest(store)
    if (root / "result.json").exists():
        r = read(root / "result.json")
        if r["identity"] != ident:
            raise RuntimeError("RESULT_IDENTITY_MISMATCH")
        store.verified(r["checkpoint"])
        return
    tr, parent = initial(c, output, seed, arm, device)
    if current:
        tr.restore(store.verified(current))
    else:
        current = save(tr, store)
    start = time.perf_counter()
    while tr.step < c["training"]["steps"]:
        if tr.step in c["training"]["proposal_steps"] and not any(
            e["at_step"] == tr.step for e in tr.proposals
        ):
            if tr.step == 100:
                historical, _ = baseline(c, output, seed, arm, 100)
                proof = compare_step100(tr.state(), historical)
                # Archived exposure and CE provide the saved route/batch trajectory comparison.
                if tr.curve != historical["curve"]:
                    raise RuntimeError("HISTORICAL_CONTROL_NOT_COMPARABLE:curve")
                atomic(root / "compatibility.json", proof)
            execute(tr, data, root, ident, lambda: durable_request(output, seed, arm))
        else:
            row = tr.update(data)
            if tr.step % c["training"]["print_every"] == 0:
                print(json.dumps(dict(seed=seed, arm=arm, step=tr.step, ce=row["ce"])), flush=True)
        if current["step"] != tr.step and (
            tr.step % c["training"]["checkpoint_every"] == 0 or tr.step == 500
        ):
            current = save(tr, store)
            print("[R3I_SNAPSHOT]", seed, arm, tr.step, flush=True)
    if (
        tr.step != c["training"]["steps"]
        or tr.data_cursor != c["training"]["steps"] * c["training"]["tokens_per_step"]
    ):
        raise RuntimeError("MAIN_BUDGET_MISMATCH")
    current = save(tr, store) if current["step"] != tr.step else current
    atomic(
        root / "result.json",
        dict(
            status="TRAINING_COMPLETE",
            identity=ident,
            checkpoint=current,
            parent_checkpoint=parent,
            proposals=tr.proposals,
            cost=tr.cost,
            training_wall_seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
            if str(device).startswith("cuda")
            else None,
            alignment=measure(tr.model, c, data, tr.epsilon()),
        ),
    )
    print("[R3I_SNAPSHOT]", seed, arm, "COMPLETE", flush=True)


def barrier(c, output):
    for seed in c["seeds"]:
        for arm in c["new_arms"]:
            root = directory(output, seed, arm)
            r = read(root / "result.json")
            if r["identity"] != dict(identity(c), seed=seed, arm=arm):
                raise RuntimeError("TERMINAL_IDENTITY_MISMATCH")
            state = Store(root, r["identity"]).verified(r["checkpoint"])
            if state["step"] != 500 or state["data_cursor"] != 256000:
                raise RuntimeError("TERMINAL_BUDGET_MISMATCH")


def worker(c, output, folder, index, count, phase):
    backend = HubBackend(c["storage"]["private_repo"])
    data = assets(
        c,
        backend,
        folder,
        ("train", "calibration", "admission") if phase == "train" else ("development",),
    )
    jobs = [(seed, arm) for seed in c["seeds"] for arm in c["new_arms"]]
    if phase == "evaluate":
        barrier(c, output)
    for seed, arm in jobs[index::count]:
        device = "cuda:%s" % index
        if phase == "train":
            run_job(c, output, seed, arm, data, device)
        else:
            root = directory(output, seed, arm)
            r = read(root / "result.json")
            if "quality" in r:
                continue
            state = Store(root, r["identity"]).verified(r["checkpoint"])
            tr, _ = initial(c, output, seed, arm, device)
            tr.restore(state)
            r["quality"] = quality(tr.model, c, data, "development")
            r["status"] = "COMPLETED"
            atomic(root / "result.json", r)
            print("[R3I_SNAPSHOT]", seed, arm, "EVALUATED", flush=True)


def durable_request(output, seed, arm):
    request = Path(output) / "upload-requests" / ("%s-%s-%s" % (seed, arm, uuid.uuid4().hex))
    atomic(request, {"seed": seed, "arm": arm})
    print("[R3I_SNAPSHOT]", seed, arm, "DURABLE_REQUEST", flush=True)
    while not request.with_suffix(".ack").exists():
        if request.with_suffix(".error").exists():
            raise RuntimeError("DURABLE_REMOTE_SAVE_FAILED")
        time.sleep(0.2)


def snapshot(backend, output, c, prior):
    ident = identity(c)
    manifest = verified_manifest(backend, prior, ident) if prior else None
    private = Path(output) / "private-checkpoints"
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        target = Path(temporary)
        for receipts in private.rglob("receipts"):
            parent = receipts.parent
            rows = []
            for file in receipts.glob("*.json"):
                row = read(file)
                Store(parent, row["identity"]).verified(row)
                rows.append(row)
                for src in (file, parent / row["file"]):
                    dest = target / src.relative_to(private)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        try:
                            os.link(src, dest)
                        except OSError:
                            shutil.copyfile(src, dest)
            if rows:
                atomic(
                    target / parent.relative_to(private) / "latest.json",
                    max(rows, key=lambda x: x["step"]),
                )
            if (parent / "result.json").exists():
                shutil.copyfile(
                    parent / "result.json", target / parent.relative_to(private) / "result.json"
                )
        return publish_segment(
            backend,
            target,
            ident,
            uuid.uuid4().hex,
            pointer_path="r3i/" + ident["run_id"] + "/latest.json",
            prior_manifest=manifest,
        )


def automatic(c, path, output, folder, report_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ident = identity(c)
    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise RuntimeError("OUTPUT_IDENTITY_MISMATCH")
    atomic(output / "identity.json", ident)
    secrets()
    backend = HubBackend(c["storage"]["private_repo"])
    if not backend.info().private:
        raise RuntimeError("PRIVATE_HF_REQUIRED")
    backend.check_write()
    revision = backend.info().sha
    pointer = backend.optional("r3i/" + ident["run_id"] + "/latest.json", revision)
    prior = {**pointer, "hf_repo": backend.repo, "hf_revision": revision} if pointer else None
    if prior:
        restore_segment(backend, prior, output / "private-checkpoints", ident, uuid.uuid4().hex)
    restore_baselines(c, backend, output / "baseline-r3h")
    failures = []
    last = [0.0]

    def callback(force):
        nonlocal prior
        if not list((output / "private-checkpoints").glob("jobs/*/receipts/*.json")):
            return
        requests = list((output / "upload-requests").glob("*"))
        requests = [p for p in requests if not p.suffix]
        if not force and not requests and time.monotonic() - last[0] < 180:
            return
        try:
            prior = snapshot(backend, output, c, prior)
            atomic(output / "HF_REFERENCE.json", prior)
            for request in requests:
                atomic(request.with_suffix(".ack"), prior)
                request.unlink()
        except Exception as error:
            atomic(
                output / "STORAGE_WARNING.json", dict(type=type(error).__name__, message=str(error))
            )
            for request in requests:
                atomic(request.with_suffix(".error"), {"type": type(error).__name__})
        last[0] = time.monotonic()

    try:
        if not report_only:
            if torch.cuda.device_count() != 2:
                raise RuntimeError("TWO_GPUS_REQUIRED")
            restore_parents(c, backend, output / "parent-r3c")
            # Reuse the existing two-worker coordinator; only the new entrypoint and event label differ.
            globals_copy = dict(original_runner.workers.__globals__, BASE=BASE)
            exec(
                inspect.getsource(original_runner.workers).replace(
                    "[R3H_SNAPSHOT]", "[R3I_SNAPSHOT]"
                ),
                globals_copy,
            )
            workers = globals_copy["workers"]
            # Forward R3i transaction events through the existing queue dispatcher.
            workers(c, path, output, folder, "train", callback)
            barrier(c, output)
            workers(c, path, output, folder, "evaluate", callback)
    except Exception as error:
        failures.append(dict(type=type(error).__name__, message=str(error)))
    finally:
        if not report_only:
            callback(True)
    result = report(c, output, failures, prior)
    from .publication import publish

    try:
        publication = publish(output)
    except Exception as error:
        publication = dict(
            status="LOCAL_RESULTS_RETAINED_PUBLICATION_RETRY_PENDING",
            type=type(error).__name__,
            message=str(error),
        )
    atomic(output / "PUBLICATION.json", publication)
    print(json.dumps(dict(status=result["status"], publication=publication)), flush=True)
    if failures:
        raise RuntimeError("R3I_PARTIAL_SEE_REPORTS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--report-existing", action="store_true")
    p.add_argument("--output")
    p.add_argument("--assets", default="/kaggle/working/r3i-assets")
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--phase", choices=["train", "evaluate"])
    a = p.parse_args()
    c = read(a.config)
    if canonical(c) != (BASE / "config/releases.lock").read_text().strip():
        raise RuntimeError("FROZEN_CONFIG_CHANGED")
    out = a.output or "/kaggle/working/r3i-output-" + source_sha()[:12]
    if a.worker is not None:
        worker(c, out, a.assets, a.worker, a.workers, a.phase)
    elif a.run_all or a.report_existing:
        automatic(c, Path(a.config).resolve(), out, a.assets, a.report_existing)
    else:
        p.error("--run-all or --report-existing required")
