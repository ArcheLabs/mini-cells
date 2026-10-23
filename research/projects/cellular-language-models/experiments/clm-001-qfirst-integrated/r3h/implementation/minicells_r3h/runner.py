"""One launch: verified parents -> 24 independent Cell-only jobs -> terminal barrier -> return."""

import argparse
import copy
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid
import torch
from minicells_artifacts.hf_storage import HubBackend
from minicells_r3e.runner import secrets, cost_estimate
from minicells_r3d.evaluation import sync
from .parent_import import (
    BASE,
    ROOT,
    config,
    identity,
    read,
    atomic,
    fingerprint,
    restore_parents,
    parent_state,
    build,
    fingerprints,
)
from .evaluation import assets, windows, quality
from .alignment import temperature, measure
from .trainer import Trainer
from .storage import Store, save, restore_latest, remote_snapshot
from .reports import write


def jobdir(output, seed, arm):
    return Path(output) / "private-checkpoints/jobs" / ("seed-%s-arm-%s" % (seed, arm))


def latest(store):
    rows = [read(p) for p in (store.root / "receipts").glob("*.json")]
    return max(rows, key=lambda r: r["step"]) if rows else None


def monitored(tr, c, data):
    work = copy.deepcopy(tr.model.work)
    begin = time.perf_counter()
    metrics = measure(tr.model, c, data, tr.epsilon())
    extra = {k: tr.model.work[k] - work[k] for k in work}
    tr.model.work = work
    tr.monitor_seconds += time.perf_counter() - begin
    tr.alignment.append(dict(step=tr.step, metrics=metrics, additional_forward_work=extra))


@torch.no_grad()
def prepare_model(c, parent, seed, architecture, device, data, cached=None):
    m = build(c, parent, seed, architecture + "-F", device, data, evidence=cached)
    fit = cached["temperature"] if cached is not None else temperature(m, c, data)
    _, x, _ = next(windows(data, c, "calibration", device, 1, labels=False))
    out = m(x)
    if not torch.isfinite(out.logits).all() or int(out.operator_rows.sum()) != 8 * x.numel():
        raise RuntimeError("INITIAL_FORWARD_INVALID")
    if any(p.requires_grad for n, p in m.named_parameters() if not n.startswith("cells.")):
        raise RuntimeError("ONLY_CELLS_MAY_TRAIN")
    # A must still reproduce the inherited D500 operators and pair mixing exactly.
    if not m.graph:
        from minicells_r3e.runner import parity

        parity(m, parent, x)
    evidence = dict(
        seed=seed,
        architecture=architecture,
        **fingerprints(m),
        parent_source_sha=c["parent"]["source_sha"],
        parent_checkpoint_sha256=c["parent"]["checkpoint_sha256"][str(seed)],
        initial_logit_hash=fingerprint(out.logits),
        initial_path_hash=fingerprint(out.actions),
        margin=m.margin_evidence,
        temperature=fit,
        initial_quality=cached["initial_quality"]
        if cached is not None
        else quality(m, c, data, "calibration"),
    )
    if cached is not None and evidence != cached:
        raise RuntimeError("PREPARATION_IDENTITY_CHANGED")
    m.work = {k: 0 for k in m.work}
    return m, evidence, x


def train_job(c, output, seed, arm, template, evidence, x, data):
    directory = jobdir(output, seed, arm)
    ident = identity(c, seed, arm)
    store = Store(directory, ident)
    row = latest(store)
    if (directory / "result.json").exists():
        done = read(directory / "result.json")
        if (
            done["identity"] != ident
            or row is None
            or row["step"] != c["training"]["steps"]
            or done["checkpoint"] != row
        ):
            raise RuntimeError("COMPLETED_JOB_IDENTITY_INVALID")
        store.verified(row)
        return
    m = copy.deepcopy(template)
    m.routing_mode = arm[-1]
    m.temperature = evidence["temperature"]["temperature"] if arm[-1] == "T" else 1.0
    with torch.no_grad():
        out = m(x)
        if (
            fingerprint(out.logits) != evidence["initial_logit_hash"]
            or fingerprint(out.actions) != evidence["initial_path_hash"]
        ):
            raise RuntimeError("FOUR_ARM_INITIAL_FORWARD_MISMATCH")
    if fingerprints(m) != {k: evidence[k] for k in fingerprints(m)}:
        raise RuntimeError("FOUR_ARM_INITIAL_WEIGHTS_MISMATCH")
    m.work = {k: 0 for k in m.work}
    tr = Trainer(m, c, arm)
    tr.parent_identity = copy.deepcopy(evidence)
    optimizer_hash = fingerprint(tr.cell_opt.state_dict())
    if arm[-1] != "F":
        baseline = read(jobdir(output, seed, arm[0] + "-F") / "result.json")
        if (
            optimizer_hash != baseline["initial_optimizer_fingerprint"]
            or evidence != baseline["initial"]
        ):
            raise RuntimeError("FOUR_ARM_INITIAL_OPTIMIZER_OR_INPUT_MISMATCH")
    if row:
        tr.restore(store.verified(row))
    else:
        monitored(tr, c, data)
        row = save(tr, store)
    if next(m.parameters()).device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(next(m.parameters()).device)
    sync(next(m.parameters()).device)
    m.sampling_enqueue_seconds = getattr(m, "sampling_enqueue_seconds", 0.0)
    begin = time.perf_counter()
    print("[R3h]", seed, arm, "restored", tr.step, "/", c["training"]["steps"], flush=True)
    while tr.step < c["training"]["steps"]:
        r = tr.update(data)
        if tr.step % c["training"]["print_every"] == 0:
            print(json.dumps(dict(seed=seed, arm=arm, step=tr.step, ce=r["ce"])), flush=True)
        if tr.step % c["training"]["checkpoint_every"] == 0:
            monitored(tr, c, data)
            tr.verify_frozen()
            row = save(tr, store)
            print("[R3H_SNAPSHOT]", seed, arm, tr.step, flush=True)
    tr.verify_frozen()
    sync(next(m.parameters()).device)
    if tr.data_cursor != c["training"]["steps"] * c["training"]["tokens_per_step"]:
        raise RuntimeError("TRAIN_TOKEN_BUDGET_CHANGED")
    final = fingerprint(m.state_dict())
    initial_cells = template.cells.state_dict()
    squared = sum(
        float((p.detach() - initial_cells[n]).double().square().sum())
        for n, p in m.cells.state_dict().items()
    )
    cost = copy.deepcopy(tr.cost)
    cost.update(
        main_forward_work=copy.deepcopy(m.work),
        main_forward_matmul_estimate=cost_estimate(m.work, c, m.graph),
        backward_scope="3X_FORWARD_IS_ANALYTIC_UPPER_PROXY_NOT_MEASURED_FLOPS",
        forward_backward_upper_proxy=3 * cost_estimate(m.work, c, m.graph),
        monitoring_seconds=tr.monitor_seconds,
        sampling_cpu_enqueue_seconds=m.sampling_enqueue_seconds,
        sampling_timing_scope="CPU_ENQUEUE_WALL_NOT_SYNCHRONIZED_GPU_KERNEL_TIME",
        wall_training_this_session_seconds=time.perf_counter() - begin,
        peak_training_allocated_bytes=torch.cuda.max_memory_allocated(next(m.parameters()).device)
        if next(m.parameters()).device.type == "cuda"
        else None,
        total_parameters=sum(p.numel() for p in m.parameters()),
        trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
        cell_parameter_delta_l2=squared**0.5,
        same_tokens_is_not_same_flops=True,
    )
    result = dict(
        status="TRAINING_COMPLETE_TERMINAL_PENDING",
        identity=ident,
        checkpoint=row,
        initial=evidence,
        initial_optimizer_fingerprint=optimizer_hash,
        final_model_fingerprint=final,
        training_cost=cost,
        temperature=m.temperature,
        alignment=tr.alignment,
        confirmation="SEALED",
    )
    atomic(directory / "result.json", result)
    print("[R3H_SNAPSHOT]", seed, arm, "COMPLETE", flush=True)


def evaluate_job(c, output, seed, arm, data, device):
    directory = jobdir(output, seed, arm)
    result = read(directory / "result.json")
    if result["status"] == "COMPLETED":
        return
    store = Store(directory, identity(c, seed, arm))
    state = store.verified(result["checkpoint"])
    if (
        state["step"] != c["training"]["steps"]
        or state["data_cursor"] != c["training"]["steps"] * c["training"]["tokens_per_step"]
    ):
        raise RuntimeError("TERMINAL_BUDGET_INVALID")
    from .routing import Model

    m = Model(c, seed, arm).to(device)
    m.load_state_dict(state["model"], strict=True)
    m.temperature = state["r3h"]["temperature"]
    if (
        state["r3h"]["routing_mode"] != arm[-1]
        or state["r3h"]["parent_identity"] != result["initial"]
    ):
        raise RuntimeError("TERMINAL_ROUTER_IDENTITY_CHANGED")
    if fingerprint(m.state_dict()) != result["final_model_fingerprint"]:
        raise RuntimeError("TERMINAL_MODEL_CHANGED")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    result["quality"] = quality(m, c, data, "development")
    result["quality"]["peak_inference_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None
    )
    result["status"] = "COMPLETED"
    atomic(directory / "result.json", result)
    print("[R3H_SNAPSHOT]", seed, arm, "EVALUATED", flush=True)


def barrier(c, output):
    for seed in c["seeds"]:
        for arm in c["arms"]:
            d = jobdir(output, seed, arm)
            r = read(d / "result.json")
            if (
                r["identity"] != identity(c, seed, arm)
                or r["checkpoint"]["step"] != c["training"]["steps"]
            ):
                raise RuntimeError("ALL_24_TRAINING_BARRIER_NOT_REACHED")
            Store(d, r["identity"]).verified(r["checkpoint"])
    return True


def worker(c, output, folder, index, count, phase):
    backend = HubBackend(c["storage"]["private_repo"])
    roles = (
        ("train", "calibration", "admission") if phase in ("prepare", "train") else ("development",)
    )
    data = assets(c, backend, folder, roles)
    if phase == "evaluate":
        barrier(c, output)
    jobs = [(s, a) for s in c["seeds"] for a in "AB"]
    for seed, architecture in jobs[index::count]:
        if phase in ("prepare", "train"):
            parent = parent_state(c, Path(output) / "parent-r3c", seed)
            prepared = Path(output) / "preparation" / ("%s-%s.json" % (seed, architecture))
            cached = read(prepared) if prepared.exists() else None
            if phase == "train" and cached is None:
                raise RuntimeError("INITIAL_PREPARATION_NOT_COMPLETE")
            model, evidence, x = prepare_model(
                c, parent, seed, architecture, "cuda:%s" % index, data, cached=cached
            )
            if phase == "prepare":
                atomic(prepared, evidence)
                continue
            for mode in "FHTM":
                train_job(c, output, seed, architecture + "-" + mode, model, evidence, x, data)
        else:
            for mode in "FHTM":
                evaluate_job(c, output, seed, architecture + "-" + mode, data, "cuda:%s" % index)


def workers(c, path, output, folder, phase, callback):
    count = min(2, torch.cuda.device_count())
    events = queue.Queue()
    procs = []
    threads = []

    def forward(p, index):
        for line in p.stdout:
            print("[GPU %s]" % index, line.rstrip(), flush=True)
            if line.startswith("[R3H_SNAPSHOT]"):
                events.put(line)

    for index in range(count):
        command = [
            sys.executable,
            "-u",
            str(BASE / "run.py"),
            "--config",
            str(path),
            "--output",
            str(output),
            "--assets",
            str(folder),
            "--worker",
            str(index),
            "--workers",
            str(count),
            "--phase",
            phase,
        ]
        p = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        t = threading.Thread(target=forward, args=(p, index))
        t.start()
        procs.append(p)
        threads.append(t)
    while any(p.poll() is None for p in procs):
        try:
            events.get(timeout=0.5)
        except queue.Empty:
            continue
        callback(False)
    for t in threads:
        t.join()
    callback(True)
    if any(p.wait() != 0 for p in procs):
        raise RuntimeError("WORKER_FAILED_CHECK_RETAINED_OUTPUT")


def automatic(c, path, output, folder):
    ident = identity(c)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise RuntimeError("OUTPUT_IDENTITY_CONFLICT")
    atomic(output / "identity.json", ident)
    start = time.time()
    failures = []
    prior = None
    backend = None
    last = [0.0]
    folder = Path(folder or "/kaggle/working/r3h-assets")
    try:
        secrets()
        if torch.cuda.device_count() < 2:
            raise RuntimeError("TWO_GPUS_REQUIRED_NO_TRAINING_STARTED")
        backend = HubBackend(c["storage"]["private_repo"])
        backend.check_write()
        if not backend.info().private:
            raise RuntimeError("HF_PRIVATE_REPO_REQUIRED")
        pointer = "r3h/" + ident["run_id"] + "/latest.json"
        revision = backend.info().sha
        ref = backend.optional(pointer, revision)
        private = output / "private-checkpoints"
        if ref:
            prior = {**ref, "hf_repo": backend.repo, "hf_revision": revision}
            if not list(private.glob("jobs/*/receipts/*.json")):
                restore_latest(backend, prior, private, ident, uuid.uuid4().hex)
        session = uuid.uuid4().hex

        def callback(force):
            nonlocal prior
            if (
                not force
                and time.monotonic() - last[0] < c["storage"]["remote_snapshot_min_seconds"]
            ):
                return
            begin = time.perf_counter()
            try:
                prior = remote_snapshot(backend, private, output, ident, pointer, session, prior)
                atomic(
                    output / "HF_PUBLICATION.json",
                    dict(status="PUBLISHED_VERIFIED", reference=prior),
                )
            except Exception as error:
                atomic(
                    output / "HF_PUBLICATION.json",
                    dict(status="LOCAL_RETAINED_REMOTE_RETRY_PENDING", type=type(error).__name__),
                )
                print(
                    "[R3h] HF retry pending; local checkpoints retained:",
                    type(error).__name__,
                    flush=True,
                )
            atomic(output / "UPLOAD_COST.json", dict(last_seconds=time.perf_counter() - begin))
            last[0] = time.monotonic()

        restore_parents(c, backend, output / "parent-r3c")
        assets(c, backend, folder, ("train", "calibration", "admission"))
        print(
            "[R3h] verified input -> 24 jobs, 12000 updates, 6144000 tokens; no independent pilot",
            flush=True,
        )
        workers(c, path, output, folder, "prepare", callback)
        workers(c, path, output, folder, "train", callback)
        barrier(c, output)
        assets(c, backend, folder, ("development",))
        workers(c, path, output, folder, "evaluate", callback)
    except Exception as error:
        failures.append(dict(type=type(error).__name__, message=str(error)))
        print("[R3h] failed:", type(error).__name__, str(error), flush=True)
    results = {
        str(s): {
            a: read(jobdir(output, s, a) / "result.json")
            for a in c["arms"]
            if (jobdir(output, s, a) / "result.json").exists()
        }
        for s in c["seeds"]
    }
    report = write(
        c,
        output,
        results,
        failures,
        dict(**ident, started_at_unix=start, finished_at_unix=time.time(), hf_reference=prior),
    )
    publication = {"status": "NOT_PUBLISHED"}
    if os.environ.get("GITHUB_TOKEN"):
        try:
            from .publication import publish

            publication = publish(output)
        except Exception as error:
            publication = dict(
                status="LOCAL_RETAINED", type=type(error).__name__, message=str(error)
            )
    atomic(output / "PUBLICATION.json", publication)
    print(
        json.dumps(dict(status=report["status"], publication=publication, output=str(output))),
        flush=True,
    )
    if failures or report["status"] != "COMPLETED":
        raise RuntimeError("R3H_PARTIAL_SEE_RETAINED_OUTPUT")
    return report


def report_existing(c, output):
    """Read verified historical outputs only. No model forward or optimizer construction."""
    output = Path(output)
    ident = read(output / "identity.json")
    if ident["config_sha256"] != canonical(c) or ident["experiment_id"] != c["experiment_id"]:
        raise RuntimeError("RECOVERY_IDENTITY_MISMATCH")
    backend = None
    if not list((output / "private-checkpoints").glob("jobs/*/result.json")):
        secrets()
        backend = HubBackend(c["storage"]["private_repo"])
        revision = backend.info().sha
        pointer = backend.optional("r3h/" + ident["run_id"] + "/latest.json", revision)
        if not pointer:
            raise RuntimeError("SAVED_OUTPUT_OR_HF_REFERENCE_REQUIRED")
        restore_latest(
            backend,
            {**pointer, "hf_repo": backend.repo, "hf_revision": revision},
            output / "private-checkpoints",
            ident,
            uuid.uuid4().hex,
        )
    results = {}
    failures = []
    for seed in c["seeds"]:
        results[str(seed)] = {}
        for arm in c["arms"]:
            d = jobdir(output, seed, arm)
            if not (d / "result.json").exists():
                failures.append({"code": "RESULT_MISSING", "seed": seed, "arm": arm})
                continue
            r = read(d / "result.json")
            if any(r["identity"].get(k) != v for k, v in ident.items()):
                raise RuntimeError("RESULT_IDENTITY_MISMATCH")
            Store(d, r["identity"]).verified(r["checkpoint"])
            results[str(seed)][arm] = r
    prior = read(output / "hf-reference.json") if (output / "hf-reference.json").exists() else None
    report = write(
        c,
        output,
        results,
        failures,
        dict(
            **ident,
            hf_reference=prior,
            recovery_source_sha=identity(c)["source_sha"],
            optimizer_updates=0,
        ),
    )
    secrets()
    from .publication import publish

    publication = publish(output)
    atomic(output / "PUBLICATION.json", publication)
    print(
        json.dumps(
            {"status": report["status"], "publication": publication, "optimizer_updates": 0}
        ),
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--report-existing", action="store_true")
    p.add_argument("--output")
    p.add_argument("--assets")
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--phase", choices=["prepare", "train", "evaluate"], default="train")
    a = p.parse_args()
    c = config(a.config)
    out = a.output or "/kaggle/working/r3h-output-" + identity(c)["source_sha"][:12]
    if a.report_existing:
        report_existing(c, out)
    elif a.worker is not None:
        worker(c, out, a.assets, a.worker, a.workers, a.phase)
    elif a.run_all:
        automatic(c, Path(a.config).resolve(), out, a.assets)
    else:
        p.error("--run-all required")
