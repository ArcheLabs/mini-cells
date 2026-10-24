"""One launch: functional checks, assets, finite Tiny Overfit, reporting and return."""

import argparse
import copy
import os
import sys
import platform
from importlib import metadata
import numpy as np
from pathlib import Path
import time
from types import SimpleNamespace
import uuid
import torch
from minicells_artifacts.hf_storage import HubBackend, error_code
from minicells_r3e.runner import secrets
from minicells_r3k.reuse import bind
from minicells_r3m import anchor, engine, training
from minicells_r3m.storage import restore as restore_private
from minicells_r3j.storage import Store
from minicells_r3me import study as previous
from . import protocol, data, learnability, specialization, report
from .protocol import BASE, atomic, digest, read, fingerprint, source_sha
from .storage import SnapshotPublisher

location = previous.location
jobs = previous.jobs
completed = previous.completed
training_barrier = previous.training_barrier
worker_pool = bind(previous.worker_pool, BASE=BASE)
advance = bind(training.advance, schedule=protocol.schedule)


def training_runner(c, output, ident, seed, arm):
    def provider(tr, historical, research, environment, unused):
        return data.batches(tr, historical, research, tr.step - c["anchor_step"], arm, c)[0]

    def no_proposal(*a, **kw):
        raise RuntimeError("R3MF_CONNECTION_REORGANIZATION_FORBIDDEN")

    return bind(engine.run_arm, batches=provider, advance=advance, propose=no_proposal)


def runtime_identity(c):
    return protocol.identity(c, source_sha())


def analyze_job(c, output, runtime, assets, research, tokenizer, ident, seed, arm, device):
    root = location(output, ident, seed, arm)
    aid = dict(ident, seed=seed, arm=arm)
    store = Store(root, aid)
    tiny = c["execution_phase"] == "tiny"
    evaluations = {}
    first_model = last_model = None
    initial_cost = None
    initial_work = None
    for step in c["barriers"]:
        target = root / ("analysis-step-%06d.json" % step)
        receipt = read(store.receipt_file(step))
        payload = store.verified(receipt)
        if payload["trainer"]["step"] != c["anchor_step"] + step:
            raise RuntimeError("BARRIER_STEP_CHANGED")
        if step == 0:
            initial_cost = copy.deepcopy(payload["trainer"]["cost"])
            initial_work = copy.deepcopy(payload["trainer"]["work"])
        if target.exists():
            row = read(target)
            if row["identity"] != aid or row["checkpoint_sha256"] != receipt["sha256"]:
                raise RuntimeError("ANALYSIS_IDENTITY_CHANGED")
        else:
            tr = training.restore(runtime, payload["trainer"], seed, device)
            started = time.perf_counter()
            metrics = learnability.measure(tr.model, research, tokenizer, c, tiny=tiny)
            row = dict(
                identity=aid,
                relative_step=step,
                checkpoint_sha256=receipt["sha256"],
                metrics=metrics,
            )
            if not tiny:
                row["interventions"] = specialization.interventions(
                    tr.model, research, tokenizer, c
                )
                row["gradient"] = specialization.gradient_conflict(tr, assets, research, seed, c)
            row["analysis_seconds"] = time.perf_counter() - started
            atomic(target, row)
            print("[R3M_SNAPSHOT]", seed, arm, "ANALYZED", step, flush=True)
        evaluations[str(step)] = row
        if step == 0:
            first_model = fingerprint(
                {k: v for k, v in payload["trainer"]["model"].items() if k.startswith("cells.")}
            )
        if step == c["continuation_steps"]:
            last_model = fingerprint(
                {k: v for k, v in payload["trainer"]["model"].items() if k.startswith("cells.")}
            )
    final = payload
    tr = training.restore(runtime, final["trainer"], seed, device)
    parameters = dict(
        total=sum(p.numel() for p in tr.model.parameters()),
        trainable=sum(p.numel() for p in tr.model.parameters() if p.requires_grad),
        cells=sum(p.numel() for p in tr.model.cells.parameters()),
        connection=sum(p.numel() for p in tr.model.connection.parameters()),
    )
    work = {
        k: v - initial_work.get(k, 0)
        for k, v in final["trainer"]["work"].items()
        if isinstance(v, (int, float))
    }
    if final["aux"]["shadow_updates"] or final["aux"]["transactions"]:
        raise RuntimeError("UNEXPECTED_CONNECTION_EVENT")
    terminal = final["trainer"]["cost"]
    diff = {
        k: terminal[k] - initial_cost.get(k, 0)
        for k in terminal
        if isinstance(terminal[k], (int, float))
    }
    result = dict(
        identity=aid,
        status="COMPLETE",
        evaluations=evaluations,
        curve=final["aux"]["curve"],
        parameter_updated=first_model != last_model,
        connection_frozen=True,
        continuation_cost=diff,
        parameter_counts=parameters,
        observed_parameter_work=work,
        main_updates=final["aux"]["main_updates"],
        tokens=c["continuation_steps"] * c["tokens_per_step"],
        cell_calls=c["continuation_steps"] * c["tokens_per_step"] * 8,
        checkpoint_integrity="HASH_VERIFIED_ALL_BARRIERS",
        resume=read(root / "RESTORE_CHECK.json"),
    )
    atomic(root / "EVALUATION.json", result)
    return result


def restore_check(c, runtime, root, ident, device):
    """No extra update: restore the actual initial/latest state and compare all semantic fields."""
    store = Store(root, ident)
    latest = read(root / "latest.json")
    payload = store.verified(latest)
    before = training.semantic(payload["trainer"])
    tr = training.restore(runtime, payload["trainer"], ident["seed"], device)
    after = training.semantic(tr.state())
    if fingerprint(before) != fingerprint(after):
        raise RuntimeError("RESTORE_FULL_STATE_MISMATCH")
    result = dict(
        exact_full_state=True,
        checkpoint_sha256=latest["sha256"],
        step=tr.step,
        rng_cursor_optimizer_verified=True,
        extra_optimizer_updates=0,
        cross_session_verified=False,
        scope="LOCAL_FULL_STATE_RESTORE; remote readback not cross-session proof",
    )
    atomic(root / "RESTORE_CHECK.json", result)
    return result


def worker(c, output, folder, index, count, phase):
    from tokenizers import Tokenizer

    secrets()
    ident = runtime_identity(c)
    runtime = anchor.runtime(c)
    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    original = protocol.load()
    assets = data.r3h_train_assets(runtime, backend, Path(folder) / "r3h-train", ("train",))
    research, manifest = data.load(folder, original)
    tokenizer = Tokenizer.from_file(str(Path(folder) / "r3h-train/tokenizer.json"))
    if digest(Path(folder) / "r3h-train/tokenizer.json") != c["model"]["tokenizer_sha256"]:
        raise RuntimeError("TOKENIZER_CHANGED")
    device = "cuda:%d" % index
    current_seed = None
    frozen = None
    for seed, arm in jobs(c)[index::count]:
        root = location(output, ident, seed, arm)
        aid = dict(ident, seed=seed, arm=arm)
        if phase.endswith("train"):
            if completed(c, output, ident, seed, arm):
                continue
            if current_seed != seed:
                parent, evidence = anchor.load(
                    c, backend, Path(output) / "parent-r3h", seed, device
                )
                frozen = copy.deepcopy(parent.state())
                current_seed = seed
            atomic(root / "ANCHOR.json", evidence)
            fixed = SimpleNamespace(state=lambda: copy.deepcopy(frozen))
            execute = training_runner(c, output, ident, seed, arm)
            execute(
                c, copy.deepcopy(runtime), fixed, aid, root, assets, research, "N", "STABLE", device
            )
            restore_check(c, runtime, root, aid, device)
            previous._durable_request(previous.control_root(output, ident), seed, arm)
        else:
            analyze_job(c, output, runtime, assets, research, tokenizer, ident, seed, arm, device)


def publication(output):
    # Reuse current-main ancestry and result-only allowlist, with F's own destination.
    return bind(
        previous.publication, BASE=BASE, reports=SimpleNamespace(ARTIFACTS=report.ARTIFACTS)
    )(output)


def run(c, path, output, folder, phase="tiny"):
    protocol.validate(c)
    # This release intentionally has no pilot bypass flag or environment override.
    if phase != "tiny":
        raise RuntimeError("R3MF_PILOT_PENDING_TINY_RESULT_INDEPENDENT_REVIEW_AND_AUTHORIZATION")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("TWO_VISIBLE_T4_REQUIRED_TO_RESTORE_PARENT_CUDA_RNG_LAYOUT")
    q = protocol.execution_config(c, phase)
    ident = runtime_identity(q)
    output, folder = Path(output) / ident["run_id"], Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    secrets()
    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    info = backend.info()
    if not info.private:
        raise RuntimeError("HF_PRIVATE_REPOSITORY_REQUIRED")
    backend.check_write()
    private = output / "private-checkpoints" / ident["run_id"]
    private.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex
    prior = backend.optional("r3m-f/" + ident["run_id"] + "/latest.json", info.sha)
    if prior:
        prior = dict(prior, hf_repo=backend.repo, hf_revision=info.sha)
        restore_private(backend, prior, private, output, ident, session, q["barriers"])
    publisher = SnapshotPublisher(backend, private, output, ident, session, prior)
    pending = previous.control_root(output, ident) / "upload-requests"
    last = [0.0]
    publication_errors = []

    def persist(force=False):
        requests = [p for p in pending.glob("*") if not p.suffix]
        if (
            not force
            and not requests
            and time.monotonic() - last[0] < c["execution"]["remote_snapshot_min_seconds"]
        ):
            return
        try:
            ref = publisher.publish(force)
            for p in requests:
                atomic(p.with_suffix(".ack"), ref)
                p.unlink()
            atomic(
                output / "HF_PUBLICATION.json", dict(status="PUBLISHED_VERIFIED", hf_reference=ref)
            )
        except Exception as error:
            detail = dict(status="LOCAL_RETAINED_RETRY_PENDING", **error_code(error))
            publication_errors.append(detail)
            atomic(output / "HF_PUBLICATION.json", detail)
            for p in requests:
                atomic(p.with_suffix(".error"), detail)
                p.unlink()
        last[0] = time.monotonic()

    failures = []
    manifest = {}
    start = time.time()
    try:
        runtime = anchor.runtime(c)
        _, _, manifest, _ = data.prepare(c, runtime, backend, folder)
        atomic(private / "INPUT_IDENTITY.json", dict(identity=ident, manifest=manifest))
        print(
            "[R3m-F] preflight passed; 8 fixed Train facts, 200 updates, 102400 tokens; no pilot or A2/A3",
            flush=True,
        )
        if not all(completed(q, output, ident, s, a) for s, a in jobs(q)):
            worker_pool(q, path, output, folder, "tiny-train", persist)
        training_barrier(q, output, ident)
        worker_pool(q, path, output, folder, "tiny-evaluate", persist)
    except Exception as error:
        failures.append(dict(type=type(error).__name__, code=str(error)[:240]))
    finally:
        persist(True)
    result = report.write(q, output, ident, manifest, failures)
    result.update(
        runtime_environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            torch=torch.__version__,
            tokenizers=metadata.version("tokenizers"),
            cuda=torch.version.cuda,
            devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            worker_omp_num_threads=1,
            worker_mkl_num_threads=1,
        ),
        command=list(sys.argv),
        started_at_unix=start,
        finished_at_unix=time.time(),
        hf_reference=publisher.prior,
        publication_errors=publication_errors,
    )
    atomic(output / "reports/RUN_MANIFEST.json", result)
    if os.environ.get("GITHUB_TOKEN"):
        try:
            result["publication"] = publication(output)
        except Exception as error:
            result["publication"] = dict(status="LOCAL_RETAINED", **error_code(error))
    else:
        result["publication"] = dict(status="LOCAL_RETAINED", reason="GITHUB_TOKEN_UNAVAILABLE")
    result["publication_status"] = result["publication"]["status"]
    result["remote_commit_sha"] = result["publication"].get("remote_commit_sha")
    # Local receipt records the actual verified push; the remote commit contains its pre-push manifest.
    atomic(output / "reports/RUN_MANIFEST.json", result)
    atomic(output / "RUN_STATUS.json", result)
    print(result, flush=True)
    if failures:
        raise RuntimeError("R3MF_FAILED_SEE_RUN_STATUS_RETAINED_OUTPUT")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(protocol.CONFIG))
    p.add_argument("--mode", choices=["auto", "cpu-gate", "pilot"], default="auto")
    p.add_argument("--output", default="/kaggle/working/r3mf-output")
    p.add_argument("--assets", default="/kaggle/working/r3mf-assets")
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--phase",
        default="tiny-train",
        choices=["tiny-train", "tiny-evaluate", "pilot-train", "pilot-evaluate"],
    )
    a = p.parse_args()
    c = protocol.load(a.config)
    if a.worker is not None:
        phase = "tiny" if a.phase.startswith("tiny") else "pilot"
        if phase == "pilot":
            raise RuntimeError("R3MF_PILOT_NOT_RELEASED")
        worker(
            protocol.execution_config(c, phase), a.output, a.assets, a.worker, a.workers, a.phase
        )
        return
    gate = protocol.cpu_gate(c)
    print(gate, flush=True)
    if a.mode == "cpu-gate":
        atomic(Path(a.output) / "CPU_GATE.json", gate)
        return
    run(c, Path(a.config).resolve(), a.output, a.assets, "pilot" if a.mode == "pilot" else "tiny")
