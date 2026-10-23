"""One launch -> identity checks -> all six arms/three independent seeds -> return. No pilot."""

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
from minicells_artifacts.hf_storage import HubBackend, error_code
from minicells_r3c_moe_bridge.data import load_assets, private_asset_subset
from minicells_r3c_moe_bridge.model import BridgeModel
from .common import (
    BASE,
    ROOT,
    ID,
    ARMS,
    atomic,
    read,
    canonical,
    configuration,
    identity,
    source_sha,
    seed_all,
)
from .stable_connection import build
from .training import Trainer
from .adaptation import execute
from .evaluation import terminal, reference, paths, sync
from .storage import (
    Store,
    save,
    restore_parents,
    parent_state,
    restore_latest,
    remote_snapshot,
    public_publish,
    cleanup_transactions,
)
from .reports import write


def secrets():
    for name in ("HF_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(name, "").strip():
            continue
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret(name).strip()
            if value:
                os.environ[name] = value
        except Exception:
            pass
    if not os.environ.get("HF_TOKEN"):
        from huggingface_hub import get_token

        token = get_token()
        if token:
            os.environ["HF_TOKEN"] = token
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("EXISTING_HF_TOKEN_REQUIRED_NO_TRAINING_STARTED")


def cost_estimate(work, c, graph):
    m = c["engine"]["model"]
    d = m["width"]
    h = m["cell_hidden"]
    r = m["router_hidden"]
    s = m["step_width"]
    v = m["vocab_size"]
    n = c["engine"]["training"]["sequence_length"]
    return int(
        work.get("cell_rows", 0) * 4 * d * h
        + work.get("gate_rows", 0) * 2 * ((d + s) * r + r * 6)
        + work.get("policy_rows", 0) * 2 * (d + (15 if graph else 4)) * (6 if graph else 15)
        + work.get("input_rows", 0) * 6 * d * d
        + work.get("context_rows", 0) * (4 * d * d + 4 * n * d)
        + work.get("head_rows", 0) * 2 * d * v
    )


@torch.no_grad()
def parity(model, parent, x):
    if model.graph:
        out = model(x)
        if (
            out.actions.shape != (x.numel(), 8)
            or not ((out.actions >= 0) & (out.actions < 6)).all()
            or int(out.operator_rows.sum()) != 8 * x.numel()
        ):
            raise RuntimeError("GRAPH_EXECUTION_INVALID")
        return
    old = BridgeModel(model.c, "D", model.seed).to(x.device)
    old.load_state_dict(parent["model"], strict=True)
    expected = old(x)
    actual = model(x)
    ids = torch.stack([model.connection.base_action(s.logits) for s in expected.stages], -1)
    if not torch.equal(expected.logits, actual.logits) or not torch.equal(ids, actual.actions):
        raise RuntimeError("ZERO_CORRECTION_D_PARITY_FAILED")


def job(c, data, output, source, seed, arm, parent, device):
    seed_all(seed)
    directory = Path(output) / "private-checkpoints/jobs" / ("seed-%s-arm-%s" % (seed, arm))
    ident = identity(c, source, seed, arm)
    store = Store(directory, ident)
    model = build(c, parent, seed, arm, device)
    refs = reference(model, c, data)
    initial = paths(model, refs)
    parity(model, parent, refs[0])
    model.work = {k: 0 for k in model.work}
    tr = Trainer(model, c, arm)
    rows = [read(p) for p in (directory / "receipts").glob("*.json")]
    latest = max(rows, key=lambda r: r["step"]) if rows else None
    if (directory / "result.json").exists():
        result = read(directory / "result.json")
        if (
            latest is None
            or result["identity"] != ident
            or latest["step"] != c["training"]["steps"]
            or result["checkpoint"] != latest
        ):
            raise RuntimeError("COMPLETED_RESULT_IDENTITY_INVALID")
        store.verified(latest)
        return result
    if latest:
        tr.restore(store.verified(latest))
    else:
        latest = save(tr, store)
    sync(device)
    begin = time.perf_counter()
    print("[R3e]", seed, arm, "restored", tr.step, "/", c["training"]["steps"], flush=True)
    while tr.step < c["training"]["steps"]:
        due = tr.step in c["training"]["proposal_steps"] and arm in ("A-T", "A-C", "B-C")
        if due and not any(p["at_step"] == tr.step for p in tr.proposals):
            execute(tr, data, directory, ident)
            # The retained branch's Cell/optimizer/RNG is now one coherent official state.
            latest = save(tr, store) if tr.step != latest["step"] else latest
            if tr.step != latest["step"]:
                raise RuntimeError("ADAPTATION_CHECKPOINT_COMMIT_FAILED")
            # Once official state has a new step, obsolete private shadow tensors need not persist.
            if tr.step > tr.proposals[-1]["at_step"]:
                cleanup_transactions(directory, tr.step)
        else:
            row = tr.update(data)
            if tr.step % c["training"]["print_every"] == 0:
                print(
                    json.dumps({"seed": seed, "arm": arm, "step": tr.step, "ce": row["ce"]}),
                    flush=True,
                )
        if tr.step % c["training"]["checkpoint_every"] == 0:
            # A rejected proposal at the same Cell step is recovered from its transaction until next update.
            if tr.step != latest["step"]:
                latest = save(tr, store)
            # Main state now includes completed proposal decisions; old branch tensors are redundant.
            cleanup_transactions(directory, tr.step)
            print("[R3E_SNAPSHOT]", seed, arm, tr.step, flush=True)
    latest = save(tr, store)
    tr.verify_frozen()
    sync(device)
    wall = time.perf_counter() - begin
    if (
        tr.step != c["training"]["steps"]
        or tr.data_cursor != tr.step * c["training"]["tokens_per_step"]
    ):
        raise RuntimeError("MAIN_BUDGET_INVALID")
    main_work = copy.deepcopy(model.work)
    with torch.no_grad():
        final_paths = paths(model, refs)
        drift = float(
            torch.cat([(a != b).any(-1).float() for a, b in zip(initial, final_paths)]).mean()
        )
    if arm in ("A-F", "B-F", "B-R") and any(
        torch.count_nonzero(p) for p in model.connection.parameters()
    ):
        raise RuntimeError("FROZEN_CONNECTION_CHANGED")
    cost = copy.deepcopy(tr.cost)
    cost["main_work"] = main_work
    main_flops = cost_estimate(main_work, c, model.graph)
    extra_flops = cost_estimate(cost["work"], c, model.graph)
    cost.update(
        main_forward_matmul_estimate=main_flops,
        additional_forward_matmul_estimate=extra_flops,
        forward_backward_upper_proxy=3 * (main_flops + extra_flops),
        wall_training_seconds=wall,
        estimation_scope="Analytic executed linear and dense QK/AV; 3x backward proxy is not measured hardware FLOPs",
        proposal_optimizer_updates=tr.policy_updates,
        proposals=len(tr.proposals),
        total_cell_updates_including_discarded=tr.step + cost["discarded_adaptation_updates"],
        same_total_compute=False,
    )
    g = torch.tensor([r["cell_gradient_norms"] for r in tr.curve])
    coverage = {
        "zero_gradient_fraction_by_cell": (g == 0).float().mean(0).tolist(),
        "mean_gradient_norm_by_cell": g.mean(0).tolist(),
        "cell_calls": torch.tensor([r["cell_counts"] for r in tr.curve]).sum(0).tolist(),
    }
    proposals = copy.deepcopy(tr.proposals)
    # Training gradients and optimizer are released before isolated terminal memory measurement.
    del tr
    for p in model.parameters():
        p.grad = None
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    quality = terminal(model, c, data)
    result = {
        "status": "COMPLETED",
        "identity": ident,
        "checkpoint": latest,
        "quality": quality,
        "training_cost": cost,
        "proposals": proposals,
        "gradient_coverage": coverage,
        "fixed_weight_route_drift": drift,
        "actual_kaggle_persistence_verified": False,
        "confirmation": "SEALED",
    }
    atomic(directory / "result.json", result)
    print("[R3E_SNAPSHOT]", seed, arm, "COMPLETE", flush=True)
    return result


def worker(c, output, assets, source, index, count, device):
    data = load_assets(c["engine"], assets)
    for seed in c["seeds"][index::count]:
        parent = parent_state(c, Path(output) / "parent-r3c", seed)
        for arm in ARMS:
            job(c, data, output, source, seed, arm, parent, device)


def workers(c, config_path, output, assets, checkpoint_callback):
    count = min(c["training"]["max_workers"], torch.cuda.device_count())
    procs = []
    threads = []
    events = queue.Queue()

    def forward(proc, index):
        for line in proc.stdout:
            print("[GPU %s]" % index, line.rstrip(), flush=True)
            if line.startswith("[R3E_SNAPSHOT]"):
                events.put(line)

    for i in range(count):
        cmd = [
            sys.executable,
            "-u",
            str(BASE / "run.py"),
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--assets",
            str(assets),
            "--worker",
            str(i),
            "--workers",
            str(count),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        thread = threading.Thread(target=forward, args=(proc, i))
        thread.start()
        procs.append(proc)
        threads.append(thread)
    while any(p.poll() is None for p in procs):
        try:
            events.get(timeout=0.5)
        except queue.Empty:
            continue
        checkpoint_callback(False)
    for t in threads:
        t.join()
    checkpoint_callback(True)
    return [p.wait() for p in procs]


def automatic(c, path, output, assets):
    source = source_sha()
    run = canonical({"source": source, "config": canonical(c)})
    ident = {
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "run_id": run,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise RuntimeError("OUTPUT_IDENTITY_CONFLICT")
    atomic(output / "identity.json", ident)
    manifest = {
        **ident,
        "started_at_unix": time.time(),
        "confirmation": "SEALED",
        "authorization": "USER_R3E_SPEC_SECTION_18",
        "planned_cell_steps": 9000,
        "planned_main_tokens": 4608000,
        "parent_reference": c["parent"]["hf_reference"],
    }
    failures = []
    results = {}
    backend = None
    prior = None
    try:
        secrets()
        backend = HubBackend(c["storage"]["private_repo"])
        backend.check_write()
        if not backend.info().private:
            raise RuntimeError("HF_PRIVATE_REPO_REQUIRED")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_UNAVAILABLE_NO_TRAINING_STARTED")
        private = output / "private-checkpoints"
        pointer = "r3e/" + run + "/latest.json"
        session = uuid.uuid4().hex
        ref = backend.optional(pointer, backend.info().sha)
        if ref and not list(private.glob("jobs/*/receipts/*.json")):
            prior = {**ref, "hf_repo": backend.repo, "hf_revision": backend.info().sha}
            restore_latest(backend, prior, private, ident, session)
            atomic(output / "hf-reference.json", prior)
        elif (output / "hf-reference.json").exists():
            prior = read(output / "hf-reference.json")
        restore_parents(c, backend, output / "parent-r3c")
        assets = Path(assets or "/kaggle/working/r3c-pilot-assets-d8a0fd0da03e")
        if (assets / "manifest.json").exists():
            load_assets(c["engine"], assets)
        else:
            private_asset_subset(
                backend, c["engine"]["input"]["asset_bundle_ref"], assets, c["engine"]
            )
        last_upload = [0.0]

        def checkpoint_callback(force):
            nonlocal prior
            if (
                not force
                and time.monotonic() - last_upload[0] < c["storage"]["remote_snapshot_min_seconds"]
            ):
                return
            try:
                ref = remote_snapshot(backend, private, output, ident, pointer, session, prior)
                if ref:
                    prior = ref
            except Exception as error:
                atomic(
                    output / "reports/storage-warning.json",
                    {
                        "status": "LOCAL_CHECKPOINTS_RETAINED_REMOTE_RETRY_PENDING",
                        **error_code(error),
                    },
                )
                print(
                    "[R3e] remote upload deferred; local checkpoints retained:",
                    type(error).__name__,
                    flush=True,
                )
            last_upload[0] = time.monotonic()

        print(
            "[R3e] identities passed -> all six arms x three seeds; no additional pilot or approval",
            flush=True,
        )
        codes = workers(c, path, output, assets, checkpoint_callback)
        if any(codes):
            failures.append({"code": "WORKER_FAILED", "exit_codes": codes})
    except Exception as error:
        failures.append(
            {
                "code": getattr(
                    error,
                    "code",
                    str(error)
                    if str(error).isupper() and " " not in str(error)
                    else type(error).__name__,
                ),
                **error_code(error),
            }
        )
    for seed in c["seeds"]:
        per = {}
        for arm in ARMS:
            p = (
                output
                / "private-checkpoints/jobs"
                / ("seed-%s-arm-%s" % (seed, arm))
                / "result.json"
            )
            if p.exists():
                per[arm] = read(p)
        results[str(seed)] = per
    manifest.update(
        hf_reference=prior, finished_at_unix=time.time(), actual_kaggle_persistence_verified=False
    )
    report = write(c, results, source, output, failures, manifest)
    publication = {"status": "NOT_REQUESTED_OR_GITHUB_TOKEN_UNAVAILABLE"}
    if os.environ.get("GITHUB_TOKEN"):
        try:
            publication = public_publish(report, output)
        except Exception as error:
            publication = {"status": "LOCAL_RESULTS_RETAINED", **error_code(error)}
    atomic(output / "reports/PUBLICATION.json", publication)
    print(
        json.dumps(
            {
                "status": report["status"],
                "hypotheses": report["hypotheses"],
                "publication": publication,
                "hf_reference": prior,
                "output": str(output),
            }
        ),
        flush=True,
    )
    if report["status"] != "COMPLETED":
        raise RuntimeError("R3E_INCOMPLETE_SEE_RETAINED_REPORTS")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--assets")
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    c = configuration(a.config)
    output = a.output or str(Path("/kaggle/working") / ("r3e-output-" + source_sha()[:12]))
    if a.worker is not None:
        worker(c, output, a.assets, source_sha(), a.worker, a.workers, "cuda:%s" % a.worker)
    elif a.run_all:
        automatic(c, Path(a.config).resolve(), output, a.assets)
    else:
        p.error("--run-all required")
