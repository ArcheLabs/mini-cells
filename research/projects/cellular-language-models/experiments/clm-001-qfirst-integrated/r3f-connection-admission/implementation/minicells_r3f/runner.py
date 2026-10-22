"""Diagnostics -> single-seed admission trial -> locked full matrix -> terminal Development."""

from pathlib import Path
import argparse
import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import torch
from minicells_artifacts.hf_storage import HubBackend, error_code
from minicells_r3e.runner import secrets, cost_estimate
from minicells_r3e.stable_connection import Model as LegacyModel
from minicells_r3e.common import ARMS
from minicells_r3e.reports import write as original_write
from .common import BASE, ROOT, ID, configuration, read, canonical, atomic, source_sha, identity
from .storage import (
    original_states,
    restore_parents,
    parent_state,
    Store,
    remote_snapshot,
    restore_latest,
)
from .stable_connection import build
from .diagnostics import diagnostic
from .data import ensure, load
from .common import digest
from .jobs import train_job
from .evaluation import terminal


def directory(output, seed, arm):
    return Path(output) / "private-checkpoints/jobs" / ("seed-%s-arm-%s" % (seed, arm))


def trial_gate(c, results):
    observations = []
    for arm, row in results.items():
        for event in row["proposals"]:
            if (
                event["at_step"] > c["admission"]["trial_end_step"]
                or event["at_step"] not in c["training"]["proposal_steps"]
            ):
                continue
            observations.append(
                {
                    "arm": arm,
                    "at_step": event["at_step"],
                    "trust_region_passed": event.get("trust_region_passed", False),
                    "cell_adaptation_executed": event.get("cell_adaptation_executed", False),
                    "adapt_steps": event.get("adapt_steps", 0),
                    "final_quality_accepted": event.get("final_quality_accepted", False),
                    "chosen_scale": event.get("chosen_scale"),
                    "reason": event.get("reason"),
                }
            )
    passed = not c.get("synthetic", False) and any(
        r["trust_region_passed"]
        and r["cell_adaptation_executed"]
        and r["adapt_steps"] == c["training"]["adapt_steps"]
        for r in observations
    )
    return {
        "status": "REAL_TRIAL_ADMISSION_PASS"
        if passed
        else "REAL_TRIAL_NO_CANDIDATE_ADMITTED_STOP",
        "passed": passed,
        "seed": c["admission"]["trial_seed"],
        "observations": observations,
        "thresholds_relaxed": False,
        "synthetic_evidence_accepted_as_real": False,
    }


def readonly_stage(c, data, backend, output, device):
    historical, ref = original_states(c, backend, Path(output) / "original-r3e")
    old_events = [event for state in historical.values() for event in state["proposals"]]
    old_outcome = {
        "verified_proposals": len(old_events),
        "trust_admitted": sum(e.get("chosen_scale") is not None for e in old_events),
        "quality_accepted": sum(e["accepted"] for e in old_events),
        "adaptation_executed": sum(e.get("adapt_steps", 0) > 0 for e in old_events),
    }
    report = {
        "status": "READ_ONLY_DIAGNOSTICS_COMPLETE",
        "verified_original_outcome": old_outcome,
        "historical_reference": ref,
        "training_updates": 0,
        "source_sha": source_sha(),
        "config_sha256": canonical(c),
        "models": {},
    }
    for seed in c["seeds"]:
        parent = parent_state(c, Path(output) / "parent-r3c", seed)
        for arm in ("A-T", "A-C", "B-C"):
            corrected = build(c, parent, seed, arm, device, data)
            # Calibrate only at D500. Final R3e weights are used as a separate read-only diagnostic snapshot.
            old = LegacyModel(c, seed, arm).to(device)
            old.load_state_dict(historical[(seed, arm)]["model"], strict=True)
            old.route_switch_margin = corrected.route_switch_margin.clone()
            old.route_numeric_tolerance = corrected.route_numeric_tolerance.clone()
            old.margin_evidence = copy.deepcopy(corrected.margin_evidence)
            weights = corrected.state_dict()
            weights.update(old.state_dict())
            corrected.load_state_dict(weights, strict=True)
            for p in old.parameters():
                p.requires_grad_(False)
            for p in corrected.parameters():
                p.requires_grad_(False)
            report["models"][str(seed) + "/" + arm] = {
                "historical": diagnostic(old, c, data, historical=True),
                "repaired": diagnostic(corrected, c, data, historical=True),
            }
            print("[R3f] read-only diagnostic completed", seed, arm, flush=True)
    atomic(Path(output) / "reports/ROUTING_DIAGNOSTICS.json", report)
    return report


def evaluate_job(c, data, output, source, seed, arm, parent, device):
    root = directory(output, seed, arm)
    row = read(root / "latest.json")
    ident = identity(c, source, seed, arm)
    state = Store(root, ident).verified(row)
    if row["step"] != c["training"]["steps"]:
        raise RuntimeError("TERMINAL_EVALUATION_BEFORE_FULL_BUDGET")
    result = read(root / "result.json")
    if result["identity"] != ident or result["checkpoint"] != row:
        raise RuntimeError("TERMINAL_RESULT_IDENTITY_CHANGED")
    if "quality" in result:
        return result
    model = build(c, parent, seed, arm, device, data)
    model.load_state_dict(state["model"], strict=True)
    del state
    result["quality"] = terminal(model, c, data)
    result["status"] = "COMPLETED"
    atomic(root / "result.json", result)
    return result


def worker(c, output, assets, source, index, count, phase):
    roles = (
        ("train", "calibration") if phase in ("train", "trial") else ("calibration", "development")
    )
    data = load(c, assets, roles)
    if phase == "trial":
        seed = c["admission"]["trial_seed"]
        parent = parent_state(c, Path(output) / "parent-r3c", seed)
        for arm in c["admission"]["trial_arms"][index::count]:
            train_job(
                c,
                data,
                output,
                source,
                seed,
                arm,
                parent,
                "cuda:%s" % index,
                c["admission"]["trial_end_step"],
            )
        return
    for seed in c["seeds"][index::count]:
        parent = parent_state(c, Path(output) / "parent-r3c", seed)
        for arm in ARMS:
            if phase == "train":
                train_job(c, data, output, source, seed, arm, parent, "cuda:%s" % index)
            else:
                evaluate_job(c, data, output, source, seed, arm, parent, "cuda:%s" % index)


def workers(c, path, output, assets, phase, callback):
    count = min(c["training"]["max_workers"], torch.cuda.device_count())
    procs = []
    threads = []
    events = queue.Queue()

    def lines(proc, i):
        for line in proc.stdout:
            print("[GPU %s]" % i, line.rstrip(), flush=True)
            if line.startswith("[R3F_SNAPSHOT]"):
                events.put(True)

    for i in range(count):
        cmd = [
            sys.executable,
            "-u",
            str(BASE / "run.py"),
            "--config",
            str(path),
            "--output",
            str(output),
            "--assets",
            str(assets),
            "--worker",
            str(i),
            "--workers",
            str(count),
            "--phase",
            phase,
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        t = threading.Thread(target=lines, args=(proc, i))
        t.start()
        procs.append(proc)
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
    if any(p.wait() for p in procs):
        raise RuntimeError("R3F_WORKER_FAILED_LOCAL_CHECKPOINTS_RETAINED")


def gather(c, output):
    result = {}
    for seed in c["seeds"]:
        per = {}
        for arm in ARMS:
            p = directory(output, seed, arm) / "result.json"
            if p.exists():
                r = read(p)
                if r.get("status") == "COMPLETED":
                    per[arm] = r
        result[str(seed)] = per
    return result


def reports(c, output, source, manifest, failures):
    # Reuse fixed NLL/Story bootstrap/Holm logic; provenance/experiment identity belongs to R3f.
    results = gather(c, output)
    report = original_write(c, results, source, output, failures, manifest)
    report["experiment_id"] = ID
    report["origin_r3e"] = c["origin_r3e"]
    report["status"] = "COMPLETE" if report["status"] == "COMPLETED" else "BLOCKED_OR_PARTIAL"
    atomic(output / "reports/METRICS.json", report)
    if not (output / "reports/ROUTING_DIAGNOSTICS.json").exists():
        atomic(
            output / "reports/ROUTING_DIAGNOSTICS.json",
            {
                "status": "NOT_RUN_INPUT_ACCESS_BLOCKED",
                "source_sha": source,
                "config_sha256": canonical(c),
                "real_states": 0,
                "failures": failures,
            },
        )
    if not (output / "reports/SINGLE_SEED_TRIAL.json").exists():
        atomic(
            output / "reports/SINGLE_SEED_TRIAL.json",
            {
                "status": "NOT_RUN",
                "passed": False,
                "reason": "READ_ONLY_STAGE_NOT_COMPLETED",
                "full_matrix_allowed": False,
            },
        )
    events = []
    costs = {}
    for seed in c["seeds"]:
        for arm in ARMS:
            root = directory(output, seed, arm)
            latest = root / "latest.json"
            if not latest.exists():
                continue
            row = read(latest)
            state = Store(root, row["identity"]).verified(row)
            costs.setdefault(str(seed), {})[arm] = {
                "committed_cell_step": state["step"],
                "main_tokens": state["data_cursor"],
                "training": state["cost"],
                "main_work": state["work"],
                "main_forward_matmul_estimate": cost_estimate(
                    state["work"], c, arm.startswith("B-")
                ),
                "extra_forward_matmul_estimate": cost_estimate(
                    state["cost"]["work"], c, arm.startswith("B-")
                ),
                "scope": "Committed update costs; current-job wall times reported separately; interrupted uncommitted compute not reconstructible",
            }
            for p in state["proposals"]:
                events.append({"seed": seed, "arm": arm, **p})
    atomic(output / "reports/TRAINING_COST.json", costs)
    admission_rows = {}
    for seed in c["seeds"]:
        for arm in ("A-T", "A-C", "B-C"):
            rows = [r for r in events if r["seed"] == seed and r["arm"] == arm]
            n = len(rows)
            admission_rows[str(seed) + "/" + arm] = {
                "proposals": n,
                "trust_region_passed": sum(r["trust_region_passed"] for r in rows),
                "adaptation_executed": sum(r["cell_adaptation_executed"] for r in rows),
                "final_quality_accepted": sum(r["final_quality_accepted"] for r in rows),
                "admission_rate": sum(r["trust_region_passed"] for r in rows) / n if n else None,
                "quality_acceptance_rate": sum(r["final_quality_accepted"] for r in rows) / n
                if n
                else None,
                "events": rows,
            }
    atomic(output / "reports/ADAPTATION_RESULTS.json", admission_rows)
    trace = [
        {"seed": e["seed"], "arm": e["arm"], "at_step": e["at_step"], **a}
        for e in events
        for a in e["attempts"]
    ]
    (output / "reports/TRUST_REGION_TRACE.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in trace)
    )
    atomic(
        output / "reports/R3E_COMPARISON.json",
        {
            "origin": c["origin_r3e"],
            "original_reported_by_user": {
                "proposals": 36,
                "trust_admitted": 0,
                "adaptation_executed": 0,
            },
            "current_admissions": admission_rows,
            "scientific_quality": report["nll"],
            "interpretation": "New margin policy and paired RNG; not an unchanged R3e rerun. No claims of specialization.",
        },
    )
    finding = output / "reports/SCIENTIFIC-FINDINGS.md"
    text = finding.read_text().replace("R3e exploratory", "R3f admission repair exploratory")
    text += "\nOriginal R3e has not been overwritten. Mechanism admission/adaptation is separate from terminal quality.\n"
    if failures:
        text += "No full-matrix release or scientific conclusion may be inferred from an incomplete run.\n"
    finding.write_text(text)
    return report


def automatic(c, path, output, assets):
    if c.get("synthetic"):
        raise RuntimeError("SYNTHETIC_CONFIGURATION_CANNOT_RELEASE_REAL_MATRIX")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = source_sha()
    run = canonical({"source": source, "config": canonical(c)})
    ident = {
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "run_id": run,
    }
    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise RuntimeError("OUTPUT_IDENTITY_CONFLICT")
    atomic(output / "identity.json", ident)
    manifest = {
        **ident,
        "confirmation": "SEALED",
        "stage": "A_READONLY_DIAGNOSTICS",
        "actual_kaggle_persistence_verified": False,
    }
    failures = []
    prior = None
    try:
        secrets()
        backend = HubBackend(c["storage"]["private_repo"])
        backend.check_write()
        if not backend.info().private:
            raise RuntimeError("PRIVATE_REPO_REQUIRED")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_UNAVAILABLE_NO_REAL_TRIAL")
        private = output / "private-checkpoints"
        session = uuid.uuid4().hex
        pointer = "r3f/" + run + "/latest.json"
        revision = backend.info().sha
        ref = backend.optional(pointer, revision)
        if ref and not list(private.glob("jobs/*/receipts/*.json")):
            prior = {**ref, "hf_repo": backend.repo, "hf_revision": revision}
            restore_latest(backend, prior, private, ident, session)
        elif (output / "hf-reference.json").exists():
            prior = read(output / "hf-reference.json")
        restore_parents(c, backend, output / "parent-r3c")
        assets = Path(assets or "/kaggle/working/r3c-pilot-assets-d8a0fd0da03e")
        calibration_data = ensure(c, backend, assets, ("calibration",))
        readonly_stage(c, calibration_data, backend, output, "cuda:0")
        last = [0.0]

        def callback(force):
            nonlocal prior
            if (
                not force
                and time.monotonic() - last[0] < c["storage"]["remote_snapshot_min_seconds"]
            ):
                return
            try:
                prior = (
                    remote_snapshot(backend, private, output, ident, pointer, session, prior)
                    or prior
                )
            except Exception as error:
                atomic(
                    output / "reports/storage-warning.json",
                    {"status": "LOCAL_RETAINED_RETRY_PENDING", **error_code(error)},
                )
            last[0] = time.monotonic()

        manifest["stage"] = "C_SINGLE_SEED_TRIAL"
        ensure(c, backend, assets, ("train", "calibration"))
        workers(c, path, output, assets, "trial", callback)
        seed = c["admission"]["trial_seed"]
        trial = {}
        for arm in c["admission"]["trial_arms"]:
            root = directory(output, seed, arm)
            row = read(root / "latest.json")
            state = Store(root, identity(c, source, seed, arm)).verified(row)
            if row["step"] < c["admission"]["trial_end_step"]:
                raise RuntimeError("SINGLE_SEED_TRIAL_INCOMPLETE")
            trial[arm] = {
                "identity": row["identity"],
                "checkpoint": row,
                "proposals": state["proposals"],
            }
        gate = trial_gate(c, trial)
        atomic(output / "reports/SINGLE_SEED_TRIAL.json", gate)
        prior = remote_snapshot(backend, private, output, ident, pointer, session, prior) or prior
        if not gate["passed"]:
            raise RuntimeError("SINGLE_SEED_NO_ADMISSION_STOP_NO_THRESHOLD_RELAXATION")
        atomic(
            output / "reports/MATRIX_RELEASE.json",
            {
                "identity": ident,
                "calibration_margins_frozen": True,
                "calibration_diagnostic_sha256": digest(
                    output / "reports/ROUTING_DIAGNOSTICS.json"
                ),
                "real_trial_gate": gate,
                "code_config_assets_locked_before_development": True,
                "development_accessed": False,
            },
        )
        manifest["stage"] = "D_FULL_TRAINING"
        workers(c, path, output, assets, "train", callback)
        # Only now may Development arrays be loaded and terminal evaluations begin.
        manifest["stage"] = "D_TERMINAL_EVALUATION"
        ensure(c, backend, assets, ("calibration", "development"))
        workers(c, path, output, assets, "evaluate", callback)
        manifest["stage"] = "COMPLETE"
    except Exception as error:
        code = getattr(error, "code", None) or (
            str(error) if str(error).isupper() and " " not in str(error) else type(error).__name__
        )
        failures.append({"code": code, **error_code(error)})
    manifest["hf_reference"] = prior
    report = reports(c, output, source, manifest, failures)
    publication = {"status": "NOT_REQUESTED_OR_GITHUB_TOKEN_UNAVAILABLE"}
    if os.environ.get("GITHUB_TOKEN"):
        try:
            from .publication import publish

            publication = publish(output)
        except Exception as error:
            publication = {"status": "LOCAL_RETAINED", **error_code(error)}
    atomic(output / "reports/PUBLICATION.json", publication)
    print(
        json.dumps(
            {
                "status": report["status"],
                "stage": manifest["stage"],
                "failures": failures,
                "output": str(output),
                "publication": publication,
            }
        ),
        flush=True,
    )
    if failures:
        raise RuntimeError("R3F_STOPPED_SEE_RETAINED_DIAGNOSTICS")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--publish-existing", action="store_true")
    p.add_argument("--output")
    p.add_argument("--assets")
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--phase", choices=["train", "trial", "evaluate"])
    a = p.parse_args()
    c = configuration(a.config)
    out = a.output or str(Path("/kaggle/working") / ("r3f-output-" + source_sha()[:12]))
    if a.publish_existing:
        from .publication import publish

        if not a.output:
            p.error("--publish-existing requires the retained --output directory")
        print(json.dumps(publish(out)))
    elif a.worker is not None:
        worker(c, out, a.assets, source_sha(), a.worker, a.workers, a.phase)
    elif a.run_all:
        automatic(c, Path(a.config).resolve(), out, a.assets)
    else:
        p.error("--run-all required")
