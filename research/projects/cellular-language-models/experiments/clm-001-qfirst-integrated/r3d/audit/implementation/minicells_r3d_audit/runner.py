"""One-launch read-only audit. No training workers, optimizer or HF write gate."""

import argparse
from pathlib import Path
import os
import subprocess
import sys
import threading
import time
import torch
from minicells_artifacts.hf_storage import error_code
from .common import BASE, ROOT, atomic, canonical, read, configuration, ID, failure_code
from .inputs import Inputs, secret
from .job import run as run_job
from .reporting import aggregate, markdown, plots


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_reports(output, c, jobs, failures, identity, provenance):
    summary = aggregate(c, jobs, failures, identity, provenance)
    folder = Path(output) / "reports"
    atomic(folder / "AUDIT_SUMMARY.json", summary)
    for name, key in [
        ("COUNTERFACTUAL_VALUES", "counterfactual"),
        ("POLICY_RANKING", "policy_ranking"),
        ("POLICY_INTERPOLATION", "interpolation"),
    ]:
        atomic(
            folder / (name + ".json"),
            {seed: {arm: row[key] for arm, row in per.items()} for seed, per in jobs.items()},
        )
    atomic(folder / "CHECKPOINT_ATTRIBUTION.json", summary["attribution"])
    (folder / "SCIENTIFIC-FINDINGS.md").write_text(markdown(summary))
    return summary


@torch.inference_mode()
def seed_jobs(c, original, historical, output, local, assets, seed, device, audit_source):
    inputs = Inputs(c, original, historical, output, local)
    data, _ = inputs.assets(assets)
    # An immutable reference selected by the coordinator is reused from local disk/cache.
    for arm in c["checkpoint_steps"]:
        unavailable = Path(output) / "private/unavailable" / ("%s-%s.json" % (seed, arm))
        if unavailable.exists():
            atomic(
                Path(output) / "private/failures" / ("%s-%s.json" % (seed, arm)), read(unavailable)
            )
            continue
        try:
            result = run_job(inputs, c, original, historical, data, seed, arm, device, audit_source)
            atomic(Path(output) / "private/completed" / ("%s-%s.json" % (seed, arm)), result)
            stale = Path(output) / "private/failures" / ("%s-%s.json" % (seed, arm))
            if stale.exists():
                stale.unlink()
        except Exception as error:
            failure = {
                "seed": seed,
                "arm": arm,
                "step": c["checkpoint_steps"][arm],
                **error_code(error),
                "code": failure_code(error),
                "status": "ANALYSIS_FAILED_CHECKPOINT_NOT_SUBSTITUTED",
            }
            # A stable missing object/name may be useful, but arbitrary HTTP exception text is private.
            atomic(Path(output) / "private/failures" / ("%s-%s.json" % (seed, arm)), failure)
            print("[R3d-A] failed", seed, arm, failure["code"], failure["type"], flush=True)
    atomic(Path(output) / "private/input-provenance" / ("%s.json" % seed), inputs.provenance())


def execute_workers(c, path, output, local, assets, audit_source):
    count = (
        min(c["runtime"]["max_workers"], torch.cuda.device_count())
        if torch.cuda.is_available()
        else 1
    )
    if count == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for seed in c["seeds"]:
            seed_jobs(
                c, *configuration(path)[1:], output, local, assets, seed, device, audit_source
            )
        return [0]
    processes = []
    readers = []

    def forward(proc, index):
        for line in proc.stdout:
            print("[GPU %s] %s" % (index, line.rstrip()), flush=True)

    for i in range(count):
        args = [
            sys.executable,
            "-u",
            str(BASE / "run.py"),
            "--config",
            str(path),
            "--output",
            str(output),
            "--assets",
            str(assets),
            "--input",
            str(local),
            "--worker",
            str(i),
            "--workers",
            str(count),
        ]
        proc = subprocess.Popen(
            args,
            cwd=ROOT,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        thread = threading.Thread(target=forward, args=(proc, i))
        thread.start()
        processes.append(proc)
        readers.append(thread)
    codes = [p.wait() for p in processes]
    for thread in readers:
        thread.join()
    return codes


def collect(output, c):
    jobs = {}
    failures = []
    output = Path(output)
    for seed in c["seeds"]:
        per = {}
        for arm in c["checkpoint_steps"]:
            p = output / "private/completed" / ("%s-%s.json" % (seed, arm))
            f = output / "private/failures" / ("%s-%s.json" % (seed, arm))
            if f.exists():
                failures.append(read(f))
            elif p.exists():
                per[arm] = read(p)
            else:
                failures.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "step": c["checkpoint_steps"][arm],
                        "status": "NOT_COMPLETED_OR_INPUT_NOT_AVAILABLE",
                    }
                )
        if per:
            jobs[str(seed)] = per
    return jobs, failures


def automatic(c, original, historical, config_path, output, local=None, assets=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_source = source_sha()
    identity = {
        "experiment_id": ID,
        "audit_source_sha": audit_source,
        "audit_config_sha256": canonical(c),
        "training_source_sha": c["training_source_sha"],
        "training_config_sha256": c["training_config_sha256"],
        "seeds": c["seeds"],
        "data_roles": list(c["sampling"]["roles"]),
        "assets_manifest_sha256": original["engine"]["input"]["assets_manifest_sha256"],
        "tokenizer_sha256": original["engine"]["input"]["tokenizer_sha256"],
        "confirmation": "SEALED",
    }
    if (output / "identity.json").exists() and read(output / "identity.json") != identity:
        raise RuntimeError("AUDIT_OUTPUT_IDENTITY_CONFLICT")
    atomic(output / "identity.json", identity)
    inputs = Inputs(c, original, historical, output, local)
    started = time.perf_counter()
    failure = None
    codes = []
    try:
        data, asset_path = inputs.assets(assets)
        del data
        # Resolve actual final identities, not every historical intermediate checkpoint.
        available = 0
        for seed in c["seeds"]:
            for arm in c["checkpoint_steps"]:
                marker = output / "private/unavailable" / ("%s-%s.json" % (seed, arm))
                try:
                    inputs.descriptor(seed, arm)
                    available += 1
                    if marker.exists():
                        marker.unlink()
                except Exception as error:
                    atomic(
                        marker,
                        {
                            "seed": seed,
                            "arm": arm,
                            "step": c["checkpoint_steps"][arm],
                            "code": failure_code(error),
                            **error_code(error),
                            "status": "INPUT_UNAVAILABLE_NO_SUBSTITUTE",
                        },
                    )
        if not available:
            raise RuntimeError("NO_REQUIRED_CHECKPOINT_AVAILABLE")
        print(
            "[R3d-A] inputs resolved. Read-only 18 checkpoints; 2,880 sampled states / 43,200 all-action evaluations. No tests/Pilot/training phase.",
            flush=True,
        )
        codes = execute_workers(c, config_path, output, inputs.local, asset_path, audit_source)
    except Exception as error:
        failure = {
            **error_code(error),
            "code": failure_code(error),
            "scope": "INPUT_OR_EXECUTION_UNAVAILABLE_NO_RANDOM_SUBSTITUTE",
        }
        print("[R3d-A] stopped:", failure["code"], flush=True)
    jobs, failures = collect(output, c)
    if failure:
        failures.insert(0, failure)
    summary = write_reports(output, c, jobs, failures, identity, inputs.provenance())
    summary["runtime"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else ["CPU"],
        "worker_exit_codes": codes,
    }
    atomic(output / "reports/AUDIT_SUMMARY.json", summary)
    if jobs:
        plots(summary, output / "reports/figures")
    publication = {"status": "NOT_REQUESTED_OR_GITHUB_CREDENTIAL_NOT_AVAILABLE"}
    if secret("GITHUB_TOKEN", required=False):
        try:
            from .publication import publish

            print(
                "[R3d-A] analysis saved; optional aggregate-only GitHub return begins", flush=True
            )
            publication = publish(output, summary)
        except Exception as error:
            publication = {
                "status": "OPTIONAL_RETURN_FAILED_LOCAL_RESULTS_RETAINED",
                **error_code(error),
            }
    atomic(output / "reports/PUBLICATION.json", publication)
    print(
        {
            "status": summary["status"],
            "completed_checkpoints": summary["completed_checkpoints"],
            "recommendation": summary["conclusions"]["recommendation"],
            "training_updates": 0,
            "optimizer_instances": 0,
            "output": str(output),
            "publication": publication,
        },
        flush=True,
    )
    if summary["status"] != "READ_ONLY_AUDIT_COMPLETE":
        raise RuntimeError("READ_ONLY_AUDIT_INCOMPLETE_SEE_SAVED_REPORTS")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--output", default="/kaggle/working/r3d-audit-output")
    p.add_argument("--input")
    p.add_argument("--assets")
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    c, original, historical = configuration(a.config)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if a.worker is not None:
        torch.cuda.set_device(a.worker)
        for seed in c["seeds"][a.worker :: a.workers]:
            seed_jobs(
                c,
                original,
                historical,
                a.output,
                a.input,
                a.assets,
                seed,
                "cuda:" + str(a.worker),
                source_sha(),
            )
    elif a.run_all:
        automatic(c, original, historical, Path(a.config).resolve(), a.output, a.input, a.assets)
    else:
        p.error("use --run-all; only the predefined read-only analysis executes")
