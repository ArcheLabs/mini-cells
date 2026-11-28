"""A3.3: isolate the effect of an extra SDK Write before an identical restored Query."""
import argparse
from collections import Counter
import inspect
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import torch
from minicells_native import MiniCells

from a31_norm import describe_norm
from common import publishable, write
from m1_continuation_isolation import (
    BOUNDS_FILE, BOUNDS_SHA256, PRE_LAYERS, PRE_TENSORS,
    compare_pair, frozen_bounds, imported, produce, start_differences,
)
from m1_full_state_bridge import baseline, check_environment, dataset, load_json, probe_inputs
from m1_l2norm_fixed_e2e import fixed_forward
from m1_live_vs_restored import trial, warm
from m1_reference_capacity import digest_file
from r1_equivalence import load_trace

FORMAT = "minicells.m1.full-state.process-history.v1"
A32_COMMIT = "45f95af8a66cf4a5ee9d9b88ebd4a3e754e52a9b"
ORDER = ("P0", "A1", "B1", "B2", "A2")
ARMS = {"P0": "live_reference", "A1": "no_write", "A2": "no_write",
        "B1": "extra_write", "B2": "extra_write"}


@torch.inference_mode()
def worker(out, label):
    from fla.modules import l2norm

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.startswith("GPU-") or "," in gpu:
        raise RuntimeError("Pin all workers to one physical GPU UUID")
    if "self.configs[0]" not in inspect.getsource(type(l2norm.l2norm_fwd_kernel).run):
        raise RuntimeError("Installed Triton lacks the audited singleton config branch")
    source = describe_norm(l2norm.l2_norm, l2norm)
    calls = Counter()
    stage = "initialization"
    control = None
    work = out / "work"
    archive = work / "prefix.cell"
    torch.set_num_threads(4)
    with (fixed_forward(l2norm, calls, lambda: stage) as policy,
          MiniCells.open_preview(work / f"registry-{label}") as mc):
        env = check_environment(mc)
        if label == "P0":
            write(out / "environment.json", env)
        elif env != load_json(out / "environment.json"):
            raise RuntimeError("Worker differs from the reference runtime identity")
        texts, writes, query = probe_inputs(dataset(out), mc._backbone)
        stage = "warmup"
        warm(mc._backbone, writes, query)
        if label == "P0":
            stage = "write"
            cell, cache, _ = produce(mc, texts, work)
            archive.chmod(0o444)
        else:
            shared = load_json(work / "shared.json")
            if digest_file(archive) != shared["bundle_sha256"]:
                raise RuntimeError("Read-only reference Bundle SHA-256 changed")
            if ARMS[label] == "extra_write":
                stage = "write"
                # Reproduce P0's SDK write and capture path in a separate directory.
                # Its Cache and Bundle are never used as the query starting state.
                with TemporaryDirectory(prefix=f"discarded-{label}-", dir=work) as temp:
                    discarded, native, state = produce(mc, texts, Path(temp))
                    control = dict(start_differences=start_differences(shared["pre"], state),
                                   bundle_sha256=digest_file(Path(temp) / "prefix.cell"))
                    mc.cells.delete(discarded.id)
                    del native
            stage = "import"
            cell, cache = imported(mc, archive)
        stage = "query"
        trial(mc, out, label, cell, cache, writes, query, ARMS[label])
    expected = {"warmup", "query"}
    if label == "P0" or ARMS[label] == "extra_write":
        expected.add("write")
    if {key[0] for key in calls} != expected:
        raise RuntimeError(f"Unexpected fixed L2 Norm coverage for {label}: {sorted(calls)}")
    keys = ("segment", "D", "T", "NB", "BD", "BT", "num_warps", "num_stages", "num_ctas")
    write(out / f"policy-{label}.json", dict(
        label=label, arm=ARMS[label], pid=os.getpid(), gpu_uuid=gpu, source=source,
        fixed_config=policy, discarded_write=control,
        launches=[dict(zip(keys, key, strict=True), calls=count) for key, count in sorted(calls.items())]))


def interpretation(pairs):
    exact = lambda a, b: pairs[f"{a}_vs_{b}"]["exact"]
    if not (exact("A1", "A2") and exact("B1", "B2")):
        return "WITHIN_ARM_DIFFERENCE"
    joined_arms = exact("A1", "B1")
    matches_p0 = {arm: exact("P0", label) for arm, label in (("A", "A1"), ("B", "B1"))}
    if joined_arms:
        return "ALL_EXACT" if matches_p0["A"] else "ARMS_MATCH_DIFFER_FROM_LIVE"
    if matches_p0["B"] and not matches_p0["A"]:
        return "WRITE_HISTORY_REPRODUCES_LIVE"
    if matches_p0["A"] and not matches_p0["B"]:
        return "WRITE_HISTORY_SHIFTS_AWAY_FROM_LIVE"
    return "ARMS_DIFFER_NEITHER_MATCHES_LIVE"


def evaluate(out, result):
    work = out / "work"
    policies = {label: load_json(out / f"policy-{label}.json") for label in ORDER}
    receipts = {label: load_json(work / f"{label}.json") for label in ORDER}
    shared = load_json(work / "shared.json")
    expected = shared["pre"]
    if expected["layer_count"] != PRE_LAYERS or len(expected["tensor_sha256"]) != PRE_TENSORS:
        raise RuntimeError("Reference Bundle must have 24 layers and 263 tensors")
    if (len({r["pid"] for r in policies.values()}) != len(ORDER)
            or len({r["gpu_uuid"] for r in policies.values()}) != 1
            or len({r["input_sha256"] for r in receipts.values()}) != 1
            or any(policy["pid"] != receipts[label]["pid"]
                   or policy["arm"] != ARMS[label]
                   or policy["fixed_config"] != policies["P0"]["fixed_config"]
                   or policy["source"] != policies["P0"]["source"]
                   for label, policy in policies.items())):
        raise RuntimeError("Worker independence, inputs, device or normalization policy mismatch")
    write(out / "initial_state_comparison.json", dict(run_order=ORDER, reference=shared, trials=receipts))
    if (digest_file(work / "prefix.cell") != shared["bundle_sha256"]
            or any(r["start_failures"] or not r.get("bundle_unchanged")
                   or r["bundle_sha256"] != shared["bundle_sha256"]
                   or start_differences(expected, r["state"]) for r in receipts.values())):
        raise RuntimeError("A worker failed the strict start-state or Bundle integrity gate")

    bounds = load_json(out / BOUNDS_FILE)
    pairs = {}
    with (out / "pair_fields.jsonl").open("w", encoding="utf-8") as stream:
        for a, b in combinations(ARMS, 2):
            key = f"{a}_vs_{b}"
            summary, rows = compare_pair(load_trace(work / f"trace-{a}"),
                                         load_trace(work / f"trace-{b}"), bounds)
            exact = summary["passed"] and all(row.get("max_absolute") == 0 for row in rows)
            pairs[key] = dict(summary, exact=exact)
            for row in rows:
                stream.write(json.dumps(dict(pair=key, **row), allow_nan=False) + "\n")
    pattern = interpretation(pairs)
    passed = all(pair["passed"] for pair in pairs.values())
    write(out / "process_history.json", dict(pattern=pattern, pairs=pairs))
    result.update(
        status="PASS" if passed else "FAIL", engineering_status="PASS",
        state_equivalence="PASS" if passed else "FAIL",
        diagnosis=f"{pattern}. {sum(p['passed'] for p in pairs.values())}/{len(pairs)} pairs within frozen bounds.",
        gates={"Exact start state": "PASS", "Read-only Bundle": "PASS", "Fixed L2 Norm": "PASS"},
        next_action="Review the comparison pattern before a targeted first-block operator trace. R1-B remains closed.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", choices=ORDER)
    args = parser.parse_args()
    out = args.output
    if args.worker:
        try:
            worker(out, args.worker)
        except Exception as exc:
            write(out / f"worker_error-{args.worker}.json", dict(error=type(exc).__name__, message=str(exc)))
            raise
        return
    (out / "work").mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="A3.3 Process History Isolation", status="INVALID", engineering_status="FAIL",
                  state_equivalence="INCOMPLETE", bridge_status="NOT_RUN_BY_SCOPE",
                  capacity_status="NOT_RUN_BY_SCOPE", a32_results_commit=A32_COMMIT,
                  frozen_bounds_sha256=BOUNDS_SHA256, scientific_status="DEVELOPMENT_ONLY")
    try:
        baseline(out)
        frozen_bounds(out)
        for label in ORDER:
            proc = subprocess.run([sys.executable, __file__, str(out.resolve()), "--worker", label])
            if proc.returncode:
                error = out / f"worker_error-{label}.json"
                raise RuntimeError(f"Worker {label}: {load_json(error) if error.exists() else proc.returncode}")
        evaluate(out, result)
    except Exception as exc:
        result.update(diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
