"""M1-002A-R1-A2: identical-start continuation in three independent processes."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from minicells_native import MiniCells
from minicells_native.full_state import FullState
from common import canonical_hash, publishable, write
from m1_full_state_bridge import baseline, check_environment, dataset, load_json, probe_inputs
from m1_reference_capacity import digest_file
from r1_equivalence import compare, ensure_readonly, load_trace, save_trace, trace

FORMAT = "minicells.m1.full-state.continuation-isolation.v1"
R1_COMMIT = "4e65d9b5ca3012151baf6c4e241337703d3e228d"
BOUNDS_FILE = "live_bounds_cross_prefix.json"
BOUNDS_SHA256 = "549b9b145b19f7a0231949e8fb650e11fac39057d46c104b9cadb22c08c693d3"
BOUNDS_URL = (f"https://raw.githubusercontent.com/ArcheLabs/mini-cells/{R1_COMMIT}/"
              "artifacts/experiments/minicells-m1-002a-r1-full-state-bridge/e86a3c239473/" + BOUNDS_FILE)
INSTANCES = ("A", "B", "C")
PAIRS = (("A", "B"), ("B", "C"), ("A", "C"))
PRE_LAYERS = 24
PRE_TENSORS = 263


def frozen_bounds(out):
    with urlopen(BOUNDS_URL, timeout=30) as response:
        data = response.read()
    if sha256(data).hexdigest() != BOUNDS_SHA256:
        raise ValueError("Published R1 Live-vs-Live bound SHA-256 mismatch")
    bounds = json.loads(data)
    if len(bounds) != 554 or not all(math.isfinite(value) and value >= 0 for value in bounds.values()):
        raise ValueError("Invalid frozen bounds")
    (out / BOUNDS_FILE).write_bytes(data)


def fingerprint(structure, tensors):
    return dict(structure_sha256=canonical_hash(structure),
                cursor=structure["sequence_length"], layer_count=structure["layer_count"],
                tensor_sha256={k: sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                               for k, t in sorted(tensors.items())})


def start_differences(expected, actual):
    differences = [key for key in ("structure_sha256", "cursor", "layer_count")
                   if expected[key] != actual[key]]
    before, now = expected["tensor_sha256"], actual["tensor_sha256"]
    differences.extend(key for key in sorted(before.keys() | now.keys()) if before.get(key) != now.get(key))
    return differences


def trace_fingerprint(result):
    structures, tensors = result
    return {phase: fingerprint(structures[phase],
                              {k.split(".", 1)[1]: t for k, t in tensors.items() if k.startswith(phase + ".")})
            for phase in ("pre", "post")}


def produce(mc, texts, work, *, require_historical_count=True):
    """Capture at the SDK write boundary and retain that exact original live object."""
    cell = mc.cells.create("Original live snapshot", full_state=True)
    captured = []
    capture = FullState.capture

    def record(current_backbone, cache):
        state = capture(current_backbone, cache)
        captured.append((cache, fingerprint(state.structure, state.tensors)))
        return state

    with patch.object(FullState, "capture", side_effect=record):
        mc.cells.write(cell.id, texts[0], resume=False)
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one capture from the original Live Cache")
    native, expected = captured.pop()
    if (expected["layer_count"] != PRE_LAYERS
            or (require_historical_count and len(expected["tensor_sha256"]) != PRE_TENSORS)):
        raise ValueError("The frozen prefix no longer yields 24 layers and 263 tensors")
    mc.cells.export(cell.id, work / "prefix.cell")
    digest = digest_file(work / "prefix.cell")
    if digest_file(mc.cells._store.path(cell.id)) != digest:
        raise RuntimeError("Exported bytes differ from the source Cell")
    write(work / "shared.json", dict(bundle_sha256=digest, pre=expected))
    return cell, native, expected


def imported(mc, archive):
    cell = mc.cells.import_cell(archive)
    state = mc.cells._state(cell.id)
    return cell, state.restore(mc._backbone)


@torch.inference_mode()
def worker(out, instance):
    torch.set_num_threads(4)
    work = out / "work"
    with MiniCells.open_preview(work / instance) as mc:
        env = check_environment(mc)
        if instance == "A":
            write(out / "environment.json", env)
        elif env != load_json(out / "environment.json"):
            raise RuntimeError("A2 workers do not share the frozen runtime identity")
        backbone = mc._backbone
        texts, writes, query = probe_inputs(dataset(out), backbone)
        query_sha = canonical_hash(dict(writes=writes[:1], query=query))
        archive = work / "prefix.cell"
        if instance == "A":
            cell, native, snapshot = produce(mc, texts, work)
            shared = load_json(work / "shared.json")
        else:
            shared = load_json(work / "shared.json")
            if digest_file(archive) != shared["bundle_sha256"]:
                raise RuntimeError("Input Bundle SHA-256 changed")
            cell, native = imported(mc, archive)
            state = FullState.capture(backbone, native)
            snapshot = fingerprint(state.structure, state.tensors)
            del state
        before = digest_file(mc.cells._store.path(cell.id))
        receipt = dict(instance=instance, pid=os.getpid(), bundle_sha256=before,
                       query_sha256=query_sha, start=snapshot,
                       primary_path="original_live_object" if instance == "A" else "restored_cache",
                       start_failures=start_differences(shared["pre"], snapshot))
        if before != shared["bundle_sha256"]:
            receipt["start_failures"].append("bundle_sha256")
        path = out / f"instance_{instance}.json"
        write(path, receipt)
        if receipt["start_failures"]:
            print(f"A2_START_FAILED {instance}: {receipt['start_failures'][:4]}", flush=True)
            return
        pristine = deepcopy(native)
        # The primary teacher-forced trace mutates the actual native Cache object.
        # Alternate candidate scoring forks the pristine pre-query in-memory copy.
        result = ensure_readonly(mc, cell, lambda: trace(
            backbone, lambda: deepcopy(pristine), query, primary_cache=native))
        states = trace_fingerprint(result)
        if states["pre"] != snapshot:
            raise RuntimeError("The primary query did not start from the verified Cache")
        if digest_file(archive) != before:
            raise RuntimeError("Query changed the shared Bundle")
        save_trace(work / f"trace_{instance}", result)
        receipt.update(states=states, bundle_unchanged=True)
        write(path, receipt)
        print(f"A2_COMPLETED {instance} PID={os.getpid()}", flush=True)


def compare_pair(reference, observed, original_bounds):
    # Exact start is checked bitwise in each worker. The comparison also uses a
    # zero pre-query numeric bound as an independent check.
    bounds = {k: 0.0 if k.startswith("pre.") else v for k, v in original_bounds.items()}
    summary, rows = compare(reference, observed, bounds)
    summary["fields_without_historical_bound"] = sorted((reference[1].keys() | observed[1].keys()) - bounds.keys())
    return summary, rows


def interpretation(pairs):
    ab, bc, ac = (pairs[key]["passed"] for key in ("A_vs_B", "B_vs_C", "A_vs_C"))
    if ab and bc and ac:
        return "MATCHED_STATE_CONTINUATION_PASS", "Evaluate the remaining R1-A Append gate before opening R1-B."
    if not bc:
        return "RESTORED_WORKERS_DIVERGED", "Investigate GPU execution reproducibility with identical starting state."
    return "LIVE_RESTORED_DIVERGED", "Inspect the live and restored execution paths from the matched starting state."


def evaluate(out, result):
    receipts = {k: load_json(out / f"instance_{k}.json") for k in INSTANCES}
    if (len({r["pid"] for r in receipts.values()}) != 3
            or len({r["query_sha256"] for r in receipts.values()}) != 1):
        raise RuntimeError("Expected independent processes and identical query inputs")
    failed = {k: r["start_failures"] for k, r in receipts.items() if r["start_failures"]}
    if failed:
        result.update(status="FAIL", engineering_status="PASS", state_equivalence="FAIL",
                      diagnosis=f"START_STATE_MISMATCH: {failed}",
                      next_action="Inspect Bundle import, tensor mapping and Cache metadata.")
        write(out / "comparisons.json", dict(start_failures=failed, pairs={}))
        return
    if not all(r.get("bundle_unchanged") for r in receipts.values()):
        raise RuntimeError("A worker did not verify the read-only Bundle")
    bounds = load_json(out / BOUNDS_FILE)
    traces = {k: load_trace(out / "work" / f"trace_{k}") for k in INSTANCES}
    pairs = {}
    summary_path = out / "pair_fields.jsonl"
    with summary_path.open("w", encoding="utf-8") as stream:
        for a, b in PAIRS:
            key = f"{a}_vs_{b}"
            pairs[key], fields = compare_pair(traces[a], traces[b], bounds)
            for field in fields:
                stream.write(json.dumps(dict(pair=key, **field), allow_nan=False) + "\n")
    # Publish lossless per-token observed outputs. The much larger raw Cache
    # tensors stay in the Kaggle run; their SHA-256 and shapes are published.
    save_file({f"{instance}.{key}": tensor for instance, (_, tensors) in traces.items()
               for key, tensor in tensors.items() if key.startswith("query.")}, out / "query_traces.safetensors")
    write(out / "cache_structures.json", {key: state for key, (state, _) in traces.items()})
    write(out / "comparisons.json", dict(start_failures={}, pairs=pairs))
    diagnosis, next_action = interpretation(pairs)
    passed = all(p["passed"] for p in pairs.values())
    result.update(status="PASS" if passed else "FAIL", engineering_status="PASS",
                  state_equivalence="PASS" if passed else "FAIL", diagnosis=diagnosis,
                  next_action=next_action, a2_pairs=pairs,
                  gates={key: "PASS" if p["passed"] else "FAIL" for key, p in pairs.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--instance", choices=INSTANCES)
    args = parser.parse_args()
    out = args.output
    if args.instance:
        try:
            worker(out, args.instance)
        except Exception as exc:
            write(out / f"worker_error_{args.instance}.json", dict(type=type(exc).__name__, message=str(exc)))
            raise
        return
    (out / "work").mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="M1-002A-R1-A2 Cross-Process Continuation Isolation",
                  status="INVALID", engineering_status="FAIL", state_equivalence="NOT_RUN",
                  bridge_status="NOT_RUN_BY_SCOPE", capacity_status="NOT_RUN_BY_SCOPE",
                  r1_results_commit=R1_COMMIT, frozen_bounds_sha256=BOUNDS_SHA256)
    try:
        baseline(out)
        frozen_bounds(out)
        result["state_equivalence"] = "INCOMPLETE"
        for instance in INSTANCES:
            proc = subprocess.run([sys.executable, __file__, str(out), "--instance", instance])
            if proc.returncode:
                error = out / f"worker_error_{instance}.json"
                raise RuntimeError(f"Worker {instance}: {load_json(error) if error.exists() else proc.returncode}")
        evaluate(out, result)
    except Exception as exc:
        result.update(diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
