"""M1-002A-R1-A3: repeated exact-start continuation and first-block localization."""
import argparse
from copy import deepcopy
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from minicells_native import MiniCells
from minicells_native.full_state import FullState

from a3_operators import Observer, alignment, instrument, replay, save_observations
from common import canonical_hash, publishable, write
from m1_continuation_isolation import (
    BOUNDS_FILE, BOUNDS_SHA256, compare_pair, fingerprint, frozen_bounds,
    imported, produce, start_differences, trace_fingerprint,
)
from m1_full_state_bridge import baseline, check_environment, dataset, load_json, probe_inputs
from m1_reference_capacity import digest_file
from r1_equivalence import ensure_readonly, live, load_trace, save_trace, trace

FORMAT = "minicells.m1.full-state.live-vs-restored.v1"
A2_COMMIT = "81c89c6834f4cd6fbb24954d52b3190246be5807"
GROUPS = ("P1", "P2", "P3")
LABELS = ("P0", *(f"{p}-{i}" for p in GROUPS for i in range(1, 4)))


def formal_pairs():
    # Within-process repeats, all Live comparisons, and matched trial indices across processes.
    yield from ((LABELS[0], label) for label in LABELS[1:])
    for group in GROUPS:
        yield from combinations((f"{group}-{i}" for i in range(1, 4)), 2)
    for a, b in combinations(GROUPS, 2):
        yield from ((f"{a}-{i}", f"{b}-{i}") for i in range(1, 4))


def execution_inputs(backbone, writes, query, cache):
    first_answer = next(iter(query["candidates"].values()))
    primary = query["tokens"] + first_answer[:-1]
    # The pinned block receives no explicit attention mask or position IDs.
    return dict(write_token_ids=writes[0], query_prefix=query["tokens"],
                candidate_order=list(query["candidates"]), candidates=query["candidates"],
                attention_mask=None, position_ids=None, cache_position=None,
                cache_cursor=cache.get_seq_length(), primary_chunks=[primary],
                input_shape=[1, len(primary)], profile=backbone.identity["execution_profile"])


def warm(backbone, writes, query):
    cache = live(backbone, writes[:1])
    backbone.step(query["tokens"] + next(iter(query["candidates"].values()))[:-1], [cache], [1.0])
    torch.cuda.synchronize()


def trial(mc, out, label, cell, cache, writes, query, primary):
    work = out / "work"
    shared = load_json(work / "shared.json")
    archive = work / "prefix.cell"
    # Check the live/restored object itself prior to executing any query.
    actual = FullState.capture(mc._backbone, cache)
    start = fingerprint(actual.structure, actual.tensors)
    differences = start_differences(shared["pre"], start)
    bundle_sha = digest_file(mc.cells._store.path(cell.id))
    if bundle_sha != shared["bundle_sha256"] or digest_file(archive) != bundle_sha:
        differences.append("bundle_sha256")
    inputs = execution_inputs(mc._backbone, writes, query, cache)
    receipt = dict(label=label, pid=os.getpid(), inputs=inputs,
                   input_sha256=canonical_hash(inputs), state=start, start_failures=differences,
                   bundle_sha256=bundle_sha, primary_path=primary)
    write(work / f"{label}.json", receipt)
    if differences:
        return
    del actual
    pristine = deepcopy(cache)
    result = ensure_readonly(mc, cell, lambda: trace(
        mc._backbone, lambda: deepcopy(pristine), query, primary_cache=cache))
    if digest_file(archive) != bundle_sha or trace_fingerprint(result)["pre"] != start:
        raise RuntimeError("Bundle or verified query start changed during the trial")
    save_trace(work / f"trace-{label}", result)
    receipt.update(bundle_unchanged=True, fingerprints=trace_fingerprint(result))
    write(work / f"{label}.json", receipt)
    print(f"A3_COMPLETED {label} PID={os.getpid()}", flush=True)


@torch.inference_mode()
def worker(out, phase):
    torch.set_num_threads(4)
    with MiniCells.open_preview(out / "work" / f"registry-{phase}") as mc:
        env = check_environment(mc)
        if phase == "P0":
            write(out / "environment.json", env)
        elif env != load_json(out / "environment.json"):
            raise RuntimeError("Frozen worker environment mismatch")
        work = out / "work"
        texts, writes, query = probe_inputs(dataset(out), mc._backbone)
        if phase in ("P0", "P2", "P3"):
            warm(mc._backbone, writes, query)
            if phase == "P0":
                original, cache, _ = produce(mc, texts, work, require_historical_count=False)
                trial(mc, out, "P0", original, cache, writes, query, "original_live_object")
                mc.cells.delete(original.id)
                del cache
            repeat_group = "P1" if phase == "P0" else phase
            for i in range(1, 4):
                cell, restored = imported(mc, work / "prefix.cell")
                trial(mc, out, f"{repeat_group}-{i}", cell, restored, writes, query, "fresh_import")
                mc.cells.delete(cell.id)
            return
        if phase.startswith("observe"):
            warm(mc._backbone, writes, query)
            cell, restored = imported(mc, work / "prefix.cell")
            shared = load_json(work / "shared.json")
            state = FullState.capture(mc._backbone, restored)
            if start_differences(shared["pre"], fingerprint(state.structure, state.tensors)):
                raise RuntimeError("Operator probe started from a different Cache")
            del state
            observer = Observer()
            primary = query["tokens"] + next(iter(query["candidates"].values()))[:-1]
            with instrument(mc._backbone, observer):
                ensure_readonly(mc, cell, lambda: mc._backbone.step(primary, [restored], [1.0]))
            if digest_file(work / "prefix.cell") != shared["bundle_sha256"]:
                raise RuntimeError("Operator probe changed the Bundle")
            save_observations(work / phase, observer)
            return
        selection = load_json(out / "first_divergence.json")
        warm(mc._backbone, writes, query)
        if phase == "deterministic":
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
        left = load_json(work / "observe-1.json")
        selected = next(row for row in left if row["key"] == selection["key"])
        all_inputs = load_file(work / "observe-1.safetensors")
        if phase == "deterministic":
            # Deterministic operator replay is a separate post-localization lane.
            payload = replay(mc._backbone, selected["operator"], selected["input"], all_inputs)
            save_trace(work / phase, payload)
            return
        payload = replay(mc._backbone, selected["operator"], selected["input"], all_inputs)
        save_trace(work / phase, payload)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        try:
            with torch.profiler.profile(activities=activities) as profiler:
                replay(mc._backbone, selected["operator"], selected["input"], all_inputs)
                torch.cuda.synchronize()
            installed = sorted({e.name for e in profiler.events() if str(e.device_type).endswith("CUDA")})
            write(work / f"{phase}-kernels.json", dict(kernels=installed))
        except (OSError, RuntimeError) as exc:
            write(work / f"{phase}-kernels.json", dict(kernels=[], profiler_error=str(exc)))


def formal(out):
    work = out / "work"
    receipts = {label: load_json(work / f"{label}.json") for label in LABELS}
    phases = {name: {receipt["pid"] for label, receipt in receipts.items()
                    if label.startswith(name + "-")} for name in GROUPS}
    if (phases["P1"] != {receipts["P0"]["pid"]} or any(len(pids) != 1 for pids in phases.values())
            or len(set.union(*phases.values())) != 3):
        raise ValueError("The prescribed three-process layout did not run")
    starts = {k: v["start_failures"] for k, v in receipts.items() if v["start_failures"]}
    if len({v["input_sha256"] for v in receipts.values()}) != 1:
        starts["inputs"] = "Input signatures differ"
    if len({v["bundle_sha256"] for v in receipts.values()}) != 1:
        starts["bundle"] = "Bundle SHA-256 differs"
    write(out / "initial_state_comparison.json", dict(failures=starts, trials={
        k: dict(pid=v["pid"], state=v["state"], inputs=v["inputs"], primary_path=v["primary_path"])
        for k, v in receipts.items()}))
    write(out / "bundle_integrity.json", {k: {n: v.get(n) for n in ("bundle_sha256", "bundle_unchanged")}
                                         for k, v in receipts.items()})
    if starts:
        return None
    if not all(v["bundle_unchanged"] for v in receipts.values()):
        raise RuntimeError("Bundle read-only check failed")
    write(out / "post_query_state.json", {k: v["fingerprints"]["post"] for k, v in receipts.items()})
    bounds = load_json(out / BOUNDS_FILE)
    pairs = {}
    with (out / "layer_comparison.jsonl").open("w", encoding="utf-8") as stream:
        for a, b in formal_pairs():
            key = f"{a}_vs_{b}"
            pairs[key], fields = compare_pair(load_trace(work / f"trace-{a}"), load_trace(work / f"trace-{b}"), bounds)
            for row in fields:
                stream.write(json.dumps(dict(pair=key, **row), allow_nan=False) + "\n")
    write(out / "execution_repetitions.json", pairs)
    # Lossless hidden states/logits are published separately per trial to bound file sizes.
    for key in LABELS:
        _, tensors = load_trace(work / f"trace-{key}")
        save_file({k: value for k, value in tensors.items() if k.startswith("query.")},
                  out / f"query-{key}.safetensors")
    return pairs


def locate(out):
    work = out / "work"
    left, right = work / "observe-1", work / "observe-2"
    first, rows = alignment(load_json(left.with_suffix(".json")), load_json(right.with_suffix(".json")),
                            load_file(left.with_suffix(".safetensors")), load_file(right.with_suffix(".safetensors")),
                            load_json(out / BOUNDS_FILE),
                            load_json(work / "P0.json")["inputs"]["input_shape"][1])
    with (out / "operator_trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write(out / "first_divergence.json", first or {"status": "UNRESOLVED"})
    return first, all(r["status"] == "PASS" for r in rows)


def run_replay(out, spawn):
    selection = load_json(out / "first_divergence.json")
    work = out / "work"
    rows = load_json(work / "observe-1.json")
    operator = next(row for row in rows if row["key"] == selection["key"])
    inputs = {key: tensor for key, tensor in load_file(work / "observe-1.safetensors").items()
              if key.startswith(selection["key"] + ".input.")}
    save_file(inputs, out / "replay_inputs.safetensors")
    write(out / "replay_inputs.json", dict(operator=operator["operator"], arguments=operator["input"],
                                          source=selection, execution_profile="t4-fp16-recurrent-v1"))
    phases = ("replay-1", "replay-2", "deterministic")
    errors = {phase: error for phase in phases if (error := spawn(phase))}
    if any(phase in errors for phase in phases[:2]):
        write(out / "operator_replay.json", dict(status="BLOCKED", errors=errors))
        return "BLOCKED"
    structures = []
    results = []
    for phase in phases[:2]:
        schema, tensors = load_trace(work / phase)
        structures.append(schema)
        results.append(tensors)
    equal = (structures[0] == structures[1] and results[0].keys() == results[1].keys()
             and all(torch.equal(results[0][key], results[1][key]) for key in results[0]))
    status = "IDENTICAL_INPUT_DIVERGENCE" if not equal else "NOT_REPRODUCED"
    deterministic_match = None
    if "deterministic" not in errors:
        schema, tensors = load_trace(work / "deterministic")
        deterministic_match = [schema == s and tensors.keys() == t.keys()
                               and all(torch.equal(tensors[k], t[k]) for k in tensors)
                               for s, t in zip(structures, results, strict=True)]
    write(out / "operator_replay.json", dict(status=status, errors=errors,
          actual_kernels={phase: load_json(work / f"{phase}-kernels.json") for phase in phases[:2]},
          deterministic=dict(settings={"torch": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                             exact_matches_formal_replays=deterministic_match)))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", choices=("P0", "P2", "P3", "observe-1", "observe-2",
                                            "replay-1", "replay-2", "deterministic"))
    args = parser.parse_args()
    out = args.output
    if args.phase:
        try:
            worker(out, args.phase)
        except Exception as exc:
            write(out / f"worker_error_{args.phase}.json", dict(type=type(exc).__name__, message=str(exc)))
            raise
        return
    (out / "work").mkdir(parents=True)
    (out / "README.md").write_text(
        "# M1-002A-R1-A3\n\nSee decision.json, execution_repetitions.json and operator_replay.json.\n",
        encoding="utf-8")
    started = time.monotonic()
    result = dict(experiment="M1-002A-R1-A3 Live Cache vs Full-State Cell", status="INVALID",
                  engineering_status="FAIL", state_equivalence="INCOMPLETE",
                  bridge_status="NOT_RUN_BY_SCOPE", capacity_status="NOT_RUN_BY_SCOPE",
                  a2_commit=A2_COMMIT, frozen_bounds_sha256=BOUNDS_SHA256)

    def spawn(phase):
        env = dict(os.environ)
        if phase == "deterministic":
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        completed = subprocess.run([sys.executable, __file__, str(out), "--phase", phase], env=env)
        if completed.returncode:
            error = out / f"worker_error_{phase}.json"
            return load_json(error) if error.exists() else completed.returncode
        return None

    try:
        baseline(out)
        frozen_bounds(out)
        for phase in ("P0", "P2", "P3"):
            error = spawn(phase)
            if error:
                raise RuntimeError(f"Worker {phase}: {error}")
        pairs = formal(out)
        result["engineering_status"] = "PASS"
        if pairs is None:
            result.update(state_equivalence="INVALID", diagnosis="START_STATE_OR_INPUT_MISMATCH")
            return
        gates = {key: "PASS" if row["passed"] else "FAIL" for key, row in pairs.items()}
        equivalent = all(value == "PASS" for value in gates.values())
        result.update(gates=gates, state_equivalence="PASS" if equivalent else "FAIL")
        errors = {phase: error for phase in ("observe-1", "observe-2") if (error := spawn(phase))}
        if errors:
            write(out / "operator_replay.json", dict(status="BLOCKED", errors=errors))
            result.update(status="BLOCKED", diagnosis="Operator observation could not be completed.")
            return
        first, operator_pass = locate(out)
        if first:
            replay_status = run_replay(out, spawn)
        else:
            replay_status = "NOT_RUN_NO_OPERATOR"
            write(out / "operator_replay.json", dict(status=replay_status))
        if equivalent and operator_pass:
            result.update(status="PASS", diagnosis="R1_A3_EQUIVALENCE_PASS",
                          next_action="Review the remaining original R1-A lifecycle gate independently.")
        elif replay_status == "BLOCKED":
            result.update(status="BLOCKED", diagnosis="Independent operator replay could not be completed.")
        elif not equivalent and replay_status == "IDENTICAL_INPUT_DIVERGENCE":
            result.update(status="FAIL_ROOT_CAUSE_IDENTIFIED",
                          diagnosis=f"Independent replay reproduced divergence at {first['operator']}.")
        else:
            result.update(status="FAIL_ROOT_CAUSE_UNRESOLVED",
                          diagnosis="A numerical gate failed. Replay did not isolate a causal operator.")
    except Exception as exc:
        result.update(status="INVALID", engineering_status="FAIL", diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
