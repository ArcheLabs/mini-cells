"""M1-002A-R1: full-state equivalence, lifecycle and frozen bridge."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time
import urllib.request
from unittest.mock import patch

import torch

from minicells_native import MiniCells
from minicells_native.full_state import FullState
from common import canonical_hash, environment, publishable, write
from m1_bridge import ARMS, DATASET_SHA256 as CANONICAL, measure, scores
from m1_value_swap import score
from m1_capacity import saved_cell
from m1_reference_capacity import digest_file
from r1_equivalence import (
    cache_audit, check_restored_tensors, compare, ensure_readonly, envelope,
    live, load_trace, restored_trace, save_trace, trace,
)

FORMAT = "minicells.m1.capacity-bridge.full-state.v1"
HISTORICAL_COMMIT = "d9fca132573de69da91b164dcdc44f66e7b41ca3"
HISTORICAL = f"https://raw.githubusercontent.com/ArcheLabs/mini-cells/{HISTORICAL_COMMIT}/artifacts/experiments/minicells-m1-002a-bridge/115bf4b2219e/"
DATASET_SHA256 = "ccb9b0fcf3134baf5f378f480d36e981a153a999377f8e9ac424680b6b97a3cf"
BASELINE = dict(reference=[12, 12], anchor=[12, 12], encoding=[11, 7],
                candidates=[8, 0], wording=[12, 0], combined=[6, 0])
PATHS = ("A", "B", "C", "D", "E")
CONTRASTS = (("C", "B"), ("D", "B"), ("E", "C"), ("E", "D"), ("E", "A"), ("E", "B"))
IDENTITY = dict(model_id="linear-moe-hub/MoM-Gated-Deltanet-1.3B",
                model_revision="5c8be94196cb1827e820077680ac2ee46ac949bc",
                tokenizer_revision="5c8be94196cb1827e820077680ac2ee46ac949bc",
                mom_revision="125038087b71b635c171cec857bd0f21d10955b6",
                execution_profile="t4-fp16-recurrent-v1", dtype="float16",
                tokenizer_hash="90e3e115a745594abd2167554dcf472ea00d4852465c03b5482309d9036f4c49",
                weights_sha256="08820287a91c2bc10f1fafe8b149dbf027ae51768530027c711dc240f7d53771")
PACKAGES = {"torch": "2.11.0+cu130", "triton": "3.6.0", "transformers": "4.51.3",
            "flash-linear-attention": "0.2.2", "minicells-native-sdk": "0.1.0"}


def load_json(file):
    return json.loads(file.read_text())


def baseline(out):
    for target in ("bridge_dataset.json", "bridge_statistics.json"):
        with urllib.request.urlopen(HISTORICAL + target, timeout=30) as response:
            data = response.read()
        file = out / ("historical_statistics.json" if target == "bridge_statistics.json" else target)
        file.write_bytes(data)
    items = dataset(out)
    stats = load_json(out / "historical_statistics.json")
    if ({key: [o["full_flips"], o["cell_flips"]] for key, o in stats["arms"].items()} != BASELINE
            or len(items) != 12):
        raise RuntimeError("Historical M1-002A outcome differs from the frozen baseline")


def dataset(out):
    path = out / "bridge_dataset.json"
    items = load_json(path)
    if digest_file(path) != DATASET_SHA256 or canonical_hash(items) != CANONICAL:
        raise ValueError("Historical dataset file or canonical SHA-256 mismatch")
    return items


def check_environment(mc):
    env = environment()
    if (env["gpu"] != "Tesla T4" or env["capability"] != [7, 5] or env["cuda"] != "13.0"
            or env["packages"] != PACKAGES or any(mc.model_info[k] != v for k, v in IDENTITY.items())):
        raise RuntimeError("Runtime identity differs from the frozen M1-002A environment")
    return dict(runtime=env, model=mc.model_info)


def probe_inputs(items, backbone):
    first = items[0]["arms"]["reference"]
    second = items[1]["arms"]["reference"]
    texts = [first["texts"][first["labels"][0]], second["texts"][second["labels"][0]]]
    writes = [backbone.tokenizer.encode(text, add_special_tokens=False) for text in texts]
    return texts, writes, first["wordings"][0]


def record(out, stage, summary, rows, **extra):
    with (out / "state_equivalence_layers.jsonl").open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(check=stage, **row), allow_nan=False) + "\n")
    file = out / "state_equivalence.json"
    info = load_json(file) if file.exists() else {}
    info[stage] = {**summary, "pid": os.getpid(), **extra}
    write(file, info)
    print(f"R1_STATE_CHECK {stage} {'PASS' if summary['passed'] else 'FAIL'}", flush=True)
    if not summary["passed"]:
        raise RuntimeError(f"State equivalence failed at {stage}: {summary['failures'][:4]}")


def equivalence(mc, out, phase):
    backbone = mc._backbone
    texts, writes, query = probe_inputs(dataset(out), backbone)
    work = out / "work"
    original = work / "prefix.cell"
    appended = work / "append.cell"
    if phase == "write":
        cell = mc.cells.create("R1 Full-state", full_state=True)
        captured = []
        capture = FullState.capture

        def observe(backbone, cache):
            state = capture(backbone, cache)
            captured.append(state)
            return state

        with patch.object(FullState, "capture", side_effect=observe):
            cell = mc.cells.write(cell.id, texts[0], resume=False)
        mc.cells.export(cell.id, original)
        write(work / "prefix_fingerprint.json", dict(bundle_sha256=digest_file(original)))
        if len(captured) != 1:
            raise RuntimeError("Expected one post-write capture from the SDK")
        serialized = check_restored_tensors(mc, cell, expected=captured.pop())
        # Freeze separate within-process envelopes before reading the persisted Cell.
        for label, parts in (("prefix", writes[:1]), ("append", writes)):
            ref = trace(backbone, lambda: live(backbone, parts), query)
            repeat = trace(backbone, lambda: live(backbone, parts), query)
            bounds = envelope(ref, repeat)
            write(out / f"live_bounds_same_{label}.json", bounds)
            save_trace(work / f"live-{label}", ref)
            if label == "prefix":
                actual = restored_trace(mc, cell, query)
                audit = cache_audit(actual[0]["pre"])
                write(out / "state_schema.json", dict(format_version=2, state_schema_version=2,
                                                        structure=actual[0]["pre"], audit=audit))
                if not all(all(layer["recurrent_banks"]) and all(all(b) for b in layer["qkv_routed_shared"])
                           for layer in audit):
                    missing = [str(l["layer"]) for l in audit
                               if not all(l["recurrent_banks"]) or not all(all(b) for b in l["qkv_routed_shared"])]
                    record(out, "cache_coverage", dict(passed=False, failures=missing), [], audit=audit)
                result, rows = compare(ref, actual, bounds)
                record(out, "same_process_roundtrip", result, rows, tensor_restore=serialized)
        return

    label, parts = ("prefix", writes[:1]) if phase == "restore" else ("append", writes)
    baseline_ref = load_trace(work / f"live-{label}")
    fresh = trace(backbone, lambda: live(backbone, parts), query)
    bounds = envelope(baseline_ref, fresh)
    # Also include the independently measured same-process live envelope, never restored results.
    same = load_json(out / f"live_bounds_same_{label}.json")
    bounds = {key: max(value, same[key]) for key, value in bounds.items()}
    write(out / f"live_bounds_cross_{label}.json", bounds)
    archive = original if phase == "restore" else appended
    cell = mc.cells.import_cell(archive)
    expected_digest = load_json(work / f"{label}_fingerprint.json")["bundle_sha256"]
    if digest_file(mc.cells._store.path(cell.id)) != expected_digest:
        raise RuntimeError("Cross-process import changed the exported bundle")
    actual = restored_trace(mc, cell, query)
    result, rows = compare(fresh, actual, bounds)
    record(out, "fresh_process_import" if phase == "restore" else "fresh_process_append",
           result, rows, tensor_restore=check_restored_tensors(mc, cell))
    if phase == "restore":
        mc.cells.append(cell.id, texts[1], resume=False)
        mc.cells.export(cell.id, appended)
        write(work / "append_fingerprint.json", dict(bundle_sha256=digest_file(appended)))
    else:
        fork = mc.cells.fork(cell.id, "R1 isolated fork")
        source = mc.cells._state(cell.id)
        forked = mc.cells._state(fork.id)
        same_state = (source.structure == forked.structure and source.tensors.keys() == forked.tensors.keys()
                      and all(torch.equal(source.tensors[k], forked.tensors[k]) for k in source.tensors))
        if fork.id == cell.id or not same_state:
            raise RuntimeError("Fork did not preserve state with a new Cell ID")
        ensure_readonly(mc, cell, lambda: mc.query(query["text"], cells=[cell.id], max_new_tokens=1))
        ensure_readonly(mc, fork, lambda: mc.query(query["text"], cells=[fork.id], max_new_tokens=1))
        mc.cells.delete(fork.id)
        mc.cells.delete(cell.id)
        record(out, "sdk_lifecycle", dict(passed=not mc.cells.list(), failures=[]), [],
               operations=["create", "write", "append", "read", "export", "import", "fork", "query", "delete"])


def scored(backbone, factory, arm, label):
    results = []
    for query in arm["wordings"]:
        # Same batching and teacher forcing as M1-002A.
        if len(query["candidates"][label]) == 1:
            logp = scores(backbone, factory, query)
            per_token = {key: [value] for key, value in logp.items()}
        else:
            evaluated = {key: score(backbone, factory(), query["tokens"], ids)
                         for key, ids in query["candidates"].items()}
            logp = {key: row["logp"] for key, row in evaluated.items()}
            per_token = {key: row["token_logp"] for key, row in evaluated.items()}
        margin = logp[label] - max(v for k, v in logp.items() if k != label)
        results.append(dict(query=query["text"], logp=logp, token_logp=per_token, margin=margin,
                            gold_nll=-logp[label] / len(query["candidates"][label])))
    return results


def bridge_case(mc, item):
    backbone = mc._backbone
    groups, rows = {}, {}
    for name, arm in item["arms"].items():
        groups.setdefault(tuple(arm["texts"].values()), []).append((name, arm))
        rows[name] = dict(no_memory=measure(backbone, lambda: backbone.stream({}), arm),
                          paths={k: {} for k in PATHS})
    for cases in groups.values():
        labels, texts = cases[0][1]["labels"], cases[0][1]["texts"]
        for label in labels:
            text = texts[label]
            tokens = backbone.tokenizer.encode(text, add_special_tokens=False)
            cell = mc.cells.create(f"full-{item['index']}-{label}", full_state=True)
            try:
                mc.cells.write(cell.id, text, resume=False)
                state = mc.cells._state(cell.id)
                with saved_cell(mc, [text], 64, f"legacy-{item['index']}-{label}") as (legacy, _):
                    factories = {"A": lambda: live(backbone, [tokens]), "B": legacy,
                                 "C": lambda: state.restore(backbone, cursor=False),
                                 "D": lambda: state.restore(backbone, conv=False),
                                 "E": lambda: state.restore(backbone)}
                    def evaluate():
                        for name, arm in cases:
                            for key, factory in factories.items():
                                rows[name]["paths"][key][label] = scored(backbone, factory, arm, label)
                    ensure_readonly(mc, cell, evaluate)
            finally:
                mc.cells.delete(cell.id)
    return dict(index=item["index"], arms=rows)


def summarize(rows):
    arms, gates = {}, {}
    for name in ARMS:
        candidates = [row["arms"][name] for row in rows]
        rates, hits = {}, {}
        for path in PATHS:
            per_case = [[w for values in row["paths"][path].values() for w in values] for row in candidates]
            hits[path] = [all(w["margin"] > 0 for w in case) for case in per_case]
            rates[path] = dict(flips=sum(hits[path]), n=len(rows),
                               mean_margin=mean(w["margin"] for case in per_case for w in case),
                               mean_gold_nll=mean(w["gold_nll"] for case in per_case for w in case))
        contrasts = {}
        for a, b in CONTRASTS:
            paired = list(zip(hits[a], hits[b], strict=True))
            contrasts[f"{a}_vs_{b}"] = dict(gains=sum(x and not y for x, y in paired),
                                            losses=sum(y and not x for x, y in paired))
        first_label = [next(iter(row["paths"]["A"])) for row in candidates]
        prior = sum(all(w["margins"][label] > 0 for w in row["no_memory"])
                    for label, row in zip(first_label, candidates, strict=True))
        e_a = max(abs(e["logp"][value] - a["logp"][value])
                  for row in candidates for label, wordings in row["paths"]["A"].items()
                  for a, e in zip(wordings, row["paths"]["E"][label], strict=True)
                  for value in a["logp"])
        arms[name] = dict(paths=rates, no_memory_original_hits=prior, contrasts=contrasts,
                          max_abs_candidate_logp_E_vs_A=e_a)
        if name in ("reference", "anchor", "combined"):
            for path in ("A", "E"):
                gates[f"{name}_{path}"] = "PASS" if rates[path]["flips"] >= 10 else "FAIL"
        if name == "combined":
            gates["combined_no_memory"] = "PASS" if prior <= 1 else "FAIL"
    return dict(arms=arms, gates=gates, historical=BASELINE)


@torch.inference_mode()
def worker(out, phase):
    torch.set_num_threads(4)
    with MiniCells.open_preview(out / "work" / phase) as mc:
        runtime = check_environment(mc)
        if phase == "write":
            write(out / "environment.json", runtime)
        elif runtime != load_json(out / "environment.json"):
            raise RuntimeError("The worker runtime identity changed")
        if phase != "bridge":
            equivalence(mc, out, phase)
            return
        rows = []
        with (out / "bridge_cases.jsonl").open("w", encoding="utf-8") as stream:
            for item in dataset(out):
                row = bridge_case(mc, item)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                print(f"R1_BRIDGE_COMPLETED {len(rows)}/12", flush=True)
        write(out / "bridge_statistics.json", summarize(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", choices=("write", "restore", "verify", "bridge"))
    args = parser.parse_args()
    out = args.output
    if args.phase:
        try:
            worker(out, args.phase)
        except Exception as exc:
            write(out / f"worker_error_{args.phase}.json", dict(error=type(exc).__name__, message=str(exc)))
            raise
        return
    out.mkdir(parents=True)
    (out / "work").mkdir()
    started = time.monotonic()
    result = dict(experiment="M1-002A-R1 Full-State Persistence Bridge", status="INVALID",
                  engineering_status="FAIL", capacity_status="NOT_RUN_BY_SCOPE",
                  state_equivalence="NOT_RUN", bridge_status="NOT_RUN")
    try:
        baseline(out)
        result.update(historical_commit=HISTORICAL_COMMIT, dataset_sha256=DATASET_SHA256,
                      dataset_canonical_sha256=CANONICAL)
        result["state_equivalence"] = "STARTED"
        for phase in ("write", "restore", "verify"):
            outcome = subprocess.run([sys.executable, __file__, str(out), "--phase", phase])
            if outcome.returncode:
                receipt = out / "state_equivalence.json"
                checks = load_json(receipt) if receipt.exists() else {}
                failed = {key: value["failures"] for key, value in checks.items() if not value["passed"]}
                if not failed:
                    error = out / f"worker_error_{phase}.json"
                    raise RuntimeError(f"{phase} worker failed: {load_json(error) if error.exists() else outcome.returncode}")
                result.update(status="FAIL", engineering_status="PASS", state_equivalence="FAIL",
                              diagnosis=f"State equivalence failed in {phase}: {failed}",
                              next_action="Investigate the recorded state divergence before running the bridge.")
                return
        checks = load_json(out / "state_equivalence.json")
        expected = ("same_process_roundtrip", "fresh_process_import", "fresh_process_append")
        if len({checks[key]["pid"] for key in expected}) != len(expected):
            raise RuntimeError("State equivalence did not use three separate processes")
        result["state_equivalence"] = "STATE_EQUIVALENCE_PASS"
        result["bridge_status"] = "STARTED"
        bridge = subprocess.run([sys.executable, __file__, str(out), "--phase", "bridge"])
        if bridge.returncode:
            err = out / "worker_error_bridge.json"
            raise RuntimeError(f"Bridge worker failed: {load_json(err) if err.exists() else bridge.returncode}")
        stats = load_json(out / "bridge_statistics.json")
        passed = "FAIL" not in stats["gates"].values()
        result.update(status="PASS" if passed else "FAIL", engineering_status="PASS",
                      bridge_status="PASS" if passed else "FAIL", gates=stats["gates"],
                      r1_bridge=stats, diagnosis=("State equivalence and bridge gates passed." if passed else
                                                "State equivalence passed. The frozen bridge gate failed."),
                      next_action="Review the state ablations and live results before selecting a follow-up.")
    except Exception as exc:
        for key in ("state_equivalence", "bridge_status"):
            if result[key] == "STARTED":
                result[key] = "INCOMPLETE"
        result.update(diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
