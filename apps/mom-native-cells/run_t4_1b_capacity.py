"""MINICELLS-T4-1B-CAPACITY-001. Run model audit before starting CUDA work."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DECISION_FORMAT = "minicells.t4.1b.capacity.001.v1"
GIB = 2**30
PROMPTS = (128, 512, 1024)
CELLS = (1, 2, 4, 8)
BASELINE_REVISION = "f4081422ea2cf01b1b5d646cac05cad62a5d5438"
STAGES = ("load", "storage", "scoring", "generation", "recovery")


def write(out, filename, value):
    dest = out / filename
    pending = dest.with_suffix(dest.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(dest)


def write_decision(out, status, diagnosis):
    write(out, "decision.json", dict(format=DECISION_FORMAT, status=status,
                                    diagnosis=diagnosis, generation_quality="NOT_EVALUATED"))


def read(out, name):
    return json.loads((out / name).read_text(encoding="utf-8"))


def environment():
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,"
                                 "compute_cap", "--format=csv,noheader,nounits"],
                                text=True, capture_output=True, timeout=10, check=True)
        names = ("name", "total_mib", "free_mib", "driver", "compute_capability")
        devices = [dict(zip(names, (x.strip() for x in line.split(",")), strict=True))
                   for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        devices = []
    return dict(timestamp=datetime.now(timezone.utc).isoformat(), python=sys.version,
                torch=torch_version, physical_devices=devices,
                hardware_status="T4_PRESENT" if any("Tesla T4" in d["name"] for d in devices) else "NON_TARGET")


def validate_target(config):
    target = config.get("approved_target")
    if target is None:
        return config["reason"]
    required = ("id", "revision", "repository_designation", "weights_sha256", "tokenizer_hash",
                "tokenizer_revision")
    if not all(isinstance(target.get(k), str) and target[k] for k in required):
        return "The target manifest lacks checkpoint provenance or immutable identity."
    if (not re.fullmatch("[0-9a-f]{40}", target["revision"])
            or not re.fullmatch("[0-9a-f]{64}", target["weights_sha256"])
            or not re.fullmatch("[0-9a-f]{64}", target["tokenizer_hash"])):
        return "The target manifest has invalid revision or SHA-256 fields."
    if target["revision"] == BASELINE_REVISION:
        return "The historical SDK checkpoint is excluded from this experiment."
    from minicells_native.backend import MODEL, REVISION, TOKENIZER_REV

    if (target["id"], target["revision"], target["tokenizer_revision"]) != (MODEL, REVISION, TOKENIZER_REV):
        return "The designated checkpoint is not registered by the existing Native MoM loader."
    return None


def matrix():
    cases = {}
    for n in CELLS:
        for p in PROMPTS:
            for mode in ("manual", "auto"):
                cases[n, p, 12, mode] = 1
    for p in PROMPTS:
        cases[0, p, 12, "no_memory"] = 1
    for p in (128, 1024):
        for g in (64, 128):
            cases[0, p, g, "no_memory"] = 1
            for n in (1, 8):
                for mode in ("manual", "auto"):
                    cases[n, p, g, mode] = 1
    for case in ((0, 128, 12, "no_memory"), (1, 128, 12, "auto"),
                 (8, 1024, 128, "auto"), (8, 1024, 128, "manual")):
        cases[case] = 3
    return cases


def build_prompts(tokenizer):
    # Build exact token sequences from actual encoded material. The prefix length
    # is deliberate and is recorded as token IDs, without re-encoding a decoded prefix.
    base = tokenizer.encode(" Capacity measurement for independently stored memory cells." * 256,
                            add_special_tokens=False)
    if len(base) < max(PROMPTS):
        raise ValueError("Tokenizer fixture is too short")
    return {str(n): base[:n] for n in PROMPTS}


def checked_runtime(meter, out, target, storage, record):
    import torch
    from minicells import MiniCells

    with meter.measure(record):
        runtime = MiniCells.open(target["id"], storage, device="cuda", dtype="fp16", max_cells=8)
    try:
        identity = runtime.model_info
        for key, name in (("model_revision", "revision"), ("weights_sha256", "weights_sha256"),
                          ("tokenizer_hash", "tokenizer_hash"), ("tokenizer_revision", "tokenizer_revision")):
            if identity.get(key) != target[name]:
                raise ValueError(f"Loaded checkpoint identity mismatch: {key}")
        backbone = runtime._backbone
        model = backbone.model
        config = model.config.to_dict()
        limit = config.get("max_position_embeddings")
        if not isinstance(limit, int):
            raise ValueError("No confirmed context length in checkpoint configuration")
        counts = Counter()
        weight_bytes = 0
        for param in model.parameters():
            if param.requires_grad or param.dtype != torch.float16:
                raise ValueError("Model must be frozen with FP16 parameters")
            counts[str(param.dtype)] += param.numel()
            weight_bytes += param.numel() * param.element_size()
        if model.training or sum(counts.values()) < 1_000_000_000:
            raise ValueError("Loaded model is not an evaluated 1B-class checkpoint")
        embedding = model.get_input_embeddings().weight.numel()
        write(out, "model_manifest.json", dict(identity=identity, config=config,
              total_parameters=sum(counts.values()), backbone_parameters=sum(p.numel() for p in model.model.parameters()),
              input_embedding_parameters=embedding, parameter_dtypes=dict(counts),
              parameter_bytes=weight_bytes, context_limit=limit,
              precision="FP16 weights; native recurrent buffers retain FP32"))
        return runtime
    except BaseException:
        runtime.close()
        raise


def hashes(mc):
    from minicells_native.backend import sha

    return {cell.id: sha(mc.cells._store.path(cell.id)) for cell in mc.cells.list()}


def query(mc, token_ids, generated, selection, meter, record, *, stop_on_eos=False):
    from capacity_meter import QueryProbe
    from minicells_native.query import run_query

    before = hashes(mc)
    probe = QueryProbe(meter.torch, mc._backbone.tokenizer.eos_token_id, stop_on_eos)
    try:
        with meter.measure(record):
            with mc._operation():
                result = run_query(mc, "capacity fixture", selection, generated, 0,
                                   token_ids=token_ids, probe=probe)
        if result.mode == "auto" and (
                set(result.cell_scores) != set(before)
                or len(result.selected_cell_ids) != min(len(before), mc._top_k)):
            raise RuntimeError("Automatic selection did not score all registered Cells")
        if result.generation_tokens != generated and not (
                stop_on_eos and probe.eos_positions and probe.eos_positions[-1] == result.generation_tokens):
            raise RuntimeError("Generation stopped before reaching its requested limit")
        if hashes(mc) != before:
            raise RuntimeError("Query modified persistent Cell state")
        record.update(status="PASS", prompt_token_ids=token_ids, requested_new_tokens=generated,
                      result=asdict(result), **probe.report())
    except BaseException:
        record.update(operation=probe.stage, **probe.report())
        raise


def cell_schema(mc):
    cells, shapes, sizes = mc.cells.list(), {}, {}
    for cell in cells:
        state = mc.cells._state(cell.id)
        sizes[cell.id] = sum(t.numel() * t.element_size() for t in state.values())
        for key, tensor in state.items():
            desc = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device))
            if shapes.setdefault(key, desc) != desc or desc["dtype"] != "torch.float32":
                raise ValueError("Cells have inconsistent state schema")
    config = mc._backbone.model.config
    return dict(layers=config.num_hidden_layers, routed_banks=config.num_memories,
                shared=config.shared_mem, per_cell_cpu_tensor_bytes=sizes,
                total_cpu_tensor_bytes=sum(sizes.values()), tensors=shapes,
                registry_zip_bytes={cell.id: mc.cells._store.path(cell.id).stat().st_size for cell in cells},
                gpu_working_state="Measured per query in peak_observed_recurrent_bytes")


def run_stage(stage, out, work, target, meter):
    from minicells import MiniCells
    from minicells_native.multicell_fusion import FusionReader
    from minicells_native.query import score_cells, select_cells
    from capacity_meter import QueryProbe

    filenames = dict(load="model_load.json", storage="cell_storage.json", scoring="read_all_profile.json",
                     generation="generation_profile.json", recovery="recovery_profile.json")
    filename = filenames[stage]
    root = work / ("restored" if stage == "recovery" else "loading" if stage == "load" else "registry")
    model_load = {}
    with checked_runtime(meter, out, target, root, model_load) as mc:
        prompts = build_prompts(mc._backbone.tokenizer)
        write(out, "prompts.json", prompts)
        limit = min(2048, mc._backbone.model.config.max_position_embeddings)
        if stage == "load":
            write(out, filename, {**model_load, "status": "PASS"})
            smoke = {}
            query(mc, prompts["128"], 12, [], meter, smoke, stop_on_eos=True)
            if not smoke["result"]["generation_tokens"]:
                raise RuntimeError("Smoke reached EOS before emitting any tokens")
            write(out, "no_memory_smoke.json", smoke)
            return

        if stage == "storage":
            for item in mc.cells.list():
                mc.cells.delete(item.id)
            facts = ("Alice plays violin in London.", "Bob grows flowers in Glasgow.")
            example = []
            for i, fact in enumerate(facts):
                example.append(mc.cells.write(mc.cells.create(f"distinct-{i}").id, fact).id)
            distinct = {}
            query(mc, prompts["128"], 12, "auto", meter, distinct, stop_on_eos=True)
            for cell_id in example:
                mc.cells.delete(cell_id)
            results = []
            first = mc.cells.create("capacity-1")
            seed_write = {}
            with meter.measure(seed_write):
                mc.cells.write(first.id, facts[0])
            for count in CELLS:
                action = {}
                with meter.measure(action):
                    while len(mc.cells.list()) < count:
                        mc.cells.fork(first.id, f"capacity-{len(mc.cells.list())+1}")
                schema = cell_schema(mc)
                if len(set(schema["per_cell_cpu_tensor_bytes"].values())) != 1:
                    raise RuntimeError("Capacity replicas have unequal state sizes")
                ids = [c.id for c in mc.cells.list()]
                with meter.torch.inference_mode():
                    left = mc._backbone.stream(mc.cells._state(ids[0]))
                    right = mc._backbone.stream(mc.cells._state(ids[0]))
                    for a, b in zip(left.to_legacy_cache(), right.to_legacy_cache(), strict=True):
                        for x, y in zip(a["recurrent_state"], b["recurrent_state"], strict=True):
                            if x is not None and x.data_ptr() == y.data_ptr():
                                raise RuntimeError("Working streams share recurrent tensor storage")
                    del left, right
                mc.cells.export_collection(work / f"cells-{count}.zip")
                results.append(dict(count=count, creation_and_fork=action, hashes=hashes(mc), schema=schema))
                write(out, filename, dict(status="RUNNING", write=seed_write, distinct_knowledge=distinct,
                                          cases=results))
            write(out, "cell_schema.json", {"status": "MEASURED", **schema})
            write(out, filename, dict(status="PASS", write=seed_write, distinct_knowledge=distinct,
                                      cases=results))
            return

        if stage == "recovery":
            for item in mc.cells.list():
                mc.cells.delete(item.id)
            report = {"model_load": model_load}
            expected = read(out, "cell_storage.json")["cases"][-1]["hashes"]
            report["import"] = {}
            with meter.measure(report["import"]):
                mc.cells.import_collection(work / "cells-8.zip")
            if hashes(mc) != expected:
                raise RuntimeError("Imported collection does not match original bundle hashes")
            report["query"] = {}
            query(mc, prompts["128"], 12, "auto", meter, report["query"])
            cid = sorted(expected)[0]
            report["continued_write"] = {}
            with meter.measure(report["continued_write"]):
                mc.cells.append(cid, "A new fact was appended after recovery.")
            after = hashes(mc)
            if after[cid] == expected[cid] or any(after[k] != v for k, v in expected.items() if k != cid):
                raise RuntimeError("Continued writing did not preserve isolation")
            report["after_write_query"] = {}
            query(mc, prompts["128"], 12, "auto", meter, report["after_write_query"])
            write(out, filename, {**report, "status": "PASS", "hashes_before": expected,
                                  "hashes_after": after})
            return

        rows, failures = [], read(out, "failures.json")
        configurations = ({(n, p, 0, "auto"): 3 if (n, p) == (8, 1024) else 1
                           for n in CELLS for p in PROMPTS} if stage == "scoring" else matrix())

        def execute(runtime, n, prompt, generated, mode, record):
            if prompt + generated > limit:
                record.update(status="CONTEXT_LIMIT", context_limit=limit)
                return
            if stage == "generation":
                selection = ("auto" if mode == "auto" else [] if mode == "no_memory"
                             else sorted(runtime.cells._registry)[:2])
                query(runtime, prompts[str(prompt)], generated, selection, meter, record)
                return
            before = hashes(runtime)
            probe = QueryProbe(meter.torch, runtime._backbone.tokenizer.eos_token_id)
            reader = FusionReader(runtime._backbone)
            try:
                with meter.measure(record):
                    with runtime._operation():
                        _, ids = runtime._candidate_ids("auto")
                        scores = score_cells(runtime, reader, prompts[str(prompt)], ids, probe)
                        with probe.phase("selector"):
                            ranking, probe.selected = select_cells(runtime, scores)
                if set(scores) != set(ids) or hashes(runtime) != before:
                    raise RuntimeError("Read All skipped a registered Cell or altered state")
                record.update(status="PASS", scores=scores, ranking=ranking, **probe.report())
            except BaseException:
                record.update(operation=probe.stage, **probe.report())
                raise

        def cases(runtime, n):
            for (count, prompt, generated, mode), repeats in configurations.items():
                if count != n:
                    continue
                item = dict(registered_cells=count, prompt_tokens=prompt, requested_new_tokens=generated,
                            mode=mode, repetitions_expected=repeats, trials=[])
                rows.append(item)
                for rep in range(-1, repeats):
                    record = dict(kind="warm_up" if rep < 0 else "measured", index=rep)
                    item["trials"].append(record)
                    try:
                        execute(runtime, count, prompt, generated, mode, record)
                    except meter.torch.OutOfMemoryError as error:
                        record.update(status="OOM", error=str(error))
                    except Exception as error:
                        record.update(status="ERROR", error=str(error))
                        failures.append({**{k: v for k, v in item.items() if k != "trials"}, **record,
                                         "stage": stage})
                        write(out, "failures.json", failures)
                        raise
                    finally:
                        write(out, filename, rows)
                    if record["status"] != "PASS":
                        failures.append({**{k: v for k, v in item.items() if k != "trials"}, **record,
                                         "stage": stage})
                        write(out, "failures.json", failures)
                        if record["status"] == "OOM":
                            gc.collect()
                            meter.torch.cuda.empty_cache()
                            # Confirm the model still runs before proceeding to other configurations.
                            health = {}
                            query(runtime, prompts["128"], 1, [], meter, health)
                        break
                measured = [r for r in item["trials"] if r["kind"] == "measured" and r["status"] == "PASS"]
                if measured:
                    durations = [r["seconds"] for r in measured]
                    item.update(median_seconds=statistics.median(durations), max_seconds=max(durations),
                                stddev_seconds=statistics.stdev(durations) if len(durations) > 1 else None)
                write(out, filename, rows)
                print(stage, count, prompt, generated, mode, item["trials"][-1]["status"], flush=True)

        for count in sorted({case[0] for case in configurations}):
            if count == 8:
                cases(mc, count)
                continue
            with MiniCells._from_backbone(mc._backbone, work / f"{stage}-{count}") as subset:
                if count and not subset.cells.list():
                    subset.cells.import_collection(work / f"cells-{count}.zip")
                cases(subset, count)


def summarize(out, *, blocked=None):
    def receipt(name):
        file = out / name
        return read(out, name) if file.is_file() else None

    def passes(row):
        return row and row.get("status") == "PASS"

    def complete(row):
        return sum(t["kind"] == "measured" and t["status"] == "PASS"
                   for t in row["trials"]) == row["repetitions_expected"]

    load, smoke = receipt("model_load.json"), receipt("no_memory_smoke.json")
    storage, restore = receipt("cell_storage.json"), receipt("recovery_profile.json")
    scoring = receipt("read_all_profile.json") or []
    generation = receipt("generation_profile.json") or []
    scoring_ok = len(scoring) == len(CELLS) * len(PROMPTS) and all(map(complete, scoring))
    generation_ok = len(generation) == len(matrix()) and all(map(complete, generation))
    simple = any(c["registered_cells"] == 1 and c["mode"] == "manual" and complete(c) for c in generation)
    config = receipt("environment.json") or {}
    gpu = config.get("runtime_gpu") or {}
    operations = []

    def collect_records(item):
        if isinstance(item, list):
            for child in item:
                collect_records(child)
        elif isinstance(item, dict):
            if "peak" in item and "other_process_peak" in item:
                operations.append(item)
            for child in item.values():
                collect_records(child)

    for row in (load, smoke, storage, restore, scoring, generation):
        collect_records(row)
    peak = max((r["peak"]["device_used"] for r in operations), default=None)
    other = max((r["other_process_peak"] for r in operations), default=None)
    clean_memory = bool(gpu and peak is not None and peak <= gpu["threshold_bytes"]
                       and other < GIB and not any(r["device_sampler_errors"] for r in operations))
    gates = dict(A=bool(passes(load) and passes(smoke)), B=bool(simple and passes(storage)),
                 C=bool(passes(storage) and scoring_ok), D=generation_ok,
                 E=clean_memory, F=bool(passes(restore)))
    failures = receipt("failures.json") or []
    integrity_error = any(row.get("status") == "ERROR" for row in failures)
    if blocked or not gates["A"]:
        status = "BLOCKED"
    elif integrity_error:
        status = "FAIL"
    elif all(gates.values()):
        status = "FULL PASS"
    elif gates["B"]:
        status = "PARTIAL PASS"
    else:
        status = "FAIL"
    maximum = {}
    for mode in ("auto", "manual", "no_memory"):
        success = [(r["registered_cells"], r["prompt_tokens"], r["requested_new_tokens"])
                   for r in generation if r["mode"] == mode and complete(r)]
        maximum[mode] = [case for case in success if not any(
            candidate != case and all(a >= b for a, b in zip(candidate, case, strict=True))
            for candidate in success)]
    identity = receipt("model_manifest.json") or {}
    summary = dict(status=status, reason=blocked, model_identity=identity.get("identity"),
        joined_artifacts={"model": "model_manifest.json", "environment": "environment.json",
                          "schema": "cell_schema.json"}, gates={
        k: "PASS" if v else "NOT_RUN" if status == "BLOCKED" else "NOT_MET" for k, v in gates.items()},
        maximum_measured_configurations=maximum, peak_device_bytes=peak, peak_other_process_bytes=other,
        runtime_gpu=gpu or None, generation_quality="NOT_EVALUATED",
        run_id=out.name, timestamp=datetime.now(timezone.utc).isoformat())
    write(out, "summary.json", summary)
    write_decision(out, status, blocked or status)
    (out / "report.md").write_text(f"""# MINICELLS-T4-1B-CAPACITY-001

**{status}**. {blocked or 'See gate results and the measured configurations below.'}

| Gate | Decision |
| --- | --- |
""" + "".join(f"| {k} | {v} |\n" for k, v in summary["gates"].items()) + f"""
Maximum measured configurations by mode (registered Cells, input tokens, new tokens):
`{json.dumps(maximum)}`.

Observed peak device memory: `{peak}` bytes.
Observed peak from other processes: `{other}` bytes.
Generation quality: `NOT_EVALUATED`. This report concerns capacity only.
""", encoding="utf-8")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", type=Path, default=HERE / "capacity_target.json")
    parser.add_argument("--stage", choices=("all", "audit", *STAGES), default="all")
    parser.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    config = json.loads(args.target.read_text(encoding="utf-8"))
    reason = validate_target(config)
    if not args.worker:
        out.mkdir(parents=True, exist_ok=args.stage in STAGES)
        if not (out / "environment.json").exists():
            write(out, "environment.json", environment())
            write(out, "model_manifest.json", config)
            write(out, "cell_schema.json", {"status": "UNMEASURED", "reason": reason})
        if not (out / "failures.json").exists():
            write(out, "failures.json", [])
        if reason or args.stage == "audit":
            summarize(out, blocked=reason or "Audit only; no GPU workload requested.")
            print(reason or "Audit complete.", flush=True)
            return
        stages = STAGES if args.stage == "all" else (args.stage,)
        (out / "work").mkdir(exist_ok=True)
        artifacts = {"load": "no_memory_smoke.json", "storage": "cell_storage.json",
                     "scoring": "read_all_profile.json", "generation": "generation_profile.json",
                     "recovery": "recovery_profile.json"}
        for stage in stages:
            prior = out / artifacts[stage]
            if prior.exists():
                archive = out / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archive.mkdir(parents=True)
                prior.replace(archive / prior.name)
                previous = read(out, "failures.json")
                write(archive, "failures.json", [row for row in previous if row["stage"] == stage])
                write(out, "failures.json", [row for row in previous if row["stage"] != stage])
            print(f"Stage {stage}", flush=True)
            result = subprocess.run([sys.executable, "-u", __file__, "--output", str(out),
                                     "--target", str(args.target.resolve()), "--worker", stage])
            if result.returncode and stage == "load":
                summarize(out, blocked="Model or initial GPU smoke could not run; see failures.json.")
                return
            if result.returncode:
                break
        summarize(out)
        return

    if reason:
        raise SystemExit(reason)
    try:
        import torch
        from capacity_meter import Meter

        if torch.cuda.device_count() != 1 or "Tesla T4" not in torch.cuda.get_device_name():
            raise RuntimeError("NON_TARGET: expected exactly one visible Tesla T4")
        if torch.cuda.get_device_capability() != (7, 5):
            raise RuntimeError("NON_TARGET: expected compute capability 7.5")
        torch.set_num_threads(2)
        meter = Meter()
        try:
            meter.baseline = meter.sample()
            gpu = dict(name=torch.cuda.get_device_name(), compute_capability=[7, 5],
                       cuda_runtime=torch.version.cuda, torch=str(torch.__version__),
                       free_bytes=torch.cuda.mem_get_info()[0], **meter.baseline)
            gpu["threshold_bytes"] = min(13 * GIB, gpu["device_total"] - 3 * GIB)
            env = read(out, "environment.json")
            write(out, "environment.json", {**env, "runtime_gpu": gpu})
            run_stage(args.worker, out, out / "work", config["approved_target"], meter)
        finally:
            meter.close()
    except Exception as exc:
        failures = read(out, "failures.json")
        failures.append(dict(stage=args.worker, status="ERROR", type=type(exc).__name__, error=str(exc)))
        write(out, "failures.json", failures)
        raise


if __name__ == "__main__":
    main()
