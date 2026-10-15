"""Same-device synthetic optimizer pilot; no automatic scientific training transition."""

import copy
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from exp006.reuse import canonical_sha256, file_sha256
from .io import write_json
from pd001.engine import step
from pd001.accounting import TrainingGemmCounter
from .config import ARMS, engine_config, source_sha, counts
from .models import make_model


def gpu_check(devices, *, isolated=False):
    names = []
    if not devices:
        raise ValueError("device list empty")
    for device in devices:
        if not str(device).startswith("cuda"):
            raise RuntimeError("real pilot/training expects T4 CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use CPU preflight")
        name = torch.cuda.get_device_name(device)
        if "T4" not in name:
            raise RuntimeError("pilot hardware differs from declared Kaggle T4: " + name)
        free, total = torch.cuda.mem_get_info(device)
        if free < 1024**3:
            raise RuntimeError("less than 1 GiB free; preserve work and free the GPU")
        names.append({"device": device, "name": name, "free_bytes": free, "total_bytes": total})
    if isolated:
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        )
        import os

        other = [p for p in apps.splitlines() if p.strip().isdigit() and int(p) != os.getpid()]
        if other:
            raise RuntimeError("another GPU process is running; stop it before serial performance pilot")
    return names


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark(
    c, output, device="cpu", *, allow_gpu=False, microbatches=(1, 2), warmup=None, steps=None, fixture=False
):
    if str(device).startswith("cuda") and not allow_gpu:
        raise RuntimeError("explicit --allow-gpu-pilot required")
    code_hashes = {f.name: file_sha256(f) for f in Path(__file__).parent.glob("*.py")}
    hardware = (
        gpu_check([device], isolated=True)
        if str(device).startswith("cuda")
        else [{"device": "cpu", "name": "CPU"}]
    )
    warmup = c["performance"]["warmup_steps"] if warmup is None else warmup
    steps = c["performance"]["measure_steps"] if steps is None else steps
    if (not fixture and (not 3 <= warmup <= 5 or not 10 <= steps <= 20)) or steps < 1 or warmup < 0:
        raise ValueError("pilot requires 3-5 warmup and 10-20 measured steps")
    output = Path(output)
    if output.exists():
        raise FileExistsError("performance evidence immutable")
    output.mkdir(parents=True)
    rows = []
    started = time.perf_counter()
    for micro in microbatches:
        if micro not in (1, 2):
            raise ValueError("only declared microbatch 1/2")
        for arm in ARMS:
            local = copy.deepcopy(c)
            local["training"]["microbatch"] = micro
            model = make_model(local, arm).to(device)
            train = engine_config(local)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=local["training"]["learning_rate"],
                weight_decay=local["training"]["weight_decay"],
            )
            generators = [torch.Generator(device=device).manual_seed(c["seed"] + 1)]
            data = np.arange(8192, dtype=np.int32) % c["model"]["vocab_size"]
            starts = np.arange(local["training"]["effective_batch"]) * (
                local["training"]["sequence_length"] + 1
            )
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            values = []
            print(f"[Pilot] {arm}, microbatch {micro}, {warmup}+{steps} steps", flush=True)
            for index in range(1, warmup + steps + 1):
                synchronize(device)
                tick = time.perf_counter()
                step(model, opt, train, index, starts, data, generators, device)
                synchronize(device)
                if index > warmup:
                    values.append(time.perf_counter() - tick)
            peak = torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None
            # Representative ONE microbatch only: do not amplify instrumentation by
            # profiling the entire accumulated optimizer step. Timing above remains batch8.
            profile_train = copy.deepcopy(train)
            profile_train["training"]["effective_batch"] = micro
            # One extra offline dispatch + profiler microbatch, never in measured timings.
            activities = [torch.profiler.ProfilerActivity.CPU]
            if str(device).startswith("cuda"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as profiler:
                with TrainingGemmCounter() as gemms:
                    step(
                        model,
                        opt,
                        profile_train,
                        warmup + steps + 1,
                        starts[:micro],
                        data,
                        generators,
                        device,
                    )
            profile = {
                e.key: {"count": e.count, "cpu_self_us": e.self_cpu_time_total}
                for e in profiler.key_averages()
                if any(
                    k in e.key for k in ("nonzero", "index_copy", "_local_scalar_dense", "mm", "multinomial")
                )
            }
            trace = output / (arm + "-m" + str(micro) + "-trace.json")
            profiler.export_chrome_trace(str(trace))
            raw = __import__("json").loads(trace.read_text())["traceEvents"]
            kernels = [e for e in raw if e.get("cat") == "kernel"]
            rows.append(
                {
                    "arm": arm,
                    "microbatch": micro,
                    "p50_seconds": float(np.median(values)),
                    "p90_seconds": float(np.percentile(values, 90)),
                    "step_seconds": values,
                    "peak_gpu_bytes": peak,
                    "supported_training_gemm_flops": gemms.flops,
                    "flops_scope": "one representative microbatch, supported GEMM forward/backward lower bound; not total accumulated-step FLOPs",
                    "profile_target_tokens": micro * c["training"]["sequence_length"],
                    "operator_profile": profile,
                    "kernel_launches": len(kernels) if str(device).startswith("cuda") else None,
                    "kernel_ms": sum(e.get("dur", 0) for e in kernels) / 1000 if kernels else None,
                    "trace": trace.name,
                    "synthetic_data": True,
                    "fixture": fixture,
                }
            )
            write_json(output / "pilot-partial.json", {"rows": rows, "scientific_decision": False})
            del model, opt, generators, profiler, raw, kernels
    selected = [r for r in rows if r["microbatch"] == c["training"]["microbatch"]]
    times = {r["arm"]: r["p90_seconds"] * c["training"]["steps"] for r in selected}
    # Conservative plan adds 50% + 5 minutes for data, diagnostics, eval, saving/report.
    train_wall = max(times["CLM-S16"], times["CLM-S8"] + times["M0-TINY"])
    projection = train_wall * 1.5 + 300
    medians = {r["arm"]: r["p50_seconds"] * c["training"]["steps"] for r in selected}
    lower = max(medians["CLM-S16"], medians["CLM-S8"] + medians["M0-TINY"]) * 1.25 + 180
    result = {
        "experiment_id": c["experiment_id"],
        "source_sha": source_sha(),
        "source_file_sha256": code_hashes,
        "config_sha256": canonical_sha256(c),
        "device": str(device),
        "hardware": hardware,
        "parameters": counts(c),
        "rows": rows,
        "projected_training_seconds": train_wall,
        "projected_total_seconds": projection if str(device).startswith("cuda") else None,
        "projected_total_range_seconds": [lower, projection] if str(device).startswith("cuda") else None,
        "projection_scope": "two T4s; S16 vs S8 then M0; p90 step times +50% and 300s; excludes this pilot",
        "end_to_end_seconds": time.perf_counter() - started,
        "synthetic": True,
        "engineering_fixture": fixture,
        "within_target": projection <= c["performance"]["target_wall_seconds"]
        if str(device).startswith("cuda")
        else None,
        "scientific_decision": False,
        "status": "GPU_PILOT_COMPLETED" if str(device).startswith("cuda") else "CPU_ENGINEERING_ONLY",
    }
    write_json(output / "pilot.json", result)
    return result


def validate_pilot(c, pilot):
    if (
        pilot.get("status") != "GPU_PILOT_COMPLETED"
        or pilot.get("engineering_fixture")
        or pilot.get("config_sha256") != canonical_sha256(c)
        or pilot.get("source_sha") != source_sha()
        or len(pilot.get("hardware", [])) != 1
        or "T4" not in pilot["hardware"][0]["name"]
    ):
        raise RuntimeError("review an actual same-source/config single-T4 pilot first")
    rows = [r for r in pilot["rows"] if r["microbatch"] == c["training"]["microbatch"]]
    if (
        set(r["arm"] for r in rows) != set(ARMS)
        or len(rows) != 3
        or any(len(r["step_seconds"]) < 10 for r in rows)
    ):
        raise ValueError("complete three-arm same-microbatch pilot required")
