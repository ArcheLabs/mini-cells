"""Synthetic, bounded performance evidence. Never loads scientific or holdout assets.

Each worker is a fresh process. B0 loads the actual read-only baseline package;
B1--B6 run single-factor cumulative instrumentation ablations of that package.
The timing scaffold excludes checkpoint I/O and final evaluation, explicitly.
"""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch

BASELINE = "2e77369b0df1ce704863a815dacc48301736dde0"
RELATIVE = (
    "research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth"
)
CASES = ("B0", "B1", "B2", "B3", "B4", "B5", "B6-m2", "B6-m4", "LIGHT")


def flags(case):
    if case not in CASES:
        raise ValueError("unknown ablation")
    level = 6 if case.startswith("B6") else 5 if case == "LIGHT" else int(case[1])
    return dict(
        capture=level < 1,
        diagnostics=level < 2,
        counter=level < 3,
        dispatch=level < 4,
        probe_interval=1 if level < 5 else 100,
        routing=case != "LIGHT",
        legacy_checks=case != "LIGHT",
    )


def paths(root):
    base = Path(root) / RELATIVE
    return [
        base / "implementation",
        base.parent / "implementation",
        Path(root) / "src",
        Path(root) / "shared/python",
    ]


def synchronized(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def worker(a):
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
    # Import the chosen source only AFTER constructing its isolated import roots.
    for path in paths(a.root):
        sys.path.insert(0, str(path))
    from pd001.configs import load, arm_config
    from pd001.runner import fixture_config
    from pd001.models import make_model
    from pd001.accounting import Counter
    from pd001.training import _train_registered
    from exp006.reuse import make_batch, file_sha256

    spec = importlib.util.spec_from_file_location("performance_engine", a.engine)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    c, p = load()
    c = copy.deepcopy(c)
    if a.fixture:
        model_config = fixture_config(c, a.arm)
        c["training"].update(effective_batch=4, sequence_length=4)
    else:
        model_config = arm_config(c, a.arm)
    if a.case == "B6-m2":
        c["training"]["microbatch"] = 2
    if a.case == "B6-m4":
        c["training"]["microbatch"] = 4
    c["training"]["steps"] = a.warmup + a.steps
    c["training"]["tokens"] = (
        c["training"]["steps"] * c["training"]["effective_batch"] * c["training"]["sequence_length"]
    )
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = make_model(model_config, a.arm, 17).to(a.device).train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=c["training"]["learning_rate"], weight_decay=c["training"]["weight_decay"]
    )
    mechanisms = c["arms"][a.arm]["mechanism"]
    generators = [
        torch.Generator(device=a.device).manual_seed(117 + i)
        for i in range(1 if mechanisms == "J0" or a.arm == "M0" else 2)
    ]
    rng = np.random.default_rng(17)
    tokens = np.arange(512, dtype=np.int32) % model_config["vocab_size"]
    starts = list(range(0, 400, c["training"]["sequence_length"]))
    x, y = make_batch(
        tokens, np.asarray(starts[:1]), c["training"]["sequence_length"], torch.device(a.device)
    )
    model.eval()
    with torch.no_grad():
        initial = model(x)
        initial_actions = initial.actions.detach().cpu() if a.arm != "M0" else None
    f = flags(a.case)
    elapsed, timeline = [], []
    profile_summary = {}
    gpu_environment = None
    if a.device.startswith("cuda"):
        try:
            gpu_environment = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            gpu_environment = "nvidia-smi unavailable; clocks/load not measured"
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(a.device)
    if a.case == "B0":
        # Execute the ORIGINAL entire training function. Save is replaced only to
        # exclude persistence I/O; completed-step boundaries remain at original save.
        import pd001.training as old

        old.arm_config = lambda config, arm: model_config
        old.make_model = lambda *args, **kwargs: model
        old.load_bundle = lambda *args, **kwargs: {
            "tokens": tokens,
            "starts": starts[:2],
            "groups": [0, 1],
            "evaluation_sha": "d" * 64,
        }
        tick = [None]
        original_zero = torch.optim.AdamW.zero_grad

        def zero(optimizer, *args, **kwargs):
            synchronized(a.device)
            tick[0] = time.perf_counter()
            return original_zero(optimizer, *args, **kwargs)

        torch.optim.AdamW.zero_grad = zero

        def save(path, model, optimizer, index, *args, **kwargs):
            synchronized(a.device)
            if index > a.warmup:
                elapsed.append((time.perf_counter() - tick[0]) * 1000)
            Path(path).write_bytes(b"performance-fixture-not-a-resumable-checkpoint")

        old.save = save
        old.write_curve = lambda *args: None
        c["training"]["checkpoint_every"] = 1
        assets = {
            "train": tokens,
            "train_starts": starts,
            "manifest_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
        }
        _train_registered(
            c,
            p,
            a.arm,
            17,
            "development",
            BASELINE,
            assets,
            "synthetic",
            Path(a.scratch) / "original-run",
            a.device,
            False,
        )
    else:
        from contextlib import nullcontext

        with Counter(model) if f["counter"] else nullcontext() as counter:
            for index in range(1, a.warmup + a.steps + 1):
                synchronized(a.device)
                started = time.perf_counter()
                chosen = rng.choice(starts, size=c["training"]["effective_batch"], replace=True)
                row = engine.step(
                    model,
                    opt,
                    c,
                    index,
                    chosen,
                    tokens,
                    generators,
                    a.device,
                    **{k: v for k, v in f.items() if k not in ("counter", "probe_interval")},
                )
                if index % f["probe_interval"] == 0:
                    if counter is not None:
                        counter.enabled = False
                    row.update(
                        engine.probes(model, x, y, 17, index, initial_actions, legacy=f["legacy_checks"])
                    )
                    if counter is not None:
                        counter.enabled = True
                synchronized(a.device)
                if index > a.warmup:
                    elapsed.append((time.perf_counter() - started) * 1000)
                timeline.append({"step": index, "ce": row["ce"]})
    measured_peak = torch.cuda.max_memory_allocated(a.device) if a.device.startswith("cuda") else None
    # Profiler is deliberately OUTSIDE the measured interval. Capture one tiny step.
    model.train()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if a.device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if a.case == "B0":
        profiler = torch.profiler.profile(activities=activities)

        class ProfileDone(Exception):
            pass

        def zero_profile(optimizer, *args, **kwargs):
            profiler.start()
            return original_zero(optimizer, *args, **kwargs)

        def end_profile(*args, **kwargs):
            profiler.stop()
            raise ProfileDone("bounded original-source profile completed")

        torch.optim.AdamW.zero_grad = zero_profile
        old.save = end_profile
        profiled_config = copy.deepcopy(c)
        profiled_config["training"]["steps"] = 1
        profiled_config["training"]["tokens"] = (
            c["training"]["effective_batch"] * c["training"]["sequence_length"]
        )
        try:
            _train_registered(
                profiled_config,
                p,
                a.arm,
                17,
                "development",
                BASELINE,
                assets,
                "synthetic",
                Path(a.scratch) / "original-profile",
                a.device,
                False,
            )
        except ProfileDone:
            pass
    else:
        chosen = rng.choice(starts, size=c["training"]["effective_batch"], replace=True)
        with torch.profiler.profile(activities=activities) as profiler:
            from contextlib import nullcontext

            with Counter(model) if f["counter"] else nullcontext() as counter:
                engine.step(
                    model,
                    opt,
                    c,
                    a.warmup + a.steps + 1,
                    chosen,
                    tokens,
                    generators,
                    a.device,
                    **{k: v for k, v in f.items() if k not in ("counter", "probe_interval")},
                )
                if f["probe_interval"] == 1:
                    if counter:
                        counter.enabled = False
                    engine.probes(
                        model, x, y, 17, a.warmup + a.steps + 1, initial_actions, legacy=f["legacy_checks"]
                    )
    events = profiler.key_averages()
    interesting = (
        "aten::item",
        "aten::_local_scalar_dense",
        "aten::nonzero",
        "aten::index_copy",
        "aten::clone",
        "aten::multinomial",
    )
    profile_summary = {
        e.key: {"count": e.count, "self_cpu_time_us": e.self_cpu_time_total}
        for e in events
        if e.key in interesting
    }
    trace = Path(a.scratch) / "trace.json"
    profiler.export_chrome_trace(str(trace))
    raw = json.loads(trace.read_text()).get("traceEvents", [])
    cuda_events = [e for e in raw if e.get("cat") == "kernel"]
    sync_events = [e for e in raw if "Synchronize" in e.get("name", "")]
    inventory = {
        name: file_sha256(Path(a.root) / RELATIVE / "implementation/pd001" / name)
        for name in ("training.py", "models.py", "losses.py", "router.py", "diagnostics.py", "accounting.py")
    }
    scheduled_ms = None
    if a.case == "LIGHT":
        synchronized(a.device)
        diagnostic_started = time.perf_counter()
        engine.step(
            model,
            opt,
            c,
            100,
            chosen,
            tokens,
            generators,
            a.device,
            capture=True,
            diagnostics=True,
            routing=True,
        )
        engine.probes(model, x, y, 17, 100, initial_actions)
        synchronized(a.device)
        scheduled_ms = (time.perf_counter() - diagnostic_started) * 1000
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.root, text=True).strip()
    result = {
        "source_sha": head,
        "arm": a.arm,
        "case": a.case,
        "device": a.device,
        "device_name": torch.cuda.get_device_name(a.device)
        if a.device.startswith("cuda")
        else platform.processor() or "CPU",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "precision": "float32",
        "effective_batch": c["training"]["effective_batch"],
        "microbatch": c["training"]["microbatch"],
        "sequence_length": c["training"]["sequence_length"],
        "resident_parameters": sum(p.numel() for p in model.parameters()),
        "measured_steps": a.steps,
        "warmup_steps": a.warmup,
        "step_times_ms": elapsed,
        "optimizer_step_ms_p50": float(np.median(elapsed)),
        "optimizer_step_ms_p95": float(np.percentile(elapsed, 95)),
        "target_tokens_per_second": a.steps
        * c["training"]["effective_batch"]
        * c["training"]["sequence_length"]
        / (sum(elapsed) / 1000),
        "measured_target_tokens": a.steps
        * c["training"]["effective_batch"]
        * c["training"]["sequence_length"],
        "cuda_peak_memory_bytes": measured_peak,
        "gpu_environment_before_run": gpu_environment,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=a.root, text=True).strip()
        ),
        "engine_sha256": file_sha256(a.engine),
        "cuda_kernel_time_ms": sum(e.get("dur", 0) for e in cuda_events) / 1000
        if a.device.startswith("cuda")
        else None,
        "host_sync_count": len(sync_events) if a.device.startswith("cuda") else None,
        "nonzero_calls": sum(e.count for e in events if e.key == "aten::nonzero"),
        "kernel_launch_count": len(cuda_events) if a.device.startswith("cuda") else None,
        "counter_enabled": f["counter"],
        "capture_enabled": f["capture"],
        "dispatch_enabled": f["dispatch"],
        "probe_interval": f["probe_interval"],
        "scheduled_diagnostic_step_ms": scheduled_ms,
        "diagnostic_timing_scope": "one extra diagnostic+probe optimizer step at index 100; outside timing/profiler window",
        "flops_scope": "separate offline profiler; no inferred total FLOPs",
        "scientific_decision": False,
        "formal_confirmation": False,
        "synthetic_data": True,
        "fixture": a.fixture,
        "pilot": a.warmup < 5 or a.steps < 20,
        "gpu_runs": int(a.device.startswith("cuda")),
        "timing_scope": "optimizer step including enabled diagnostics/probes; excludes checkpoint IO, final evaluation; B0 original function; other cases equivalent extracted step",
        "source_file_hashes": inventory,
        "profiler_window": "one extra step, outside timing",
        "operator_profile": profile_summary,
        "trace_path": str(trace),
        "timeline": timeline,
        "limitations": [
            "CPU scalar operators are not CUDA synchronizations",
            "memory is allocated peak, not reserved/device process memory",
            "synthetic inputs do not measure dataset IO",
            "instrumentation ablations use old-source components; LIGHT uses repaired source",
            "B0 wall time and profiler both call the original frozen training function",
        ],
    }
    Path(a.result).write_text(json.dumps(result, indent=2) + "\n")


def run(c, args):
    from pd001.configs import ROOT
    from pd001.runner import write

    if args.steps < 1 or args.steps > 30 or args.warmup < 0 or args.warmup > 10 or not 1 <= args.repeats <= 5:
        raise ValueError("performance microbenchmark is bounded: 1--30 measured + 0--10 warmup, 1--5 repeats")
    if args.device.startswith("cuda") and not args.allow_gpu_microbenchmark:
        raise RuntimeError(
            "Kaggle user must explicitly enable --allow-gpu-microbenchmark; no full training is authorized"
        )
    if not args.all_arms and args.arm not in ("M0", "P50-F16"):
        raise ValueError("performance comparison supports the frozen M0/P50-F16 pair")
    baseline = Path(args.baseline_root).resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=baseline, text=True).strip()
    if (
        actual != BASELINE
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=baseline, text=True).strip()
    ):
        raise RuntimeError("baseline must be a clean checkout of actual run SHA " + BASELINE)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("choose a fresh performance evidence directory")
    output.mkdir(parents=True)
    rows = []
    for arm in ["M0", "P50-F16"] if args.all_arms else [args.arm]:
        for case in args.cases:
            for repeat in range(args.repeats):
                scratch = output / (arm + "-" + case + "-" + str(repeat))
                scratch.mkdir()
                root = ROOT if case == "LIGHT" else baseline
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--root",
                    str(root),
                    "--engine",
                    str(Path(__file__).with_name("engine.py")),
                    "--case",
                    case,
                    "--arm",
                    arm,
                    "--device",
                    args.device,
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                    "--scratch",
                    str(scratch),
                    "--result",
                    str(scratch / "result.json"),
                ]
                if args.perf_fixture:
                    command.append("--fixture")
                print("[Performance] " + arm + " " + case + " repeat " + str(repeat + 1), flush=True)
                with (scratch / "worker.log").open("w") as log:
                    subprocess.run(
                        command,
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    )
                rows.append(json.loads((scratch / "result.json").read_text()))
    summary = []
    for arm in sorted({r["arm"] for r in rows}):
        for case in args.cases:
            values = [r["optimizer_step_ms_p50"] for r in rows if r["arm"] == arm and r["case"] == case]
            summary.append(
                {
                    "arm": arm,
                    "case": case,
                    "repeat_median_ms": float(np.median(values)),
                    "repeat_range_ms": [min(values), max(values)],
                    "repeats": len(values),
                }
            )
    result = {
        "baseline_sha": BASELINE,
        "status": "PERFORMANCE_MICROBENCHMARK_COMPLETED",
        "scientific_decision": False,
        "READY_FOR_RETRAIN": "NO",
        "reason": "user review and actual Kaggle persistence rehearsal required; CPU fixtures cannot approve GPU retraining",
        "rows": rows,
        "summary": summary,
    }
    write(output / "ablation.json", result)
    write(output / "baseline.json", {"source_sha": BASELINE, "rows": [r for r in rows if r["case"] == "B0"]})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    for key in ("root", "engine", "case", "arm", "device", "scratch", "result"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--fixture", action="store_true")
    worker(parser.parse_args())
