"""v0.4 isolated formal runner; no optimizer or retraining command exists here."""

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
import uuid
from pathlib import Path

import torch

from . import formal_storage
from .common import BASE, NATIVE, ROOT, digest, sha, source, write

STAGES = (
    "e1-preflight",
    "e1a-correctness",
    "e1a-benchmark",
    "e1b-correctness",
    "e1b-benchmark",
    "e1c-benchmark",
    "e1-analyze",
)


class BudgetStop(RuntimeError):
    pass


def configuration():
    c = json.loads((BASE / "formal_config.json").read_text())
    if (
        sha(c) != (BASE / "formal_config.sha256").read_text().strip()
        or digest(BASE / "PROTOCOL.v0.4.md") != c["protocol_sha256"]
    ):
        raise RuntimeError("FORMAL_CONFIG_OR_PROTOCOL_CHANGED")
    if (
        c["training_authorized"]
        or c["max_context"] != 128
        or list(c["assets"]) != ["T1", "X3", "X4"]
        or c["gpu_worker_budget_seconds"] != 7200
    ):
        raise RuntimeError("FORMAL_SCOPE_CHANGED")
    return c


def verify_execution_source():
    paths = [
        BASE / "implementation",
        BASE / "run.py",
        BASE / "formal_config.json",
        BASE / "formal_config.sha256",
        NATIVE,
        ROOT / "src/minicells/language_models.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *[str(p.relative_to(ROOT)) for p in paths]],
        cwd=ROOT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("FORMAL_EXECUTION_CODE_DIFFERS_FROM_PINNED_COMMIT")


def hardware():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE_NO_FORMAL_GPU_RESULT")
    properties = torch.cuda.get_device_properties(0)

    def query(args):
        return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=10).strip()

    identity = query(
        [
            "--id=0",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = query(
        ["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"]
    )
    device_uuid = identity.split(",")[0].strip()
    foreign = [
        line
        for line in processes.splitlines()
        if line.split(",")[0].strip() == device_uuid
        and line.split(",")[1].strip() != str(os.getpid())
    ]
    if foreign:
        raise RuntimeError("OTHER_GPU_WORKLOAD_PRESENT_NO_TIMING")
    return {
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "huggingface-hub", "matplotlib")
        },
        "identity": identity,
        "uuid": device_uuid,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "process_check": "NO_OTHER_PROCESS_ON_SELECTED_GPU",
        "pid": os.getpid(),
        "power_clock_observation": query(
            [
                "--id=0",
                "--query-gpu=power.draw,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def device_contract(env):
    return {key: env[key] for key in ("identity", "uuid", "torch", "cuda", "python")}


class Runner:
    def __init__(self, root, assets, publish=False):
        self.root = Path(root)
        self.directory = Path(assets)
        self.root.mkdir(parents=True, exist_ok=True)
        verify_execution_source()
        self.c = configuration()
        self.publish = publish
        self.deadline = None
        path = self.root / "RUN_MANIFEST.json"
        if not path.exists() or self._pristine(path):
            try:
                formal_storage.restore(self.root, self.c, source())
            except (OSError, RuntimeError, ValueError, ImportError) as error:
                self.restore_failure = (
                    str(error) if str(error).startswith("FORMAL_") else type(error).__name__
                )
        if path.exists():
            self.m = json.loads(path.read_text())
        else:
            self.m = {
                "experiment_id": "EPCLM-001",
                "scope": "E1_FORMAL_INFERENCE_ONLY",
                "run_id": uuid.uuid4().hex,
                "source_sha": source(),
                "config_sha256": sha(self.c),
                "optimizer_updates": 0,
                "stages": {},
                "gpu_worker_seconds_charged": 0.0,
                "gpu_worker_seconds_observed": 0.0,
                "status": "NOT_RUN",
            }
        if self.m["source_sha"] != source() or self.m["config_sha256"] != sha(self.c):
            raise RuntimeError("EXISTING_RUN_SOURCE_OR_CONFIG_CHANGED")
        interrupted = []
        for key, record in self.m["stages"].items():
            if record.get("status") == "RUNNING":
                record.update(
                    status="INTERRUPTED", failure_code="PROCESS_INTERRUPTED_DURABLE_BUDGET_RETAINED"
                )
                interrupted.append(key)
        if interrupted:
            self.m["status"] = "INCOMPLETE"
            write(
                self.root / "INTERRUPTED_STAGES.json",
                {"stages": interrupted, "budget_retained": True},
            )
        self.save()
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @staticmethod
    def _pristine(path):
        if not path.exists():
            return True
        value = json.loads(path.read_text())
        return value.get("gpu_worker_seconds_charged", 0) == 0 and all(
            k == "e1-preflight" and v.get("status") != "COMPLETE"
            for k, v in value.get("stages", {}).items()
        )

    def resources(self):
        write(
            self.root / "RESOURCE_MANIFEST.json",
            {
                "gpu_worker_seconds_observed": self.m["gpu_worker_seconds_observed"],
                "gpu_worker_seconds_charged": self.m["gpu_worker_seconds_charged"],
                "budget_seconds": 7200,
                "training_optimizer_updates": 0,
                "unmeasured_dram_bytes": None,
                "unmeasured_energy_joules": None,
                "budget_scope": "GPU-worker wall time; check between cases; hard watchdog at remaining budget",
                "kernel_time_scope": "separate profiler only; CUDA stream intervals include submission gaps",
            },
        )

    def save(self):
        write(self.root / "RUN_MANIFEST.json", self.m)

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def output(self, name, value):
        write(self.root / name, value)
        if getattr(self, "active_stage", None):
            self.m["stages"][self.active_stage].setdefault("progress", {})[name] = digest(
                self.root / name
            )
            self.save()

    def check(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise BudgetStop("FORMAL_BUDGET_EXHAUSTED_NO_NEW_CASE")

    def complete(self, stage):
        record = self.m["stages"].get(stage, {})
        if record.get("status") != "COMPLETE":
            return False
        if stage == "e1-analyze" and record.get("dependencies") != self.analysis_dependencies():
            return False
        for name, value in record["outputs"].items():
            if not (self.root / name).is_file() or digest(self.root / name) != value:
                raise RuntimeError("COMPLETED_STAGE_OUTPUT_CHANGED")
        return True

    def require(self, stage):
        if not self.complete(stage):
            raise RuntimeError("REQUIRED_STAGE_NOT_COMPLETE_" + stage)

    def assets(self):
        result = self.read("ASSET_MANIFEST.json")
        if not all(
            (self.directory / row["files"]["inference.pt"]).exists()
            for row in result["models"].values()
        ):
            from .formal_assets import prepare

            restored = prepare(self.directory, self.c)
            if restored["inputs_sha256"] != result["inputs_sha256"]:
                raise RuntimeError("RESTORED_INPUT_IDENTITY_CHANGED")
        return result

    def model(self, key):
        from .formal_assets import load_model

        return load_model(self.assets(), key, "cuda:0", self.directory)

    @contextlib.contextmanager
    def gpu(self):
        from .retrain import lock

        with lock("/tmp/epclm001-controlled-retrain-gpu.lock"):
            env = hardware()
            if "device_contract" in self.m and self.m["device_contract"] != device_contract(env):
                raise RuntimeError("FROZEN_DEVICE_CHANGED_NO_MIXED_GPU_SAMPLES")
            self.m["device_contract"] = device_contract(env)
            remaining = 7200 - self.m["gpu_worker_seconds_charged"]
            if remaining <= 0:
                raise BudgetStop("DURABLE_FORMAL_BUDGET_EXHAUSTED")
            old = self.m["gpu_worker_seconds_charged"]
            self.m["gpu_worker_seconds_charged"] = 7200.0
            self.save()
            # A lost process retains this reservation in the independent controller.
            formal_storage.persist(self.root, self.c)
            begin = time.monotonic()
            self.deadline = begin + remaining
            # A hard stop cannot release the durable reservation: interrupted runs fail closed.
            watchdog = threading.Timer(remaining, lambda: os._exit(124))
            watchdog.daemon = True
            watchdog.start()
            try:
                yield env
            finally:
                watchdog.cancel()
                elapsed = time.monotonic() - begin
                self.m["gpu_worker_seconds_observed"] += elapsed
                self.m["gpu_worker_seconds_charged"] = min(7200.0, old + elapsed)
                self.deadline = None
                self.save()
                torch.cuda.empty_cache()

    def analysis_dependencies(self):
        return sha({key: value for key, value in self.m["stages"].items() if key != "e1-analyze"})

    def execute(self, stage):
        if self.complete(stage):
            return
        previous = self.m["stages"].get(stage, {})
        for name, value in previous.get("progress", {}).items():
            if not (self.root / name).exists() or digest(self.root / name) != value:
                raise RuntimeError("PARTIAL_STAGE_OUTPUT_CHANGED_NO_REPLAY")
        self.active_stage = stage
        self.m["stages"][stage] = {
            "status": "RUNNING",
            "progress": previous.get("progress", {}),
            "previous_attempts": previous.get("previous_attempts", [])
            + (
                [{k: v for k, v in previous.items() if k != "previous_attempts"}]
                if previous
                else []
            ),
        }
        self.m["status"] = "RUNNING"
        self.save()
        try:
            getattr(self, stage.replace("-", "_"))()
            record = self.m["stages"][stage]
            record.update(status="COMPLETE", outputs=dict(record["progress"]))
            if stage == "e1-analyze":
                record["dependencies"] = self.analysis_dependencies()
        except (OSError, RuntimeError, ValueError, AssertionError, TypeError, ImportError) as error:
            record = self.m["stages"][stage]
            record.update(
                status="RESOURCE_STOP" if isinstance(error, BudgetStop) else "FAILED",
                failure_type=type(error).__name__,
                failure_code=str(error).splitlines()[0][:200],
            )
            failures = (
                self.read("FAILED_CASES.json") if (self.root / "FAILED_CASES.json").exists() else []
            )
            failures.append(
                {
                    "stage": stage,
                    "case": getattr(self, "current_case", None),
                    "failure": record["failure_code"],
                }
            )
            write(self.root / "FAILED_CASES.json", failures)
            self.m["status"] = "INCOMPLETE"
            self.save()
            raise
        finally:
            self.active_stage = None
        self.save()
        self.resources()
        formal_storage.persist(self.root, self.c)
        if self.publish:
            formal_storage.publish(self.root)

    def e1_preflight(self):
        if hasattr(self, "restore_failure"):
            raise RuntimeError("FORMAL_CONTROLLER_RESTORE_" + self.restore_failure)
        self.output("PERSISTENCE_PREFLIGHT.json", formal_storage.probe(self.root, self.c))
        from .formal_assets import prepare

        self.output("ASSET_MANIFEST.json", prepare(self.directory, self.c))
        self.output("ENVIRONMENT.json", hardware())
        self.output(
            "EXECUTION_INPUTS.json",
            {
                str(p.relative_to(ROOT)): digest(p)
                for p in sorted((BASE / "implementation").rglob("*.py"))
            },
        )

    def e1a_correctness(self):
        self.require("e1-preflight")
        from .formal_correctness import check_model

        assets = self.assets()
        result = {}
        with self.gpu():
            for key in self.c["assets"]:
                self.check()
                model = self.model(key)
                result[key] = check_model(
                    model, assets["inputs"], self.c, assets["models"][key]["inference_sha256"], self
                )
                del model
                torch.cuda.empty_cache()
                self.output("E1A_PARITY.json", result)

    def freeze(self):
        env = hardware()
        data = {
            "source_sha": source(),
            "config_sha256": sha(self.c),
            "device_contract": device_contract(env),
            "dtype": "float32",
            "warmup": self.c["warmup"],
            "repeats": self.c["repeats"],
            "input_sha256": self.assets()["inputs_sha256"],
            "workloads": self.c["workloads"],
            "event_points": self.c["event_points"],
            "tolerances": self.c["tolerances"],
            "statistics": self.c["statistics"],
            "plan_selection": "All points fixed before pilot; no outcome-based reduction",
        }
        path = self.root / "FORMAL_FREEZE.json"
        if path.exists() and self.read("FORMAL_FREEZE.json") != data:
            raise RuntimeError("FORMAL_FREEZE_CHANGED")
        if not path.exists():
            self.output("FORMAL_FREEZE.json", data)

    def inference_samples(self, keys, workloads, repeats, warmup, checkpoint_name, cross=False):
        from .formal_measure import analytic_work, request

        assets = self.assets()
        rows = (
            self.read(checkpoint_name)["records"] if (self.root / checkpoint_name).exists() else []
        )
        done = {(r["model"], r["workload"], r["mode"], r["repeat"]) for r in rows}
        variants = (
            [
                ("T1", "full"),
                ("T1", "A0"),
                ("X3", "full"),
                ("X3", "A0"),
                ("X3", "A1"),
                ("X4", "full"),
                ("X4", "A0"),
                ("X4", "A1"),
            ]
            if cross
            else [(key, mode) for key in keys for mode in ("full", "A0", "A1")]
        )
        for workload in workloads:
            if all(
                (key, workload["id"], mode, repeat) in done
                for key, mode in variants
                for repeat in range(repeats)
            ):
                continue
            for repeat in range(warmup + repeats):
                order = variants[repeat % len(variants) :] + variants[: repeat % len(variants)]
                for order_index, (key, mode) in enumerate(order):
                    if repeat >= warmup and (key, workload["id"], mode, repeat - warmup) in done:
                        continue
                    self.current_case = {
                        "model": key,
                        "mode": mode,
                        "workload": workload["id"],
                        "repeat": repeat - warmup,
                    }
                    self.check()
                    model = self.model(key)
                    ids = torch.tensor(
                        [
                            assets["inputs"][str(workload["offset"])][
                                : workload["prefill"] + workload["decode"]
                            ]
                        ],
                        dtype=torch.long,
                        device="cuda:0",
                    )
                    sample = request(model, ids, workload["prefill"], mode)
                    if repeat >= warmup:
                        sample.update(
                            model=key,
                            workload=workload["id"],
                            repeat=repeat - warmup,
                            order_index=order_index,
                            flops=analytic_work(key, mode, workload["prefill"], workload["decode"]),
                        )
                        rows.append(sample)
                        self.output(checkpoint_name, {"records": rows})
                    del model, ids
                    torch.cuda.empty_cache()
        return rows

    def e1a_benchmark(self):
        self.require("e1a-correctness")
        with self.gpu():
            pilot = self.inference_samples(
                ["X3", "X4"],
                [self.c["workloads"][0], self.c["workloads"][-1]],
                self.c["pilot_repeats"],
                self.c["pilot_warmup"],
                "pilot/E1A_TIMING.json",
            )
            self.output(
                "pilot/E1A_TIMING.json", {"records": pilot, "scope": "PILOT_ONLY_NOT_FORMAL"}
            )
            self.freeze()
            self.output(
                "E1A_TIMING.json",
                {
                    "records": self.inference_samples(
                        ["X3", "X4"],
                        self.c["workloads"],
                        self.c["repeats"],
                        self.c["warmup"],
                        "E1A_TIMING.json",
                    )
                },
            )

    def e1b_correctness(self):
        self.require("e1-preflight")
        from .formal_events import semantics, special_semantics

        rows = []
        with self.gpu():
            for kind in self.c["event_kernels"]:
                for point in self.c["event_points"]:
                    self.check()
                    rows.append(
                        {
                            "point": point["id"],
                            "kernel": kind,
                            "report": semantics(point, kind, self.c, "cuda:0"),
                        }
                    )
                    self.output("E1B_PARITY.json", {"records": rows})
            self.output("E1B_SPECIAL_SEMANTICS.json", special_semantics(self.c, "cuda:0"))

    def e1b_benchmark(self):
        self.require("e1b-correctness")
        from .formal_events import samples

        rows = (
            self.read("E1B_TIMING.json")["records"]
            if (self.root / "E1B_TIMING.json").exists()
            else []
        )
        with self.gpu():
            pilot = []
            for kind in self.c["event_kernels"]:
                for point in (self.c["event_points"][1], self.c["event_points"][5]):
                    pilot += samples(
                        point,
                        kind,
                        self.c,
                        "cuda:0",
                        self,
                        self.c["pilot_repeats"],
                        self.c["pilot_warmup"],
                    )
            self.output(
                "pilot/E1B_TIMING.json", {"records": pilot, "scope": "PILOT_ONLY_NOT_FORMAL"}
            )
            self.freeze()
            for kind in self.c["event_kernels"]:
                for point in self.c["event_points"]:
                    self.check()
                    old = [r for r in rows if r["point"] == point["id"] and r["kernel"] == kind]
                    if len(old) == self.c["repeats"] * 4:
                        continue
                    other = [
                        r for r in rows if not (r["point"] == point["id"] and r["kernel"] == kind)
                    ]
                    self.current_case = {"kernel": kind, "point": point["id"]}
                    local = samples(
                        point,
                        kind,
                        self.c,
                        "cuda:0",
                        self,
                        self.c["repeats"],
                        self.c["warmup"],
                        existing=old,
                        progress=lambda local, other=other: self.output(
                            "E1B_TIMING.json", {"records": other + local}
                        ),
                    )
                    rows = other + local
                    self.output("E1B_TIMING.json", {"records": rows})

            from .formal_profile import events

            if not (self.root / "EVENT_PROFILER.json").exists():
                events(self)

    def e1c_benchmark(self):
        self.require("e1a-correctness")
        self.require("e1a-benchmark")
        self.freeze()
        with self.gpu():
            self.output(
                "E1C_TIMING.json",
                {
                    "records": self.inference_samples(
                        list(self.c["assets"]),
                        self.c["workloads"],
                        self.c["repeats"],
                        self.c["warmup"],
                        "E1C_TIMING.json",
                        cross=True,
                    )
                },
            )
            from .formal_profile import models

            if not (self.root / "MODEL_PROFILER.json").exists():
                models(self)
            assets = self.assets()
            self.output(
                "E1C_QUALITY.json",
                {
                    key: {
                        k: row[k]
                        for k in (
                            "quality",
                            "quality_reused",
                            "quality_evaluation_dtype",
                            "quality_file_sha256",
                            "quality_evaluator_sha256",
                            "quality_scope",
                            "dataset_manifest_sha256",
                        )
                    }
                    for key, row in assets["models"].items()
                },
            )

    def e1_analyze(self):
        from .formal_analysis import analyze

        analyze(self.root, self.c, self.m)
        for path in [
            *self.root.glob("*.md"),
            self.root / "DECISIONS.json",
            self.root / "RAW_SUMMARY.csv",
            self.root / "SAMPLE_STATISTICS.json",
            self.root / "COMPETITIVE_TABLE.csv",
            self.root / "INFERENCE_COVERAGE.json",
            self.root / "FIGURE_INDEX.json",
            *self.root.glob("figures/*"),
        ]:
            if path.is_file():
                self.m["stages"][self.active_stage]["progress"][
                    str(path.relative_to(self.root))
                ] = digest(path)


def main():
    parser = argparse.ArgumentParser(description="v0.4 inference only; never trains models")
    parser.add_argument("command", choices=(*STAGES, "e1-run", "e1-publish"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("/kaggle/working/epclm001-e1-assets"))
    parser.add_argument("--publish-github", action="store_true")
    args = parser.parse_args()
    from .retrain import lock

    with lock(args.output / ".formal-controller.lock"):
        runner = Runner(args.output, args.assets, args.publish_github)
        if args.publish_github:
            formal_storage.git_preflight(args.output)
        try:
            if args.command == "e1-publish":
                formal_storage.persist(args.output, runner.c)
                formal_storage.publish(args.output)
            else:
                requested = STAGES if args.command == "e1-run" else (args.command,)
                for stage in requested:
                    runner.execute(stage)
                if args.command == "e1-run":
                    runner.m["status"] = "FORMAL_COMPLETE_REVIEW_REQUIRED"
                    runner.save()
        except (OSError, RuntimeError, ValueError, AssertionError, TypeError, ImportError):
            from .formal_analysis import analyze

            if runner.m["stages"].get("e1-analyze", {}).get("status") == "COMPLETE":
                runner.m["stages"]["e1-analyze"]["status"] = "STALE_AFTER_STAGE_FAILURE"
                runner.save()
            analyze(runner.root, runner.c, runner.m)
            runner.resources()
            if args.publish_github:
                try:
                    formal_storage.persist(runner.root, runner.c)
                except (OSError, RuntimeError, ValueError) as error:
                    write(runner.root / "PERSISTENCE_FAILURE.json", {"type": type(error).__name__})
                formal_storage.publish(runner.root)
            raise
        finally:
            runner.resources()
        if args.publish_github:
            formal_storage.persist(runner.root, runner.c)
            formal_storage.publish(runner.root)
        print(
            json.dumps(
                {
                    "status": runner.m["status"],
                    "output": str(args.output),
                    "optimizer_updates": 0,
                    "publication": runner.read("PUBLICATION.json")
                    if (runner.root / "PUBLICATION.json").exists()
                    else None,
                }
            )
        )


if __name__ == "__main__":
    main()
