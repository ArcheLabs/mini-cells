"""Re-run the original A3 equivalence matrix with one audited L2 Norm config."""
import argparse
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import torch

from a31_norm import config, describe_norm
from common import publishable, write
from m1_continuation_isolation import BOUNDS_SHA256, frozen_bounds
from m1_full_state_bridge import baseline, load_json
from m1_l2norm_autotune import FIXED, fixed_config
import m1_live_vs_restored as a3

FORMAT = "minicells.m1.full-state.l2norm-fixed-e2e.v1"
A3_COMMIT = "818c9625f46f9450c49500cfab54a4c9dbc71101"
GRID_COMMIT = "df39c7d6394b1b5175597abb5a36ae550272db11"
PHASES = ("P0", "P2", "P3")


@contextmanager
def fixed_forward(module, calls, segment):
    kernel = module.l2norm_fwd_kernel
    chosen = fixed_config(kernel)
    original = module.l2norm_fwd

    @wraps(original)
    def observed(x, *args, **kwargs):
        if x.dtype != torch.float16 or x.shape[-1] != 256:
            raise RuntimeError("Unexpected normalization input outside the audited FP16 D=256 branch")
        result = original(x, *args, **kwargs)
        launch = config(module, x)
        if kernel.best_config is not chosen or launch["num_warps"] != FIXED["num_warps"] or launch["BT"] != FIXED["BT"]:
            raise RuntimeError("Triton did not use the configured singleton L2 Norm policy")
        if result.dtype != x.dtype or result.shape != x.shape:
            raise RuntimeError("The fixed path changed L2 Norm dtype or shape")
        calls[(segment(), launch["D"], launch["T"], launch["NB"], launch["BD"],
               launch["BT"], launch["num_warps"], launch["num_stages"], launch["num_ctas"])] += 1
        return result

    with (patch.object(kernel, "configs", [chosen]),
          patch.object(module, "l2norm_fwd", observed)):
        yield dict(chosen.all_kwargs())


def worker(out, phase):
    from fla.modules import l2norm

    cuda_id = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_id.startswith("GPU-") or "," in cuda_id:
        raise RuntimeError("A single physical GPU UUID is required")
    if "self.configs[0]" not in inspect.getsource(type(l2norm.l2norm_fwd_kernel).run):
        raise RuntimeError("Installed Triton lacks the audited singleton config branch")
    source = describe_norm(l2norm.l2_norm, l2norm)
    calls = Counter()
    segments = ["initialization"]

    def labeled(fn, label):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            segments.append(label(*args, **kwargs))
            try:
                return fn(*args, **kwargs)
            finally:
                segments.pop()
        return wrapped

    with (fixed_forward(l2norm, calls, lambda: segments[-1]) as selected,
          patch.object(a3, "warm", labeled(a3.warm, lambda *_: "warmup")),
          patch.object(a3, "produce", labeled(a3.produce, lambda *_, **__: "write")),
          patch.object(a3, "trial", labeled(a3.trial, lambda _mc, _out, label, *_: f"query/{label}"))):
        a3.worker(out, phase)
    expected = {"warmup"}
    if phase == "P0":
        expected.update(("write", "query/P0"))
    group = "P1" if phase == "P0" else phase
    expected.update(f"query/{group}-{i}" for i in range(1, 4))
    missing = expected - {fields[0] for fields in calls}
    if missing:
        raise RuntimeError(f"Missing fixed L2 Norm coverage: {sorted(missing)}")
    keys = ("segment", "D", "T", "NB", "BD", "BT", "num_warps", "num_stages", "num_ctas")
    rows = [dict(zip(keys, fields, strict=True), calls=count) for fields, count in sorted(calls.items())]
    write(out / f"policy-{phase}.json", dict(phase=phase, pid=os.getpid(), gpu_uuid=cuda_id,
                                            source=source, fixed_config=selected, launches=rows))
    print(f"A3_FIXED_POLICY_VERIFIED {phase} launches={sum(calls.values())}", flush=True)


def evaluate(out, result):
    policies = {phase: load_json(out / f"policy-{phase}.json") for phase in PHASES}
    if (len({receipt["pid"] for receipt in policies.values()}) != 3
            or len({receipt["gpu_uuid"] for receipt in policies.values()}) != 1
            or any(r["fixed_config"] != policies["P0"]["fixed_config"] for r in policies.values())):
        raise RuntimeError("Workers did not share the same fixed config and physical GPU")
    pairs = a3.formal(out)
    if pairs is None:
        result.update(state_equivalence="INVALID", diagnosis="Fixed-policy initial state or input mismatch.")
        return
    planned = {f"{a}_vs_{b}" for a, b in a3.formal_pairs()}
    if set(pairs) != planned:
        raise RuntimeError("Incomplete A3 comparison matrix")
    failed = [key for key, values in pairs.items() if not values["passed"]]
    with (out / "layer_comparison.jsonl").open() as stream:
        rows = (json.loads(line) for line in stream)
        differing = {row["pair"] for row in rows
                     if row.get("max_absolute", 0) != 0 or row["status"] != "PASS"}
    exact = sorted(planned - differing)
    write(out / "fixed_config_summary.json", dict(pairs=len(pairs), pairs_passed=len(pairs)-len(failed),
            exact_under_recorded_metric=exact, failed_pairs=failed, config=policies["P0"]["fixed_config"]))
    if failed:
        result.update(status="FAIL", engineering_status="PASS", state_equivalence="FAIL",
                      diagnosis=f"{len(failed)}/{len(pairs)} original-equivalence pairs exceeded the frozen bounds.",
                      next_action="Investigate the first remaining divergence under the verified fixed configuration.")
    else:
        result.update(status="PASS", engineering_status="PASS", state_equivalence="PASS",
                      diagnosis=f"All {len(pairs)} A3 pairs passed under the fixed diagnostic policy.",
                      next_action="Review the separate cross-process Append gate before opening R1-B.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", choices=PHASES)
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
    result = dict(
        experiment="A3.2 Fixed L2 Norm End-to-End Equivalence", status="INVALID", engineering_status="FAIL",
        state_equivalence="INCOMPLETE", bridge_status="NOT_RUN_BY_SCOPE", capacity_status="NOT_RUN_BY_SCOPE",
        original_a3_commit=A3_COMMIT, grid_results_commit=GRID_COMMIT, frozen_bounds_sha256=BOUNDS_SHA256,
        intervention="native L2 Norm singleton config for every forward call")
    try:
        baseline(out)
        frozen_bounds(out)
        for phase in PHASES:
            child = subprocess.run([sys.executable, __file__, str(out), "--worker", phase])
            if child.returncode:
                error = out / f"worker_error-{phase}.json"
                raise RuntimeError(f"Worker {phase}: {load_json(error) if error.exists() else child.returncode}")
        evaluate(out, result)
    except Exception as exc:
        result.update(diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
