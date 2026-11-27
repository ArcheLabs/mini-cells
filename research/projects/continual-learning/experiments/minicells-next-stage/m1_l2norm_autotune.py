"""A3.1 focused intervention: native Triton autotuning versus one fixed config."""
import argparse
from copy import deepcopy
import inspect
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from a31_norm import Boundary, FIELDS, describe_norm, instrument, metrics
from common import publishable, write
from m1_continuation_isolation import BOUNDS_FILE, compare_pair, frozen_bounds
from m1_full_state_bridge import baseline, check_environment, dataset, load_json, probe_inputs
from m1_l2norm_isolation import (
    A3_COMMIT, BUNDLE_SHA256, BundleUnavailable, PROCESSES, get_a3, recover_bundle, validated_start,
)
from m1_reference_capacity import digest_file
from minicells_native import MiniCells
from r1_equivalence import ensure_readonly, live, trace

FORMAT = "minicells.m1.full-state.l2norm-autotune.v1"
FIXED = {"BT": 32, "num_warps": 4}
MODES = ("natural", "fixed")


def fixed_config(kernel):
    options = [option for option in kernel.configs
               if option.kwargs.get("BT") == FIXED["BT"] and option.num_warps == FIXED["num_warps"]]
    if len(kernel.configs) != 25 or len(options) != 1 or options[0].pre_hook is not None:
        raise ValueError("Pinned FLA L2 Norm Autotuner does not expose the expected 25 configs")
    return options[0]


def config_key(receipt):
    launch = receipt["kernel"]
    return launch["BT"], launch["num_warps"], launch["num_stages"], launch["num_ctas"]


def validate_calls(calls, fixed):
    for axis in "QK":
        kernel = calls[axis]["kernel"]
        bt, warps, stages, ctas = config_key(calls[axis])
        if kernel["D"] != 256 or kernel["kernel"] != "l2norm_fwd_kernel" or None in (bt, warps, stages, ctas):
            raise RuntimeError(f"Did not observe the expected Triton D=256 branch for {axis}")
        if fixed and (bt, warps) != (FIXED["BT"], FIXED["num_warps"]):
            raise RuntimeError(f"Fixed-config call used different launch parameters for {axis}")


class PinnedBoundary(Boundary):
    def __init__(self, fn, module, chosen):
        super().__init__(fn, module)
        self.chosen = chosen

    def norm(self, *args, **kwargs):
        if not self.active or self.norm_calls >= 2:
            return super().norm(*args, **kwargs)
        # A singleton selects the existing config directly in Triton 3.6.
        with patch.object(self.module.l2norm_fwd_kernel, "configs", [self.chosen]):
            return super().norm(*args, **kwargs)


def score_trial(mc, archive, expected, query, observer=None):
    cell, cache = validated_start(mc, archive, expected)
    pristine = deepcopy(cache)

    def execute():
        return trace(mc._backbone, lambda: deepcopy(pristine), query, primary_cache=cache)

    try:
        if observer is None:
            scores = ensure_readonly(mc, cell, execute)
        else:
            with instrument(mc._backbone, observer):
                scores = ensure_readonly(mc, cell, execute)
        if digest_file(archive) != BUNDLE_SHA256:
            raise RuntimeError("Diagnostic modified the historical A3 Bundle")
        return scores
    finally:
        mc.cells.delete(cell.id)


@torch.inference_mode()
def worker(out, role):
    from fla.modules import l2norm
    from triton import knobs

    torch.set_num_threads(4)
    work = out / "work"
    with MiniCells.open_preview(work / role) as mc:
        backbone = mc._backbone
        environment = check_environment(mc)
        gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
            raise RuntimeError("An exact shared physical GPU UUID is required")
        fn = backbone._recurrent_execution.ops.l2_norm
        if fn is not l2norm.l2_norm:
            raise RuntimeError("MoM resolves to a different normalization function")
        kernel = l2norm.l2norm_fwd_kernel
        chosen = fixed_config(kernel)
        if "self.configs[0]" not in inspect.getsource(type(kernel).run):
            raise RuntimeError("The installed Triton no longer has the audited single-config path")
        expected = load_json(work / "a3_reference.json")
        archive = Path(load_json(work / "archive.json")["path"])
        _, writes, query = probe_inputs(dataset(out), backbone)
        primary = query["tokens"] + next(iter(query["candidates"].values()))[:-1]
        if (writes[0] != expected["inputs"]["write_token_ids"]
                or list(query["candidates"]) != expected["inputs"]["candidate_order"]
                or query["candidates"] != expected["inputs"]["candidates"]
                or [primary] != expected["inputs"]["primary_chunks"]):
            raise ValueError("A3 inputs or chunk boundaries differ")

        # Same scratch warm-up as A3. Formal native behavior is observed before any intervention.
        scratch = live(backbone, writes[:1])
        backbone.step(primary, [scratch], [1.0])
        del scratch
        control = score_trial(mc, archive, expected["state"], query)
        natural = Boundary(fn, l2norm)
        observed = score_trial(mc, archive, expected["state"], query, natural)
        validation, fields = compare_pair(control, observed, load_json(out / BOUNDS_FILE))
        driver = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid, "--query-gpu=driver_version",
                                          "--format=csv,noheader"], text=True, timeout=10).strip()
        receipt = dict(pid=os.getpid(), gpu_uuid=gpu_uuid, driver=driver, environment=environment,
                       source=describe_norm(fn, l2norm), bundle_sha256=BUNDLE_SHA256,
                       cache_env=os.environ.get("TRITON_CACHE_AUTOTUNING"),
                       effective_cache_knob=bool(knobs.autotuning.cache),
                       effective_autotuner_disk_cache=kernel.cache_results,
                       autotune_key=list(kernel.keys), fixed_config=dict(chosen.all_kwargs()),
                       instrumentation=dict(summary=validation, fields=fields))
        write(out / f"worker-{role}.json", receipt)
        if not validation["passed"]:
            return
        del control, observed
        validate_calls(natural.calls, fixed=False)
        fixed = PinnedBoundary(fn, l2norm, chosen)
        score_trial(mc, archive, expected["state"], query, fixed)
        validate_calls(fixed.calls, fixed=True)
        tensors = {}
        receipt["modes"] = {}
        for mode, observer in zip(MODES, (natural, fixed), strict=True):
            values, metadata = observer.export()
            tensors.update({f"{mode}.{key}": value for key, value in values.items()})
            receipt["modes"][mode] = dict(calls=observer.calls, tensors=metadata)
        save_file(tensors, out / f"boundary-{role}.safetensors")
        write(out / f"worker-{role}.json", receipt)
        print(f"A31_AUTOTUNE_COMPLETED {role}", flush=True)


def pair_metrics(arrays, receipts, a, b, mode):
    left, right = (receipts[k]["modes"][mode] for k in (a, b))
    fields = {}
    for key in FIELDS:
        x, y = (arrays[role][f"{mode}.{key}"] for role in (a, b))
        m, n = (item["tensors"][key] for item in (left, right))
        similar = (x.shape == y.shape and x.dtype == y.dtype)
        values = metrics(x, y) if similar else dict(bitwise_equal=False, shape_or_dtype_mismatch=True)
        fields[key] = dict(values, layout_equal=all(m[field] == n[field] for field in
                                                    ("shape", "dtype", "stride", "storage_offset")))
    for axis in "QK":
        x, y = (item["calls"][axis] for item in (left, right))
        slot = x["first_recurrent_routed_slot"]
        fields[axis] = dict(config_equal=config_key(x) == config_key(y),
                            norm_arguments_equal=x["arguments"] == y["arguments"],
                            same_first_routed_slot=slot is not None and slot == y["first_recurrent_routed_slot"])
    return fields


def bitwise(fields, name):
    measurement = fields[name]
    return measurement["bitwise_equal"] and measurement["layout_equal"]


def classify(pairs, unchanged_inputs):
    fixed_equal = all(bitwise(lanes["fixed"], name) for lanes in pairs.values() for name in FIELDS)
    changed_pins = []
    for pair, lanes in pairs.items():
        native, fixed = lanes["natural"], lanes["fixed"]
        for axis in "QK":
            pre, post, recur = (f"{axis}_{end}" for end in ("PRE_NORM", "POST_NORM", "RECURRENT"))
            if (not native[axis]["config_equal"] and native[axis]["norm_arguments_equal"]
                    and native[axis]["same_first_routed_slot"] and fixed[axis]["same_first_routed_slot"]
                    and bitwise(native, pre) and not native[post]["bitwise_equal"]
                    and not native[recur]["bitwise_equal"]):
                changed_pins.append(dict(pair=pair, axis=axis))
    if fixed_equal and unchanged_inputs and changed_pins:
        return "AUTOTUNE_HYPOTHESIS_SUPPORTED", changed_pins
    if not unchanged_inputs or any(not bitwise(lanes[mode], f"{axis}_PRE_NORM")
                                  for lanes in pairs.values() for mode in MODES for axis in "QK"):
        return "PRE_NORM_INPUT_DIVERGED", changed_pins
    if not fixed_equal:
        return "DIVERGENCE_WITH_FIXED_CONFIG", changed_pins
    return "INCONCLUSIVE", changed_pins


def evaluate(out, result):
    receipts = {role: load_json(out / f"worker-{role}.json") for role in PROCESSES}
    if (len({r["pid"] for r in receipts.values()}) != 3
            or len({(r["gpu_uuid"], r["driver"]) for r in receipts.values()}) != 1
            or any(r["environment"] != receipts["P0"]["environment"] for r in receipts.values())
            or any(r["source"] != receipts["P0"]["source"] for r in receipts.values())
            or any(r["fixed_config"] != receipts["P0"]["fixed_config"] for r in receipts.values())):
        raise RuntimeError("Driver, device, source or diagnostic config is not comparable across processes")
    write(out / "autotuning.json", {role: {k: v for k, v in r.items() if k != "instrumentation"}
                                   for role, r in receipts.items()})
    if not all(r["instrumentation"]["summary"]["passed"] for r in receipts.values()):
        result.update(status="INVALID_INSTRUMENTATION", engineering_status="PASS",
                      diagnosis="Unobserved and observed native execution exceeded the original envelope.")
        return
    arrays = {role: load_file(out / f"boundary-{role}.safetensors") for role in PROCESSES}
    unchanged = {}
    for role in PROCESSES:
        expected = receipts[role]["modes"]
        unchanged[role] = {axis: all(
            expected["natural"]["tensors"][f"{axis}_PRE_NORM"][key]
            == expected["fixed"]["tensors"][f"{axis}_PRE_NORM"][key]
            for key in ("sha256", "shape", "dtype", "stride", "storage_offset"))
            for axis in "QK"}
    pairs = {f"{a}_vs_{b}": {mode: pair_metrics(arrays, receipts, a, b, mode) for mode in MODES}
             for a, b in combinations(PROCESSES, 2)}
    outcome, evidence = classify(pairs, all(v for axes in unchanged.values() for v in axes.values()))
    write(out / "comparisons.json", dict(status=outcome, pairs=pairs, unchanged_pre_norm=unchanged,
                                         differentiating_cases=evidence))
    result.update(status=outcome, diagnostic_result=outcome, engineering_status="PASS",
                  diagnosis=outcome, next_action="Retain the original R1-A gate pending separate end-to-end validation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", choices=PROCESSES)
    args = parser.parse_args()
    out = args.output
    if args.worker:
        try:
            worker(out, args.worker)
        except Exception as exc:
            write(out / f"worker_error-{args.worker}.json", dict(type=type(exc).__name__, message=str(exc)))
            raise
        return
    (out / "work").mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="M1-002A-R1-A3.1 Focused L2 Norm Autotune Diagnostic",
                  status="INCONCLUSIVE", engineering_status="FAIL", state_equivalence="NOT_RUN_BY_SCOPE",
                  bridge_status="NOT_RUN_BY_SCOPE", capacity_status="NOT_RUN_BY_SCOPE", a3_commit=A3_COMMIT)
    try:
        recover_bundle(out, os.getenv("A3_BUNDLE_PATH"), os.getenv("A3_KAGGLE_REF"))
        get_a3(out)
        baseline(out)
        frozen_bounds(out)
        for role in PROCESSES:
            child = subprocess.run([sys.executable, __file__, str(out), "--worker", role])
            if child.returncode:
                path = out / f"worker_error-{role}.json"
                raise RuntimeError(f"{role}: {load_json(path) if path.exists() else child.returncode}")
        evaluate(out, result)
    except BundleUnavailable as exc:
        print(f"A3_BUNDLE_BLOCKED {exc}", flush=True)
        result.update(status="BLOCKED", diagnosis=str(exc))
    except Exception as exc:
        result.update(diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
