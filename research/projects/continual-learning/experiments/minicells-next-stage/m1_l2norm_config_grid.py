"""Replay the verified A3.1 Q/K inputs through the full native L2 Norm grid."""
import argparse
from hashlib import sha256
import json
import os
import subprocess
from pathlib import Path
import time
from urllib.request import urlopen
from unittest.mock import patch

import torch
from safetensors.torch import load

from a31_norm import config, describe, describe_norm, metrics, restore, tensor_bytes
from common import environment, publishable, write
from m1_full_state_bridge import PACKAGES

FORMAT = "minicells.m1.full-state.l2norm-config-grid.v1"
INPUT_COMMIT = "3647b6894ce70f94abd3fcf53329c9eb3c75f741"
INPUT_ROOT = (
    f"https://raw.githubusercontent.com/ArcheLabs/mini-cells/{INPUT_COMMIT}/artifacts/experiments/"
    "minicells-m1-002a-r1-a3-1-l2norm-autotune/5f1c177d2c0c/"
)
GRID = tuple((bt, warps) for bt in (8, 16, 32, 64, 128) for warps in (1, 2, 4, 8, 16))
ROLES = ("P0", "P2", "P3")


def verified_inputs():
    def read(name):
        with urlopen(INPUT_ROOT + name, timeout=30) as response:
            return response.read()

    manifest = json.loads(read("manifest.json"))["artifacts_sha256"]

    def checked(name):
        data = read(name)
        if sha256(data).hexdigest() != manifest[name]:
            raise ValueError(f"Published input checksum mismatch for {name}")
        return data

    receipts = json.loads(checked("autotuning.json"))
    comparisons = json.loads(checked("comparisons.json"))
    if any(not v for group in comparisons["unchanged_pre_norm"].values() for v in group.values()):
        raise ValueError("Published A3.1 run changed a PRE_NORM input")
    for lane in comparisons["pairs"].values():
        for mode in ("natural", "fixed"):
            for axis in "QK":
                field = lane[mode][f"{axis}_PRE_NORM"]
                if not field["bitwise_equal"] or not field["layout_equal"]:
                    raise ValueError("Published A3.1 inputs do not match across lanes")
    archive = load(checked("boundary-P0.safetensors"))
    for axis in "QK":
        for receipt in receipts.values():
            if receipt["modes"]["natural"]["calls"][axis]["arguments"] != receipts["P0"]["modes"]["natural"]["calls"][axis]["arguments"]:
                raise ValueError("A3.1 norm arguments differ across input processes")
        for mode in ("natural", "fixed"):
            for field in ("PRE_NORM", "POST_NORM"):
                key = f"{axis}_{field}"
                expected = receipts["P0"]["modes"][mode]["tensors"][key]["sha256"]
                if sha256(tensor_bytes(archive[f"{mode}.{key}"])).hexdigest() != expected:
                    raise ValueError(f"Published tensor hash mismatch for {mode}.{key}")
    return archive, receipts, {k: manifest[k] for k in ("autotuning.json", "comparisons.json", "boundary-P0.safetensors")}


def select_configs(kernel):
    choices = {(choice.kwargs["BT"], choice.num_warps): choice for choice in kernel.configs}
    if sorted(choices) != list(GRID) or len(kernel.configs) != len(GRID):
        raise RuntimeError("The native FLA kernel no longer exposes the frozen 25-config grid")
    if len({(choice.num_stages, choice.num_ctas) for choice in choices.values()}) != 1:
        raise RuntimeError("Grid unexpectedly contains distinct stages or CTA counts")
    if any(choice.pre_hook is not None for choice in choices.values()):
        raise RuntimeError("Frozen grid unexpectedly modifies inputs in a config pre-hook")
    return choices


def run_axis(fn, module, view, params, original, native_config, fixed, fixed_config):
    kernel = module.l2norm_fwd_kernel
    choices = select_configs(kernel)
    if view.shape[-1] != 256 or view.dtype != torch.float16:
        raise ValueError("Replay requires the captured FP16 D=256 branch")
    before = sha256(tensor_bytes(view)).hexdigest()
    reference = None
    rows = []
    for key in GRID:
        with patch.object(kernel, "configs", [choices[key]]):
            output = fn(view, **params)
            observed = config(module, view)
        if (observed["BT"], observed["num_warps"]) != key:
            raise RuntimeError(f"Triton selected a different config for {key}")
        if view.dtype != output.dtype or view.shape != output.shape:
            raise RuntimeError("FLA replay output dtype or shape changed")
        observed_bytes = sha256(tensor_bytes(view)).hexdigest()
        if observed_bytes != before:
            raise RuntimeError("Native norm modified its input")
        output = output.detach().to("cpu", copy=True)
        if reference is None:
            reference = output
        row = dict(BT=key[0], num_warps=key[1], launch=observed, input_sha256=before,
                   output_sha256=sha256(tensor_bytes(output)).hexdigest(), vs_first=metrics(reference, output))
        if key == native_config:
            row["vs_archived_native"] = metrics(original, output)
        if key == fixed_config:
            row["vs_archived_fixed"] = metrics(fixed, output)
        rows.append(row)
    return rows


def outcome(rows):
    if not any("vs_archived_native" in row for row in rows) or not any("vs_archived_fixed" in row for row in rows):
        raise ValueError("Replay did not cover published native and fixed launch configs")
    mismatch = any(not v["bitwise_equal"] for row in rows for k, v in row.items() if k.startswith("vs_archived"))
    if mismatch:
        return "ARCHIVED_CONTEXT_MISMATCH"
    if any(not row["vs_first"]["bitwise_equal"] for row in rows):
        return "CONFIG_DEPENDENT_OUTPUT"
    return "CONFIG_GRID_BITWISE_EQUAL"


@torch.inference_mode()
def run(out, result):
    from fla.modules import l2norm

    env = environment()
    if (torch.cuda.device_count() != 1 or env["gpu"] != "Tesla T4" or env["capability"] != [7, 5]
            or env["cuda"] != "13.0" or env["packages"] != PACKAGES):
        raise RuntimeError("Replay runtime differs from the published A3.1 T4 environment")
    archive, receipts, hashes = verified_inputs()
    source = describe_norm(l2norm.l2_norm, l2norm)
    p0 = receipts["P0"]
    # Use the observed source receipt while bypassing all model and cache code.
    if p0["source"] != source:
        raise RuntimeError("Installed FLA implementation differs from the capture source")
    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not gpu_uuid or not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
        raise RuntimeError("Expected a single physical GPU UUID")
    driver = subprocess.check_output(
        ["nvidia-smi", "-i", gpu_uuid, "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, timeout=10).strip()
    write(out / "input_provenance.json", dict(
        input_commit=INPUT_COMMIT, input_hashes=hashes, environment=env, source=source,
        observed_natural_configs={r: receipts[r]["modes"]["natural"]["calls"] for r in ROLES},
        captured_gpu_uuid=p0["gpu_uuid"], captured_driver=p0["driver"], gpu_uuid=gpu_uuid, driver=driver))
    all_rows, outcomes = {}, {}
    for axis in "QK":
        native = p0["modes"]["natural"]
        fixed = p0["modes"]["fixed"]
        parameters = dict(native["calls"][axis]["arguments"])
        if parameters.get("output_dtype"):
            parameters["output_dtype"] = getattr(torch, parameters["output_dtype"].removeprefix("torch."))
        meta = native["tensors"][f"{axis}_PRE_NORM"]
        view = restore(archive[f"natural.{axis}_PRE_NORM"], meta, "cuda:0")
        if describe(view)["data_ptr_alignment_256"] != meta["data_ptr_alignment_256"]:
            raise RuntimeError("Restored PRE_NORM alignment differs from the capture")
        native_cfg, fixed_cfg = (entry["calls"][axis]["kernel"] for entry in (native, fixed))
        rows = run_axis(l2norm.l2_norm, l2norm, view, parameters, archive[f"natural.{axis}_POST_NORM"],
                        (native_cfg["BT"], native_cfg["num_warps"]), archive[f"fixed.{axis}_POST_NORM"],
                        (fixed_cfg["BT"], fixed_cfg["num_warps"]))
        all_rows[axis] = rows
        outcomes[axis] = outcome(rows)
        print(f"L2NORM_GRID {axis} {len(rows)} {outcomes[axis]}", flush=True)
    status = next(value for value in ("ARCHIVED_CONTEXT_MISMATCH", "CONFIG_DEPENDENT_OUTPUT",
                                      "CONFIG_GRID_BITWISE_EQUAL") if value in outcomes.values())
    write(out / "config_grid.json", dict(status=status, outcomes=outcomes, axes=all_rows,
              unique_output_hashes={axis: len({row["output_sha256"] for row in rows})
                                    for axis, rows in all_rows.items()}))
    decisions = {
        "ARCHIVED_CONTEXT_MISMATCH": "Check the difference between standalone and model execution before attributing the result.",
        "CONFIG_DEPENDENT_OUTPUT": "Use the differing config pair as the targeted reproduction for the L2 Norm boundary.",
        "CONFIG_GRID_BITWISE_EQUAL": "Investigate the failing A3 normalization boundary with matched input and runtime metadata.",
    }
    result.update(status=status, engineering_status="PASS", diagnosis=status, next_action=decisions[status])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="A3.1 Native L2 Norm Config Grid", status="INVALID", engineering_status="FAIL",
                  state_equivalence="NOT_RUN_BY_SCOPE", bridge_status="NOT_RUN_BY_SCOPE",
                  capacity_status="NOT_RUN_BY_SCOPE")
    try:
        run(out, result)
    except Exception as exc:
        result.update(diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
