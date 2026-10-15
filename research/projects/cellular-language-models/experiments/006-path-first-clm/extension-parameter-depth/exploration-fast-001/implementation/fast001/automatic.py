"""One user launch: fail-fast checks, data, short pilot, training, offline return."""

import json
import time
from pathlib import Path
from exp006.reuse import canonical_sha256
from .config import ARMS, freeze, authorize
from .entry import preflight
from .data import prepare, load_assets
from .benchmark import benchmark, gpu_check, validate_pilot
from .schedule import matrix
from .evaluate import evaluate, profile
from .interventions import suite
from .report import report
from .resume import export
from .io import write_json


def run(c, output, devices):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("[1/5] Automatic CPU functional preflight", flush=True)
    preflight(c, output / "cpu-gate")
    gpu_check(devices)
    print("[2/5] Automatic independent development data checks", flush=True)
    assets = output / "assets"
    if not assets.exists():
        prepare(c, assets)
    load_assets(c, assets)
    config_path = output / "execution-config.json"
    if config_path.exists():
        frozen = json.loads(config_path.read_text())
        if frozen["approval"].get("base_config_sha256") != canonical_sha256(c):
            raise ValueError("existing development config differs; preserve its output")
        for device in devices:
            authorize(frozen, device, "train")
        print("[3/5] Reusing matching recorded execution identity", flush=True)
    else:
        print("[3/5] Short same-GPU performance precheck; training follows automatically", flush=True)
        reference = output / "pilot-reference.json"
        pilot_path = (
            Path(json.loads(reference.read_text())["path"])
            if reference.exists()
            else output / "pilot/pilot.json"
        )
        if pilot_path.is_file():
            pilot = json.loads(pilot_path.read_text())
            validate_pilot(c, pilot)
        else:
            # Preserve incomplete measurements; a new attempt never overwrites them.
            folder = output / "pilot"
            if folder.exists():
                folder = output / ("pilot-retry-" + str(time.time_ns()))
            pilot = benchmark(
                c, folder, devices[0], allow_gpu=True, microbatches=(c["training"]["microbatch"],)
            )
            write_json(reference, {"path": str((folder / "pilot.json").resolve())})
        print("Estimated development wall seconds:", pilot["projected_total_seconds"], flush=True)
        # User requested automatic development; target is advisory, never a kill or a new form.
        frozen = freeze(c, pilot, config_path, approve=True, accept_slow=True)
    print("[4/5] Starting CLM-S16, CLM-S8; next free device runs M0-TINY", flush=True)
    result = matrix(
        frozen,
        config_path.resolve(),
        assets.resolve(),
        output.resolve(),
        devices,
        arms=["CLM-S16", "CLM-S8", "M0-TINY"],
        resume=True,
    )
    if result["status"] == "DEVELOPMENT_COMPLETED":
        print("[5/5] All workers joined; offline evaluation, diagnostics and report", flush=True)
        for arm in ARMS:
            folder = output / "runs" / arm
            if not (folder / "evaluation.json").exists():
                evaluate(frozen, arm, assets, folder, devices[0])
            if not (folder / "hardware-profile.json").exists():
                profile(frozen, arm, assets, folder, devices[0])
            if arm != "M0-TINY" and not (folder / "interventions.json").exists():
                suite(frozen, arm, assets, folder, devices[0])
        result = report(frozen, output)
    if any((output / "runs").glob("*/checkpoint.pt")):
        destination = output / ("portable-" + str(time.time_ns()))
        export(output, destination)
        result["portable_output"] = str(destination)
    result["end_to_end_seconds"] = time.perf_counter() - started
    write_json(output / "automatic-result.json", result)
    print("Save Version to retain checkpoints and results:", output, flush=True)
    return result
