"""Independent registered arm workers; no profiling during concurrent training."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from .configs import BASE, authorize


def matrix(c, p, arms, seed, role, source, assets, output, config_path, devices):
    for arm in arms:
        for device in devices:
            authorize(c, p, arm, seed, role, "train", source, device)
    if len(set(arms)) != len(arms) or not devices or len(set(devices)) != len(devices):
        raise ValueError("unique arm IDs and distinct worker devices required")
    output = Path(output)

    def queue(worker):
        results = []
        for arm in arms[worker :: len(devices)]:
            folder = output / arm
            command = [
                sys.executable,
                str(BASE / "run_pd001.py"),
                "--mode",
                "train",
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--role",
                role,
                "--source-commit",
                source,
                "--config",
                str(config_path),
                "--assets",
                str(assets),
                "--output",
                str(folder),
                "--device",
                devices[worker],
            ]
            folder.parent.mkdir(parents=True, exist_ok=True)
            with (folder.parent / (arm + "-worker.log")).open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            results.append({"arm": arm, "device": devices[worker], "run": str(folder)})
        return results

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        runs = [r for group in pool.map(queue, range(len(devices))) for r in group]
    # Return only after all training processes exit; caller's separate profile mode is serial.
    return {
        "runs": runs,
        "workers_finished": True,
        "profiling_phase": "separate serial registered invocation",
        "scientific_decision": False,
    }
