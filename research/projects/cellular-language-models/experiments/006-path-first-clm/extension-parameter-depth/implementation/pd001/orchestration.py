"""Independent registered arm workers; no profiling during concurrent training."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import os
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
            if (folder / "checkpoint.pt").exists():
                command.append("--resume")
            folder.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Train] {arm} on {devices[worker]}", flush=True)
            with (folder.parent / (arm + "-worker.log")).open("a") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{arm}] {line.rstrip()}", flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
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
