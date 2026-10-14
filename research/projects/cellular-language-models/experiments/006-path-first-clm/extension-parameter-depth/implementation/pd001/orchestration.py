"""Independent registered arm workers; no profiling during concurrent training."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import os
import sys
import json
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
            paused = (folder / "pause.json").exists() and not (folder / "metrics.json").exists()
            if paused:
                marker = json.loads((folder / "pause.json").read_text())
                if marker.get("status") != "PAUSED_RESUMABLE":
                    raise RuntimeError("invalid pause evidence")
            results.append(
                {
                    "arm": arm,
                    "device": devices[worker],
                    "run": str(folder),
                    "status": "PAUSED_RESUMABLE" if paused else "DEVELOPMENT_COMPLETED",
                }
            )
            if paused:
                break
        return results

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        runs = [r for group in pool.map(queue, range(len(devices))) for r in group]
    # Return only after all training processes exit; caller's separate profile mode is serial.
    return {
        "runs": runs,
        "workers_finished": len(runs) == len(arms) and all(r["status"] != "PAUSED_RESUMABLE" for r in runs),
        "status": "PAUSED_RESUMABLE"
        if any(r["status"] == "PAUSED_RESUMABLE" for r in runs)
        else "TRAINING_WORKERS_COMPLETED",
        "profiling_phase": "separate serial registered invocation",
        "scientific_decision": False,
    }
