"""Two initial CLMs; next free device picks up the MoE. No profiling overlap."""

from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from pathlib import Path
import subprocess
import sys
import os
from .io import write_json
from .config import BASE, ARMS, authorize


def matrix(c, config_path, assets, output, devices, arms=None, *, resume=False, max_session_seconds=None):
    arms = list(arms or ("CLM-S16", "CLM-S8", "M0-TINY"))
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("three FAST arms only")
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("distinct worker devices required")
    for device in devices:
        authorize(c, device, "train")
    if any(str(d).startswith("cuda") for d in devices):
        from .benchmark import gpu_check

        gpu_check(devices)
    devices = devices[: len(arms)]
    pending = Queue()
    for arm in arms[len(devices) :]:
        pending.put(arm)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def work(index):
        results = []
        arm = arms[index]
        while True:
            folder = output / "runs" / arm
            folder.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(BASE / "run_fast001.py"),
                "--mode",
                "train",
                "--arm",
                arm,
                "--config",
                str(config_path),
                "--assets",
                str(assets),
                "--output",
                str(folder),
                "--device",
                devices[index],
            ]
            if resume and (folder / "checkpoint.pt").exists():
                cmd.append("--resume")
            if max_session_seconds is not None:
                cmd += ["--max-session-seconds", str(max_session_seconds)]
            try:
                print(f"[Train] {arm} on {devices[index]}", flush=True)
                with (output / (arm + "-worker.log")).open("a") as log:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env={
                            **os.environ,
                            "PYTHONUNBUFFERED": "1",
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                        },
                    )
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print("[" + arm + "] " + line.rstrip(), flush=True)
                    code = process.wait()
                state = (
                    "FAILED"
                    if code
                    else "PAUSED_RESUMABLE"
                    if (folder / "pause.json").exists()
                    else "DEVELOPMENT_COMPLETED"
                )
                results.append({"arm": arm, "device": devices[index], "status": state, "exit_code": code})
                if state != "DEVELOPMENT_COMPLETED":
                    break
            except Exception as error:
                results.append(
                    {"arm": arm, "device": devices[index], "status": "FAILED", "error": str(error)}
                )
                break
            try:
                arm = pending.get_nowait()
            except Empty:
                break
        return results

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        rows = [r for group in pool.map(work, range(len(devices))) for r in group]
    result = {
        "status": "DEVELOPMENT_COMPLETED"
        if len(rows) == len(arms) and all(r["status"] == "DEVELOPMENT_COMPLETED" for r in rows)
        else "INCOMPLETE",
        "workers": rows,
        "workers_joined": True,
        "scientific_decision": False,
        "unfinished_arms": [
            a for a in arms if not any(r["arm"] == a and r["status"] == "DEVELOPMENT_COMPLETED" for r in rows)
        ],
    }
    write_json(output / "schedule.json", result)
    return result
