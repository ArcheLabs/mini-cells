"""Standard-library session supervisor, also embedded verbatim before Notebook checkout."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SESSION_SECONDS = 7200
SAVE_RESERVE_SECONDS = 600
TERMINATION_RESERVE_SECONDS = 10
DEADLINE_ENV = "PD001_SESSION_DEADLINE_MONOTONIC"


def session_deadline():
    now = time.monotonic()
    requested = float(os.environ.get(DEADLINE_ENV, now + SESSION_SECONDS))
    if not now < requested < float("inf"):
        raise TimeoutError("PD-001 session budget exhausted; preserve Output")
    deadline = min(requested, now + SESSION_SECONDS)
    os.environ[DEADLINE_ENV] = str(deadline)
    return deadline


def pause_due():
    value = os.environ.get(DEADLINE_ENV)
    return value is not None and time.monotonic() >= float(value) - SAVE_RESERVE_SECONDS


class SessionBudget:
    def __init__(self, status_path):
        self.deadline = session_deadline()
        self.status_path = Path(status_path)

    def record(self, status):
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "status": status, "wall_limit_seconds": SESSION_SECONDS,
            "remaining_seconds": max(0, self.deadline - time.monotonic()),
            "scientific_decision": False,
            "note": "Retain Kaggle Output; only completed checkpoints/evidence can be restored."
        }, indent=2) + "\n")
        temporary.replace(self.status_path)

    def run(self, command, **kwargs):
        remaining = self.deadline - time.monotonic() - TERMINATION_RESERVE_SECONDS
        if remaining <= 0:
            self.record("SESSION_TIME_LIMIT")
            raise TimeoutError("PD-001 two-hour session limit; preserve Output")
        # One group contains the launcher, benchmark workers and training workers.
        # Killing only the immediate launcher can leave CUDA work running.
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        try:
            code = process.wait(timeout=remaining)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # Group members may outlive an already exited leader.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
            self.record("SESSION_TIME_LIMIT" if isinstance(error, subprocess.TimeoutExpired) else "INTERRUPTED")
            raise
        if code:
            self.record("FAILED")
            raise subprocess.CalledProcessError(code, command)
        return code
