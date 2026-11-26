"""Run and publish Native MoM 1.3B capacity after the T4 preflight passes."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

SOURCE = "https://github.com/ArcheLabs/mini-cells.git"
BRANCH = "codex/native-mom-13b-checkpoint-audit"
APP = "apps/mom-native-cells"
EXPERIMENT = "minicells-t4-1b-capacity-001"
RUN_STAGE = "all"


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def t4_uuid():
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,compute_cap", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )
    for line in devices.splitlines():
        uuid, name, capability = (item.strip() for item in line.split(","))
        if (name == "Tesla T4" and capability == "7.5"
                and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid)):
            return uuid
    return ""


def main():
    with tempfile.TemporaryDirectory(prefix="minicells-capacity-") as tmp:
        repo = Path(tmp) / "source"
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "--branch", BRANCH, SOURCE, str(repo), timeout=180)
        run("git", "sparse-checkout", "set", "scripts/research", APP, cwd=repo, timeout=120)
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / APP))
        from run_t4_1b_capacity import DECISION_FORMAT, validate_target, write_decision
        from scripts.research.publish_experiment_results import (
            ExperimentSpec, load_github_token, prepare_artifacts, push_results,
        )

        load_github_token("GITHUB_TOKEN")
        resume = os.getenv("MINICELLS_CAPACITY_PUBLISH_ID")
        if resume and not re.fullmatch(r"[0-9a-f]{12}", resume):
            raise ValueError("Invalid publication retry ID")
        run_id = resume or uuid.uuid4().hex[:12]
        name = f"{EXPERIMENT}-{run_id}"
        output = Path("/kaggle/working") / name
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=repo).strip()
        version_id = os.getenv("KAGGLE_SCRIPT_VERSION_ID")
        if not resume:
            try:
                manifest = repo / APP / "capacity_target.json"
                target = json.loads(manifest.read_text(encoding="utf-8"))
                reason = validate_target(target)
                if reason:
                    raise RuntimeError(reason)
                from importlib.metadata import version

                constraints = Path(tmp) / "constraints.txt"
                constraints.write_text(f"torch=={version('torch')}\ntriton=={version('triton')}\n")
                run("git", "sparse-checkout", "add", "src", cwd=repo, timeout=120)
                run("git", "checkout", "--ignore-skip-worktree-bits", "HEAD", "--",
                    "pyproject.toml", "README.md", "LICENSE", cwd=repo, timeout=120)
                run(sys.executable, "-m", "pip", "install", "-c", str(constraints),
                    str(repo) + "[mom]", "nvidia-ml-py", "psutil", timeout=240)
                try:
                    visible_gpu = t4_uuid()
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"T4 UUID detection failed: {exc}", flush=True)
                    visible_gpu = ""
                print(f"CUDA_VISIBLE_DEVICES={visible_gpu or '<no T4>'}", flush=True)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible_gpu}
                run(sys.executable, "-u", str(repo / APP / "run_t4_1b_capacity.py"),
                    "--output", str(output), "--stage", RUN_STAGE, timeout=7200, env=env)
            except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                output.mkdir(exist_ok=True)
                write_decision(output, "BLOCKED", f"Capacity runner did not finish: {exc}")
        receipt = output / "decision.json"
        value = json.loads(receipt.read_text(encoding="utf-8"))
        if resume:
            source_commit, version_id = value["source_commit"], value["script_version_id"]
            current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repo).strip()
            if source_commit != current:
                raise ValueError("Publishing this receipt requires the original source commit")
        else:
            value.update(source_commit=source_commit, script_version_id=version_id, run_id=run_id)
            receipt.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        relative = "results/" + name
        artifact = f"artifacts/experiments/{EXPERIMENT}/{run_id}"
        shutil.copytree(output, repo / relative, ignore=shutil.ignore_patterns("work", "*.tmp"))
        branch = "kaggle/" + name
        run("git", "sparse-checkout", "add", artifact, cwd=repo, timeout=120)
        files = tuple(sorted(p.relative_to(repo / relative).as_posix()
                             for p in (repo / relative).rglob("*") if p.is_file()))
        spec = ExperimentSpec(relative, artifact, files, branch, DECISION_FORMAT)
        bundle = prepare_artifacts(repo, name, spec, version_id)
        push_results(repo, bundle, name, branch, "GITHUB_TOKEN", "origin/main")
        print(f"Published: {branch} ({value['status']})", flush=True)
        if value["status"] != ("PASS" if RUN_STAGE == "preflight" else "FULL PASS"):
            raise SystemExit(f"{value['status']}: see the published report")


if __name__ == "__main__":
    main()
