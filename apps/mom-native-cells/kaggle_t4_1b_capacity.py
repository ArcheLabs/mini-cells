"""Publish the model audit, or run the capacity suite if the model gate is satisfied."""
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
BRANCH = "codex/minicells-t4-1b-capacity-001"
APP = "apps/mom-native-cells"
EXPERIMENT = "minicells-t4-1b-capacity-001"


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


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
            manifest = repo / APP / "capacity_target.json"
            target = json.loads(manifest.read_text(encoding="utf-8"))
            # Current release fails closed with standard library only, without a CUDA install.
            if not validate_target(target):
                from importlib.metadata import version
                constraints = Path(tmp) / "constraints.txt"
                constraints.write_text(f"torch=={version('torch')}\ntriton=={version('triton')}\n")
                run("git", "sparse-checkout", "add", "src", cwd=repo, timeout=120)
                run("git", "checkout", "--ignore-skip-worktree-bits", "HEAD", "--",
                    "pyproject.toml", "README.md", "LICENSE", cwd=repo, timeout=120)
                run(sys.executable, "-m", "pip", "install", "-c", str(constraints),
                    str(repo) + "[mom]", "nvidia-ml-py", "psutil", timeout=240)
            try:
                run(sys.executable, "-u", str(repo / APP / "run_t4_1b_capacity.py"),
                    "--output", str(output), timeout=7200)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
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
        if value["status"] != "FULL PASS":
            raise SystemExit(f"{value['status']}: see the published report")


if __name__ == "__main__":
    main()
