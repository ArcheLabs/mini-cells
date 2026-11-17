"""Kaggle Python script. Enable Internet and select a GPU accelerator."""
from pathlib import Path
import json
import os
import shutil
import uuid
import subprocess
import sys

SOURCE = "https://github.com/ArcheLabs/mini-cells.git"
BRANCH = "codex/titans-p1-1-content-verification"
SMOKE = False
SCRIPT_VERSION_ID = ""
UPSTREAM = "https://github.com/sarahpannn/titans-pytorch.git"
REVISION = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"


def run(*args, cwd=None, timeout=None):
    subprocess.run(args, cwd=cwd, check=True, timeout=timeout)


def main():
    script_version = os.getenv("KAGGLE_SCRIPT_VERSION_ID") or SCRIPT_VERSION_ID
    if not (script_version.isdecimal() and int(script_version) > 0):
        script_version = None
        print("KAGGLE_SCRIPT_VERSION_ID unavailable; recording exact Git source commit",
              flush=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
    root = Path("/kaggle/working")
    repo, upstream = root / "layer-location-source", root / "layer-location-titans"
    print("SOURCE_FETCH_START", flush=True)
    run("git", "clone", "--progress", "--depth", "1", "--single-branch",
        "--filter=blob:none", "--sparse", "--branch", BRANCH, SOURCE, str(repo),
        timeout=180)
    run("git", "sparse-checkout", "set", "scripts/research",
        "research/titans-qwen3-06b/p1-meta",
        "research/titans-qwen3-06b/p1-1",
        cwd=repo, timeout=180)
    print("SOURCE_FETCH_COMPLETE", flush=True)
    sys.path.insert(0, str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results,
    )
    load_github_token("GITHUB_TOKEN")
    upstream.mkdir()
    run("git", "init", str(upstream))
    run("git", "fetch", "--depth", "1", UPSTREAM, REVISION, cwd=upstream, timeout=180)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=upstream)
    # Use the pinned checkout directly. Preserve Kaggle's matched Torch stack.
    import tomllib
    from importlib.metadata import version

    dependencies = tomllib.loads((upstream / "pyproject.toml").read_text())["project"]["dependencies"]
    constraints = root / "layer-location-constraints.txt"
    constraints.write_text("torch==" + version("torch") + "\n")
    run(sys.executable, "-m", "pip", "install", "-c", str(constraints),
        "transformers==4.57.6", "datasets==4.4.1", "assoc-scan==0.0.6", "accelerated-scan==0.3.1",
        *(item for item in dependencies if not item.startswith("torch>=")))
    run(sys.executable, "-c",
        "import torch; from transformers import Qwen3ForCausalLM; "
        "assert torch.cuda.is_available(), 'Kaggle GPU unavailable'; "
        "print(torch.__version__, torch.cuda.get_device_name())")
    directory = repo / "research/titans-qwen3-06b/p1-1"
    run(sys.executable, "-m", "unittest", "-v", cwd=directory)
    output = root / ("titans-p1-1-" + uuid.uuid4().hex[:12])
    log = root / f"{output.name}.log"
    command = [sys.executable, "run.py", str(upstream), str(output)]
    if SMOKE:
        command.append("--smoke")
    with log.open("w") as stream, subprocess.Popen(
        command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        for line in process.stdout:
            stream.write(line)
            print(line, end="", flush=True)
        completed = subprocess.CompletedProcess(command, process.wait())
    output.mkdir(exist_ok=True)
    shutil.move(log, output / "run.log")
    kind = "minicells.titans-p1-1.v1"
    decision_path = output / "decision.json"
    decision = json.loads(decision_path.read_text()) if completed.returncode == 0 else {
        "format": kind, "status": "RUN_FAILED", "engineering_status": "FAIL",
        "scientific_status": "NOT_EVALUATED", "returncode": completed.returncode,
    }
    decision["run_id"] = output.name
    decision["kaggle_script_version_id"] = script_version
    decision["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    decision_path.write_text(json.dumps(decision, indent=2))
    source = f"results/{output.name}"
    shutil.copytree(output, repo / source)
    branch = f"kaggle/{output.name}"
    spec = ExperimentSpec(source, f"artifacts/experiments/{output.name}",
                          tuple(sorted(file.relative_to(output).as_posix()
                                       for file in output.rglob("*") if file.is_file())),
                          branch, kind)
    run("git", "sparse-checkout", "add", spec.artifact_dir, cwd=repo, timeout=180)
    folder = prepare_artifacts(repo, "titans-p1-1", spec,
                               script_version or None)
    push_results(repo, folder, "titans-p1-1", branch,
                 "GITHUB_TOKEN", "origin/main")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    print({"status": "REMOTE_SHA_VERIFIED", "branch": branch, "commit": sha}, flush=True)
    completed.check_returncode()


if __name__ == "__main__":
    main()
