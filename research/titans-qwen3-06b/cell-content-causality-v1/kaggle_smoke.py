"""Kaggle Python script. Enable Internet and select a GPU accelerator."""
from pathlib import Path
import json
import os
import shutil
import uuid
import subprocess
import sys

SOURCE = "https://github.com/ArcheLabs/mini-cells.git"
BRANCH = "codex/titans-cell-content-causality-v1"
UPSTREAM = "https://github.com/sarahpannn/titans-pytorch.git"
REVISION = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"


def run(*args, cwd=None, timeout=None):
    subprocess.run(args, cwd=cwd, check=True, timeout=timeout)


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    root = Path("/kaggle/working")
    repo, upstream = root / "layer-location-source", root / "layer-location-titans"
    print("SOURCE_FETCH_START", flush=True)
    run("git", "clone", "--progress", "--depth", "1", "--single-branch",
        "--filter=blob:none", "--sparse", "--branch", BRANCH, SOURCE, str(repo),
        timeout=180)
    run("git", "sparse-checkout", "set", "scripts/research",
        "research/titans-qwen3-06b/cell-content-causality-v1",
        "research/titans-qwen3-06b/layer-location-v1", cwd=repo, timeout=180)
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
        "transformers==4.57.6", "assoc-scan==0.0.6", "accelerated-scan==0.3.1",
        *(item for item in dependencies if not item.startswith("torch>=")))
    run(sys.executable, "-c",
        "import torch; from transformers import Qwen3ForCausalLM; "
        "assert torch.cuda.is_available(), 'Kaggle GPU unavailable'; "
        "print(torch.__version__, torch.cuda.get_device_name())")
    directory = repo / "research/titans-qwen3-06b/cell-content-causality-v1"
    run(sys.executable, "-m", "unittest", "-v", cwd=directory)
    output = root / ("cell-content-causality-" + uuid.uuid4().hex[:12])
    log = root / f"{output.name}.log"
    command = [sys.executable, "smoke.py", str(upstream), str(output)]
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
    kind = "minicells.titans-cell-content-causality.v1"
    decision = {"format": kind, "run_id": output.name,
                "status": "RUN_COMPLETE" if completed.returncode == 0 else "RUN_FAILED",
                "returncode": completed.returncode}
    (output / "decision.json").write_text(json.dumps(decision, indent=2))
    source = f"results/{output.name}"
    shutil.copytree(output, repo / source)
    branch = f"kaggle/{output.name}"
    spec = ExperimentSpec(source, f"artifacts/experiments/{output.name}",
                          tuple(sorted(file.name for file in output.iterdir())),
                          branch, kind)
    folder = prepare_artifacts(repo, "titans-cell-content-causality", spec,
                               os.getenv("KAGGLE_SCRIPT_VERSION_ID"))
    push_results(repo, folder, "titans-cell-content-causality", branch,
                 "GITHUB_TOKEN", "origin/main")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    print({"status": "REMOTE_SHA_VERIFIED", "branch": branch, "commit": sha}, flush=True)
    completed.check_returncode()


if __name__ == "__main__":
    main()
