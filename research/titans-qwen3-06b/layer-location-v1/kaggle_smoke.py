"""Kaggle Python script. Enable Internet and select a GPU accelerator."""
from pathlib import Path
import json
import os
import shutil
import uuid
import subprocess
import sys

SOURCE = "https://github.com/ArcheLabs/mini-cells.git"
BRANCH = "codex/titans-layer-location-v1"
UPSTREAM = "https://github.com/sarahpannn/titans-pytorch.git"
RUN_EPISODE_TIMING = False
REVISION = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    root = Path("/kaggle/working")
    repo, upstream = root / "layer-location-source", root / "layer-location-titans"
    run("git", "clone", "--depth", "1", "--branch", BRANCH, SOURCE, str(repo))
    if RUN_EPISODE_TIMING:
        sys.path.insert(0, str(repo))
        from scripts.research.publish_experiment_results import load_github_token
        load_github_token("GITHUB_TOKEN")
    upstream.mkdir()
    run("git", "init", str(upstream))
    run("git", "fetch", "--depth", "1", UPSTREAM, REVISION, cwd=upstream)
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
    directory = repo / "research/titans-qwen3-06b/layer-location-v1"
    run(sys.executable, "-m", "unittest", "-v", cwd=directory)
    if not RUN_EPISODE_TIMING:
        output = root / ("layer-location-preflight-" + uuid.uuid4().hex[:12])
        run(sys.executable, "memory_preflight.py", str(upstream), str(output), cwd=directory)
        return
    if RUN_EPISODE_TIMING:
        output = root / ("layer-location-timing-" + uuid.uuid4().hex[:12])
        run(sys.executable, "train.py", str(upstream), str(output),
            "--timing-only", cwd=directory)
        publish(repo, output)


def publish(repo, output):
    sys.path.insert(0, str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, prepare_artifacts, push_results,
    )

    format_id = "minicells.titans-layer-location.timing.v1"
    (output / "decision.json").write_text(json.dumps({
        "format": format_id, "status": "TIMING_COMPLETE", "optimizer_updates": 0,
        "run_id": output.name,
    }, indent=2))
    source = f"results/{output.name}"
    branch = f"kaggle/{output.name}"
    shutil.copytree(output, repo / source)
    spec = ExperimentSpec(
        source, f"artifacts/experiments/titans-layer-location/{output.name}",
        ("decision.json", "data.json", "layer_3/warmup.json", "layer_22/warmup.json"),
        branch, format_id,
    )
    folder = prepare_artifacts(repo, "titans-layer-location", spec,
                               os.getenv("KAGGLE_SCRIPT_VERSION_ID"))
    push_results(repo, folder, "titans-layer-location", branch, "GITHUB_TOKEN", "origin/main")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    receipt = {"status": "REMOTE_SHA_VERIFIED", "branch": branch, "commit": sha}
    (output / "publication.json").write_text(json.dumps(receipt, indent=2))
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
