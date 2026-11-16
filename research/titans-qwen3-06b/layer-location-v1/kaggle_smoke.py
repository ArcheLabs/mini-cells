"""Kaggle Python script. Enable Internet and select a GPU accelerator."""
from pathlib import Path
import os
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
    upstream.mkdir()
    run("git", "init", str(upstream))
    run("git", "fetch", "--depth", "1", UPSTREAM, REVISION, cwd=upstream)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=upstream)
    # Titans pins Torch 2.9; Kaggle's optional media extensions are built
    # for its preinstalled Torch. This experiment only uses text.
    run(sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio")
    run(sys.executable, "-m", "pip", "install", "transformers==4.57.6", str(upstream))
    run(sys.executable, "-c",
        "import torch; from transformers import Qwen3ForCausalLM; "
        "assert torch.cuda.is_available(), 'Kaggle GPU unavailable'; "
        "print(torch.__version__, torch.cuda.get_device_name())")
    directory = repo / "research/titans-qwen3-06b/layer-location-v1"
    run(sys.executable, "-m", "unittest", "-v", cwd=directory)
    run(sys.executable, "smoke.py", str(upstream), cwd=directory)
    if RUN_EPISODE_TIMING:
        run(sys.executable, "train.py", str(upstream), str(root / "layer-location-timing"),
            "--timing-only", cwd=directory)


if __name__ == "__main__":
    main()
