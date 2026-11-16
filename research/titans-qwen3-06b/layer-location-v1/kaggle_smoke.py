"""Kaggle Python script. Enable Internet and select a GPU accelerator."""
from pathlib import Path
import os
import subprocess
import sys

SOURCE = "https://github.com/ArcheLabs/mini-cells.git"
BRANCH = "codex/titans-layer-location-v1"
UPSTREAM = "https://github.com/sarahpannn/titans-pytorch.git"
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
    run(sys.executable, "-m", "pip", "install", "transformers==4.57.6", str(upstream))
    directory = repo / "research/titans-qwen3-06b/layer-location-v1"
    run(sys.executable, "-m", "unittest", "-v", cwd=directory)
    run(sys.executable, "smoke.py", str(upstream), cwd=directory)


if __name__ == "__main__":
    main()
