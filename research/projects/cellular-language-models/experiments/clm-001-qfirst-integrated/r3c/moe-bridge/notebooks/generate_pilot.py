"""Build the released short-pilot Notebook with an implementation commit pin."""

from pathlib import Path
import json
import sys

sha = sys.argv[1]
if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
    raise ValueError("full source SHA required")
paths = [
    "/pyproject.toml",
    "/src/",
    "/shared/python/",
    "/research/projects/cellular-language-models/engineering/kaggle-continuation/",
    "/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/",
    "/research/projects/cellular-language-models/experiments/006-path-first-clm/",
    "/research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/",
    "/research/projects/continual-learning/context/04-continual-learning/scripts/research/",
    "/research/projects/cellular-language-models/context/08-hybrid-clm/scripts/research/",
]
entry = "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/run.py"
asset_sha = json.loads((Path(__file__).resolve().parents[1] / "config/pilot.json").read_text())[
    "input"
]["assets_manifest_sha256"]
code = f"""from pathlib import Path
import importlib.util, os, subprocess, sys
SOURCE_SHA={sha!r}
ROOT=Path('/kaggle/working/r3c-pilot-code-'+SOURCE_SHA[:12])
OUTPUT=Path('/kaggle/working/r3c-pilot-output-'+SOURCE_SHA[:12])
ASSETS=Path('/kaggle/working/r3c-pilot-assets-{asset_sha[:12]}')
def run(*args, **kwargs):
    print('[R3c Pilot]', *map(str,args), flush=True)
    if str(args[0])=='git':kwargs.setdefault('timeout',45)
    return subprocess.run(list(map(str,args)),check=True,**kwargs)
# Use installed PyTorch; only add the existing storage dependency if absent.
if importlib.util.find_spec('huggingface_hub') is None:
    run(sys.executable,'-m','pip','install','--quiet','huggingface_hub>=0.23,<2')
ROOT.mkdir(parents=True,exist_ok=True)
run('git','init','--quiet',ROOT)
remote=subprocess.run(['git','-C',str(ROOT),'remote','get-url','origin'],capture_output=True,text=True)
if remote.returncode:
    run('git','-C',ROOT,'remote','add','origin','https://github.com/ArcheLabs/mini-cells.git')
run('git','-C',ROOT,'config','remote.origin.promisor','true')
run('git','-C',ROOT,'config','remote.origin.partialclonefilter','blob:none')
run('git','-C',ROOT,'sparse-checkout','init','--no-cone')
run('git','-C',ROOT,'sparse-checkout','set','--no-cone',*{paths!r})
head=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
if head.returncode or head.stdout.strip()!=SOURCE_SHA:
    run('git','-C',ROOT,'fetch','--depth=1','--filter=blob:none','origin',SOURCE_SHA)
    run('git','-C',ROOT,'checkout','--detach','FETCH_HEAD')
actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
if actual!=SOURCE_SHA:raise RuntimeError('Source identity mismatch')
env=dict(os.environ,HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_ETAG_TIMEOUT='10',HF_HUB_DOWNLOAD_TIMEOUT='30',TOKENIZERS_PARALLELISM='false')
print('Automatic short pilot: max 40 optimizer updates / 20,480 Train tokens. No formal 500-step training.',flush=True)
try:
    run(sys.executable,ROOT/{entry!r},'--mode','pilot','--output',OUTPUT,'--assets',ASSETS,cwd=ROOT,env=env)
finally:
    print('Retain this Kaggle Output, including local Checkpoints:',OUTPUT,flush=True)
"""
nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# R3c MoE Bridge — automatic short GPU Pilot\n",
                "Use the original bingjiangli/minicells Notebook. Enable Internet, select a GPU, and keep existing HF_TOKEN / GITHUB_TOKEN secrets enabled. Run All once. Checks, fixed assets, bounded joint training and HF/GitHub result return are automatic.\n",
                "At most 40 updates / 20,480 Train tokens; an invalid balance candidate set ends with NO_GO after 15 updates and saves the diagnostic result. No manual authorization/hash field or full test suite. No formal 500-Step training or Confirmation. Checkpoint copies stay in Kaggle Output even if remote upload fails.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(True),
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path(__file__).with_name("kaggle-pilot.ipynb").write_text(json.dumps(nb, indent=2) + "\n")
