"""Generate CPU-only one-launch Notebook; source pin supplied by release engineering."""

from pathlib import Path
import json
import sys

sha = sys.argv[1]
if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
    raise ValueError("source SHA required")
base = "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge"
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
code = f"""from pathlib import Path
import os, subprocess, sys
SOURCE_SHA={sha!r}
ROOT=Path('/kaggle/working/r3c-bridge-code-'+SOURCE_SHA[:12])
OUTPUT=Path('/kaggle/working/r3c-bridge-cpu-'+SOURCE_SHA[:12])
def run(*args, **kwargs):
    print('[R3c CPU]', *map(str,args), flush=True)
    return subprocess.run(list(map(str,args)),check=True,**kwargs)
ROOT.mkdir(parents=True, exist_ok=True)
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
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
run(sys.executable,ROOT/{base!r}/'run.py','--mode','engineering','--output',OUTPUT,cwd=ROOT,env=env)
print('CPU synthetic engineering complete. Save Version retains:',OUTPUT)
print('No GPU, real assets, or Confirmation were used. No additional Notebook test suite.')
"""
nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# R3c MoE Bridge — CPU engineering only\n",
                "Run All once with accelerator **None**. This release is not a GPU training release. Uses installed PyTorch and NumPy; no tokens or secret setup required. Saves local synthetic checkpoint and aggregate reports. Public code submission/CI are separate from Kaggle Output persistence.\n",
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
Path(__file__).with_name("kaggle-smoke.ipynb").write_text(json.dumps(nb, indent=2) + "\n")
# No fake development notebook is shipped: the real GPU release is pending authorization and measurements.
