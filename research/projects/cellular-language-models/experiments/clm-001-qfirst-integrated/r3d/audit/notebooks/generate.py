"""Reuse R3d's pinned single-launch shell; replace only the read-only entry."""

from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parents[1]
sha = sys.argv[1]
if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
    raise ValueError("full source SHA required")
nb = json.loads((BASE.parent / "notebooks/kaggle-development.ipynb").read_text())
code = "".join(nb["cells"][1]["source"])
old = next(line for line in code.splitlines() if line.startswith("SOURCE_SHA="))
code = (
    code.replace(old, "SOURCE_SHA=" + repr(sha))
    .replace("r3d-code-", "r3d-audit-code-")
    .replace("r3d-output-", "r3d-audit-output-")
    .replace("[R3d]", "[R3d-A]")
)
code = code.replace(
    "ROOT/'scripts/run_r3d.py'",
    "ROOT/'research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/audit/run.py'",
)
code = code.replace("r3d/config/frozen.json'", "r3d/audit/config/frozen.json'")
code = code.replace(
    "# Use installed PyTorch; only add the existing storage dependency if absent.",
    "# Use installed PyTorch; this Notebook never starts a test suite or a Trainer.",
)
code = code.replace(
    "ROOT.mkdir(parents=True,exist_ok=True)",
    "if importlib.util.find_spec('matplotlib') is None:\n    run(sys.executable,'-m','pip','install','--quiet','matplotlib')\nROOT.mkdir(parents=True,exist_ok=True)",
)
code = code.replace(
    "R3d automatic full exploratory experiment: 3 seeds; F0/L/M/F/P/J; 5,700 job steps plus bounded J policy updates. Confirmation sealed.",
    "R3d-A read-only mechanism audit: 18 checkpoints, 43,200 all-action evaluations; zero training and optimizers. Confirmation sealed.",
)
code = code.replace(
    "Retain this Kaggle Output, including local Checkpoints:",
    "Retain this Kaggle Output with private audit scores and read-only input cache:",
)
code = code.replace("'/scripts/run_r3d.py', ", "")
nb["cells"][0]["source"] = [
    "# R3d-A — Read-Only Mechanism Audit\n",
    "Use original bingjiangli/minicells. Internet, preferably T4 x2. Existing HF_TOKEN needs READ access only; GITHUB_TOKEN is optional aggregate return. Run All once. No training, optimizer, Smoke, Pilot or manual intermediate steps.\n",
    "Prefers existing R3d Output, otherwise verifies immutable private HF inputs. Confirmation stays sealed. Save Version retains private analysis outputs.\n",
]
nb["cells"][1]["source"] = code.splitlines(True)
(BASE / "notebooks/kaggle-audit.ipynb").write_text(json.dumps(nb, indent=2) + "\n")
