"""Derive the launcher from the verified R3c one-launch Notebook."""

from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parents[1]
sha = sys.argv[1]
if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
    raise ValueError("full implementation SHA required")
nb = json.loads((BASE.parent / "r3c/moe-bridge/notebooks/kaggle-development.ipynb").read_text())
code = "".join(nb["cells"][1]["source"])
old = next(line for line in code.splitlines() if line.startswith("SOURCE_SHA="))
code = (
    code.replace(old, "SOURCE_SHA=" + repr(sha))
    .replace("r3c-development-code-", "r3d-code-")
    .replace("r3c-development-output-", "r3d-output-")
    .replace("[R3c Development]", "[R3d]")
)
code = code.replace("*['/pyproject.toml',", "*['/scripts/run_r3d.py', '/pyproject.toml',")
code = code.replace(
    "ROOT/'research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/run.py','--mode','development'",
    "ROOT/'scripts/run_r3d.py','--config',ROOT/'research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/config/frozen.json','--run-all'",
)
code = code.replace(
    "Automatic exploratory development: 500 steps x 3 seeds; A/B/C/D on up to two GPUs; E aliases C. Confirmation sealed.",
    "R3d automatic full exploratory experiment: 3 seeds; F0/L/M/F/P/J; 5,700 job steps plus bounded J policy updates. Confirmation sealed.",
)
nb["cells"][0]["source"] = [
    "# R3d — Long-Horizon Credit & Joint Learning\n",
    "Original bingjiangli/minicells; T4 x2, Internet, existing HF_TOKEN / GITHUB_TOKEN. Run All once. Checks automatically continue into all six arms and three seeds. No separate Pilot or test suite.\n",
    "Uses pinned R3c D Step500 checkpoints; no random substitute. L/M 200 steps, F/P/J 500 steps. Cell and Connection remain joint in J. Save Version must retain Output, even if remote storage fails. Confirmation stays sealed.\n",
]
nb["cells"][1]["source"] = code.splitlines(True)
(BASE / "notebooks/kaggle-development.ipynb").write_text(json.dumps(nb, indent=2) + "\n")
