"""Reuse the existing R3d launch shell, replacing only R3e identity/entry and banner."""

from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parents[1]
sha = sys.argv[1]
if len(sha) != 40 or any(v not in "0123456789abcdef" for v in sha):
    raise ValueError("full source SHA required")
nb = json.loads((BASE.parent / "r3d/notebooks/kaggle-development.ipynb").read_text())
code = "".join(nb["cells"][1]["source"])
old = next(line for line in code.splitlines() if line.startswith("SOURCE_SHA="))
code = (
    code.replace(old, "SOURCE_SHA=" + repr(sha))
    .replace("r3d-code-", "r3e-code-")
    .replace("r3d-output-", "r3e-output-")
    .replace("[R3d]", "[R3e]")
)
code = code.replace("/scripts/run_r3d.py", "/scripts/run_r3e.py").replace(
    "ROOT/'scripts/run_r3d.py'", "ROOT/'scripts/run_r3e.py'"
)
code = code.replace("r3d/config/frozen.json", "r3e/config/frozen.json")
code = code.replace(
    "R3d automatic full exploratory experiment: 3 seeds; F0/L/M/F/P/J; 5,700 job steps plus bounded J policy updates. Confirmation sealed.",
    "R3e: A-F/A-T/A-C and B-F/B-R/B-C, 3 seeds, 500 Cell updates each; automatic checks then all arms. Confirmation SEALED.",
)
nb["cells"][0]["source"] = [
    "# MiniCells R3e — Stable Co-Adaptive Learning & Sequential Cell Graph\n",
    "Original bingjiangli/minicells; T4 x2, Internet, existing HF_TOKEN (private repo write) and GITHUB_TOKEN. Run All once. No separate tests, Pilot or approval.\n",
    "Initializes each arm from its R3c D500, never from R3d Connection. Six arms x three independent seeds. Saves full state every 50 Cell updates; HF transfers serialized and deduplicated. Save Version retains Output. Calibration is selection data; final Development is exploratory; Confirmation SEALED.\n",
]
nb["cells"][1]["source"] = code.splitlines(True)
(BASE / "notebooks/kaggle-development.ipynb").write_text(json.dumps(nb, indent=2) + "\n")
