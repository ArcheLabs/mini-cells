"""Reuse the R3e fixed-source notebook shell; the new controller enforces all four stages."""

from pathlib import Path
import sys
import json

BASE = Path(__file__).resolve().parents[1]
sha = sys.argv[1]
if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
    raise ValueError("full source SHA required")
nb = json.loads((BASE.parent / "r3e/notebooks/kaggle-development.ipynb").read_text())
code = "".join(nb["cells"][1]["source"])
old = next(x for x in code.splitlines() if x.startswith("SOURCE_SHA="))
code = (
    code.replace(old, "SOURCE_SHA=" + repr(sha))
    .replace("r3e-code-", "r3f-code-")
    .replace("r3e-output-", "r3f-output-")
    .replace("[R3e]", "[R3f]")
)
code = code.replace("run_r3e.py", "run_r3f.py").replace(
    "r3e/config/frozen.json", "r3f-connection-admission/config/frozen.json"
)
code = code.replace(
    "R3e: A-F/A-T/A-C and B-F/B-R/B-C, 3 seeds, 500 Cell updates each; automatic checks then all arms. Confirmation SEALED.",
    "R3f: read-only Calibration diagnostics -> one-seed A-C/B-C admission trial -> full matrix ONLY if real admission gate passes -> terminal Development. No threshold relaxation. Confirmation SEALED.",
)
nb["cells"][0]["source"] = [
    "# R3f — Connection Admission Repair\n",
    "Original bingjiangli/minicells. T4 x2, Internet, existing HF_TOKEN/GITHUB_TOKEN. Run All once. Historical R3e code/results remain unchanged.\n",
    "Automatically runs read-only diagnostics and the single-seed admission trial. If no effective candidate enters K-step adaptation, STOP with diagnostics; never relax thresholds. Full matrix continues automatically only after the real gate passes. No unit test suite in Kaggle; Development is final-only. Save Version retains Output.\n",
]
nb["cells"][1]["source"] = code.splitlines(True)
(BASE / "notebooks/kaggle-development.ipynb").write_text(json.dumps(nb, indent=2) + "\n")
