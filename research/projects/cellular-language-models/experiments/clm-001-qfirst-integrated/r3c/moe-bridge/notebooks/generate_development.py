"""Reuse the released Pilot launcher template for one-launch development."""

from pathlib import Path
import json
import sys

sha = sys.argv[1]
if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
    raise ValueError("full source SHA required")
folder = Path(__file__).resolve().parent
nb = json.loads((folder / "kaggle-pilot.ipynb").read_text())
code = "".join(nb["cells"][1]["source"])
old_sha = next(line for line in code.splitlines() if line.startswith("SOURCE_SHA="))
code = code.replace(old_sha, "SOURCE_SHA=" + repr(sha))
code = code.replace("r3c-pilot-code-", "r3c-development-code-")
code = code.replace("r3c-pilot-output-", "r3c-development-output-")
code = code.replace("[R3c Pilot]", "[R3c Development]")
code = code.replace("'--mode','pilot'", "'--mode','development'")
code = code.replace(
    "TOKENIZERS_PARALLELISM='false'",
    "TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'",
)
code = code.replace(
    "Automatic short pilot: max 40 optimizer updates / 20,480 Train tokens. No formal 500-step training.",
    "Automatic exploratory development: 500 steps x 3 seeds; A/B/C/D on up to two GPUs; E aliases C. Confirmation sealed.",
)
nb["cells"][0]["source"] = [
    "# R3c MoE Bridge — one-launch development\n",
    "Use the original bingjiangli/minicells Notebook. Select T4 x2, enable Internet and existing HF_TOKEN / GITHUB_TOKEN. Run All once. Checks automatically continue to development; no manual fields or test suite.\n",
    "500 steps per model x3 paired seeds; physical A/B/C/D, E is explicitly derived from C at lambda=0. Reuses verified Pilot state when compatible. Checkpoints stay in Output; Save Version must retain it. No Confirmation or scientific PASS.\n",
]
nb["cells"][1]["source"] = code.splitlines(True)
(folder / "kaggle-development.ipynb").write_text(json.dumps(nb, indent=2) + "\n")
