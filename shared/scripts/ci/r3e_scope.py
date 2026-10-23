"""Keep unrelated legacy suites out of an isolated R3e-only change."""

import os
from pathlib import Path
import subprocess

PREFIX = (
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3j/",
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3i/",
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3h/",
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3g/",
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-v2/",
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/",
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-connection-admission/",
)
ALLOWED = {
    "scripts/run_r3j.py",
    ".github/workflows/r3j-long-horizon.yml",
    "scripts/run_r3i.py",
    ".github/workflows/r3i-stable-coadaptation.yml",
    "scripts/run_r3h.py",
    ".github/workflows/r3h-routing-alignment.yml",
    "scripts/run_r3g.py",
    ".github/workflows/r3g-coadaptation-attribution.yml",
    "scripts/run_r3f_v2.py",
    ".github/workflows/r3f-v2-local-connection.yml",
    ".github/workflows/clm001-qfirst-cpu.yml",
    ".github/workflows/hybrid-clm-ci.yml",
    ".github/workflows/research-notebook-ci.yml",
    ".github/workflows/r3e-stable-coadaptation.yml",
    "shared/scripts/ci/r3e_scope.py",
    "scripts/run_r3e.py",
    "scripts/run_r3f.py",
    ".github/workflows/r3f-connection-admission.yml",
}


def required(paths):
    return (
        not paths
        or not any(p.startswith(PREFIX) for p in paths)
        or any(not p.startswith(PREFIX) and p not in ALLOWED for p in paths)
    )


if __name__ == "__main__":
    base = os.environ.get("BASE_SHA", "")
    need = True
    if base and set(base) != {"0"} and os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        files = subprocess.check_output(
            ["git", "diff", "--name-only", base, "HEAD"], text=True
        ).splitlines()
        need = required(files)
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as f:
            f.write("run_legacy=" + str(need).lower() + "\n")
    print("Legacy suites required:", need, "; R3e uses its focused CPU correctness checks.")
