"""Skip historical model suites for an isolated R3d-A change; keep normal/manual runs."""

from pathlib import Path
import os
import subprocess

AUDIT = (
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/audit/"
)
INFRA = {
    "shared/scripts/ci/read_only_audit_scope.py",
    ".github/workflows/r3d-long-horizon-credit-cpu.yml",
    ".github/workflows/clm001-qfirst-cpu.yml",
    ".github/workflows/hybrid-clm-ci.yml",
    ".github/workflows/research-notebook-ci.yml",
    ".github/workflows/r3d-readonly-audit.yml",
}


def historical_needed(paths):
    paths = list(paths)
    return (
        not paths
        or not any(p.startswith(AUDIT) for p in paths)
        or any(not p.startswith(AUDIT) and p not in INFRA for p in paths)
    )


def main():
    base = os.environ.get("BASE_SHA", "")
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    needed = True
    if base and set(base) != {"0"} and event != "workflow_dispatch":
        files = subprocess.check_output(
            ["git", "diff", "--name-only", base, "HEAD"], text=True
        ).splitlines()
        needed = historical_needed(files)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a") as stream:
            stream.write("run_model_tests=" + str(needed).lower() + "\n")
    print(
        "Historical model suites required: "
        + str(needed)
        + "; read-only audit has its own focused CPU checks."
    )


if __name__ == "__main__":
    main()
