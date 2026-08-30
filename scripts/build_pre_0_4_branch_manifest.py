#!/usr/bin/env python3
"""Build the pre-0.4 remote-branch audit from Git history.

This is an archival/indexing utility. It never changes refs or scientific data.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research/archive/branch-manifest-pre-0.4.json"


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=check
    )
    return result.stdout.strip()


def classify(branch: str, merged: bool) -> str:
    if merged:
        return "merged"
    if branch.startswith("kaggle/"):
        return "results-preserved"
    if branch.startswith("fix/"):
        return "hotfix-superseded"
    if branch.startswith("release/"):
        return "release-history"
    if branch.startswith(("codex/", "research/", "re-validation-")):
        return "superseded"
    return "abandoned"


def main() -> None:
    main_ref = "origin/main"
    refs = git(
        "for-each-ref",
        "--format=%(refname:short) %(objectname)",
        "refs/remotes/origin/",
    ).splitlines()
    entries = []
    for line in refs:
        ref, head = line.split()
        if ref in {"origin/HEAD", main_ref}:
            continue
        branch = ref[len("origin/") :] if ref.startswith("origin/") else ref
        base = git("merge-base", main_ref, ref)
        merged = (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ref, main_ref], cwd=ROOT
            ).returncode
            == 0
        )
        unique = git("rev-list", "--count", f"{main_ref}..{ref}")
        changed = git("diff", "--name-only", f"{base}..{ref}").splitlines()
        artifact_files = [p for p in changed if p.startswith("artifacts/experiments/")]
        missing_artifacts = [p for p in artifact_files if not (ROOT / p).is_file()]
        canonical_preserved = not missing_artifacts
        classification = classify(branch, merged)
        notes = [f"classification: {classification}"]
        if artifact_files:
            notes.append(
                f"{len(artifact_files)} experiment artifact file(s) changed since fork; "
                f"{len(missing_artifacts)} absent from main"
            )
        elif branch.startswith("kaggle/"):
            notes.append("result branch has no experiment-artifact paths absent from main")
        if missing_artifacts:
            classification = "needs-migration"
            notes.append("migration required: " + ", ".join(missing_artifacts[:8]))
        entries.append(
            {
                "branch": branch,
                "head_sha": head,
                "merge_base_main": base,
                "fully_merged_into_main": merged,
                "unique_commits": int(unique),
                "open_pr": None,
                "canonical_artifacts_preserved_in_main": canonical_preserved,
                "classification": classification,
                "disposition": "retain" if classification == "needs-migration" else "delete",
                "notes": "; ".join(notes),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} branch records to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
