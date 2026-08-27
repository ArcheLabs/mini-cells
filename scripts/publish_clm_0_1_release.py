from __future__ import annotations

import argparse

from publish_experiment_results import (
    DEFAULT_SECRET_NAME,
    ExperimentSpec,
    prepare_artifacts,
    push_results,
    repo_root,
)


SPEC = ExperimentSpec(
    source_dir="results/clm-0.1-release",
    artifact_dir="artifacts/releases/clm-0.1",
    files=(
        "decision.json",
        "conditionality-002-decision.json",
        "benchmark.json",
        "runtime.json",
        "model.pt",
        "tokenizer.json",
        "config.json",
        "MODEL_CARD.md",
        "r0-release-worker.json",
        "r1-release-worker.json",
        "r2-release-worker.json",
    ),
    branch="release/clm-0.1-artifacts",
    expected_format="minicells.clm-0.1.release.v1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish MiniCells CLM-0.1 release artifacts.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    root = repo_root()
    destination = prepare_artifacts(
        root, "clm-0.1", SPEC, args.kaggle_script_version_id
    )
    print(f"Prepared CLM-0.1 artifacts: {destination.relative_to(root)}")
    if args.push:
        push_results(
            root,
            destination,
            "clm-0.1",
            args.branch or SPEC.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Review the release bundle and re-run with --push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
