# D1 GitHub result return

This publication amendment implements the human request to repeat the same D1 training with automatic GitHub result return. The scientific config, seed, original checkpoint, parameter mask, loss functions, sample plan and 200-logical-update budget are unchanged. Older completed outputs are not overwritten. No GPU training is started by the engineering implementation task.

## Reused repository components

The adapter follows the existing R3g/R3m publication discipline: a separate sparse metadata checkout follows current main, stages explicit result paths, and verifies the remote commit after push. Authentication is the exact `publish_core_validation_007._authenticated_git_env('GITHUB_TOKEN')` context returned by `minicells_r3c_moe_bridge.publication.helper()`. It reuses the established Kaggle Secret/GIT_ASKPASS flow. Source checkouts and workflow changes are not published as result commits.

The small Git adapter applies explicit network deadlines and low-speed detection to this existing authentication path: fetch/default commands at most30s, dry-run20s, final push45s, ref inspection15s. There are no hidden network retries or training runtime limits. A single automatic dry-run before GPU work catches absent credentials or inaccessible return paths.

## Artifact contract

The allowlist contains `RESULTS.json`, `RUN_MANIFEST.json`, `BUDGET_PROOF.json`, `SAMPLE_PLAN.json`, `A_REUSED_PREDICTIONS.json`, `CPU_GATE_RUN_MANIFEST.json`, and B/C/D `ARM_RESULT.json` files. JSON must be valid and finite; file sizes, hashes, run/experiment/source/config/data/seed identity are checked before publication. Each file is limited to4MB and the result payload to12MB. Checkpoint references may remain in the manifest; checkpoint tensors, arbitrary files, environment variables and secrets are excluded.

Result files retain their original bytes. The published runtime manifest is a projection with volatile publication fields normalized to `GIT_PAYLOAD_SNAPSHOT`. `PUBLICATION_INDEX.json` lists the exact committed payload hashes, source/config identity, publisher source SHA and main parent. A commit cannot contain its own SHA; therefore final verification belongs to the Git branch/commit and local `PUBLICATION.json`, which updates the local runtime manifest only after the remote ref equals the local commit. No circular or fabricated commit hash is used.

A payload-derived branch name makes retries idempotent. An existing remote result must pass index and actual file hash verification; it is not accepted merely because a branch exists. New result commits must change only the allowlisted report paths relative to their main parent.

## Failure and recovery

The receipt distinguishes validation, local commit creation, push completion and remote verification. `PUBLISHED` requires all verification steps. A failed push or failed remote readback leaves `FAILED_LOCAL_RETAINED`, preserves training completion and exposes no unverified `remote_commit_sha`. Training errors after a successful return-path check attempt publication of the available partial reports with the original failed/partial status. A failed return-path preflight does not waste GPU work or immediately repeat the same failed network check.

Completed runs are validated and reused before any model restore. Publication retry and `--mode publish-existing` perform no optimizer updates and require no historical checkpoint load. The latter can also publish the earlier completed16df822e run while preserving its original source identity; a later rerun remains a distinct run identity.

## Validation scope

CPU tests use temporary local bare Git remotes to exercise main ancestry, workflow preservation, exact report bytes, remote SHA verification, idempotent retries, push failures, successful-push/readback failures, partial result return, corrupted artifact rejection, missing-token fail-fast behavior and bounded/redacted network errors. The actual production-control admission is checked to stop before GPU work when write preflight fails, and to skip model execution for an already complete run. No external result branch, real model training or GPU execution is created by these tests.
