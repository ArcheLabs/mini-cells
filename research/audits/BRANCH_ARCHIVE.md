# Branch archive and cleanup recommendations

The machine-readable inventory is [`BRANCH_ARCHIVE.json`](BRANCH_ARCHIVE.json).
It records local branch heads, categories, canonical paths, and a fail-closed
`safe_to_delete` recommendation. No branch was deleted during release
preparation.

A branch may be removed only after all of these checks pass:

1. no open pull request depends on it;
2. source commits remain reachable from `main`, a tag, or this documented SHA;
3. published artifact provenance records the original source SHA;
4. supported notebooks do not hardcode the branch;
5. workflows do not require it; and
6. no unmerged scientific result exists only on the branch.

The current HybridCLM and Stage-1 branches are active and must be kept until
their review/merge is complete. Historical research branches remain recoverable
through their recorded commit IDs.
