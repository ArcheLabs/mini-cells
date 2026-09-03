# Research Script Entrypoints

`script/research` is intentionally treated as an **entrypoint compatibility layer**, not as the canonical home of scientific meaning. Protocols, formal decisions, interpretation boundaries and audit conclusions belong under `research/`.

The directory is currently large because several generations of named runners/reporters/publishers were retained for exact notebook and artifact reproduction. This audit does **not** mass-move those files: published notebooks, commands, manifests and branch artifacts contain literal paths such as `scripts/research/run_core_validation_005.py`.

## Canonical structure going forward

```text
scripts/research/
  README.md                 # this policy and navigation
  run.py                    # preferred unified run dispatch when supported
  report.py                 # preferred unified report dispatch when supported
  publish.py                # preferred unified publish dispatch when supported
  _dispatch.py              # dispatch support

  # retained compatibility entrypoints
  run_*                     # historical/family-specific runners
  report_*                  # frozen reporters / decision emitters
  publish_*                 # artifact publishers
  orchestrate_*             # resumable multi-seed orchestration
  validate_*                # identity/protocol/artifact validators
  analyze_*                 # checkpoint-only/offline diagnostics
  fetch_*                   # external checkpoint/data hydration
```

The flat compatibility namespace is **legacy but supported**. New research should not add another unstructured root entrypoint when a unified dispatcher or a family package can carry it.

## Scientific families

Use the filename prefix and the corresponding path under `research/validations/` to identify ownership:

| Family | Script examples/prefix | Scientific record |
|---|---|---|
| Core continual-learning validations | `*_core_validation_*`, `*_core00*` | `research/validations/core-*` |
| Constructive CLM | `*constructive_clm*` | `research/validations/constructive-clm-*` |
| Native CLM | `*native_clm_v0*` | `research/validations/native-clm-v0-*` and `research/stages/06-native-clm/` |
| Checkpoint/address diagnostics | `analyze_*`, selected `report_*` | matching validation README/protocol; diagnostic status must remain explicit |
| Publication/orchestration infrastructure | `publish_*`, `orchestrate_*`, `_dispatch.py` | protocol/result path referenced by the entrypoint |
| Historical model/release experiments | generation/eval helpers retained for reproduction | `research/archive/`, `research/releases/`, or the referenced historical experiment |

## Path stability rule

A research entrypoint may be physically moved only when all of the following are done in the same migration:

1. search the repository for the old literal path;
2. update notebooks, docs, CI/workflows and manifests that are allowed to move;
3. leave a thin compatibility shim at the historical path when a published reproduction command uses it;
4. smoke the new target and the compatibility shim;
5. record the migration in the research audit/commit message;
6. never rewrite a frozen protocol/result merely to make its old reproduction command look current.

This rule intentionally values **reproducibility over directory aesthetics**.

## New-script policy

For new work:

- reusable logic belongs in `src/minicells/` or the relevant research package, not in a large entrypoint script;
- use `run.py` / `report.py` / `publish.py` dispatch when the experiment family is supported;
- if a new family needs multiple scripts, create a dedicated package/subdirectory rather than adding many unrelated root files;
- every formal runner must point to a frozen protocol under `research/validations/`;
- every reporter must emit only statuses allowed by that protocol;
- every publisher must preserve commit/protocol/data identities;
- checkpoint-only diagnostics must say explicitly that they consume no new formal scientific seeds.

## Cleanup phases

### Phase 1 — completed by the capability-ceiling audit

- establish this compatibility policy;
- make `research/audits/` the cross-experiment meaning layer;
- stop treating flat script placement as scientific organization;
- prohibit new unstructured root growth.

### Phase 2 — safe physical migration

Generate a reference graph for every flat entrypoint, then migrate only files whose references can be updated or shimmed safely. Good first candidates are documentation-only helpers and unreferenced diagnostics. Frozen formal runners/publishers should move last, if at all.

### Phase 3 — deletion

Delete a compatibility shim only after no supported reproduction path, notebook, workflow, release document or artifact manifest references it.

## Audit references

- [`research/audits/RESEARCH_LEDGER.md`](../../research/audits/RESEARCH_LEDGER.md)
- [`research/audits/CLM_CAPABILITY_CEILING.md`](../../research/audits/CLM_CAPABILITY_CEILING.md)

The goal is a repository where scientific history remains reproducible even while the active engineering surface becomes smaller and clearer.
