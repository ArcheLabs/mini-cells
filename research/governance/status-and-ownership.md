# Ownership and evidence status

Use the [project catalog](catalog.yaml) to assign one owner per study. Keep its
implementation and canonical study record under that project. A study on a
branch is a candidate for that project; its branch or result branch does not
make it part of main.

## Record hierarchy

| Question | Record to consult |
|---|---|
| What was planned, and what may run? | The study protocol and any explicit authorization or amendment |
| What code and inputs actually ran? | Source commit, resolved configuration, dataset/model revisions and run manifest |
| What happened in this run? | Decision file, raw metrics and published artifact commit |
| What scientific claims are supported? | Study analysis, with links to the underlying runs and controls |
| Where do I find a study? | The project README and [registry](registry.yaml) |

When records disagree, report the discrepancy and cite both sources. An index
or a later summary does not overwrite a frozen protocol or raw run.

## Keep three states separate

Record these fields in the study README or its linked decision record:

- **Execution:** planned, running, completed, failed, or stopped.
- **Engineering:** not evaluated, pass, partial, or fail; name the tested scope.
- **Scientific:** not evaluated, inconclusive, supported, or not supported;
  state the claim and dataset scope.

A published result SHA proves publication. A smoke pass proves the tested
execution path. Neither establishes a scientific result or authorizes a formal
run. Record skipped checks explicitly. Keep missing provenance visible.

## Before launch and at closure

1. Identify the owner, study ID, source ref, protocol, run mode and budget.
2. Use the [automatic launch workflow](automatic-development-workflow.md).
   Record the resolved inputs and remote result SHA without manual bookkeeping.
3. Link the decision, evidence, limitations and next action from the canonical
   study record. Keep large checkpoints in durable artifact storage and link
   them by checksum. When a branch-only result affects a project claim, land
   its evidence summary through review.
4. Update the registry and evidence map only where the study changes their
   coverage or conclusions. Preserve failures, amendments and historical paths.
