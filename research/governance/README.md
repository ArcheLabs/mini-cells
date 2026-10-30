# Research Governance

This directory contains shared records and controls that span projects. It is
not a second home for experiment code or results.

- [`catalog.yaml`](catalog.yaml) lists the research projects.
- [`registry.yaml`](registry.yaml) indexes registered studies and their status.
- [`validations/`](validations/) preserves frozen protocols and decisions.
- [`audits/`](audits/) records cross-study reviews and evidence boundaries.
- [`archive/`](archive/) preserves retired workflows and historical records.
- [`experiment-index.md`](experiment-index.md) and
  [`notebook-index.md`](notebook-index.md) explain where active material lives.
- [`historical-migration-map.md`](historical-migration-map.md) maps the former
  stage folders into current project ownership.

Treat project experiment records and frozen protocol files as authoritative.
The registry and indexes help locate them; they do not change their status.

Current development launch policy: [one-launch automatic workflow](automatic-development-workflow.md). Manual registration fields are not a prerequisite for development execution.

## MiniCells Evidence Map

Read the [Research History and Evidence Map](MINICELLS_EVIDENCE_MAP.md) for the integrated narrative. Original protocols and results remain authoritative; the Map is not a parallel registry or execution authorization.

Formal registration should link its research question, line, stable claims and existing evidence. Exploratory work records purpose, scope and budget. A study-closing PR must update its history, outcomes, affected claims and substantive assessment changes, including failures and aborted runs. Preserve amendments and identify post-result analysis explicitly. Keep unknowns and conflicts visible rather than inventing chronology or preregistration. Review the Map at major phase boundaries; [completeness audit](audits/EVIDENCE_MAP_COMPLETENESS.md).
