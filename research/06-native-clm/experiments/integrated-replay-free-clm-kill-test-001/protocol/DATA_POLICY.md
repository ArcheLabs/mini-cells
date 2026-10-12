# KT001 Data Policy

Status: **REGISTERED_BEFORE_DEVELOPMENT_RUN**

KT001 keeps the Native CLM A/B/C/D domain identities and exact upstream revisions, but it must not falsely describe a same-size sample as fully fresh when the pinned dataset cannot support a disjoint replacement.

## Pinned revisions

- TinyStories: `roneneldan/TinyStories@f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- WikiText: `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`
- CodeParrot train: `codeparrot/codeparrot-clean-train@3e6ab65f2864931e041f6a82db9b5a6ec2b71ab4`
- CodeParrot eval: `codeparrot/codeparrot-clean-valid@4db92d2ec0c1b4c41eeb439cfae16854511d9dcd`
- Dolly: `databricks/databricks-dolly-15k@bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`

## Selection rule

For each split, the builder knows how many non-empty examples the earlier first-N Native CLM builder consumed from the beginning of that split.

1. Treat that prefix as historical.
2. Prefer records strictly after that prefix.
3. Select records deterministically by salted SHA-256 min-hash (`IRF-CLM-KT001-v1`).
4. If the post-prefix remainder can provide the full registered target, select zero historical-prefix records.
5. If the pinned split is too small, select every necessary fresh post-prefix record and supplement only the unavoidable shortfall from the historical prefix using the same deterministic min-hash rule.
6. Record the exact `historical_prefix_reused` count in `manifest.json`.

A manifest with any nonzero historical-prefix reuse is **not** fully disjoint from the earlier Native CLM sample.

## Why this is necessary

Dolly-15k makes strict same-dataset/same-size disjointness impossible: the earlier Native CLM builder used 10,000 rows for D training and the following 2,000 rows for D evaluation. A second 12,000-row sample from the same roughly 15,000-row pinned dataset must overlap substantially.

The scientific priority is therefore:

1. preserve exact domain/revision comparability;
2. maximize fresh records deterministically;
3. expose unavoidable overlap rather than hiding it.

The manifest is part of formal provenance. A future experiment that changes dataset families or target sizes is a different protocol, not a silent KT001 data fix.
