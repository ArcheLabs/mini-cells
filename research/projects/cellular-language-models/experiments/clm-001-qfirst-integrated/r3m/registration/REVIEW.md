# Story isolation amendment 001 — blocked

The original protocol candidates are retained in SPEC.md and config/pilot.json. The requested replacement was audited before any formal change.

| Role | Original | Requested replacement | Outcome |
|---|---|---|---|
| Trigger Calibration | [10048,10304) | [100000,100256) | Historical train conflict |
| Pilot Report | [10304,10560) | [100256,100512) | Historical train conflict |
| Main Report | [10560,11072) | [100512,101024) | Historical train conflict |

The original candidates overlap Experiment005 train[0,50000). The replacement overlaps Cellular Computing Experiment007. Its preserved corpus manifest reports476847 consumed nonempty stories. The original loader strips/skips empty records: replay at the frozen TinyStories Revision reaches raw index476900 and exactly reproduces the old120M-token uint16 SHA256. This is an actual data-source match, despite the old manifest lacking a Revision field.

The inventory covers111 tracked JSON/YAML metadata documents, including historical corpus manifests, artifacts, explicit ranges, Train/Tokenizer prefixes and validation/confirmation declarations. Dataset reservation markers without an upper endpoint are retained as unresolved markers; they are not silently treated as infinite Confirmation. No model accessed Confirmation. No evaluation or training was performed.

The proof records1024 raw and normalized text hashes. No within-candidate normalized duplicates were found. Cross-history leakage is already certain from the reproduced training-prefix membership: any newly constructed evaluation windows from these stories would use historical training content. Window tokenization was therefore not performed and no new R3m tokenizer was trained. The required R3m tokenizer remains SHA256612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff.

A next INDEX candidate is Trigger[500000,500256), Pilot[500256,500512), Main[500512,501024), beyond the verified historical prefix and inside the pinned first shard. It has NOT been registered, read for selection, or certified for text/window uniqueness. New review and full content/window validation are required; no automatic relocation was applied.

Reproduction uses audit_registration.py and verify_historical_prefix.py. Both are CPU read-only data utilities. No optimizer, model or GPU is used. Full private R3h anchor/persistence verification remains a separate future engineering step.
