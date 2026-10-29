# Content and fragment audit evidence

The shared CPU scanner lives in `shared/python/minicells_data_registry/fragments.py` and `fragment_scan.cpp`. Source streams use fixed token identities and uint32 arrays. Polynomial uint64 fingerprints are only lookup filters: every reported match is compared elementwise. Explicit EOS/boundary markers remain in concatenated historical streams, and scanning never resets at a processing/story boundary. Candidate fragments never concatenate unrelated candidate stories.

Freeze candidate identities, source membership, tokenizer/preprocessing variants, fragment lengths, collision handling and evidence truncation rules before scanning. Record exact source/stream/index hashes and local-vs-remote durability. Count every occurrence; bounded examples must not be presented as a complete occurrence list. An actual model-visible token budget and a full-story conservative superset have different meanings and must be reported separately.

Keep three questions separate: source coverage, exact content/fragment overlap, and scientific admissibility. A partial source scan cannot establish complete isolation. Short phrase repetition, semantic overlap and a complete context/label copy are different claims. Cross-vocabulary integer token IDs are never compared directly. Unknown declarations and reserved extents remain explicit.

The first application is R3m `registration/content-isolation-001`. The original dataset usage ledger and prior failed amendments are preserved. It uses an isolated integrity auditor with no model/quality access, and does not grant Proof B or change scientific registration gates.
