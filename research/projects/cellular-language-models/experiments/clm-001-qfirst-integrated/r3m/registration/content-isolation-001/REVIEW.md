# R3m historical source coverage and fragment isolation audit001

## Answer to the scientific question

**The current evidence cannot prove complete global story-and-fragment isolation of these512 Main candidates.** A concrete conservative source union can be constructed and scanned, but global source coverage remains incomplete, and exact partial overlaps already exist inside the known union. Do not register R3m, replace candidates, relax thresholds or rerun the completed R3h sampling replay to bypass this result.

This is an integrity audit, not a model-quality experiment. Short common fragments do not automatically prove scientific contamination, a performance advantage or any algorithmic failure. All findings remain available for independent review; there is no automatic scientific admissibility decision.

## Frozen before inspection

`FRAGMENT_AUDIT_PROTOCOL.json` was committed as `d490d1b4` before reading candidate fragments. The scanner and implementation bindings followed in `f492ccbe`, also before the content scan. Diagnostic lengths are16/32/64/65 tokens per tokenizer and16/32/64 normalized words. Candidate order and512 original IDs remain the immutable Amendment003 list. Rules were not adjusted after viewing results.

The auditor used original R3m4096 tokenizer bytes and historical2048 tokenizer bytes, each in its own ID namespace. Normalized words use NFKC/casefold/Unicode-word splitting with deterministic shared IDs and a separate story-boundary sentinel. Hashes are filters only; every hit requires complete elementwise equality. Three CPU scanner checks cover all-stride positions, source boundaries/budgets, terminal windows and deliberate polynomial-hash collision rejection.

## Source universe actually examined

- Frozen original TinyStories `train` raw IDs `[0,476901)` and `validation` raw IDs `[0,4396)`.
- Full source Parquet SHA256 values verified against the original frozen snapshot.
- Historical2048 stream prefixes independently match the preserved007120,000,000-token train and1,000,000-token validation corpus hashes. Entire terminal stories add593 train tokens and130 validation tokens as explicit conservative over-exclusion; budget counts remain separate.
- All stride-1 source positions are checked, retaining EOS/sentinels across story concatenation. Scanning never resets at story boundaries. Candidate fragments do not connect different candidate stories.
- The audit4096 view has110,057,561 train tokens; historical2048 has120,000,593; normalized words have85,666,991 units. These are source-union index sizes, not new training budgets or claims of actual optimizer consumption.
- Large exact streams/patterns/complete occurrence-count tables and memberships are retained locally at the index path in the proof (about1.61GB). SHA256 and byte sizes are recorded. No remote durability or HF upload is claimed. GitHub holds compact hash/provenance evidence, not source text or model checkpoints.

## Coverage map: all58 original unpinned declarations

`HISTORICAL_SOURCE_COVERAGE.json` maps every original ledger entry without rewriting historical identities. All111 original evidence file hashes match.

| Imported evidence class | Count | What is established |
|---|---:|---|
| Exact reconstructed model-visible source streams | 50 | Encoded streams are within the reproduced007 corpus universe; original raw Revision fields remain unrecorded |
| Reused007 corpus proof | 2 | Complete historical eligible corpus identity, not all actual optimizer starts |
| R3k/R3l later research roles | 6 | Frozen loader/range containment offers conditional coverage; actual role manifests/membership remain unrecovered |
| Other pinned declared ranges | 123 | Declared bounds fall inside the conservative source union; declaration alone does not attest actual runtime membership |

These181 entries are declaration counts, not independent models or disjoint corpora. Source bounds can conservatively cover multiple possible memberships, but no such conditional coverage is silently turned into scientific PASS.

## Exact fragment findings in historical train source

Whole-story collisions remain **zero** across raw, NFKC/whitespace and normalized-word story hashes. Nevertheless258/512 candidates have at least one of the fixed diagnostic fragment matches in the train union;94 have matches in validation. These candidate sets overlap, so they must not be added.

| View | Fragment length | Candidates with a match |
|---|---:|---:|
| Original R3m4096 | 16 tokens | 236 |
| Original R3m4096 | 32 tokens | 15 |
| Original R3m4096 | 64 /65 tokens | 0 /0 |
| Historical2048 | 16 tokens | 257 |
| Historical2048 | 32 tokens | 23 |
| Historical2048 | 64 /65 tokens | 2 /2 |
| NFKC normalized words | 16 words | 154 |
| NFKC normalized words | 32 words | 6 |
| NFKC normalized words | 64 words | 0 |

The historical2048 train view contains three matching65-token patterns across two candidates. All three occur within the exact reproduced120M-token corpus budget, not just the conservative terminal tail. A separate Python re-tokenization of the original source records independently confirmed full sequence equality:

| Candidate raw ID | Historical source raw ID |65-token pattern occurrences |
|---|---:|---:|
| 498305 | 110289 | 1 |
| 485693 | 272812 | 2 overlapping patterns |

Decoded shared fragments encode to59–60 tokens under the R3m4096 tokenizer. Therefore historical65-token overlap and zero R3m64-token hits are compatible; token lengths across vocabularies are not interchangeable. No raw fragment text is published. `LONG_FRAGMENT_INDEPENDENT_CHECK.json` preserves offsets and hashes.

These are matches in reachable/eligible corpus content, not proof that those exact spans were consumed by a historical optimizer. Exact overlaps at the declared diagnostic lengths prevent an unqualified assertion that the candidate/source fragment intersection is empty. Their scientific significance remains subject to review, with no post-hoc rule change or candidate replacement.

## Why global isolation still cannot be certified

1. **Fourteen reservation markers have no recorded upper endpoint.** Current Main IDs are not proven outside those scopes. Neither the entire tail nor an empty tail is invented as the answer.
2. **Six R3k/R3l research-role asset identities remain incomplete.** Their fixed source/range code is conditional containment evidence, not actual run provenance.
3. **Two additional declared historical tokenizer identities lack exact fragment views here.** The inventory identifies `f90b...` (including Experiment0058192-vocabulary assets) and `c0fd...` (CLM0.4-mini). Normalized-word checks do not replace their exact tokenizer/preprocessing identities.
4. **Non-interval corpus references need resolution.** CLM0.4-mini calibration metadata cites a30,000,057-token base corpus and base/curriculum manifest hashes, without their membership manifests in this audit. This is a declared asset identity, not proof that training actually used it; actual relevance/containment cannot be assumed either way. Details are in `ADDITIONAL_COVERAGE_GAPS.json`.
5. **The ledger has a stated coverage boundary.** Tracked JSON/YAML declarations do not prove absence of unpublished/manual/runtime/external source use.
6. **Fixed fragment diagnostics are not arbitrary preprocessing/semantic isolation.** Differing tokenizer lengths do not provide byte-aligned context/label equivalence across every historical pipeline; paraphrases and shorter spans are not certified absent.

## Proof B disposition

`PROOF_B_SCIENTIFIC_REVIEW_PROPOSAL.md` separates verified-source containment from exact historical reproduction. It prepares the formal claim, required evidence and independent decisions without enabling the route. Complete source exclusion could in principle address data independence without certifying the original NumPy production environment. This audit does not establish the premises for such a PASS.

The correct next work is to resolve the specific source/reservation/tokenizer questions and scientifically adjudicate the retained fragment evidence. No new Main selection is performed or proposed. No additional Kaggle/GPU run is required to reproduce the findings reported here.

## Gate status and preserved history

- Tokenizer Identity: PASS, unchanged.
- Original Asset Reconstruction: PASS, unchanged.
- R3h B-H Sampling Boundary: CONSISTENT / NOT_CERTIFIED, unchanged; workline closed.
- Global Content Isolation: BLOCKED.
- Proof B: prepared proposal, not approved.
- Scientific Registration: BLOCKED.

No model was imported/evaluated, no optimizer update or GPU work occurred, and no scientific config, candidate list, historical Checkpoint, original ledger or earlier amendment was modified. All processing was isolated local CPU content integrity work.
