# Exact History protocol candidate

Use the same synthetic transcript and query for all three conditions within an episode. Keep episode IDs and complete target sequences in the evaluator. Prompts contain the query and its addressing metadata, with no answer tokens. Keep this generator separate from the EPCLM-007 candidate-ranking generator.

| Task | Generator contract | Primary score |
| --- | --- | --- |
| Exact Fact Recovery | Insert a uniquely identified fact with an ordered token payload and query by identifier | Exact equality of the complete payload |
| Exact Sequence Recovery | Insert a delimited sequence among distractor spans and query by span identifier | Exact equality of the complete token sequence |
| Position-based Recovery | Query a declared absolute token offset in the original transcript | Exact equality of the token at that offset |
| Conversation Order Recovery | Assign unique turn IDs and request their original order | Exact equality of the complete ordered turn-ID sequence |

Use free autoregressive generation with identical prompts and decoding rules, and no teacher-forced answers. Wrong length, missing output and invalid output count as misses. Report exact-match counts and denominators separately by task, length, condition and seed. Candidate ranking, semantic similarity and token-level partial credit do not enter the primary scores.

- **E0 Full KV:** restore complete KV and position state from the source transcript.
- **E0 Transcript Replay:** rebuild KV from the exact transcript in a fresh session.
- **E3 + Titans Cell-only:** load the paired shared weights and Cell into a fresh session with empty SWA and fresh Mamba state. Provide no source transcript.

Use distinct deterministic split namespaces, and verify no overlap of transcript identities or scored payload instances before evaluation. Retain the same episode IDs across arms. The source transcript and answers remain separate from the target session inputs in the Cell-only condition.

Pending freeze decisions include the symbol pool, delimiters and prompts, transcript structure, payload length ranges, example counts and split seeds, decoding stop and output limits, and whether any task supervision enters backbone training. The generator and hashes must be verified before Test materialization. The EPCLM-008 context lengths are 1024, 4096 and 16384 for Session Continuity. Reusing those lengths for Exact History requires an explicit decision.
