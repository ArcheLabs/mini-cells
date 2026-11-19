# T4 resource and recovery validation

Run resource_validation.ipynb on Tesla T4 with Internet and GITHUB_TOKEN.
This uses batch-one synthetic FP32 examples and disposable model weights.
Each arm is attempted at 1024, 4096 and 16384 input tokens. MAG uses the
canonical packet encoder and gated query path with a frozen E3. MAG
throughput counts query targets, while backbone throughput counts shifted
next-token targets. OOM is recorded at the requested length.

Recovery compares the next committed update after fresh-process restore
against uninterrupted execution at length 1024. The archive includes the
interruption checkpoints. GitHub receives reports and the execution log.
The saved Kaggle Output contains the weight archive. The notebook also uploads
the archive as a GitHub Release asset and verifies downloaded bytes before
publishing the decision. If Kaggle omits the saved-version ID, the report records
a null ID pending backfill from the actual saved version.

## Exact History proposed defaults for review

Use zero-shot greedy decoding and four disjoint synthetic validation tasks.
Use 128 episodes per task and context length, with 1024, 4096 and 16384
token contexts, and a single scored query per episode. Use distinct split
namespaces and seeds 42 for validation and 137 for sealed Test. These are
generator seeds independent of the paired model seeds. Use eight
single-token turn IDs, four-token fact payloads, eight-token sequence
payloads and one-token position targets. Sample distractors from a
disjoint token pool. Score the exact token sequence, with generation
bounded to the target length declared by task. Before freeze, audit
token IDs and delimiters against the frozen tokenizer and verify disjoint
transcript identities. These defaults are proposals pending review.
