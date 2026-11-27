# M1-002 · Single-Cell capacity and interference

## Scope

[M1-001 Value Swap](https://github.com/ArcheLabs/mini-cells/commit/7744e8c63dcb2b7085ec58b9a9a948bb694eec14) established one-fact recall under the bounded native MoM 1.3B preview. M1-002 measures target recall as the number of stored facts increases. It uses a single Tesla T4 and the same frozen model and recurrent-only Cell profile.

The first stage stores 1, 3, 8, or 16 facts. Twelve independent episodes each supply one scored target and up to fifteen other project-to-code mappings. Gold codes are randomized from twenty candidate words. Each episode uses two query wordings. At each size beyond one fact, the same target and distractor set are written in two orders. The target occupies the first or last position and the distractors are shuffled independently in each order. Results describe the sampled target at each size.

The dataset uses seed 66109 and canonical SHA-256 `a02e726996e1f257e73a80cb5b923f5a789fa1016ed93a1287570c106e7937fb`. The pinned tokenizer must reproduce it exactly. The two write APIs receive identical Token IDs at the Append boundary. In the neutral arm each filler has the same Token count as the replaced fact, preserving target offset and total length.

## Conditions

| Condition | Function |
| --- | --- |
| No memory | Fresh stream for the query |
| Live target | Query follows the single target fact in the same native stream |
| Single stored fact | One write followed by a fresh query from recurrent-only state |
| Factual | One write containing the target and other mappings, 64-Token chunks |
| Neutral | One write containing the target and length-matched neutral text, same schedule as Factual |
| Split32 | Factual text with 32-Token chunks |
| Append | First and second halves through the SDK Write and Append APIs, up to 64 Tokens per forward |
| Continuous Append schedule | Same Token IDs and forward boundaries as Append, using one continuous write stream before recurrent-only export |

Each query starts from a separate cache. Cell bundle hashes are checked before and after both wordings. Append restarts from recurrent-only state at the SDK write boundary. Its comparison with the continuous schedule measures that session-boundary behavior, including the convolution history and cursor.

## Metrics and decision

Each query is scored by full-vocabulary log probability of twenty same-length candidate answers, with one Token per answer. Gold must strictly outrank all distractors. Report log probabilities, gold NLL, gold margin, and recall for each wording, plus paired changes in margin, NLL, and recall. An episode is recalled at a capacity only if both wordings pass in both factual orders. For the single-fact arm, both wordings must pass. The interference rate has all paired episodes and orders as denominator and counts a factual miss paired with a neutral hit.

- Invalid when no-memory recall exceeds 10% or live-target recall is below 80%.
- Pass when controls are valid and single-fact recall and robust factual recall at 3, 8, and 16 each reach 80%.
- Fail when controls are valid and any factual recall gate falls below 80%.

A pass supports freezing a separate confirmation at 20–30 facts. M0 and subsequent M1 experiments have independent gates. The launcher publishes M1-002 artifacts directly to GitHub.

## Kaggle

Use `kaggle_capacity.py` as a Kaggle Python Script with a Tesla T4, internet access and the `GITHUB_TOKEN` secret. The CPU check runs in the same job before the GPU experiment. Review `report.md`, `statistics.json`, and `cases.jsonl` after publication.
