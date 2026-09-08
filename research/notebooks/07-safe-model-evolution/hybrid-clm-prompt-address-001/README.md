# Hybrid CLM Prompt Address 001

This folder contains the hosted-GPU orchestration for `HYBRID_CLM_PROMPT_ADDRESS_001`.

The experiment is a **mechanism diagnostic / engineering validation**, not a replacement for the Granite Hybrid CLM v0.1 milestone decision and not a rewrite of `CLM_CONVERSION_KILL_TEST_001`.

## Question

Does prompt-anchor address semantics remove the tokenwise-routing/history-KL confound while preserving Hybrid CLM knowledge acquisition?

The runtime contract is:

- address applicability is computed once from the prompt anchor;
- the decision is independent per Cell and non-competitive;
- candidate/answer tokens cannot change routing;
- a Cell may write only at the anchor and later positions;
- address training uses train paraphrases only;
- held-out paraphrases must route correctly before the address is frozen;
- general-history prompt anchors must have zero false-positive activations before the address is frozen.

The existing write gates are unchanged:

- history KL `<= 0.02`;
- target NLL gain `>= 0.5`;
- semantic candidate-choice accuracy `= 1.0`.

## Canonical assets

- Frozen protocol: `research/validations/hybrid-clm-prompt-address-001/protocol.json`
- Numerical evidence: `artifacts/experiments/hybrid-clm-prompt-address-001/`
- Hosted runner: `scripts/research/hybrid_clm_prompt_address_001/run.py`
- Terminal-result publisher: `scripts/research/hybrid_clm_prompt_address_001/publish.py`

The notebook is orchestration only. `seed_summary.json` and `decision.json` are the durable evidence.

## Hosted execution rules

`GITHUB_TOKEN` and `HF_TOKEN` must both be configured as Kaggle Secrets. The notebook never prints their values. The GitHub token is checked with a dry-run push before GPU execution; the HF token is required by the frozen protocol so the model is not fetched through anonymous Hub requests.

The three-fact smoke deliberately uses one GPU because all mutations belong to a single sequential lineage. Replicating the model across two GPUs provides little benefit at this size. Independent future seeds or independent full-run lineages may be dispatched one per GPU.

Scientific `PASS` and `FAIL` are both published to the branch. Infrastructure failures are not converted into scientific failures.

## Recovery

Rerun the notebook. A terminal artifact is skipped only if its protocol SHA-256 and registered implementation Git-blob map match the currently frozen protocol. Do not edit thresholds after observing GPU results and then treat the rerun as the same experiment.
