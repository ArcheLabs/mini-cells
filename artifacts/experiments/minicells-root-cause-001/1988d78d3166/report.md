# MiniCells Root Cause · Phase A

**FAIL**. Phase A Base Runtime gates evaluated. Phase B remains a separate experiment.

- tokenizer: PASS
- native_reference: PASS
- upstream_split: FAIL
- greedy_reference: PASS
- base_completion: FAIL

## Known completions, out of six

- continuation/bos=False: 2
- continuation/bos=True: 3
- fact_completion/bos=False: 4
- fact_completion/bos=True: 3
- question_answer/bos=False: 1
- question_answer/bos=True: 1

Official CausalLM no-label wrapper: FAIL.

The auxiliary results include the official wrapper probe, raw outputs, per-schedule cache positions, and full logit comparison statistics.
