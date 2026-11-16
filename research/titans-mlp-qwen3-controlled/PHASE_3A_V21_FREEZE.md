# Freeze review

**NOT_FROZEN**. Formal Test is sealed and the formal experiment has not started.

| Item | Status |
|---|---|
| Template semantics and grammar | Codex review complete, owner review pending |
| Template families and pinned tokenizer offsets | CPU checks complete |
| Dataset balance and disjoint splits | CPU unit checks complete |
| Kaggle T4 and remote publication | Pending |
| Formal configuration | Draft, formal runner integration pending |
| Owner sign-off | Pending |

Formal configuration SHA256:
`cb7948ab25cdea760483ee4cb95ed8045b250ef7766ecef79e6d8579fdff7e28`.

Template registry SHA256:
`e7fccd1fc0c1f4932e437cc19d882c5f908893cebe5693b86c4e3669eb252ba1`.

The draft specifies full-scale training, baselines, conditions and thresholds.
The current executable is the separate eight-update Smoke. A formal freeze
requires the formal runner consuming this draft, audited T4 Smoke and
publication evidence, and explicit owner approval.
