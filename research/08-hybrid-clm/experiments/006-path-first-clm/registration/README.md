# Registration and authorization — not completed

Current state: **Gate 0 engineering only; Gate 1 NOT REGISTERED**.
Both config files are competing proposals. Neither is a scientific protocol freeze.

Before Gate 1, review and freeze:

1. Dataset provenance/revision/raw source hash, independent story identities and
   duplicate audits; train-only Tokenizer, exact windows and development manifest.
2. One model scale, six-step operation budget/alpha, optimizer schedule, entropy,
   critic and MoE balance coefficients, precision and checkpoint selection rule.
3. Development seed(s), at least three untouched confirmation seeds; audit collisions
   against historical consumed/reserved seeds and record evidence. No seeds are
   silently supplied by this implementation.
4. GPU budget and explicit authorization; short/full development sequencing.
5. Every null threshold in `protocol.json`: state/connection quality and cost,
   MoE quality/latency, three causal interventions, collapse/coverage, context use
   and sampled/greedy distribution gap. Bootstrap/window/profile definitions.
6. Approved source SHA, data identities, immutable publication format and review record.

An approval revision supplies `registration/evidence.json` with experiment ID,
canonical config SHA256, exact thresholds, `user_approved: true`,
`seed_collision_audit_passed: true`, and the SHA256 of the human approval record.
Freeze its file SHA256 in `protocol.json`, set REGISTERED, clear
`registration_required`, and explicitly approve GPU development in the config.
Changing the config or thresholds invalidates that evidence. A CPU pass alone
never sets any of these fields. This repository contains no fabricated approval.

Gate 3 review precedes Gate 4. Gate 4 materialization requires sealed seed membership,
explicit confirmation authorization and `confirmation_review_approved`. It creates
a separate bundle reusing the frozen training array/Tokenizer and checking exact
story content against both earlier sets. Freeze the resulting confirmation manifest
SHA256 in a newly approved config before confirmation training. No confirmation
bundle is prepared or opened by default.

R3/STOP is a later separate research registration. All current entrypoints reject it.
Final reports compute frozen criteria and consistency across seeds but leave formal
scientific claims pending research review. The implementation cannot confer human
approval; approval evidence is a repository review contract, not a cryptographic signer.
