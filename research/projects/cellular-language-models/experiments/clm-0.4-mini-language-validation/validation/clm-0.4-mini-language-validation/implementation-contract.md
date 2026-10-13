# CLM-0.4-mini Implementation Contract

> Status: Normative implementation clarification

This file resolves implementation ambiguities without changing the scientific protocol.

## Normative precedence

When implementing CLM-0.4-mini, use this precedence order:

1. `protocol.json` for frozen architecture, gates, seeds, curriculum semantics, and decision rules;
2. `transaction-record.schema.json` and `cell-registry.schema.json` for observability structure;
3. `protocol-lock.template.json` for the pre-formal implementation lock;
4. `README.md` / `README.zh-CN.md` for human-readable rationale and explanation.

If prose appears ambiguous, do not silently choose a different scientific behavior. Follow the machine-readable contract or revise the protocol before formal execution.

## Primary variant semantics

### `local_always`

- starts from the same base checkpoint as the other variants;
- every transaction trains only the four deterministic routed **base Cells**: Top-2 in block 3 and Top-2 in block 4;
- always commits the direct candidate;
- never creates, routes, trains, or reuses private growth Cells;
- provides the primary new-learning-gain and regression-damage reference.

### `local_tx`

- uses exactly the same direct candidate support as `local_always`;
- applies the registered new-gain and dependency-scoped regression gates;
- commits a passing direct candidate and fully rolls back a failing candidate;
- never grows private Cells.

### `local_tx_growth`

- first attempts the same direct candidate as `local_tx` while no private bundle exists;
- after direct rejection, fully rolls back before creating the probationary private bundle;
- a successful spawn transaction atomically commits private Cell parameters plus private route;
- once an address owns a private bundle, later transactions train only that bundle;
- a failed private-reuse candidate rolls back and does not create a second bundle in M1.

No variant may use the hidden global oracle to make a state-transition decision.

## Transaction record shape

A transaction can contain more than one speculative candidate. Therefore the normative raw record is:

`transaction -> attempts[] -> final_decision`

Examples:

- direct pass: one `direct` attempt, then `direct-commit`;
- direct fail + growth pass: `direct` attempt followed by `spawn`, then `growth-commit`;
- direct fail + growth fail: `direct` + `spawn`, then `rollback`;
- existing private bundle pass/fail: one `private-reuse` attempt, then `private-reuse-commit` or `rollback`.

Candidate-level metrics, dependency scope, oracle result, false-safe state, structural escape, optimizer/cost counters, and timing belong inside each `attempts[]` item. Transaction-level state hashes, Cell births/deletions, route state, and final decision belong on the parent record.

## Stable addressing

`address_id` is out-of-band metadata. It must not be encoded into the model's language token stream for formal M1.

The formal base route must depend only on immutable protocol inputs such as `protocol_salt`, `layer_id`, and `address_id`. It must never depend on mutable hidden activations.

Any learned semantic router is a shadow diagnostic only and must have no causal path to candidate support, dependency scope, validation, commit, rollback, or growth.

## Formal-lock boundary

Development seed `90401` may be used only for the finite registered candidate-training grid and implementation/debug validation.

Before running `90411`, `90412`, or `90413`:

1. replace all required null values in the protocol-lock template;
2. set the lock state to `LOCKED`;
3. record protocol/code/data/tokenizer/environment hashes;
4. commit the lock file;
5. verify the tracked tree is clean.

After any formal result is observed, changes to scientific gates, architecture, data/curriculum manifests, transaction schedule, optimizer selection, candidate learning rate/steps, routing salt, or structural tolerance require a new experiment/protocol revision rather than modification of this experiment's result.
