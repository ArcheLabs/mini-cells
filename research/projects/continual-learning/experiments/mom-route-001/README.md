# MOM-ROUTE-001 — Routed Memory Feasibility

Continual Learning. [Protocol](PROTOCOL.md). Development only.

The runner scores 12 valid queries in four memory conditions, with three
additional routing probes. Each answer branch independently primes the native
MoM cache, then scores only the query and answer tokens. The existing
MA-001 scorer provides record-level Top 16 and the frozen Cell threshold is
reported separately. No Titans, training or cross-process cache restore.

`python run.py MOM_SOURCE MOM_SNAPSHOT TOKENIZER_SNAPSHOT E5_SNAPSHOT NEW_OUTPUT`

Use the checkpoint-compatible MoM/FLA environment from MOM-INT-001 and the
matching 32K tokenizer used in the upstream training recipe. Before a remote
run, pin and verify the tokenizer snapshot identity and prepare a launcher
that returns reports. No GPU execution or scientific result is claimed yet.
