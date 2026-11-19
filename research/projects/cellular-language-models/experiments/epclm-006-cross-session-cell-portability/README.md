# EPCLM-006 Cross-Session Cell Portability

The frozen E3 Samba-lite implementation is the Session backbone. The same
weights encode each explicit Cell segment and Query Packet from an empty state.
The optional MAG-style Cell reads without changing Fast Weights or Momentum.
Portable artifacts contain tensor-only state and compatibility metadata.
Target E3 state lives in a separate Runtime and may retain information after
Cell unload.

Local engineering checks:

```sh
python portability_preflight.py /tmp/epclm006-preflight-new
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_portability.py -q
```

The [Notebook](notebook.ipynb) runs CPU and T4 checks and a no-update GPU budget
probe. Select Tesla T4, enable Internet and GITHUB_TOKEN, then Save & Run All.
The run uses one training seed, 32 AdamW updates, 8,192 supplied input tokens
and 512 answer targets. Canonical Query encoding adds 16 E3 token positions
per episode beyond supplied inputs, independently of the training ledger.
C0–C3 and C3-reset share one final checkpoint and three held-out banks.
Independent Test remains sealed. Attach complete saved Output for exact-source
resume or publication retry.

The training-only optional Cell policy reserves one of four lanes and drops
each remaining lane with probability 0.20, giving a 0.15 expected batch-average
dropout. This fixed engineering default has not been tuned against validation
quality. The primary 3 pp threshold applies to per-fact answer EM, with both
paired episode-bootstrap lower bounds above zero. Joint four-answer Episode EM
is reported separately. Query templates are explicit Token-ID alternatives.
No abstention token is defined.
