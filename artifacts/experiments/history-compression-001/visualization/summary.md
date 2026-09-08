# History Compression 001 — Result Summary

Status: **HISTORY_COMPRESSION_TO_8_SUPPORTED**

| Mode | History prompts | Pass | Median heldout gain | Median eval KL | Median Top-1 | Coordinates |
|---|---:|---:|---:|---:|---:|---|
| zero_0 | 0 | 0/3 | 11.252905 | 1.92004800 | 0.34375 | E16/G4 |
| tiny_2 | 2 | 0/3 | 11.251993 | 0.29641733 | 0.81250 | E19/G14, E30/G2, E31/G15 |
| tiny_8 | 8 | 2/3 | 11.251219 | 0.02487715 | 0.96875 | E19/G14, E24/G6, E24/G15 |
| full_32 | 32 | 3/3 | 11.248345 | 0.00275664 | 1.00000 | E0/G5 |

The plot and this table are derived from durable per-mode `result.json` files. They are views, not the scientific source of truth.
