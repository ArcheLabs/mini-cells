# M1-002A-R1 · Full-State Persistence Bridge

Use the frozen MoM 1.3B FP16 T4 Developer Preview. The executable protocol verifies runtime versions and downloads the exact 12-case dataset and baseline statistics from the M1-002A result commit `d9fca132573de69da91b164dcdc44f66e7b41ca3`. Both the byte and canonical dataset hashes must match.

## Storage

`mc.cells.create(name, full_state=True)` opts into schema v2 on the pinned preview. Bundles contain `manifest.json`, `cache_structure.json`, and `cache_state.safetensors`. They preserve recurrent state, Q/K/V routed and shared convolution state, attention and FFN fields, nested container types, tensor shape/dtype/stride, observed FLA cursor, per-layer cursor view and window metadata. FLA 0.2.2 has one global cursor and no configured window for this path. MoM can advance by routed packed length, so the SDK tracks the written-token count independently. Existing v1 archives remain recurrent-only. The public Cell metadata exposes `state_completeness`.

## R1-A

Use the first original Reference write and its first query with canonical token IDs. The Append probe writes the second original Reference fact in its own call. A writer, restore/append worker and verification worker each open an independent model process. A fourth process runs R1-B after R1-A passes.

Measure two independent continuous executions before same-process recovery. A fresh process then rebuilds the live write independently. The per-field maximum of the measured within-process and cross-process live errors is frozen before importing. If the live error is zero for a tensor, require exact equality for that tensor. Validate restored tensor values, dtype, structure and token cursor in addition to comparing query outputs. Record per-token block hidden states and logits, candidate log probabilities and per-layer pre/post recurrent and convolution state difference summaries. Exercise the public SDK lifecycle, including cross-process Append, read-only Query and independent Fork. Capture actual MoM field coverage. This probe reports execution equivalence for the tested sequences.

## R1-B

Run the original six arms and publish the historical comparisons. A runs continuously. B uses the legacy recurrent-only Cell. C restores Conv with the cursor reset. D restores the cursor with Conv reset. E restores full state. Each candidate runs on a fresh cache and uses the original bridge scoring. Record candidate and token log probabilities, Gold NLL, margins and paired gains/losses. Reference, Anchor and Combined each require at least 10/12 flips in A and E. Combined no-memory original recall must be at most 1/12. Record state and bridge outcomes independently. Capacity remains `NOT_RUN_BY_SCOPE`.

## Kaggle

Run `kaggle_full_state_bridge.py` as a Python Script with Tesla T4, Internet access and the `GITHUB_TOKEN` secret. The CPU suite runs before GPU work. Successful or failed experiments publish available receipts to a separate experiment branch. Model and temporary state tensors remain local to the running job.
