# M3-B E2 baseline freeze

Gate B PASS for E2-INIT-ONLY / ENGINEERING BASELINE. The EPCLM-004 M2 result index records an engineering run. No approved trained E2 checkpoint was identified. The Stage 1 B0-P3 E6 archive is hybrid and is excluded from the E2 identity.

Frozen E2 uses the M2 Mamba-1 reference, 18 blocks, dim 256, state 20, expansion 2, conv 4, tied vocabulary 8192 embedding/head and seed 41017. It contains 10,092,288 parameters. Its plain-tensor checkpoint is 40,423,686 bytes. [Manifest and source lock](../m3-composition/ssm-titans/baselines/e2/manifest.json)

The final-normalization hook exposes [1,5,256] without altering the existing forward. The original LM Head exactly reconstructs logits. Full versus 3+2 cached logits differ by at most 1.67e-6 under the original M2 cache tolerance. Batch-one runtime has 884,736 bytes of recurrent state and zero attention KV entries. Same-shape cross-stream perturbations preserve the other lane bitwise. Comparing batch-one and batch-two execution exceeds the original elementwise tolerance at 18 and 27 positions. The diagnostic is retained with unchanged tolerances and a worst absolute difference of 2.09e-6. Use fixed batch shape for reproducibility.

A fresh interpreter reproduced checkpoint bytes from the initializer, loaded the serialized tensors using weights_only=True and reproduced validation and output hashes. Earlier attempts are preserved in attempts.json, including the original batch-shape parity failure and the initial state dict serialization failure.

Reproduce with `python m3-composition/ssm-titans/freeze_e2.py /tmp/new-e2-output` in the locked environment. The actual tensor file is retained in the local run output, with its digest and regeneration path committed. The weights have zero training updates. Scientific SSM–Titans evaluation requires separately approved trained E2 weights.
