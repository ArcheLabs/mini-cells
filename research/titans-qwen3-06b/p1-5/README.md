# P1.5 — Fast Weight Update vs. Retrieval Response

Run kaggle_p1_5.py with Accelerator None, Internet and GITHUB_TOKEN. The first episode compares the original and instrumented execution exactly before proceeding. CPU FP32, no outer training. The frozen P1.4 episode function is reused directly. The final read statistics are checked against published P1.4 per head (rtol 0.005, atol 1e-7).

The report traces swap and random perturbation norms stage by stage. Per-episode JSON records matrix and head statistics; bootstrap intervals resample paired episodes with the three seeds fixed. Norms across stages have different units, so use the relative comparisons and paired intervals when interpreting attenuation. Fixed external K does not exclude an internal address mismatch. No model or memory matrices are published.
