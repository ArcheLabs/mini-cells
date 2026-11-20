# HAM-001A — Single-binding diagnostic

HAM-001 B4 uses projected synthetic inputs, a pinned upstream store/read path, and
one-token commits. This standalone diagnostic retains that configuration for the
mechanism arm and adds a matched-geometry comparison using Qwen3-0.6B-Base Layer
3's actual frozen Q/K weights, native head normalization and position-zero RoPE.
Hidden inputs and targets are synthetic; this is not a language or tokenizer test.
The matched control repeats K into its corresponding GQA query heads. Both arms
write identical K/V at native geometry.

Controls: C0 zero; C1 native slow map, no writes; C2 one write and matched fact
query; C3 independently copied C2 state and unrelated query; C4 fresh empty write
at original K with alternate V. Compare direction of MSE improvement over the
slow map, specificity versus C3, change in preference between the two values,
and whether output beats zero. Positive directional effects can be tiny and cannot
by themselves establish reliable recall.

Run the Kaggle Python script with Accelerator None, Internet and GITHUB_TOKEN.
The launcher publishes a separate run branch and records Kaggle version when
available. If unavailable, computation is retained with incomplete provenance.
No HAM-001 rerun or H2. Local: run.py OUTPUT --titans-upstream PINNED_CHECKOUT.
