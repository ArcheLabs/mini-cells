# MiniCells HybridCLM v0.1 publication bundle

This directory contains the checked-in release configuration, publication
templates, and the rendered Space data contract. Replace the source commit
placeholder in `release-config.json` with the exact commit behind the release
tag before publication. Run `scripts/release/prepare_hybrid_clm_release.py`
with a validated mutation directory to stage a real safetensors artifact. The
foundation model is never copied into this bundle.

Publication status: **Engineering Evidence · Formal Validation Pending**.

Suggested Hugging Face surfaces:

- model repo: `archelabs-org/granite-3.1-1b-hybrid-cell-l7-k64`
- Space: `archelabs-org/MiniCells-HybridCLM`
- Collection: `MiniCells — Hybrid CLM`

The Kaggle launcher is
`notebooks/kaggle/hybrid_clm_release_v0_2_0a1.ipynb`. It defaults to a dry run
and never executes formal seeds.
