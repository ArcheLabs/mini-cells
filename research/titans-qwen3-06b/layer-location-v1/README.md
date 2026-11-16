# Titans layer location screen

Both placements (block indices 3 and 22) use the plain 512→2048→512 GELU MemoryMLP from Titans commit `c60b4efaa3a3c6df4deebd57cbccf75947b87cad`. The experiment owner approved this deviation from the original LayerNorm + Residual specification. The pinned upstream dense fast-weight implementation rejects the ResidualNorm wrapper, and its Titan-Llama integration also sets `mem_model_norm_add_residual=False`.

The adaptive inner update, momentum, chunk size of one, differentiable inner gradients, and attention-output fusion are unchanged. Run `kaggle_smoke.py` as a Kaggle Python Script with Internet and GPU enabled before starting the screening.
