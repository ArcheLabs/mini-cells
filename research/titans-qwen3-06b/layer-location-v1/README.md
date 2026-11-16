# Titans layer location screen

Both placements (block indices 3 and 22) use the plain 512→2048→512 GELU MemoryMLP from Titans commit `c60b4efaa3a3c6df4deebd57cbccf75947b87cad`. The experiment owner approved this deviation from the original LayerNorm + Residual specification. The pinned upstream dense fast-weight implementation rejects the ResidualNorm wrapper, and its Titan-Llama integration also sets `mem_model_norm_add_residual=False`.

The adaptive inner update, momentum, chunk size of one, differentiable inner gradients, and attention-output fusion are unchanged. Run `kaggle_smoke.py` as a Kaggle Python Script with Internet and GPU enabled before starting the screening.

## T4 capacity finding

The four-token smoke passed at both placements. A full-episode timing run
failed in the upstream associative scan during the forward store path,
before an optimizer update. The T4 had 14.56 GiB capacity, with 732.81 MiB
free when the scan requested another 848 MiB. PyTorch had allocated
13.70 GiB and held only 21.39 MiB reserved but unallocated. This points
to live tensor footprint, with little unused reservation to recover.

The timing diagnostics in e47f70c do not change the scan or its per-token
dense weight timeline. Do not repeat the same full workload on T4 to
validate those diagnostics. A larger GPU needs a fresh timing measurement.
Any memory-saving implementation must verify causal outputs, final Cell
state and outer gradients against the current path before screening.
