# Stage 3 outer LM training

The FP32 T4 fixture freezes Qwen and optimizes head-local initialization,
controls and fusion using next-token cross entropy. The gradient unroll
covers one historical write and one subsequent read. Native Q/K/V
projections stay frozen.

The runner saves the adapter and optimizer in Kaggle Output and verifies
exact state and prediction recovery. It compares the original E2
initialization and restored E3 on one separate token fixture. The paired
NLL is an engineering diagnostic. Corpus-level training, quality evaluation
and Stage 4 expansion remain pending.
