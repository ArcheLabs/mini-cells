# CLM Validation 001 diagnosis

Diagnosis: `CONTINUATION_DID_NOT_LEAVE_DENSE_BASIN`

The original Validation 001 result remains `FAIL / CLM_PROGRAM_SPARSITY_QUALITY_FAILURE` and
is not rewritten by Validation 001b. Its evidence establishes dense architectural success, not
a theoretical failure of conditional computation: all three real-checkpoint conversions had
TextNCA/CLM perplexity parity, logits parity, recurrent-state parity, and approximately 2.08%
receptor overhead.

The soft continuation did not reach its registered targets. The soft-75 stages remained near
`R_P = 0.9995`, and the soft-50 stages remained near `R_P = 0.996`, rather than 0.75 and 0.50.
The actual path was therefore approximately dense 8/8 → abrupt hard 6/8 → almost-dense soft →
abrupt hard 4/8. It did not constitute progressive sparse continuation.

Validation 001 also recorded zero Dynamic structural variation because its Dynamic evaluation
branch captured masks but did not call the routing-variation measurement. That zero is not valid
scientific evidence of static routing.

Finally, the whole-executor threshold `executor_ratio <= 0.5` was unreachable for a program-only
experiment. With cell activation fixed at one, the GRU remains fully active, so 4/8 FFN programs
produce an executor ratio of approximately 0.7143 rather than 0.5.

Validation 001b preserves the original result and addresses only these validation/training-harness
issues. It does not alter the CLM architecture.
