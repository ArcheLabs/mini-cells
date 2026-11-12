# EPCLM-007 Cell Retrieval Learning Validation

Run `OMP_NUM_THREADS=1 python preflight007.py /tmp/epclm007-new` in the existing PyTorch environment. Gate 0 checks shared Key and Value symbol coverage, disjoint pair partitions, a sealed Test namespace and exact oracle labels. The 1/8 reference is uniform random candidate selection. The split partition is deterministic and constrains admissible associations. Future learned screening must measure a no-Cell control rather than assume all controls have exactly uniform performance.

Gate 1 uses eight orthogonal keys, independent nonidentity target assignment, and 128 full-set writes with the unchanged frozen M3 update rule. Each retrieval ranks eight values by cosine. Its permutation control preserves the key and candidate sets and has no fixed points. The positive control requires eight correct retrievals before any E3 optimization. This fixed acquisition fixture is not a claim about all update schedules.

## Observed result

The dataset audit passed. The fixed K/V/Q positive control was BLOCKED at 5/8 correct versus 0/8 for the permuted Cell. Fast Weight delta was 4.7794. Full inner-loss and control histories are in cpu-preflight.json. Export/import parity for this fixture was downstream of the failed gate and remains pending. The frozen E3 adapter and candidate objective are implemented for the next gate. The E3 fixed-episode optimization, held-out screening, T4 and C0–C4 remain unexecuted.

EPCLM-006 remains INCONCLUSIVE. Its evaluation conflated unseen pairs with unseen symbols, included random-answer loss without a Cell, and used full-vocabulary decoding from a randomly initialized E3. This experiment uses shared symbol sets, candidate ranking, separated K/V/Q input paths and no Cell dropout. E3 is explicitly a frozen initialization. Neither the EPCLM-006 artifacts nor the frozen M3 update rule were changed. Diagnose inner acquisition before spending resources on E3.
