# Phase A.3 controlled interventions

A.3 interventions measured. Base runtime acceptance remains a separate A.4 gate.

| Arm | Schedule | Logits max absolute difference | Expert set agreement |
| --- | --- | ---: | ---: |
| original | prefill_decode | 15.63 | 76.39% |
| original | two_chunks | 8.0156 | 90.28% |
| original | tokenwise | 18.379 | 51.39% |
| packing_only | prefill_decode | 14.749 | 75.69% |
| packing_only | two_chunks | 9.0469 | 87.50% |
| packing_only | tokenwise | 18.473 | 51.39% |
| convolution_only | prefill_decode | 2.6875 | 97.22% |
| convolution_only | two_chunks | 3.6875 | 94.44% |
| convolution_only | tokenwise | 2.7656 | 96.53% |
| both | prefill_decode | 1.6348 | 100.00% |
| both | two_chunks | 0.84375 | 98.61% |
| both | tokenwise | 1.8428 | 100.00% |
