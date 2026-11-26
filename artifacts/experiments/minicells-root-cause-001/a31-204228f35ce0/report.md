# Phase A.3.1 expert convolution padding

A.3.1 padding intervention measured. A.4 remains a separate gate.

| Arm | Schedule | Logits max abs | Mean abs | In tolerance |
| --- | --- | ---: | ---: | --- |
| a3_both | prefill_decode | 1.6436 | 0.061143 | False |
| a3_both | two_chunks | 0.84375 | 0.067857 | False |
| a3_both | tokenwise | 1.873 | 0.061364 | False |
| skip_expert_padding | prefill_decode | 0.03125 | 0.0025377 | True |
| skip_expert_padding | two_chunks | 0.92969 | 0.048889 | False |
| skip_expert_padding | tokenwise | 0.023438 | 0.0025842 | True |
