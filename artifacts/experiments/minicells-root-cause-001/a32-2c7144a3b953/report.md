# Phase A.3.2 expert recurrence padding

A.3.2 expert recurrence intervention measured. A.4 remains a separate gate.

| Arm | Schedule | Logits max abs | Mean abs | In tolerance |
| --- | --- | ---: | ---: | --- |
| a31 | prefill_decode | 0.03125 | 0.0048521 | True |
| a31 | two_chunks | 0.91406 | 0.048493 | False |
| a31 | tokenwise | 0.023438 | 0.0043424 | True |
| skip_padded_delta | prefill_decode | 0.03125 | 0.0048521 | True |
| skip_padded_delta | two_chunks | 0.023438 | 0.0025996 | True |
| skip_padded_delta | tokenwise | 0.023438 | 0.0043424 | True |
