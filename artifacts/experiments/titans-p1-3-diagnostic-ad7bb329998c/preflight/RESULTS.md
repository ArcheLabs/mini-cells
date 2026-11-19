# Random binding diagnostic

| Checkpoint | Condition | Question probe Recall@1 | Same-address Recall@1 | Same-address swapped-value credit | LM candidate accuracy | Mean target log P |
|---|---|---:|---:|---:|---:|---:|
| seed_42_step_200 | correct | 0.250 | 0.250 | 0.250 | 0.250 | -5.81968 |
| seed_42_step_200 | swapped | 0.250 | 0.250 | 0.250 | 0.250 | -5.81835 |
| seed_42_step_200 | empty | 0.250 | 0.250 | 0.250 | 0.250 | -5.88970 |

## Fixed K, swapped V

| Checkpoint | Queries | Mean margin shift toward swapped value | Positive shift rate |
|---|---:|---:|---:|
| seed_42_step_200 | 4 | 1.4943304e-05 | 100.00% |

Direct probe: cosine against contextualized write V, repeated across the two Q heads per KV group.
This is a geometry probe; low scores alone cannot rule out information in another representation.
Empty has four tied retrieval candidates (0.25 credit). No outer training.
