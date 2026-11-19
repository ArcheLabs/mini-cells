# Random binding diagnostic

| Checkpoint | Condition | Question probe Recall@1 | Same-address Recall@1 | Same-address swapped-value credit | LM candidate accuracy | Mean target log P |
|---|---|---:|---:|---:|---:|---:|
| seed_42_step_000 | correct | 0.234 | 0.219 | 0.281 | 0.250 | -5.99941 |
| seed_42_step_000 | swapped | 0.234 | 0.219 | 0.281 | 0.250 | -5.99938 |
| seed_42_step_000 | empty | 0.250 | 0.250 | 0.250 | 0.250 | -6.03675 |
| seed_42_step_200 | correct | 0.234 | 0.250 | 0.266 | 0.250 | -5.93735 |
| seed_42_step_200 | swapped | 0.234 | 0.250 | 0.266 | 0.250 | -5.93642 |
| seed_42_step_200 | empty | 0.250 | 0.250 | 0.250 | 0.250 | -6.03675 |
| seed_43_step_000 | correct | 0.250 | 0.266 | 0.219 | 0.250 | -6.02530 |
| seed_43_step_000 | swapped | 0.250 | 0.266 | 0.219 | 0.250 | -6.02561 |
| seed_43_step_000 | empty | 0.250 | 0.250 | 0.250 | 0.250 | -6.03675 |
| seed_43_step_200 | correct | 0.250 | 0.328 | 0.203 | 0.250 | -5.98631 |
| seed_43_step_200 | swapped | 0.250 | 0.328 | 0.203 | 0.250 | -5.98631 |
| seed_43_step_200 | empty | 0.250 | 0.250 | 0.250 | 0.250 | -6.03675 |
| seed_44_step_000 | correct | 0.250 | 0.297 | 0.266 | 0.250 | -6.05116 |
| seed_44_step_000 | swapped | 0.250 | 0.297 | 0.266 | 0.250 | -6.05134 |
| seed_44_step_000 | empty | 0.250 | 0.250 | 0.250 | 0.250 | -6.03675 |
| seed_44_step_200 | correct | 0.219 | 0.250 | 0.250 | 0.250 | -5.98603 |
| seed_44_step_200 | swapped | 0.219 | 0.250 | 0.250 | 0.250 | -5.98626 |
| seed_44_step_200 | empty | 0.250 | 0.250 | 0.250 | 0.250 | -6.03675 |

Direct probe: cosine against contextualized write V, repeated across the two Q heads per KV group.
This is a geometry probe; low scores alone cannot rule out information in another representation.
Empty has four tied retrieval candidates (0.25 credit). No outer training.
