# Core Validation 009A Confirmation

- Status: `CONFIRMATION_INCOMPLETE`
- Scientific decision: `False`
- Locked split: `{'left_dim': 56, 'right_dim': 8}`
- Completed seeds: `[80911, 80912]`
- Missing seeds: `[80913]`

## Gate summary

|   seed |   left_dim |   right_dim |   train_median_local_action_residual |   eval_median_local_action_residual |   eval_median_frobenius_residual |   train_eval_local_action_gap |   rank1_eval_local_action_residual |   basis_parameter_count | heldout_action   | heldout_frobenius   | generalization_gap   | rank1_identity_guard   | budget   | pass   |
|-------:|-----------:|------------:|-------------------------------------:|------------------------------------:|---------------------------------:|------------------------------:|-----------------------------------:|------------------------:|:-----------------|:--------------------|:---------------------|:-----------------------|:---------|:-------|
|  80911 |         56 |           8 |                             0.179637 |                            0.302165 |                         0.338456 |                      0.122527 |                         0.0040982  |                    4096 | True             | True                | True                 | True                   | True     | True   |
|  80912 |         56 |           8 |                             0.17637  |                            0.313082 |                         0.341735 |                      0.136712 |                         0.00493993 |                    4096 | True             | True                | True                 | True                   | True     | True   |

009A confirmation concerns factor geometry only; routing, sparsity, certificates, growth and continual learning remain outside scope.
