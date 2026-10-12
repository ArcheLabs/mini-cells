# Core Validation 009A Bridge — Right-Side Collapse Robustness

- Status: `RIGHT_COLLAPSE_DIAGNOSTIC_COMPLETE`
- Scientific decision: `False` (diagnostic bridge by construction)
- Source 009A remains: `FACTORIZED_FUNCTIONAL_COORDINATES_SUPPORTED`
- Completed seeds: `[80911, 80912, 80913]`
- Missing seeds: `[]`

## Aggregate diagnostics

```json
{
  "centered_sequence_right_top1_energy": 0.12796883184222158,
  "maximum_source_reproduction_absolute_delta": 0.0,
  "mean_direction_removed_sequence_right_top1_energy": 0.11797817918855676,
  "raw_eval_two_sided_56x1_action_residual": 0.31044547830701963,
  "raw_eval_two_sided_56x8_action_residual": 0.3103777894725398,
  "raw_sequence_right_top1_energy": 0.9665393439146105,
  "raw_token_energy_weighted_right_top1_energy": 0.969114442003077,
  "raw_token_normalized_right_top1_energy": 0.9635393120036276,
  "top1_ablation_eval_residual_action_fraction": 0.007507349020233809,
  "top1_ablation_residual_right_top1_energy": 0.11707964857992785,
  "whitened_sequence_right_top1_energy": 0.022550776480442392
}
```

## Descriptive flags

```json
{
  "centering_sensitive": true,
  "mean_direction_sensitive": true,
  "post_top1_residual_still_low_dimensional": false,
  "raw_right_collapse_reproduced": true,
  "robust_common_right_direction_across_controls": false,
  "sequence_aggregation_sensitive": false,
  "top1_functionally_dominant": true,
  "whitening_sensitive": true
}
```

## Per-condition summary

|   seed | condition              |   sequence_right_top1_energy |   sequence_right_top8_energy |   sequence_right_participation_rank |   sequence_right_dim95 |   token_normalized_right_top1_energy |   token_normalized_right_top8_energy |   token_energy_weighted_right_top1_energy |   token_energy_weighted_right_top8_energy |   sequence_left_top56_energy |   token_count |   eval_right_only_n1_action_residual |   eval_two_sided_56x1_action_residual |   eval_right_only_n2_action_residual |   eval_two_sided_56x2_action_residual |   eval_right_only_n4_action_residual |   eval_two_sided_56x4_action_residual |   eval_right_only_n8_action_residual |   eval_two_sided_56x8_action_residual |
|-------:|:-----------------------|-----------------------------:|-----------------------------:|------------------------------------:|-----------------------:|-------------------------------------:|-------------------------------------:|------------------------------------------:|------------------------------------------:|-----------------------------:|--------------:|-------------------------------------:|--------------------------------------:|-------------------------------------:|--------------------------------------:|-------------------------------------:|--------------------------------------:|-------------------------------------:|--------------------------------------:|
|  80911 | raw                    |                    0.970198  |                     0.983318 |                             1.0623  |                      1 |                            0.966174  |                             0.979954 |                                 0.972661  |                                  0.98388  |                     0.963689 |         28436 |                           0.00628255 |                              0.305536 |                           0.00585535 |                              0.302228 |                           0.00478693 |                              0.302184 |                            0.0040791 |                              0.302165 |
|  80911 | centered               |                    0.127969  |                     0.329193 |                            32.1172  |                     58 |                            0.127027  |                             0.329716 |                                 0.303918  |                                  0.4964   |                     0.926642 |         28436 |                           0.844154   |                              0.853962 |                           0.797822   |                              0.816721 |                           0.717011   |                              0.74756  |                            0.631752  |                              0.66667  |
|  80911 | whitened               |                    0.0225508 |                     0.162251 |                            61.7192  |                     59 |                            0.0263991 |                             0.166887 |                                 0.0208504 |                                  0.154523 |                     0.92385  |         28436 |                           0.98514    |                              0.986572 |                           0.970001   |                              0.972425 |                           0.932243   |                              0.941256 |                            0.857652  |                              0.873022 |
|  80911 | mean_direction_removed |                    0.117978  |                     0.317371 |                            34.6623  |                     58 |                            0.119807  |                             0.318482 |                                 0.224634  |                                  0.430005 |                     0.92622  |         28436 |                           0.838989   |                              0.853938 |                           0.803949   |                              0.830888 |                           0.749613   |                              0.78536  |                            0.63671   |                              0.68037  |
|  80912 | raw                    |                    0.96232   |                     0.978398 |                             1.07973 |                      1 |                            0.958378  |                             0.975082 |                                 0.966477  |                                  0.979513 |                     0.965282 |         28436 |                           0.00777388 |                              0.313187 |                           0.00758775 |                              0.31318  |                           0.0059795  |                              0.313099 |                            0.0051135 |                              0.313082 |
|  80912 | centered               |                    0.104194  |                     0.316237 |                            37.2081  |                     58 |                            0.106486  |                             0.318827 |                                 0.290865  |                                  0.486032 |                     0.927355 |         28436 |                           0.871918   |                              0.875756 |                           0.797472   |                              0.819181 |                           0.732068   |                              0.763739 |                            0.62737   |                              0.661696 |
|  80912 | whitened               |                    0.0222447 |                     0.161789 |                            61.7443  |                     59 |                            0.024928  |                             0.166757 |                                 0.0212395 |                                  0.153552 |                     0.924915 |         28436 |                           0.983498   |                              0.984678 |                           0.968398   |                              0.970838 |                           0.941567   |                              0.948046 |                            0.866686  |                              0.880724 |
|  80912 | mean_direction_removed |                    0.104273  |                     0.310524 |                            37.6396  |                     58 |                            0.107016  |                             0.313149 |                                 0.187352  |                                  0.408474 |                     0.927012 |         28436 |                           0.873903   |                              0.879171 |                           0.806115   |                              0.817276 |                           0.762488   |                              0.776879 |                            0.630167  |                              0.663118 |
|  80913 | raw                    |                    0.966539  |                     0.982007 |                             1.07034 |                      1 |                            0.963539  |                             0.979274 |                                 0.969114  |                                  0.982926 |                     0.962321 |         28437 |                           0.00750735 |                              0.310445 |                           0.00686518 |                              0.310419 |                           0.00469016 |                              0.310399 |                            0.0041802 |                              0.310378 |
|  80913 | centered               |                    0.17256   |                     0.371677 |                            23.0505  |                     58 |                            0.173051  |                             0.375011 |                                 0.29246   |                                  0.519423 |                     0.927959 |         28437 |                           0.745795   |                              0.765063 |                           0.692124   |                              0.735441 |                           0.611332   |                              0.667145 |                            0.51271   |                              0.569635 |
|  80913 | whitened               |                    0.0231237 |                     0.16109  |                            61.704   |                     59 |                            0.0268024 |                             0.169146 |                                 0.0210322 |                                  0.154908 |                     0.923764 |         28437 |                           0.982896   |                              0.984389 |                           0.963568   |                              0.968391 |                           0.938387   |                              0.945231 |                            0.852203  |                              0.866277 |
|  80913 | mean_direction_removed |                    0.171584  |                     0.363736 |                            23.3294  |                     57 |                            0.172615  |                             0.367    |                                 0.226034  |                                  0.465672 |                     0.927896 |         28437 |                           0.755858   |                              0.778727 |                           0.70976    |                              0.73878  |                           0.648429   |                              0.68667  |                            0.531837  |                              0.581452 |

This bridge explains the 009A asymmetry only. It cannot revoke or strengthen the formal 009A support decision and does not test routing, sparsity, certificates, growth, or continual learning.
