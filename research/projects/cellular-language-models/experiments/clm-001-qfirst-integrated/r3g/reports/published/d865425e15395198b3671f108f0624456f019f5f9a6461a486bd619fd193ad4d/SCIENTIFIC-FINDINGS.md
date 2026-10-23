# R3g read-only Cell–Connection attribution

READONLY_ATTRIBUTION_AUDIT_COMPLETE

One training seed. Calibration was used by the parent to select an organization; it is not independent validation.
No optimizer updates, new candidates, new adaptation or Monte Carlo training replay. Development unread; Confirmation sealed.

## A-C

| Quantity | Value |
|---|---:|
| L_0O | 4.4168553575873375 |
| L_0N | 4.415545403957367 |
| L_OO | 4.415665343403816 |
| L_ON | 4.414222419261932 |
| L_NO | 4.415663152933121 |
| L_NN | 4.41421639919281 |
| route_effect_old_cells | -0.00144292414188385 |
| cell_effect_new_connection | -6.020069122314453e-06 |
| interaction | -3.829598426818848e-06 |
| numerical_resolution | 5.265301892265484e-07 |

Attribution: ['CELL_ADAPTATION_DETECTED', 'FAVORABLE_INTERACTION_DETECTED']
Cell parameters bitwise equal: False
Training/inference distribution assessment: {'old': {'median_normalized_entropy_averaged_over_epsilon': 0.9999999793274743, 'mean_hard_action_match_probability': 0.06671268897506773, 'expected_local_mismatch_probability': 0.9332873110249322, 'high_entropy_near_uniform': True, 'descriptive_thresholds': {'normalized_entropy_near_uniform_min': 0.99, 'hard_match_uniform_relative_deviation_max': 0.1, 'scope': 'DESCRIPTIVE_ONLY_NOT_AN_ADMISSION_OR_TRAINING_GATE'}, 'status': 'HIGH_ENTROPY_TRAIN_DETERMINISTIC_EVAL_GAP', 'statistical_confirmation': False, 'closed_loop_path_probability': False}, 'new': {'median_normalized_entropy_averaged_over_epsilon': 0.9999999790783132, 'mean_hard_action_match_probability': 0.06671277791833385, 'expected_local_mismatch_probability': 0.9332872220816661, 'high_entropy_near_uniform': True, 'descriptive_thresholds': {'normalized_entropy_near_uniform_min': 0.99, 'hard_match_uniform_relative_deviation_max': 0.1, 'scope': 'DESCRIPTIVE_ONLY_NOT_AN_ADMISSION_OR_TRAINING_GATE'}, 'status': 'HIGH_ENTROPY_TRAIN_DETERMINISTIC_EVAL_GAP', 'statistical_confirmation': False, 'closed_loop_path_probability': False}}
Next research recommendation: PRIORITIZE_TRAIN_INFERENCE_ROUTING_ALIGNMENT_STUDY
Training path difference steps: [108]
Training-distribution entropy and hard-action match probabilities are fixed-state descriptive measures, not closed-loop path agreement.
No functional specialization or general convergence claim follows from an interaction term.

## B-C

| Quantity | Value |
|---|---:|
| L_0O | 4.444146633148193 |
| L_0N | 4.444194599986076 |
| L_OO | 4.44657114893198 |
| L_ON | 4.445827133953571 |
| L_NO | 4.44657114893198 |
| L_NN | 4.445827133953571 |
| route_effect_old_cells | -0.0007440149784088135 |
| cell_effect_new_connection | 0.0 |
| interaction | 0.0 |
| numerical_resolution | 5.300725876011825e-07 |

Attribution: ['ROUTE_DOMINANT', 'NO_DETECTABLE_CELL_ADAPTATION']
Cell parameters bitwise equal: True
Training/inference distribution assessment: {'old': {'median_normalized_entropy_averaged_over_epsilon': 0.9999999747159706, 'mean_hard_action_match_probability': 0.1667513029780361, 'expected_local_mismatch_probability': 0.8332486970219639, 'high_entropy_near_uniform': True, 'descriptive_thresholds': {'normalized_entropy_near_uniform_min': 0.99, 'hard_match_uniform_relative_deviation_max': 0.1, 'scope': 'DESCRIPTIVE_ONLY_NOT_AN_ADMISSION_OR_TRAINING_GATE'}, 'status': 'HIGH_ENTROPY_TRAIN_DETERMINISTIC_EVAL_GAP', 'statistical_confirmation': False, 'closed_loop_path_probability': False}, 'new': {'median_normalized_entropy_averaged_over_epsilon': 0.9999999752559248, 'mean_hard_action_match_probability': 0.16675120782047245, 'expected_local_mismatch_probability': 0.8332487921795275, 'high_entropy_near_uniform': True, 'descriptive_thresholds': {'normalized_entropy_near_uniform_min': 0.99, 'hard_match_uniform_relative_deviation_max': 0.1, 'scope': 'DESCRIPTIVE_ONLY_NOT_AN_ADMISSION_OR_TRAINING_GATE'}, 'status': 'HIGH_ENTROPY_TRAIN_DETERMINISTIC_EVAL_GAP', 'statistical_confirmation': False, 'closed_loop_path_probability': False}}
Next research recommendation: PRIORITIZE_TRAIN_INFERENCE_ROUTING_ALIGNMENT_STUDY
Training path difference steps: []
Training-distribution entropy and hard-action match probabilities are fixed-state descriptive measures, not closed-loop path agreement.
No functional specialization or general convergence claim follows from an interaction term.

## Next research decision

If route-only benefit dominates, prioritize a separately versioned train/inference routing-alignment experiment.
If Cell adaptation or favorable interaction is resolved above numerical error, design multi-seed replication before claiming general benefit.
If evidence is incomplete or historical NLL does not reproduce, resolve the input/evaluation discrepancy first.
This audit never starts that next experiment.
