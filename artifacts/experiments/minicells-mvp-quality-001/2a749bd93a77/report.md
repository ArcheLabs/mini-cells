# MiniCells MVP quality

**FAIL**. The upper paired CI for automatic EM gain does not exceed five percentage points.

B uses prelabelled oracle selection. Lower NLL/PPL and higher exact match/F1 are better.
Paired intervals resample questions 2,000 times. The frozen primary margin is +5 pp EM for D − A on related queries. Unrelated queries and pair-versus-single are secondary.

## smoke

| Group | Arm | n | Mean NLL | Mean PPL | Exact Match |
| --- | --- | ---: | ---: | ---: | ---: |
| all | A | 9 | 5.1948 | 286.1302 | 0.0000 |
| all | B | 9 | 5.4367 | 344.7333 | 0.0000 |
| all | C | 9 | 5.8443 | 594.7933 | 0.0000 |
| all | D | 9 | 6.0980 | 949.9207 | 0.0000 |
| relevant | A | 6 | 5.7669 | 398.7511 | 0.0000 |
| relevant | B | 6 | 6.1296 | 486.6558 | 0.0000 |
| relevant | C | 6 | 6.4347 | 837.0661 | 0.0000 |
| relevant | D | 6 | 6.7062 | 1339.6795 | 0.0000 |
| single_cell | A | 3 | 5.1751 | 194.6748 | 0.0000 |
| single_cell | B | 3 | 5.9359 | 403.1891 | 0.0000 |
| single_cell | C | 3 | 5.7838 | 332.2526 | 0.0000 |
| single_cell | D | 3 | 6.4773 | 1554.3598 | 0.0000 |
| cross_cell | A | 3 | 6.3587 | 602.8274 | 0.0000 |
| cross_cell | B | 3 | 6.3234 | 570.1226 | 0.0000 |
| cross_cell | C | 3 | 7.0855 | 1341.8797 | 0.0000 |
| cross_cell | D | 3 | 6.9350 | 1124.9993 | 0.0000 |
| irrelevant | A | 3 | 4.0507 | 60.8882 | 0.0000 |
| irrelevant | B | 3 | 4.0507 | 60.8882 | 0.0000 |
| irrelevant | C | 3 | 4.6636 | 110.2476 | 0.0000 |
| irrelevant | D | 3 | 4.8818 | 170.4029 | 0.0000 |

| Group | Contrast | Metric | n | Mean | Paired 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| all | B-A | mean_target_nll | 9 | 0.2418 | [-0.0957, 0.6874] |
| all | B-A | target_perplexity | 9 | 58.6032 | [-67.1018, 195.9244] |
| all | B-A | exact_match | 9 | 0.0000 | [0.0000, 0.0000] |
| all | B-A | token_f1 | 9 | 0.0000 | [0.0000, 0.0000] |
| all | B-A | constraint_satisfied | 9 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | mean_target_nll | 9 | 0.6495 | [0.2337, 1.0528] |
| all | C-A | target_perplexity | 9 | 308.6631 | [31.7722, 630.0932] |
| all | C-A | exact_match | 9 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | token_f1 | 9 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | constraint_satisfied | 9 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | mean_target_nll | 9 | 0.9032 | [0.2889, 1.6934] |
| all | D-A | target_perplexity | 9 | 663.7905 | [72.4507, 1556.6301] |
| all | D-A | exact_match | 9 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | token_f1 | 9 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | constraint_satisfied | 9 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | mean_target_nll | 9 | -0.6614 | [-1.0815, -0.2221] |
| all | B-D | target_perplexity | 9 | -605.1873 | [-1382.7710, -76.6977] |
| all | B-D | exact_match | 9 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | token_f1 | 9 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | constraint_satisfied | 9 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | mean_target_nll | 3 | -0.0699 | [-0.3679, 0.1516] |
| all | B-best_single | target_perplexity | 3 | -60.9378 | [-279.8215, 94.3435] |
| all | B-best_single | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | mean_target_nll | 6 | 0.3628 | [-0.1481, 1.0036] |
| relevant | B-A | target_perplexity | 6 | 87.9047 | [-104.5068, 283.2473] |
| relevant | B-A | exact_match | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | token_f1 | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | constraint_satisfied | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | mean_target_nll | 6 | 0.6678 | [0.1486, 1.1747] |
| relevant | C-A | target_perplexity | 6 | 438.3150 | [34.2276, 873.4575] |
| relevant | C-A | exact_match | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | token_f1 | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | constraint_satisfied | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | mean_target_nll | 6 | 0.9393 | [0.0522, 2.1687] |
| relevant | D-A | target_perplexity | 6 | 940.9284 | [82.1636, 2249.6117] |
| relevant | D-A | exact_match | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | token_f1 | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | constraint_satisfied | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | mean_target_nll | 6 | -0.5765 | [-1.1950, -0.0318] |
| relevant | B-D | target_perplexity | 6 | -853.0237 | [-1954.7225, -81.0967] |
| relevant | B-D | exact_match | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | token_f1 | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | constraint_satisfied | 6 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | mean_target_nll | 3 | -0.0699 | [-0.3679, 0.1516] |
| relevant | B-best_single | target_perplexity | 3 | -60.9378 | [-279.8215, 94.3435] |
| relevant | B-best_single | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | mean_target_nll | 3 | 0.7609 | [0.1373, 1.8009] |
| single_cell | B-A | target_perplexity | 3 | 208.5142 | [30.3211, 478.7198] |
| single_cell | B-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | mean_target_nll | 3 | 0.6088 | [0.1541, 1.4214] |
| single_cell | C-A | target_perplexity | 3 | 137.5778 | [34.3240, 297.6472] |
| single_cell | C-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | mean_target_nll | 3 | 1.3022 | [-0.2869, 3.7611] |
| single_cell | D-A | target_perplexity | 3 | 1359.6850 | [-51.3753, 3977.0515] |
| single_cell | D-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | mean_target_nll | 3 | -0.5414 | [-1.9602, 0.4243] |
| single_cell | B-D | target_perplexity | 3 | -1151.1707 | [-3498.3317, 81.6963] |
| single_cell | B-D | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | mean_target_nll | 3 | -0.0353 | [-0.5026, 0.5762] |
| cross_cell | B-A | target_perplexity | 3 | -32.7047 | [-268.0616, 293.7101] |
| cross_cell | B-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | mean_target_nll | 3 | 0.7268 | [-0.1691, 1.4766] |
| cross_cell | C-A | target_perplexity | 3 | 739.0523 | [-105.5685, 1273.2858] |
| cross_cell | C-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | mean_target_nll | 3 | 0.5763 | [-0.1691, 1.1057] |
| cross_cell | D-A | target_perplexity | 3 | 522.1719 | [-105.5685, 910.2073] |
| cross_cell | D-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | mean_target_nll | 3 | -0.6117 | [-0.9720, -0.3335] |
| cross_cell | B-D | target_perplexity | 3 | -554.8766 | [-1033.9700, -162.4932] |
| cross_cell | B-D | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | mean_target_nll | 3 | -0.0699 | [-0.3679, 0.1516] |
| cross_cell | B-best_single | target_perplexity | 3 | -60.9378 | [-279.8215, 94.3435] |
| cross_cell | B-best_single | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | mean_target_nll | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | target_perplexity | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | mean_target_nll | 3 | 0.6128 | [-0.2077, 1.2966] |
| irrelevant | C-A | target_perplexity | 3 | 49.3594 | [-16.8842, 106.1441] |
| irrelevant | C-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | mean_target_nll | 3 | 0.8310 | [0.4723, 1.3369] |
| irrelevant | D-A | target_perplexity | 3 | 109.5146 | [24.1148, 252.6943] |
| irrelevant | D-A | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | mean_target_nll | 3 | -0.8310 | [-1.3369, -0.4723] |
| irrelevant | B-D | target_perplexity | 3 | -109.5146 | [-252.6943, -24.1148] |
| irrelevant | B-D | exact_match | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | token_f1 | 3 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | constraint_satisfied | 3 | 0.0000 | [0.0000, 0.0000] |

| Group | Routing metric | n | Mean |
| --- | --- | ---: | ---: |
| all | recall_at_2 | 6 | 0.0000 |
| all | precision_at_2 | 6 | 0.0000 |
| all | hit_at_2 | 6 | 0.0000 |
| all | mrr | 6 | 0.2530 |
| all | ndcg_at_2 | 6 | 0.0000 |
| all | all_relevant_selected | 6 | 0.0000 |
| all | false_selection | 3 | 1.0000 |
| all | false_selection_count | 3 | 2.0000 |
| all | glassworks hit | 2 | 0.0000 |
| all | railway hit | 3 | 0.0000 |
| all | observatory hit | 3 | 0.0000 |
| all | archive hit | 1 | 0.0000 |
| relevant | recall_at_2 | 6 | 0.0000 |
| relevant | precision_at_2 | 6 | 0.0000 |
| relevant | hit_at_2 | 6 | 0.0000 |
| relevant | mrr | 6 | 0.2530 |
| relevant | ndcg_at_2 | 6 | 0.0000 |
| relevant | all_relevant_selected | 6 | 0.0000 |
| relevant | glassworks hit | 2 | 0.0000 |
| relevant | railway hit | 3 | 0.0000 |
| relevant | observatory hit | 3 | 0.0000 |
| relevant | archive hit | 1 | 0.0000 |
| single_cell | recall_at_2 | 3 | 0.0000 |
| single_cell | precision_at_2 | 3 | 0.0000 |
| single_cell | hit_at_2 | 3 | 0.0000 |
| single_cell | mrr | 3 | 0.2004 |
| single_cell | ndcg_at_2 | 3 | 0.0000 |
| single_cell | all_relevant_selected | 3 | 0.0000 |
| single_cell | glassworks hit | 1 | 0.0000 |
| single_cell | railway hit | 1 | 0.0000 |
| single_cell | observatory hit | 1 | 0.0000 |
| cross_cell | recall_at_2 | 3 | 0.0000 |
| cross_cell | precision_at_2 | 3 | 0.0000 |
| cross_cell | hit_at_2 | 3 | 0.0000 |
| cross_cell | mrr | 3 | 0.3056 |
| cross_cell | ndcg_at_2 | 3 | 0.0000 |
| cross_cell | all_relevant_selected | 3 | 0.0000 |
| cross_cell | glassworks hit | 1 | 0.0000 |
| cross_cell | railway hit | 2 | 0.0000 |
| cross_cell | observatory hit | 2 | 0.0000 |
| cross_cell | archive hit | 1 | 0.0000 |
| irrelevant | false_selection | 3 | 1.0000 |
| irrelevant | false_selection_count | 3 | 2.0000 |

Accuracy and NLL by whether D selected all relevant Cells

| Group | Conditional metric | n | Mean |
| --- | --- | ---: | ---: |
| all | missing_support_A_exact_match | 6 | 0.0000 |
| all | missing_support_A_mean_nll | 6 | 5.7669 |
| all | missing_support_B_exact_match | 6 | 0.0000 |
| all | missing_support_B_mean_nll | 6 | 6.1296 |
| all | missing_support_C_exact_match | 6 | 0.0000 |
| all | missing_support_C_mean_nll | 6 | 6.4347 |
| all | missing_support_D_exact_match | 6 | 0.0000 |
| all | missing_support_D_mean_nll | 6 | 6.7062 |
| all | not_applicable_A_exact_match | 3 | 0.0000 |
| all | not_applicable_A_mean_nll | 3 | 4.0507 |
| all | not_applicable_B_exact_match | 3 | 0.0000 |
| all | not_applicable_B_mean_nll | 3 | 4.0507 |
| all | not_applicable_C_exact_match | 3 | 0.0000 |
| all | not_applicable_C_mean_nll | 3 | 4.6636 |
| all | not_applicable_D_exact_match | 3 | 0.0000 |
| all | not_applicable_D_mean_nll | 3 | 4.8818 |
| relevant | missing_support_A_exact_match | 6 | 0.0000 |
| relevant | missing_support_A_mean_nll | 6 | 5.7669 |
| relevant | missing_support_B_exact_match | 6 | 0.0000 |
| relevant | missing_support_B_mean_nll | 6 | 6.1296 |
| relevant | missing_support_C_exact_match | 6 | 0.0000 |
| relevant | missing_support_C_mean_nll | 6 | 6.4347 |
| relevant | missing_support_D_exact_match | 6 | 0.0000 |
| relevant | missing_support_D_mean_nll | 6 | 6.7062 |
| single_cell | missing_support_A_exact_match | 3 | 0.0000 |
| single_cell | missing_support_A_mean_nll | 3 | 5.1751 |
| single_cell | missing_support_B_exact_match | 3 | 0.0000 |
| single_cell | missing_support_B_mean_nll | 3 | 5.9359 |
| single_cell | missing_support_C_exact_match | 3 | 0.0000 |
| single_cell | missing_support_C_mean_nll | 3 | 5.7838 |
| single_cell | missing_support_D_exact_match | 3 | 0.0000 |
| single_cell | missing_support_D_mean_nll | 3 | 6.4773 |
| cross_cell | missing_support_A_exact_match | 3 | 0.0000 |
| cross_cell | missing_support_A_mean_nll | 3 | 6.3587 |
| cross_cell | missing_support_B_exact_match | 3 | 0.0000 |
| cross_cell | missing_support_B_mean_nll | 3 | 6.3234 |
| cross_cell | missing_support_C_exact_match | 3 | 0.0000 |
| cross_cell | missing_support_C_mean_nll | 3 | 7.0855 |
| cross_cell | missing_support_D_exact_match | 3 | 0.0000 |
| cross_cell | missing_support_D_mean_nll | 3 | 6.9350 |
| irrelevant | not_applicable_A_exact_match | 3 | 0.0000 |
| irrelevant | not_applicable_A_mean_nll | 3 | 4.0507 |
| irrelevant | not_applicable_B_exact_match | 3 | 0.0000 |
| irrelevant | not_applicable_B_mean_nll | 3 | 4.0507 |
| irrelevant | not_applicable_C_exact_match | 3 | 0.0000 |
| irrelevant | not_applicable_C_mean_nll | 3 | 4.6636 |
| irrelevant | not_applicable_D_exact_match | 3 | 0.0000 |
| irrelevant | not_applicable_D_mean_nll | 3 | 4.8818 |

## development

| Group | Arm | n | Mean NLL | Mean PPL | Exact Match |
| --- | --- | ---: | ---: | ---: | ---: |
| all | A | 56 | 5.1712 | 323.3953 | 0.0000 |
| all | B | 56 | 6.1037 | 766.5571 | 0.0000 |
| all | C | 56 | 6.0351 | 742.4905 | 0.0000 |
| all | D | 56 | 6.1916 | 805.9814 | 0.0000 |
| relevant | A | 48 | 5.2430 | 356.9492 | 0.0000 |
| relevant | B | 48 | 6.3309 | 873.9712 | 0.0000 |
| relevant | C | 48 | 6.1770 | 820.0070 | 0.0000 |
| relevant | D | 48 | 6.3501 | 883.6154 | 0.0000 |
| single_cell | A | 40 | 4.9267 | 189.6152 | 0.0000 |
| single_cell | B | 40 | 6.2768 | 830.9967 | 0.0000 |
| single_cell | C | 40 | 6.0554 | 653.9028 | 0.0000 |
| single_cell | D | 40 | 6.3526 | 916.6622 | 0.0000 |
| cross_cell | A | 8 | 6.8249 | 1193.6189 | 0.0000 |
| cross_cell | B | 8 | 6.6019 | 1088.8441 | 0.0000 |
| cross_cell | C | 8 | 6.7848 | 1650.5283 | 0.0000 |
| cross_cell | D | 8 | 6.3375 | 718.3817 | 0.0000 |
| irrelevant | A | 8 | 4.7399 | 122.0723 | 0.0000 |
| irrelevant | B | 8 | 4.7399 | 122.0723 | 0.0000 |
| irrelevant | C | 8 | 5.1840 | 277.3913 | 0.0000 |
| irrelevant | D | 8 | 5.2408 | 340.1772 | 0.0000 |

| Group | Contrast | Metric | n | Mean | Paired 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| all | B-A | mean_target_nll | 56 | 0.9325 | [0.6427, 1.2367] |
| all | B-A | target_perplexity | 56 | 443.1618 | [183.8596, 733.9749] |
| all | B-A | exact_match | 56 | 0.0000 | [0.0000, 0.0000] |
| all | B-A | token_f1 | 56 | 0.0000 | [0.0000, 0.0000] |
| all | B-A | constraint_satisfied | 56 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | mean_target_nll | 56 | 0.8640 | [0.5844, 1.1671] |
| all | C-A | target_perplexity | 56 | 419.0952 | [209.3587, 666.4569] |
| all | C-A | exact_match | 56 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | token_f1 | 56 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | constraint_satisfied | 56 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | mean_target_nll | 56 | 1.0204 | [0.7090, 1.3515] |
| all | D-A | target_perplexity | 56 | 482.5860 | [235.2056, 789.7761] |
| all | D-A | exact_match | 56 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | token_f1 | 56 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | constraint_satisfied | 56 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | mean_target_nll | 56 | -0.0879 | [-0.3575, 0.1765] |
| all | B-D | target_perplexity | 56 | -39.4243 | [-331.4702, 273.3403] |
| all | B-D | exact_match | 56 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | token_f1 | 56 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | constraint_satisfied | 56 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | mean_target_nll | 8 | 0.1305 | [-0.1741, 0.5378] |
| all | B-best_single | target_perplexity | 8 | 364.8656 | [-107.8594, 1041.0382] |
| all | B-best_single | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | mean_target_nll | 48 | 1.0879 | [0.7428, 1.4425] |
| relevant | B-A | target_perplexity | 48 | 517.0220 | [205.7676, 859.3322] |
| relevant | B-A | exact_match | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | token_f1 | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | constraint_satisfied | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | mean_target_nll | 48 | 0.9340 | [0.6167, 1.2480] |
| relevant | C-A | target_perplexity | 48 | 463.0579 | [211.7783, 734.2756] |
| relevant | C-A | exact_match | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | token_f1 | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | constraint_satisfied | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | mean_target_nll | 48 | 1.1070 | [0.7443, 1.4659] |
| relevant | D-A | target_perplexity | 48 | 526.6663 | [239.3894, 886.6351] |
| relevant | D-A | exact_match | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | token_f1 | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | constraint_satisfied | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | mean_target_nll | 48 | -0.0191 | [-0.2963, 0.2689] |
| relevant | B-D | target_perplexity | 48 | -9.6442 | [-327.9198, 342.9676] |
| relevant | B-D | exact_match | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | token_f1 | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | constraint_satisfied | 48 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | mean_target_nll | 8 | 0.1305 | [-0.1741, 0.5378] |
| relevant | B-best_single | target_perplexity | 8 | 364.8656 | [-107.8594, 1041.0382] |
| relevant | B-best_single | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | mean_target_nll | 40 | 1.3501 | [1.0174, 1.6895] |
| single_cell | B-A | target_perplexity | 40 | 641.3814 | [346.2654, 986.2387] |
| single_cell | B-A | exact_match | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | token_f1 | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | constraint_satisfied | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | mean_target_nll | 40 | 1.1288 | [0.7984, 1.4695] |
| single_cell | C-A | target_perplexity | 40 | 464.2876 | [256.2785, 701.1502] |
| single_cell | C-A | exact_match | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | token_f1 | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | constraint_satisfied | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | mean_target_nll | 40 | 1.4259 | [1.0835, 1.7564] |
| single_cell | D-A | target_perplexity | 40 | 727.0469 | [417.2093, 1106.2436] |
| single_cell | D-A | exact_match | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | token_f1 | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | constraint_satisfied | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | mean_target_nll | 40 | -0.0758 | [-0.3841, 0.2307] |
| single_cell | B-D | target_perplexity | 40 | -85.6655 | [-427.7851, 290.6313] |
| single_cell | B-D | exact_match | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | token_f1 | 40 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | constraint_satisfied | 40 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | mean_target_nll | 8 | -0.2231 | [-0.7585, 0.3662] |
| cross_cell | B-A | target_perplexity | 8 | -104.7748 | [-892.6330, 937.8381] |
| cross_cell | B-A | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | mean_target_nll | 8 | -0.0401 | [-0.6889, 0.6079] |
| cross_cell | C-A | target_perplexity | 8 | 456.9094 | [-596.4675, 1755.0346] |
| cross_cell | C-A | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | mean_target_nll | 8 | -0.4874 | [-0.8723, -0.0828] |
| cross_cell | D-A | target_perplexity | 8 | -475.2372 | [-797.0627, -172.8885] |
| cross_cell | D-A | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | mean_target_nll | 8 | 0.2644 | [-0.3305, 0.9604] |
| cross_cell | B-D | target_perplexity | 8 | 370.4623 | [-374.9294, 1436.0646] |
| cross_cell | B-D | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | mean_target_nll | 8 | 0.1305 | [-0.1741, 0.5378] |
| cross_cell | B-best_single | target_perplexity | 8 | 364.8656 | [-107.8594, 1041.0382] |
| cross_cell | B-best_single | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | mean_target_nll | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | target_perplexity | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | mean_target_nll | 8 | 0.4441 | [-0.1143, 1.0569] |
| irrelevant | C-A | target_perplexity | 8 | 155.3190 | [5.5100, 368.1719] |
| irrelevant | C-A | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | mean_target_nll | 8 | 0.5009 | [-0.1087, 1.1763] |
| irrelevant | D-A | target_perplexity | 8 | 218.1048 | [2.5271, 570.3556] |
| irrelevant | D-A | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | mean_target_nll | 8 | -0.5009 | [-1.1728, 0.1093] |
| irrelevant | B-D | target_perplexity | 8 | -218.1048 | [-570.0956, -2.4600] |
| irrelevant | B-D | exact_match | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | token_f1 | 8 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | constraint_satisfied | 8 | 0.0000 | [0.0000, 0.0000] |

| Group | Routing metric | n | Mean |
| --- | --- | ---: | ---: |
| all | recall_at_2 | 48 | 0.1250 |
| all | precision_at_2 | 48 | 0.0833 |
| all | hit_at_2 | 48 | 0.1458 |
| all | mrr | 48 | 0.2924 |
| all | ndcg_at_2 | 48 | 0.1096 |
| all | all_relevant_selected | 48 | 0.1042 |
| all | false_selection | 8 | 1.0000 |
| all | false_selection_count | 8 | 2.0000 |
| all | glassworks hit | 7 | 0.0000 |
| all | railway hit | 7 | 0.0000 |
| all | observatory hit | 7 | 0.0000 |
| all | archive hit | 7 | 0.0000 |
| all | greenhouse hit | 7 | 0.0000 |
| all | lighthouse hit | 7 | 0.0000 |
| all | foundry hit | 7 | 0.8571 |
| all | harbor hit | 7 | 0.2857 |
| relevant | recall_at_2 | 48 | 0.1250 |
| relevant | precision_at_2 | 48 | 0.0833 |
| relevant | hit_at_2 | 48 | 0.1458 |
| relevant | mrr | 48 | 0.2924 |
| relevant | ndcg_at_2 | 48 | 0.1096 |
| relevant | all_relevant_selected | 48 | 0.1042 |
| relevant | glassworks hit | 7 | 0.0000 |
| relevant | railway hit | 7 | 0.0000 |
| relevant | observatory hit | 7 | 0.0000 |
| relevant | archive hit | 7 | 0.0000 |
| relevant | greenhouse hit | 7 | 0.0000 |
| relevant | lighthouse hit | 7 | 0.0000 |
| relevant | foundry hit | 7 | 0.8571 |
| relevant | harbor hit | 7 | 0.2857 |
| single_cell | recall_at_2 | 40 | 0.1000 |
| single_cell | precision_at_2 | 40 | 0.0500 |
| single_cell | hit_at_2 | 40 | 0.1000 |
| single_cell | mrr | 40 | 0.2569 |
| single_cell | ndcg_at_2 | 40 | 0.0815 |
| single_cell | all_relevant_selected | 40 | 0.1000 |
| single_cell | glassworks hit | 5 | 0.0000 |
| single_cell | railway hit | 5 | 0.0000 |
| single_cell | observatory hit | 5 | 0.0000 |
| single_cell | archive hit | 5 | 0.0000 |
| single_cell | greenhouse hit | 5 | 0.0000 |
| single_cell | lighthouse hit | 5 | 0.0000 |
| single_cell | foundry hit | 5 | 0.8000 |
| single_cell | harbor hit | 5 | 0.0000 |
| cross_cell | recall_at_2 | 8 | 0.2500 |
| cross_cell | precision_at_2 | 8 | 0.2500 |
| cross_cell | hit_at_2 | 8 | 0.3750 |
| cross_cell | mrr | 8 | 0.4699 |
| cross_cell | ndcg_at_2 | 8 | 0.2500 |
| cross_cell | all_relevant_selected | 8 | 0.1250 |
| cross_cell | glassworks hit | 2 | 0.0000 |
| cross_cell | railway hit | 2 | 0.0000 |
| cross_cell | observatory hit | 2 | 0.0000 |
| cross_cell | archive hit | 2 | 0.0000 |
| cross_cell | greenhouse hit | 2 | 0.0000 |
| cross_cell | lighthouse hit | 2 | 0.0000 |
| cross_cell | foundry hit | 2 | 1.0000 |
| cross_cell | harbor hit | 2 | 1.0000 |
| irrelevant | false_selection | 8 | 1.0000 |
| irrelevant | false_selection_count | 8 | 2.0000 |

Accuracy and NLL by whether D selected all relevant Cells

| Group | Conditional metric | n | Mean |
| --- | --- | ---: | ---: |
| all | missing_support_A_exact_match | 43 | 0.0000 |
| all | missing_support_A_mean_nll | 43 | 5.2794 |
| all | missing_support_B_exact_match | 43 | 0.0000 |
| all | missing_support_B_mean_nll | 43 | 6.3393 |
| all | missing_support_C_exact_match | 43 | 0.0000 |
| all | missing_support_C_mean_nll | 43 | 6.1648 |
| all | missing_support_D_exact_match | 43 | 0.0000 |
| all | missing_support_D_mean_nll | 43 | 6.3275 |
| all | all_support_A_exact_match | 5 | 0.0000 |
| all | all_support_A_mean_nll | 5 | 4.9304 |
| all | all_support_B_exact_match | 5 | 0.0000 |
| all | all_support_B_mean_nll | 5 | 6.2588 |
| all | all_support_C_exact_match | 5 | 0.0000 |
| all | all_support_C_mean_nll | 5 | 6.2821 |
| all | all_support_D_exact_match | 5 | 0.0000 |
| all | all_support_D_mean_nll | 5 | 6.5439 |
| all | not_applicable_A_exact_match | 8 | 0.0000 |
| all | not_applicable_A_mean_nll | 8 | 4.7399 |
| all | not_applicable_B_exact_match | 8 | 0.0000 |
| all | not_applicable_B_mean_nll | 8 | 4.7399 |
| all | not_applicable_C_exact_match | 8 | 0.0000 |
| all | not_applicable_C_mean_nll | 8 | 5.1840 |
| all | not_applicable_D_exact_match | 8 | 0.0000 |
| all | not_applicable_D_mean_nll | 8 | 5.2408 |
| relevant | missing_support_A_exact_match | 43 | 0.0000 |
| relevant | missing_support_A_mean_nll | 43 | 5.2794 |
| relevant | missing_support_B_exact_match | 43 | 0.0000 |
| relevant | missing_support_B_mean_nll | 43 | 6.3393 |
| relevant | missing_support_C_exact_match | 43 | 0.0000 |
| relevant | missing_support_C_mean_nll | 43 | 6.1648 |
| relevant | missing_support_D_exact_match | 43 | 0.0000 |
| relevant | missing_support_D_mean_nll | 43 | 6.3275 |
| relevant | all_support_A_exact_match | 5 | 0.0000 |
| relevant | all_support_A_mean_nll | 5 | 4.9304 |
| relevant | all_support_B_exact_match | 5 | 0.0000 |
| relevant | all_support_B_mean_nll | 5 | 6.2588 |
| relevant | all_support_C_exact_match | 5 | 0.0000 |
| relevant | all_support_C_mean_nll | 5 | 6.2821 |
| relevant | all_support_D_exact_match | 5 | 0.0000 |
| relevant | all_support_D_mean_nll | 5 | 6.5439 |
| single_cell | missing_support_A_exact_match | 36 | 0.0000 |
| single_cell | missing_support_A_mean_nll | 36 | 4.9960 |
| single_cell | missing_support_B_exact_match | 36 | 0.0000 |
| single_cell | missing_support_B_mean_nll | 36 | 6.3069 |
| single_cell | missing_support_C_exact_match | 36 | 0.0000 |
| single_cell | missing_support_C_mean_nll | 36 | 6.0955 |
| single_cell | missing_support_D_exact_match | 36 | 0.0000 |
| single_cell | missing_support_D_mean_nll | 36 | 6.3515 |
| single_cell | all_support_A_exact_match | 4 | 0.0000 |
| single_cell | all_support_A_mean_nll | 4 | 4.3028 |
| single_cell | all_support_B_exact_match | 4 | 0.0000 |
| single_cell | all_support_B_mean_nll | 4 | 6.0056 |
| single_cell | all_support_C_exact_match | 4 | 0.0000 |
| single_cell | all_support_C_mean_nll | 4 | 5.6947 |
| single_cell | all_support_D_exact_match | 4 | 0.0000 |
| single_cell | all_support_D_mean_nll | 4 | 6.3619 |
| cross_cell | missing_support_A_exact_match | 7 | 0.0000 |
| cross_cell | missing_support_A_mean_nll | 7 | 6.7370 |
| cross_cell | missing_support_B_exact_match | 7 | 0.0000 |
| cross_cell | missing_support_B_mean_nll | 7 | 6.5062 |
| cross_cell | missing_support_C_exact_match | 7 | 0.0000 |
| cross_cell | missing_support_C_mean_nll | 7 | 6.5210 |
| cross_cell | missing_support_D_exact_match | 7 | 0.0000 |
| cross_cell | missing_support_D_mean_nll | 7 | 6.2040 |
| cross_cell | all_support_A_exact_match | 1 | 0.0000 |
| cross_cell | all_support_A_mean_nll | 1 | 7.4407 |
| cross_cell | all_support_B_exact_match | 1 | 0.0000 |
| cross_cell | all_support_B_mean_nll | 1 | 7.2717 |
| cross_cell | all_support_C_exact_match | 1 | 0.0000 |
| cross_cell | all_support_C_mean_nll | 1 | 8.6316 |
| cross_cell | all_support_D_exact_match | 1 | 0.0000 |
| cross_cell | all_support_D_mean_nll | 1 | 7.2717 |
| irrelevant | not_applicable_A_exact_match | 8 | 0.0000 |
| irrelevant | not_applicable_A_mean_nll | 8 | 4.7399 |
| irrelevant | not_applicable_B_exact_match | 8 | 0.0000 |
| irrelevant | not_applicable_B_mean_nll | 8 | 4.7399 |
| irrelevant | not_applicable_C_exact_match | 8 | 0.0000 |
| irrelevant | not_applicable_C_mean_nll | 8 | 5.1840 |
| irrelevant | not_applicable_D_exact_match | 8 | 0.0000 |
| irrelevant | not_applicable_D_mean_nll | 8 | 5.2408 |

## held_out

| Group | Arm | n | Mean NLL | Mean PPL | Exact Match |
| --- | --- | ---: | ---: | ---: | ---: |
| all | A | 128 | 5.7334 | 529.7728 | 0.0000 |
| all | B | 128 | 6.4747 | 1143.1209 | 0.0000 |
| all | C | 128 | 6.1686 | 661.7741 | 0.0000 |
| all | D | 128 | 6.1464 | 681.5018 | 0.0000 |
| relevant | A | 104 | 5.7631 | 505.7351 | 0.0000 |
| relevant | B | 104 | 6.6754 | 1260.6251 | 0.0000 |
| relevant | C | 104 | 6.3370 | 728.7929 | 0.0000 |
| relevant | D | 104 | 6.4305 | 799.4907 | 0.0000 |
| single_cell | A | 80 | 5.5613 | 421.0132 | 0.0000 |
| single_cell | B | 80 | 6.7871 | 1446.8273 | 0.0000 |
| single_cell | C | 80 | 6.3418 | 739.1328 | 0.0000 |
| single_cell | D | 80 | 6.4284 | 809.6264 | 0.0000 |
| cross_cell | A | 24 | 6.4358 | 788.1413 | 0.0000 |
| cross_cell | B | 24 | 6.3034 | 639.9512 | 0.0000 |
| cross_cell | C | 24 | 6.3209 | 694.3267 | 0.0000 |
| cross_cell | D | 24 | 6.4372 | 765.7048 | 0.0000 |
| irrelevant | A | 24 | 5.6048 | 633.9361 | 0.0000 |
| irrelevant | B | 24 | 5.6048 | 633.9361 | 0.0000 |
| irrelevant | C | 24 | 5.4388 | 371.3591 | 0.0000 |
| irrelevant | D | 24 | 4.9155 | 170.2167 | 0.0000 |

| Group | Contrast | Metric | n | Mean | Paired 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| all | B-A | mean_target_nll | 128 | 0.7413 | [0.5524, 0.9397] |
| all | B-A | target_perplexity | 128 | 613.3482 | [375.8222, 920.0608] |
| all | B-A | exact_match | 128 | 0.0000 | [0.0000, 0.0000] |
| all | B-A | token_f1 | 128 | 0.0000 | [0.0000, 0.0000] |
| all | B-A | constraint_satisfied | 128 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | mean_target_nll | 128 | 0.4352 | [0.2475, 0.6249] |
| all | C-A | target_perplexity | 128 | 132.0013 | [-33.9213, 285.1592] |
| all | C-A | exact_match | 128 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | token_f1 | 128 | 0.0000 | [0.0000, 0.0000] |
| all | C-A | constraint_satisfied | 128 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | mean_target_nll | 128 | 0.4130 | [0.2122, 0.6046] |
| all | D-A | target_perplexity | 128 | 151.7290 | [-20.0276, 307.6621] |
| all | D-A | exact_match | 128 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | token_f1 | 128 | 0.0000 | [0.0000, 0.0000] |
| all | D-A | constraint_satisfied | 128 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | mean_target_nll | 128 | 0.3283 | [0.1814, 0.4979] |
| all | B-D | target_perplexity | 128 | 461.6191 | [198.5744, 805.7376] |
| all | B-D | exact_match | 128 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | token_f1 | 128 | 0.0000 | [0.0000, 0.0000] |
| all | B-D | constraint_satisfied | 128 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | mean_target_nll | 24 | 0.2347 | [-0.0127, 0.4612] |
| all | B-best_single | target_perplexity | 24 | 139.0295 | [-1.0748, 263.5357] |
| all | B-best_single | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| all | B-best_single | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | mean_target_nll | 104 | 0.9124 | [0.7023, 1.1178] |
| relevant | B-A | target_perplexity | 104 | 754.8901 | [467.7487, 1096.7439] |
| relevant | B-A | exact_match | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | token_f1 | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-A | constraint_satisfied | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | mean_target_nll | 104 | 0.5739 | [0.3801, 0.7639] |
| relevant | C-A | target_perplexity | 104 | 223.0579 | [65.4304, 380.4132] |
| relevant | C-A | exact_match | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | token_f1 | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | C-A | constraint_satisfied | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | mean_target_nll | 104 | 0.6674 | [0.4641, 0.8726] |
| relevant | D-A | target_perplexity | 104 | 293.7556 | [122.4418, 466.0531] |
| relevant | D-A | exact_match | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | token_f1 | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | D-A | constraint_satisfied | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | mean_target_nll | 104 | 0.2450 | [0.0704, 0.4176] |
| relevant | B-D | target_perplexity | 104 | 461.1345 | [164.1469, 830.9986] |
| relevant | B-D | exact_match | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | token_f1 | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-D | constraint_satisfied | 104 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | mean_target_nll | 24 | 0.2347 | [-0.0127, 0.4612] |
| relevant | B-best_single | target_perplexity | 24 | 139.0295 | [-1.0748, 263.5357] |
| relevant | B-best_single | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| relevant | B-best_single | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | mean_target_nll | 80 | 1.2258 | [1.0132, 1.4418] |
| single_cell | B-A | target_perplexity | 80 | 1025.8141 | [689.2888, 1455.3465] |
| single_cell | B-A | exact_match | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | token_f1 | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-A | constraint_satisfied | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | mean_target_nll | 80 | 0.7806 | [0.5734, 0.9989] |
| single_cell | C-A | target_perplexity | 80 | 318.1196 | [113.6592, 502.9506] |
| single_cell | C-A | exact_match | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | token_f1 | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | C-A | constraint_satisfied | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | mean_target_nll | 80 | 0.8672 | [0.6494, 1.0827] |
| single_cell | D-A | target_perplexity | 80 | 388.6132 | [179.3700, 581.4417] |
| single_cell | D-A | exact_match | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | token_f1 | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | D-A | constraint_satisfied | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | mean_target_nll | 80 | 0.3586 | [0.1556, 0.5775] |
| single_cell | B-D | target_perplexity | 80 | 637.2009 | [259.1879, 1121.3232] |
| single_cell | B-D | exact_match | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | token_f1 | 80 | 0.0000 | [0.0000, 0.0000] |
| single_cell | B-D | constraint_satisfied | 80 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | mean_target_nll | 24 | -0.1324 | [-0.4969, 0.1975] |
| cross_cell | B-A | target_perplexity | 24 | -148.1901 | [-409.4056, 79.9046] |
| cross_cell | B-A | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-A | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | mean_target_nll | 24 | -0.1149 | [-0.4781, 0.2564] |
| cross_cell | C-A | target_perplexity | 24 | -93.8146 | [-363.5580, 177.8659] |
| cross_cell | C-A | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | C-A | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | mean_target_nll | 24 | 0.0014 | [-0.3825, 0.3622] |
| cross_cell | D-A | target_perplexity | 24 | -22.4364 | [-308.7353, 260.8028] |
| cross_cell | D-A | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | D-A | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | mean_target_nll | 24 | -0.1338 | [-0.4057, 0.1702] |
| cross_cell | B-D | target_perplexity | 24 | -125.7536 | [-345.0339, 84.6273] |
| cross_cell | B-D | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-D | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | mean_target_nll | 24 | 0.2347 | [-0.0127, 0.4612] |
| cross_cell | B-best_single | target_perplexity | 24 | 139.0295 | [-1.0748, 263.5357] |
| cross_cell | B-best_single | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| cross_cell | B-best_single | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | mean_target_nll | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | target_perplexity | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-A | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | mean_target_nll | 24 | -0.1660 | [-0.6685, 0.3197] |
| irrelevant | C-A | target_perplexity | 24 | -262.5769 | [-728.9096, 146.9424] |
| irrelevant | C-A | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | C-A | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | mean_target_nll | 24 | -0.6893 | [-0.9611, -0.4276] |
| irrelevant | D-A | target_perplexity | 24 | -463.7194 | [-885.7988, -115.5418] |
| irrelevant | D-A | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | D-A | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | mean_target_nll | 24 | 0.6893 | [0.4277, 0.9611] |
| irrelevant | B-D | target_perplexity | 24 | 463.7194 | [115.5460, 887.5942] |
| irrelevant | B-D | exact_match | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | token_f1 | 24 | 0.0000 | [0.0000, 0.0000] |
| irrelevant | B-D | constraint_satisfied | 24 | 0.0000 | [0.0000, 0.0000] |

| Group | Routing metric | n | Mean |
| --- | --- | ---: | ---: |
| all | recall_at_2 | 104 | 0.1875 |
| all | precision_at_2 | 104 | 0.1154 |
| all | hit_at_2 | 104 | 0.2212 |
| all | mrr | 104 | 0.3295 |
| all | ndcg_at_2 | 104 | 0.1566 |
| all | all_relevant_selected | 104 | 0.1538 |
| all | false_selection | 24 | 1.0000 |
| all | false_selection_count | 24 | 2.0000 |
| all | glassworks hit | 16 | 0.0000 |
| all | railway hit | 16 | 0.2500 |
| all | observatory hit | 16 | 0.0000 |
| all | archive hit | 16 | 0.0000 |
| all | greenhouse hit | 16 | 0.0000 |
| all | lighthouse hit | 16 | 0.0000 |
| all | foundry hit | 16 | 0.6875 |
| all | harbor hit | 16 | 0.5625 |
| relevant | recall_at_2 | 104 | 0.1875 |
| relevant | precision_at_2 | 104 | 0.1154 |
| relevant | hit_at_2 | 104 | 0.2212 |
| relevant | mrr | 104 | 0.3295 |
| relevant | ndcg_at_2 | 104 | 0.1566 |
| relevant | all_relevant_selected | 104 | 0.1538 |
| relevant | glassworks hit | 16 | 0.0000 |
| relevant | railway hit | 16 | 0.2500 |
| relevant | observatory hit | 16 | 0.0000 |
| relevant | archive hit | 16 | 0.0000 |
| relevant | greenhouse hit | 16 | 0.0000 |
| relevant | lighthouse hit | 16 | 0.0000 |
| relevant | foundry hit | 16 | 0.6875 |
| relevant | harbor hit | 16 | 0.5625 |
| single_cell | recall_at_2 | 80 | 0.1875 |
| single_cell | precision_at_2 | 80 | 0.0938 |
| single_cell | hit_at_2 | 80 | 0.1875 |
| single_cell | mrr | 80 | 0.2979 |
| single_cell | ndcg_at_2 | 80 | 0.1460 |
| single_cell | all_relevant_selected | 80 | 0.1875 |
| single_cell | glassworks hit | 10 | 0.0000 |
| single_cell | railway hit | 10 | 0.2000 |
| single_cell | observatory hit | 10 | 0.0000 |
| single_cell | archive hit | 10 | 0.0000 |
| single_cell | greenhouse hit | 10 | 0.0000 |
| single_cell | lighthouse hit | 10 | 0.0000 |
| single_cell | foundry hit | 10 | 0.7000 |
| single_cell | harbor hit | 10 | 0.6000 |
| cross_cell | recall_at_2 | 24 | 0.1875 |
| cross_cell | precision_at_2 | 24 | 0.1875 |
| cross_cell | hit_at_2 | 24 | 0.3333 |
| cross_cell | mrr | 24 | 0.4348 |
| cross_cell | ndcg_at_2 | 24 | 0.1922 |
| cross_cell | all_relevant_selected | 24 | 0.0417 |
| cross_cell | glassworks hit | 6 | 0.0000 |
| cross_cell | railway hit | 6 | 0.3333 |
| cross_cell | observatory hit | 6 | 0.0000 |
| cross_cell | archive hit | 6 | 0.0000 |
| cross_cell | greenhouse hit | 6 | 0.0000 |
| cross_cell | lighthouse hit | 6 | 0.0000 |
| cross_cell | foundry hit | 6 | 0.6667 |
| cross_cell | harbor hit | 6 | 0.5000 |
| irrelevant | false_selection | 24 | 1.0000 |
| irrelevant | false_selection_count | 24 | 2.0000 |

Accuracy and NLL by whether D selected all relevant Cells

| Group | Conditional metric | n | Mean |
| --- | --- | ---: | ---: |
| all | missing_support_A_exact_match | 88 | 0.0000 |
| all | missing_support_A_mean_nll | 88 | 5.8586 |
| all | missing_support_B_exact_match | 88 | 0.0000 |
| all | missing_support_B_mean_nll | 88 | 6.7843 |
| all | missing_support_C_exact_match | 88 | 0.0000 |
| all | missing_support_C_mean_nll | 88 | 6.3962 |
| all | missing_support_D_exact_match | 88 | 0.0000 |
| all | missing_support_D_mean_nll | 88 | 6.4942 |
| all | all_support_A_exact_match | 16 | 0.0000 |
| all | all_support_A_mean_nll | 16 | 5.2376 |
| all | all_support_B_exact_match | 16 | 0.0000 |
| all | all_support_B_mean_nll | 16 | 6.0766 |
| all | all_support_C_exact_match | 16 | 0.0000 |
| all | all_support_C_mean_nll | 16 | 6.0113 |
| all | all_support_D_exact_match | 16 | 0.0000 |
| all | all_support_D_mean_nll | 16 | 6.0799 |
| all | not_applicable_A_exact_match | 24 | 0.0000 |
| all | not_applicable_A_mean_nll | 24 | 5.6048 |
| all | not_applicable_B_exact_match | 24 | 0.0000 |
| all | not_applicable_B_mean_nll | 24 | 5.6048 |
| all | not_applicable_C_exact_match | 24 | 0.0000 |
| all | not_applicable_C_mean_nll | 24 | 5.4388 |
| all | not_applicable_D_exact_match | 24 | 0.0000 |
| all | not_applicable_D_mean_nll | 24 | 4.9155 |
| relevant | missing_support_A_exact_match | 88 | 0.0000 |
| relevant | missing_support_A_mean_nll | 88 | 5.8586 |
| relevant | missing_support_B_exact_match | 88 | 0.0000 |
| relevant | missing_support_B_mean_nll | 88 | 6.7843 |
| relevant | missing_support_C_exact_match | 88 | 0.0000 |
| relevant | missing_support_C_mean_nll | 88 | 6.3962 |
| relevant | missing_support_D_exact_match | 88 | 0.0000 |
| relevant | missing_support_D_mean_nll | 88 | 6.4942 |
| relevant | all_support_A_exact_match | 16 | 0.0000 |
| relevant | all_support_A_mean_nll | 16 | 5.2376 |
| relevant | all_support_B_exact_match | 16 | 0.0000 |
| relevant | all_support_B_mean_nll | 16 | 6.0766 |
| relevant | all_support_C_exact_match | 16 | 0.0000 |
| relevant | all_support_C_mean_nll | 16 | 6.0113 |
| relevant | all_support_D_exact_match | 16 | 0.0000 |
| relevant | all_support_D_mean_nll | 16 | 6.0799 |
| single_cell | missing_support_A_exact_match | 65 | 0.0000 |
| single_cell | missing_support_A_mean_nll | 65 | 5.6519 |
| single_cell | missing_support_B_exact_match | 65 | 0.0000 |
| single_cell | missing_support_B_mean_nll | 65 | 6.9556 |
| single_cell | missing_support_C_exact_match | 65 | 0.0000 |
| single_cell | missing_support_C_mean_nll | 65 | 6.4030 |
| single_cell | missing_support_D_exact_match | 65 | 0.0000 |
| single_cell | missing_support_D_mean_nll | 65 | 6.5134 |
| single_cell | all_support_A_exact_match | 15 | 0.0000 |
| single_cell | all_support_A_mean_nll | 15 | 5.1683 |
| single_cell | all_support_B_exact_match | 15 | 0.0000 |
| single_cell | all_support_B_mean_nll | 15 | 6.0567 |
| single_cell | all_support_C_exact_match | 15 | 0.0000 |
| single_cell | all_support_C_mean_nll | 15 | 6.0766 |
| single_cell | all_support_D_exact_match | 15 | 0.0000 |
| single_cell | all_support_D_mean_nll | 15 | 6.0603 |
| cross_cell | missing_support_A_exact_match | 23 | 0.0000 |
| cross_cell | missing_support_A_mean_nll | 23 | 6.4427 |
| cross_cell | missing_support_B_exact_match | 23 | 0.0000 |
| cross_cell | missing_support_B_mean_nll | 23 | 6.3003 |
| cross_cell | missing_support_C_exact_match | 23 | 0.0000 |
| cross_cell | missing_support_C_mean_nll | 23 | 6.3769 |
| cross_cell | missing_support_D_exact_match | 23 | 0.0000 |
| cross_cell | missing_support_D_mean_nll | 23 | 6.4400 |
| cross_cell | all_support_A_exact_match | 1 | 0.0000 |
| cross_cell | all_support_A_mean_nll | 1 | 6.2776 |
| cross_cell | all_support_B_exact_match | 1 | 0.0000 |
| cross_cell | all_support_B_mean_nll | 1 | 6.3743 |
| cross_cell | all_support_C_exact_match | 1 | 0.0000 |
| cross_cell | all_support_C_mean_nll | 1 | 5.0326 |
| cross_cell | all_support_D_exact_match | 1 | 0.0000 |
| cross_cell | all_support_D_mean_nll | 1 | 6.3743 |
| irrelevant | not_applicable_A_exact_match | 24 | 0.0000 |
| irrelevant | not_applicable_A_mean_nll | 24 | 5.6048 |
| irrelevant | not_applicable_B_exact_match | 24 | 0.0000 |
| irrelevant | not_applicable_B_mean_nll | 24 | 5.6048 |
| irrelevant | not_applicable_C_exact_match | 24 | 0.0000 |
| irrelevant | not_applicable_C_mean_nll | 24 | 5.4388 |
| irrelevant | not_applicable_D_exact_match | 24 | 0.0000 |
| irrelevant | not_applicable_D_mean_nll | 24 | 4.9155 |

## Interpretation

- integrity: PASS
- evaluators: PASS
- routing: MEASURED
- product_effect: FAIL

- Irrelevant Cell impact on unrelated questions: C − A is +0.0% exact match.
- D missed at least one labelled Cell on 88 related questions. The oracle-minus-automatic exact-match gap is +0.0%.
- The oracle exact-match effect is +0.0% relative to a base accuracy of 0.0%. A small oracle effect leaves open write, retention, read and fusion limitations.
- Pair minus best single on cross-Cell questions is +0.0% exact match.
Use the paired intervals and conditional metrics above to assess these point estimates.

Flagged attempts, fallback and parse failures: 842. See failures.json.
The default SDK has no active rejection threshold in this run. Synthetic questions use domain cues and exact integer answers. Conclusions apply to this fixed fixture.
