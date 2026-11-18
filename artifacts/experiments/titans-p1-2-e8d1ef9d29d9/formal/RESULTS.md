# Titans P1.2 attribution

Coverage: COMPLETE

| Step | Pairs | Fast delta RMS difference | Read RMS difference | Logit RMS difference | Donor − Correct NLL |
|---|---:|---:|---:|---:|---:|
| 0 | 576 | 0.0007519053 | 0.005695668 | 0.0106834 | 8.0821e-05 |
| 20 | 576 | 0.000790472 | 0.00589034 | 0.01067478 | 5.821677e-05 |
| 200 | 576 | 0.0008553339 | 0.007556816 | 0.0110353 | 2.83834e-05 |

## Update and read attribution

- Step 0: raw_gradient RMS separation 0.5492597, surprise RMS separation 0.0005267651, momentum RMS separation 0.0005623181, decay_delta RMS separation 0.0002831609, gated_read RMS separation 0.0005695668; Correct beats Donor in 58.33% of pairs.
  Paired document NLL gain 95% CI: [3.4352557526678984e-05, 0.0001292336318227926].
- Step 20: raw_gradient RMS separation 0.5515771, surprise RMS separation 0.000543337, momentum RMS separation 0.0005815664, decay_delta RMS separation 0.0002826998, gated_read RMS separation 0.0005883845; Correct beats Donor in 56.94% of pairs.
  Paired document NLL gain 95% CI: [1.446944144038037e-05, 0.00010240326325099136].
- Step 200: raw_gradient RMS separation 0.5861883, surprise RMS separation 0.000573793, momentum RMS separation 0.0006194773, decay_delta RMS separation 0.0002226283, gated_read RMS separation 0.0008144807; Correct beats Donor in 51.39% of pairs.
  Paired document NLL gain 95% CI: [-6.1011148823642926e-05, 0.00012973282072278416].

Diagnostic label: MIXED / UNRESOLVED pending review of per-head, per-document and paired statistics.
P1.1 remains INCONCLUSIVE. No training or parameter updates were performed.
