# CLM-001-R3b-CF-001

Single-seed Development, inference-only. CONFIRMATION_SEALED; MULTISEED_NOT_TESTED; CAUSAL_FEEDBACK_NOT_TESTED.

The native Greedy baseline remains the original policy. Counterfactual improvements do not replace its reported performance.

## old-analysis-preservation.json

```json
{
  "status": "NOT_MEASURED"
}
```

## baseline-parity.json

```json
{
  "status": "NOT_MEASURED"
}
```

## cf1-stop-value-calibration.json

```json
{
  "status": "NOT_MEASURED"
}
```

## cf2-forced-stop.json

```json
{
  "status": "NOT_MEASURED"
}
```

## cf3-repeat-constraint.json

```json
{
  "status": "NOT_MEASURED"
}
```

## Interpretation

Negative ΔNLL and ΔUtility support harmful over-computation on this model/distribution. Positive ΔNLL with negative ΔUtility is a cost/quality tradeoff only. Shorter paths or lower forced-end rate alone are not language-quality improvements. Mixed or incomplete evidence is INCONCLUSIVE; no unique training-loss cause is established.
