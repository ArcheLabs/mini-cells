# Alpha control

For a compatible mutation, effective change is:

```text
delta_effective = alpha * delta
```

`alpha=0` disables the mutation, `alpha=1` expresses the full learned delta,
and intermediate values are reversible expression controls. Alpha does not
measure or implement dependency on the original parent computation; the future
dependency-withdrawal variable is conceptually separate (`beta`).
