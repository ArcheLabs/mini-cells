# Unified geometry capacity screening

Run with one PyTorch CPU thread and a fresh output path.

```bash
OMP_NUM_THREADS=1 python screen.py /tmp/phase1-screen
```

The planned matrix uses 3 paired seeds, 20 independent episodes per seed,
32-dimensional normalized Gaussian keys and candidates, identical value
scaling and 128 presentations at each load. Hard-negative FPR uses the
historical fixed 0.8 cutoff and is a diagnostic at that operating point.
Confirmation and final aggregate intervals are pending.
