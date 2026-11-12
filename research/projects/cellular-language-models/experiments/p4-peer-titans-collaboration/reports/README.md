# P4 CPU validation

From the P4 directory, run the following with the existing PyTorch environment.

```sh
python m1.1-capacity/smoke.py /tmp/p4-m11-new
python m1.1-capacity/report.py /tmp/p4-m11-new/manifest.json
python m2-isolation/verify.py /tmp/p4-m2-new
```

Each run uses a new output directory. The M2 wrapper waits for the writer process to exit before launching the reader. For the completed smoke, decompress m11-smoke-manifest.json.gz to inspect full curves and episode results. The M2 inventory and result JSON document saved Cell identities and negative cases. Checkpoint payloads stay in local run output.
