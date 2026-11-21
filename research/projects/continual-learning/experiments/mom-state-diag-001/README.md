# MOM-STATE-DIAG-001 — Cache Restore and Runtime Determinism

Development diagnostic with frozen MoM 340M, one GPU and fixed short token
inputs. A continues live state; B reads the same checkpoint in A's process;
C restores the checkpoint in a new process; D1/D2 independently rebuild
the prefix in two new processes. The coordinator keeps A alive but releases
its model before launching the workers. C does not replay the prefix.

Compare initial and post-append cache, logits and both candidates' per-token
log probabilities. Report per-layer recurrent/conv hashes, differing tensor
paths, absolute/relative errors and FP16 ULP distances. Relative error uses
the left operand with a 1e-12 denominator floor. Cache attribute names and
types are inventoried; this is not proof that every implementation detail
affects continuation or that the codec covers a different FLA version.

B differing from A suggests a local restore or execution effect; compare
initial cache first. D1/D2 establish a no-restore cross-process control.
C vs D1 alone cannot isolate restoration if the rebuilt initial cache
already differs. Zero tolerance is unchanged. This cannot supersede the
historical baseline or establish recall quality.

Kaggle: run kaggle_diag.py as a Python Script on one T4, with Internet and
GITHUB_TOKEN. The launcher publishes JSON diagnostics; tensor bundles stay
in Kaggle Output. No external Cell routing or training.
