# P1.4 — Memory Signal Transfer Analysis

CPU-only FP32 replay. P1.3 ran with a GPU FP16 backbone; this replay reuses the pinned tokenizer, model revision, checkpoint parameters, token construction and episode seed. Do not interpret its absolute vector magnitudes as bitwise reproductions of P1.3.

Use Python 3.11+ with CPU Torch and the P1.3 launcher dependency pins. Check out the pinned Titans revision and extract the P1.1 artifacts as described in P1.2. Run from this directory:

```sh
python run.py /path/to/titans /path/to/p1-1-artifact /path/to/new-output --threads 4
```

Three states start empty per episode and differ only in the four selected write V vectors. Random perturbations match the swap norm per token and KV head, with the same random directions across seeds and checkpoints. Queries use the original write K. There are no outer updates.

Saved tensors are indexed by fact, query head and head coordinate. Value deltas repeat across the two query heads that share a KV head. For injected delta x and observed read delta y, report cosine, norm(y)/norm(x), projection coefficient dot(y,x)/dot(x,x), and the norm of the remaining orthogonal component (both absolute and normalized). Zero read delta has zero cosine by convention.

JSON summaries retain episode, seed and head breakdowns. Paired intervals resample the 16 episodes jointly across conditions, checkpoints and seeds; they describe episode variability conditional on those seeds. Compare swap to random and compare their difference across checkpoints. The equal-norm control tests direction sensitivity; it does not by itself prove binding recall.
