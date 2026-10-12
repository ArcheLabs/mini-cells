# MiniCells Core Validation 002 — Write Addressability under Superposition

## Question

This experiment tests one mechanism only:

> Can a neural system learn a sparse functional coordinate system from a densely superposed observation and, from a handful of changed examples alone, infer a write address whose local update changes the shared function while leaving unrelated functions nearly invariant?

The experiment does **not** test natural language, cell growth, mitosis, routing growth, 2D NCA structure, or continual creation of new coordinates. The encoder is frozen before the first edit.

The formal conclusion is binary:

- `WRITE_ADDRESSABILITY_SUPPORTED`
- `WRITE_ADDRESSABILITY_NOT_SUPPORTED`

## Synthetic ground truth

There are `F=512` latent functional atoms. Each example activates exactly `k=4` atoms:

\[
s\in\mathbb{R}^{F},\qquad \|s\|_0=k.
\]

The model never observes `s`. A fixed dense Gaussian mixing matrix with column normalization creates the observation

\[
x=As,\qquad A\in\mathbb{R}^{128\times512}.
\]

Each atom owns a reusable output function vector, collected in `V`:

\[
y=Vs,\qquad V\in\mathbb{R}^{32\times512}.
\]

The primary regime therefore has superposition load

\[
\alpha=F/d=4.
\]

We additionally report the sparse-recovery load

\[
\rho=\frac{k\log(F/k)}{d},
\]

because `F/d` alone does not characterize recoverability.

## Knowledge edit

For target atom `j`, the world changes

\[
v_j' = v_j + \delta.
\]

Therefore every affected example obeys

\[
\Delta y=s_j\delta,
\]

while every invariant example with `s_j=0` has

\[
\Delta y=0.
\]

Each edit exposes only 8 changed examples. Their three other active atoms vary. The editor receives only `(x_i, y_i')` and its current model state.

**The editor must not receive `s`, `j`, `A`, or `V`.** Those values are evaluator-only. Any implementation that uses the true target identity to choose the candidate write address invalidates the experiment.

## Learned sparse functional model

The candidate learns

\[
z=E(x),\qquad z\in\mathbb{R}^{1024},\qquad \|z\|_0\le 8,
\]

with a top-k sparse bottleneck, an auxiliary reconstruction head, and a linear functional writer

\[
\hat y=Wz.
\]

Pretraining minimizes output error plus an input-reconstruction auxiliary loss. Ground-truth `s` is never a training target.

After pretraining, the encoder and reconstruction head are frozen for the entire edit sequence. This isolates the write mechanism.

## Address inference without an oracle

For the edit examples, let

\[
r_i=y_i'-\hat y_i
\]

be the current output residual. For every learned latent coordinate `q`, solve

\[
\delta_q^*=\arg\min_\delta\sum_i\|r_i-z_{iq}\delta\|^2.
\]

The editor selects the sufficiently shared, non-degenerate coordinate with minimum residual score:

\[
q^*=\arg\min_q\sum_i\|r_i-z_{iq}\delta_q^*\|^2.
\]

It then performs one local write:

\[
w_{q^*}\leftarrow w_{q^*}+\delta_{q^*}^*.
\]

This closes the critical loop:

\[
\text{changed examples}\rightarrow\text{common functional coordinate}\rightarrow\text{write address}\rightarrow\text{local update}.
\]

## Decisive comparison

The primary causal comparison uses **the exact same pretrained sparse model**, copied before editing:

1. **Inferred addressed write** — infer one shared coordinate and update only its writer column.
2. **Global writer SGD** — keep the same frozen sparse encoder but optimize the full writer on the same 8 edit examples.

Thus parameter count, forward representation, pretrained weights, edit data, and encoder are identical. The only intended difference is the write operator.

Two contextual baselines are also trained on the same pretraining stream with approximately matched parameter budgets:

- dense MLP;
- standard top-k MoE.

They are useful context, but the formal locality claim does not depend on architecture-level comparisons with them.

## Diagnostic hierarchy

### Learned representation + oracle address

On a held-out evaluator probe set, each true feature is mapped to the learned latent with maximum absolute correlation. The target feature id may then select that latent.

This diagnostic asks whether the representation is good enough **if address discovery is solved for us**. It is not the candidate system and cannot satisfy the main claim by itself.

### Learned representation + inferred address

This is the actual candidate. The target feature id is unavailable to the editor.

### Permuted write control

The forward/read mapping is left completely unchanged. Address inference also runs normally. Only the destination of the write is permuted:

\[
q\rightarrow\pi(q).
\]

This avoids the invalid control in which the forward model is already broken before the edit. A large degradation therefore provides causal evidence that read/write alignment matters.

### Analytic oracle check

For `z=s`, changing only `w_j` gives

\[
\Delta\hat y=s_j\delta.
\]

Hence invariant examples have exactly zero output change. The test suite verifies this identity as an implementation invariant; it is not an empirical result.

## Continual sequence

The formal run applies 100 edits with no replay of previous training examples.

The schedule deliberately contains:

- new target atoms;
- repeated edits to previously changed atoms;
- edit batches in which a previously edited atom is forced to appear as a distractor.

After every edit, retained evaluator-only probes from previous targets are checked against the world’s current `V`. These probes are never used for optimization.

## Evaluation partitions

For current target `j`:

### Edit examples

The 8 examples used by the editor. Fitting them alone is not evidence of a knowledge update.

### Unseen affected combinations

Fresh combinations with `s_j != 0`. Correct behavior requires

\[
\Delta\hat y\approx s_j\delta.
\]

This tests functional update generalization.

### Invariant combinations

Fresh combinations with `s_j = 0`. Correct behavior requires

\[
\Delta\hat y\approx0.
\]

This tests write locality.

## Primary metrics

### Normalized Update Error

\[
U_j=
\frac{
\mathbb E_{A_j}\| (\hat y'-\hat y)-(y'-y)\|^2
}{
\mathbb E_{A_j}\|y'-y\|^2
}.
\]

A system that changes nothing has `U≈1`, so low leakage alone cannot pass.

### Write Leakage

\[
L_j=
\frac{
\mathbb E_{I_j}\|\hat y'-\hat y\|^2
}{
\mathbb E_{A_j}\|y'-y\|^2
}.
\]

The ideal local writer has `L=0`.

### Mechanistic prediction

For the selected learned address `q`, define

\[
Q_j=
\frac{
\mathbb E_{I_j}[z_q^2]
}{
\mathbb E_{A_j}[s_j^2]
}.
\]

For an aligned one-column write, the theory predicts that off-support representation energy controls write leakage. The experiment therefore reports the Pearson relation between `Q` and `L`, plus a log-log slope diagnostic.

## Frozen v1 gates

A seed passes only when all of the following hold:

1. **Base-world validity**: sparse, dense, and MoE pretrained models each reach normalized validation MSE <= 0.10. A weak contextual baseline is not allowed to count as evidence for the candidate.
2. **Update generalization**: inferred-address median `U <= 0.10` on unseen affected combinations.
3. **Decisive baseline validity**: global-writer median `U <= 0.15`; a baseline that simply fails to edit cannot provide a locality comparison.
4. **Locality advantage**: at those valid update-success levels,
   \[
   \operatorname{median}(L_{\rm inferred})\le0.1\operatorname{median}(L_{\rm global}).
   \]
5. **Mechanistic prediction**: Pearson correlation between the off-support activation proxy `Q` and observed `L` is >= 0.70.
6. **Causal control**: the permuted-write control’s joint median `(U+L)` is at least 2x the candidate’s.

Formal seeds are `74201`, `74202`, and `74203`. The final status is positive only if **all three** seeds pass. There is no majority-vote rescue and no post-hoc threshold tuning in the frozen v1 protocol.

## Recovery-load diagnostic

The optional sweep changes `F` and `k`, reports both `alpha=F/d` and `rho=k log(F/k)/d`, and plots Write Leakage against recovery load. It is descriptive in v1 and does not alter the primary pass/fail result.

The Gaussian mixing matrix is deliberately a favorable sparse-recovery setting. A successful Core Validation 002 should be followed by a structured-superposition test with correlated feature occurrence and coherent dictionaries; that is outside this frozen v1 protocol.

## Interpretation

A positive result supports the narrow statement:

> In this synthetic sparse additive world, a neural system can learn functional coordinates from dense superposition, infer a shared coordinate from changed examples without receiving the true atom identity, and use an aligned local parameter write to generalize the edit with substantially lower off-target interference. Moreover, the remaining interference is predicted by off-support activation of the selected functional coordinate.

It does **not** establish that:

- natural-language knowledge always admits this decomposition;
- one learned coordinate always corresponds to one semantic fact;
- SAE is the final representation;
- MiniCells already has mature continual learning;
- the functional basis can itself grow or reorganize online;
- Gaussian sparse recovery represents the difficulty of real neural representations.

## Run

Formal Kaggle/GPU run:

```bash
python scripts/research/run_core_validation_002.py --device cuda
python scripts/research/report_core_validation_002.py
```

Optional recovery-load diagnostic:

```bash
python scripts/research/run_core_validation_002.py --device cuda --sweep
python scripts/research/report_core_validation_002.py
```

CPU smoke run:

```bash
python scripts/research/run_core_validation_002.py --smoke --device cpu
python scripts/research/report_core_validation_002.py
```

Outputs are written to:

```text
results/core-validation-002-write-addressability/
```

The Kaggle notebook is:

```text
research/notebooks/04-continual-learning-core/core-validation-002-write-addressability.ipynb
```
