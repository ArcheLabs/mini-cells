# Shadow Cell Validation 001 — Copy-on-Write Functional Isolation

Status: **FROZEN / UNRUN**

This validation is deliberately **independent of the Native CLM M2/M3/M3W-0 conclusion chain**. It does not attempt to repair M2, does not consume M2/M3 formal seeds, and does not use any historical Native CLM checkpoint or language stream.

## Question

The experiment asks only:

> If new learning is isolated in an exact-clone Shadow branch, can conditional expression achieve a retention–plasticity Pareto frontier that a matched direct write / checkpoint interpolation cannot reach?

If the answer is no, Shadow Cell should not be promoted into a more complex CLM lifecycle mechanism.

## Fresh controlled token-predictive world

Every seed creates a fresh model and fresh synthetic byte-language data.

A/B have the same textual surface:

```text
p03q71x42y86=
```

The prefix is always 13 UTF-8/ASCII bytes and the model predicts exactly one next byte: an ASCII digit.

The hidden context factor is relational rather than a domain marker:

```text
A: p < q    answer = x mod 10
B: p > q    answer = y mod 10
```

`p == q` is excluded. `x` and `y` have identical marginals in A and B. Train, calibration and heldout prefixes are unique within each rule family.

This gives a real causal next-token loss while avoiding an English-vs-code style domain split.

## Fresh base model

Each registered seed trains a new NativeCLM from scratch on A only:

```text
vocab                 256 bytes
context               13
width                 384
shared blocks         4
heads                  6
FFN                    1024
Cells                  4
active Cells/token     1
Cell operator          384 x 384
Cellular Layer         after the final shared block
certificate rank       0 / disabled
```

Placing the Cellular Layer after the final shared block is intentional. Shadow expression at the answer-prediction position cannot flow through later attention, so the validation isolates operator expression rather than introducing a second routing/composition problem.

The mature Parent is the Cell with highest A-heldout Top-1 ownership after base training. The registered conflict gate requires the same Parent to own at least 80% of both A and B answer positions after routing is frozen.

## Shadow definition

The Parent operator is `W_p`. Shadow birth is exact:

```text
W_s(0) = W_p
```

The local answer-position operator is:

```text
y = W_p h + a(x) (W_s - W_p) h
```

Therefore birth is exactly function preserving for any expression value:

```text
W_s = W_p  =>  y = W_p h
```

B operator training uses `a(x)=1`, changes only `W_s`, and never supplies A examples to the Shadow/direct operator optimizer.

## Matched controls are algebraic controls

The five named arms are not treated as five unrelated models.

The direct candidate and Shadow candidate:

- start from the same exact `W_p`;
- see the same B minibatch schedule;
- use the same optimizer and hyperparameters;
- are the only trainable tensor.

They must therefore finish with the same operator within the registered numerical tolerance.

For linear Cells:

```text
Direct-Interp(m)
  = W_p + m (W_direct - W_p)

Shadow-Global(m)
  = W_p + m (W_shadow - W_p)
```

and `W_direct == W_shadow`, so these curves must match. Likewise `Shadow-Immediate(m=1)` must equal `Direct-TX`.

Failure of these identities makes the experiment inconclusive instead of creating a false Shadow advantage.

## The only new structural variable

The primary arm is:

```text
a(x) = m * g(x)
```

where `g(x)` is a linear logistic probe over the frozen base model's inference-visible Cell input at the answer-prediction position.

Validation 001 intentionally allows A/B calibration labels to train this probe. The probe:

- cannot modify the model;
- cannot enter B operator training;
- is only an upper-bound expression controller.

A heldout gate AUC below 0.90 makes the experiment `INCONCLUSIVE_GATE_CAPACITY` rather than a Shadow failure.

## Maturity curve

The frozen maturity grid is:

```text
0
0.0625
0.125
0.25
0.5
0.75
1
```

Every point reports:

```text
A_regression = max(0, A_base_accuracy - A_candidate_accuracy)
B_gain       = max(0, B_candidate_accuracy - B_base_accuracy)
B_gain/direct
```

and contributes to a registered retention–plasticity Pareto hypervolume.

The primary conditional point is selected before classification: among maturity values with `A_regression <= 0.10`, choose the one with the highest `B_gain/direct`, breaking ties toward lower maturity.

## Shuffled-gate causal control

The Shadow operator remains identical. Only gate values are deterministically permuted across the combined A/B heldout set.

This tests whether the advantage comes from expression being aligned with functional context rather than from extra parameters or globally shrinking the update.

## Registered evidence gates

Before any positive Shadow claim, every formal seed must have:

```text
base A accuracy          >= 0.95
base A NLL               <= 0.20
Parent A Top-1 share     >= 0.80
Parent B Top-1 share     >= 0.80
Direct-TX B gain         >= 0.50
gate heldout AUC         >= 0.90
all algebraic identities PASS
```

The primary conditional point must additionally satisfy:

```text
A regression             <= 0.10
B gain/direct            >= 0.90
```

and the conditional Pareto hypervolume must improve at least 20% over Direct-Interp for any isolated-Shadow advantage claim.

## Registered classifications

```text
INCONCLUSIVE_BASE_TRAINING
INCONCLUSIVE_PARENT_CONFLICT
INCONCLUSIVE_DIRECT_PLASTICITY
INCONCLUSIVE_GATE_CAPACITY
INCONCLUSIVE_IDENTITY_CONTROL
SHADOW_CELL_NOT_SUPPORTED
ISOLATED_SHADOW_ADVANTAGE_NOT_SUPPORTED
SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY
SHADOW_CELL_CONTROLLED_MATURATION_SUPPORTED
```

The strongest result additionally requires that Immediate Shadow does not already satisfy the primary gate, Conditional dominates Shadow-Global, and correct gate alignment produces the registered old-damage advantage over shuffled gates.

## Fresh seeds

Development only:

```text
95101 / 95102 / 95103
```

Formal evidence:

```text
95211 / 95212 / 95213
```

Development seeds never count as formal evidence. Formal seeds remain untouched until the canonical formal runner is deliberately executed.

## Evidence boundary

Validation 001 uses no:

- dynamic growth;
- second mitosis;
- certificate projection;
- old-example replay in operator weight training;
- natural Cell discovery;
- semantic routing intervention;
- LoRA/rank compression.

A positive result supports only the existence of a structural Shadow/conditional-expression advantage under a strong calibration gate. Replay-free autonomous control would require a separate validation.
