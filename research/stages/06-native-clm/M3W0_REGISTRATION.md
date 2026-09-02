# M3W-0 — Root Write-Drift Counterfactual Restoration

Status: **FROZEN / UNRUN checkpoint diagnostic**

## Parent result

M3L-2 closed as:

```text
NATIVE_CLM_V0_M3L2_ONLINE_ADDRESS_STATE_NOT_SUPPORTED
formal seeds = 74211 / 74212 / 74213
publish commit = 348a6cd28cda13298b6d61c01453d06e14efbd33
HF revision = 2b6ac153e926f899f038ff02c8c10041baaacb4a
```

M3L-2 preserved root-route ownership, reduced old-domain child leakage, maintained plasticity, and reduced A regression by roughly five percentage points versus the matched lineage-cosine control. It still failed the registered absolute retention, retention-advantage and mean-forgetting gates. The residual A damage appeared mostly during the first B phase.

## Question

M3W-0 asks:

> With final routing, address state and affine gates held fixed, how much of residual A forgetting is causally attributable to operator writes in the original eight roots versus operator writes in descendants?

No model is trained or updated. No new formal seed is introduced. The already-consumed M3L-2 treatment checkpoints are used only as fixed evidence.

## Exact counterfactuals

For every M3L-2 treatment checkpoint:

```text
11 FINAL
   roots final
   descendants final

01 ROOT_RESTORE
   roots -> exact M1 root weights
   descendants final

10 DESCENDANT_ROOT_RESTORE
   roots final
   every descendant -> exact M1 weight of its root ancestor

00 ALL_LINEAGE_RESTORE
   every Cell -> exact M1 weight of its root ancestor
```

The experiment deliberately does **not** claim child birth-state restoration. M3L-2 did not persist child operator tensors at birth, so those tensors cannot be reconstructed exactly after the fact.

## Why all-lineage restoration is an identity check

M3L-2 keeps the original M1 root router frozen. All descendants remain inside a root lineage and inherit the root-level probability mass. If every concrete Cell operator in a lineage is replaced by the same exact M1 root operator, the local parent/child choice becomes functionally irrelevant:

\[
W_{r,m}=W_r^{M1}\quad\forall m\in L_r.
\]

Therefore the Cellular Layer should reconstruct the M1 function despite retaining final affine gates and lineage topology. M3W-0 requires all four evaluation-domain losses to match M1 within the frozen `1e-4` tolerance. Failure makes the diagnostic `INCONCLUSIVE_IDENTITY`.

## 2×2 Shapley attribution

Let `L00`, `L10`, `L01`, `L11` be A losses under the four states above. Root and descendant write contributions are:

\[
\phi_{root}=\frac12[(L_{10}-L_{00})+(L_{11}-L_{01})],
\]

\[
\phi_{desc}=\frac12[(L_{01}-L_{00})+(L_{11}-L_{10})].
\]

They satisfy:

\[
\phi_{root}+\phi_{desc}=L_{11}-L_{00}.
\]

This avoids assigning root/descendant interaction effects arbitrarily to one side.

## Plasticity transfer diagnostic

For each new domain `D in {B,C,D}`, M3W-0 also reports how much of the final M3L-2 adaptation gain remains after restoring roots to M1:

\[
R_D=\frac{L_D^{M1}-L_D^{root\ restore}}
          {L_D^{M1}-L_D^{final}}.
\]

If root writes dominate A forgetting while `R_B,R_C,R_D >= 0.70`, descendants already carry most new-domain plasticity. If root writes dominate but those ratios are smaller, the result indicates a **write-transfer gap**: roots contain both the interference and a material share of newly acquired capability.

## Registered classifications

```text
INCONCLUSIVE_IDENTITY
ROOT_WRITE_DOMINANT_CHILDREN_CARRY_PLASTICITY
ROOT_WRITE_DOMINANT_TRANSFER_GAP
DESCENDANT_WRITE_DOMINANT
DISTRIBUTED_WRITE_DRIFT
```

Root dominance requires root Shapley fraction >= 0.60 on every source seed. Descendant dominance is symmetric. The full frozen protocol is in:

```text
research/validations/native-clm-v0-m3w0-write-drift-restoration/protocol.json
```

## Interpretation boundary

A root-dominant result would justify registering a **new** continual-language formal experiment whose only new integration variable is write-ownership separation / copy-on-write committed parents. It would not prove that copy-on-write will pass continual-learning retention gates.

M3W-0 itself cannot modify the M3L-2 formal decision and has `scientific_decision = false`.
