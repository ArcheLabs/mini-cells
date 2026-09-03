[English](README.md) | 中文

# Stage 06 — Native CLM

状态：**ACTIVE — M2 SAFE-FUNCTIONAL-WRITE REOPENED**

Stage 06 的目标仍然只有一个：在真实 token-predictive model 中建立 replay-free continual learning。M1 证明 Native Cellular architecture 可以训练；M2 是第一个真正的 continual-learning milestone，并且至今仍然 **NOT SUPPORTED**。

因此从本版本开始，M3、M3R、M3L、M3L-1、M3L-2、M3W-0 不再解释为“已经跨过 M2 后的更高 milestones”，而统一归入 **M2 failure-decomposition evidence**。

完整重开路线见：

- [M2_REOPENED_ROADMAP.zh-CN.md](M2_REOPENED_ROADMAP.zh-CN.md)
- [M2_REOPENED_ROADMAP.md](M2_REOPENED_ROADMAP.md)

## 当前路线图

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🟢 COMPLETE
        ↓
  M2  fixed-topology replay-free continual language      🔴 NOT SUPPORTED
        ↓  reopened failure analysis
  M2-R0 actual optimizer-update invariant audit          🔵 FROZEN / UNRUN
        ↓
  M2-R1 functional certificate reconstruction            ⚪ BLOCKED ON R0
        ↓
  M2-R2 fixed-topology replay-free continual language    ⚪ BLOCKED ON R0/R1
        ↓ only if formal PASS
  growth / mitosis reopened                              ⚪ BLOCKED

Historical M2 failure decomposition:
  M3   global-pool growth                                🔴 NOT SUPPORTED
  M3R  lineage-isolated read routing                     🔴 NOT SUPPORTED
  Address Diagnostic                                     🟢 QUERY GEOMETRY SEPARABLE
  M3L  rank-16 query-sketch gate                         🔴 NOT FEASIBLE
  M3L-1 address-state capacity                           🟢 LOW_RANK_CAPACITY_SUFFICIENT (rank 32)
  M3L-2 online address integration                       🔴 NOT SUPPORTED / partial retention benefit
  M3W-0 write-drift restoration                          🟡 ROOT_WRITE_DOMINANT_TRANSFER_GAP

  M4   Cell ontology / specialization                    ⚪ BLOCKED
  30M scale-up                                            ⚪ BLOCKED
```

硬规则：**M2-R2 未正式通过前，不注册新的 Native growth milestone，不进入 M4，不升级 30M。**

## Canonical substrate

```text
M1 起始参数                 12,154,368
vocab                       256 UTF-8 bytes
context                     256
shared width                384
shared blocks               6
attention heads             6
FFN width                   1536
Cellular Layers             1
initial Cells               8
active Cells/token          2
Cell operator               384 × 384 linear residual
certificate max rank        64
```

Canonical M1 checkpoint：

```text
Hugging Face  archelabsxyz/native-clm-v0
file          final-model.pt
SHA-256       91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

## M0 / M1 — 🟢

M0 建立 sparse routing、Cell-local gradients、certificate projection、dynamic spawn、optimizer enrollment、checkpoint round-trip 与 generation。

M1 证明该架构可做真实 next-token training：

```text
validation loss       5.723429 -> 0.788535
perplexity             305.9523 -> 2.2002
active Cell fraction   2/8 = 0.25
```

详见 [M1_CLOSURE.md](M1_CLOSURE.md)。

## M2 — 🔴 NOT SUPPORTED / 当前唯一未完成的核心 milestone

正式结论：

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
formal seeds = 73211 / 73212 / 73213  (consumed)
```

结果：

```text
protected mean forgetting     ~0.211502
unsafe mean forgetting        ~0.278987
protected A regression        ~0.438682
registered A regression       <=0.20
protected/unsafe plasticity    ~0.964
```

Certificate projection 有真实因果收益，但远未关闭 end-to-end retention gap。

原 M2 的固定边界仍然是我们重新验证 continual-write primitive 的基线：

```text
exact M1 start
8 Cells fixed
active Cells = 2
shared Transformer frozen
router / route keys frozen
growth disabled
B -> C -> D
learner raw replay = 0
```

## 为什么 M3 以后降级为 failure-decomposition evidence

M3 以后揭示了真实但次级的附加问题：

- M3：children 进入 global Top-K 后破坏 old-domain read ownership；
- M3R：保住 root ownership，但 lineage-local cosine boundary 不足；
- Address Diagnostic：query representation 中实际存在可分 boundary；
- M3L-1：rank-32 historical address state 足以表示该 boundary；
- M3L-2：online address state 将 A regression 相对 cosine control 稳定改善约 5pp，同时保持约 99% plasticity，但 A regression 仍约 42%；
- M3W-0：checkpoint-only 2×2 restoration 将 residual A operator damage 的约 94.4–95.2% 归因到 original-root final operator drift。

M3W-0 的注册分类为：

```text
ROOT_WRITE_DOMINANT_TRANSFER_GAP
publish commit = 7bd8d554fea89fc44ef16287453c97f90a4e3f06
scientific_decision = false
```

它同时说明简单冻结 roots 也不是答案：恢复 roots 到 M1 后，只保留约 49–50% B gain、67–68% C gain、33% D gain。Root 中混合了大量 useful plasticity 与 destructive interference。

因此当前问题不是“root 要不要学”，而是：

\[
\boxed{\text{哪些实际参数事务可以安全提交到旧 Cell？}}
\]

## M2-R0 — 🔵 Protected Update Invariant Audit

M2-R0 在任何新 certificate 研究之前先验证原 M2 以为自己满足的数学不变量。

原 protected path：

\[
G_p=G(I-Q^TQ)
\]

之后却由 AdamW 执行 element-wise preconditioning 和 decoupled weight decay。M2-R0 不再只测 projected gradient，而直接测每个 Cell/step 的真实 optimizer parameter delta：

\[
\rho=\frac{\|\Delta WQ^T\|_F}{\|\Delta W\|_F+10^{-12}}.
\]

冻结四个 matched arms：

```text
1. canonical AdamW + gradient projection + wd=.01
2. AdamW + gradient projection + wd=0
3. SGD + gradient projection + wd=0            (algebraic reference)
4. AdamW + gradient projection + final-delta projection
```

第 4 路先产生完整 AdamW proposal，再提交：

\[
U=U_{raw}(I-Q^TQ).
\]

M2-R0：

- 使用 exact M1 checkpoint；
- 只用 pinned WikiText B-train 产生真实梯度；
- 64 steps/arm；
- certificate 固定，不新增 basis；
- shared/router 不更新；
- 不做 growth；
- 不读取 A replay；
- 不消耗新的 formal seed；
- 不产生新模型 checkpoint；
- `scientific_decision=false`。

Canonical protocol：

```text
research/validations/native-clm-v0-m2r0-update-invariant-audit/protocol.json
```

Canonical Kaggle notebook：

```text
research/notebooks/06-native-clm/native-clm-v0-m2r0-update-invariant-audit-kaggle.ipynb
```

## M2-R1 — ⚪ Functional Certificate Reconstruction

R1 在 R0 关闭 optimizer-level ambiguity 以后再研究 certificate 本身，不立即进入新的 continual formal。

候选方向：

```text
A. current top-1 mean basis                 historical baseline
B. all-active probability-weighted SVD      activation coverage
C. importance-weighted activation subspace  soft/scaled protection
D. Jacobian/Fisher functional sketch         old-function protection
```

对 Cell：

\[
\delta z\approx J_i(x)p_i\Delta W_i h
\]

因此最终希望逼近的不是 activation similarity，而是：

\[
D_i(\Delta W)=\mathbb E_{old}\|J_i p_i\Delta W_i h\|^2.
\]

可持久化近似优先考虑：

\[
F_i\approx A_i\otimes B_i,
\]

\[
A_i=\mathbb E[p_i^2hh^T],\quad B_i=\mathbb E[J_i^TJ_i].
\]

R1 必须用 held-out old-function drift 验证 certificate fidelity，并显式报告 storage/rank/capacity；raw old replay 不允许成为最终机制。

## M2-R2 — ⚪ 下一次真正的 continual-learning formal

只有 R0/R1 mechanism validation 完成后，才使用**全新 untouched formal seeds**重新运行固定 8-Cell B→C→D。

只允许引入 R0/R1 已单独验证过的变量：

1. actual-update constrained optimizer；
2. selected functional certificate。

最低 end-to-end gate 保持：

```text
A absolute regression <= 20%
mean forgetting <= 15%
B/C/D phase gain >= registered plasticity floor
plasticity >= 80% matched control
shared/router frozen
zero learner replay
fixed 8-Cell topology
```

只有 M2-R2 formal PASS，才允许声明：

\[
\boxed{\text{Native CLM basic replay-free continual-write primitive supported}}
\]

## Growth 重新开放的条件

未来 growth 不再由 `loss high + rank pressure + cooldown` 独立触发，而由 safe-write infeasibility 推导。

定义 Cell 的安全更新集合：

\[
\mathcal S_i(\epsilon)=\{\Delta W:D_i(\Delta W)\le\epsilon\}.
\]

如果已有 Cell 能在 \(\mathcal S_i\) 中取得足够 new-domain gain，则 reuse/write；只有所有已有 Cell 都无法安全吸收该写入时，才重新开放 capacity allocation / mitosis。

## Evidence boundary

- M2/M3/M3R/M3L-2 formal seeds 均已消耗，不得作为 untouched evidence 重跑；
- Address Diagnostic、M3L、M3L-1、M3W-0 都是 diagnostics，不能修改历史 formal decision；
- M2-R0/R1 也都是 diagnostics；
- M2-R2 必须使用全新 formal seeds；
- 在 R2 PASS 前，M4、growth milestone 与 30M scale-up 全部 blocked。
