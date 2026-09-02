# Native CLM M2 重开路线图 — Safe Functional Write Primitive

状态：**M2 REOPENED / M2-R0 FROZEN UNRUN**

## 0. 为什么回到 M2

M1 只证明 Native Cellular architecture 可以完成真实 next-token training；M2 才是第一个端到端 replay-free continual-language milestone。

M2 的正式结果仍然是：

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
formal seeds = 73211 / 73212 / 73213  (已消耗)
protected A regression mean ≈ 0.438682
protected mean forgetting ≈ 0.211502
unsafe mean forgetting ≈ 0.278987
registered A-regression target <= 0.20
```

因此 Native CLM 在科学意义上至今没有通过 M2。

M3、M3R、Address Diagnostic、M3L、M3L-1、M3L-2、M3W-0 统一解释为 **M2 failure-decomposition series**，而不是已经跨过 M2 后的更高 continual-learning milestones。它们证明 growth/read-address 会增加额外损伤，也证明 online rank-32 address state 能消除相当一部分 read leakage；但修复后仍回到约 42–44% 的 A-regression ceiling。M3W-0 又将约 94.4–95.2% 的 residual A operator damage 归因到 original roots 的 final operator-state drift。

active question 因此回到最底层的不变量：

\[
\boxed{\text{新知识写入时，实际参数事务是否真的位于旧功能允许的安全区域？}}
\]

## 1. 当前 M2 protected write 的实现假设

对 Cell operator：

\[
y=Wh
\]

M2 certificate 保存正交 basis `Q`，并在 optimizer step 前执行：

\[
G_p=G(I-Q^\top Q).
\]

若实际更新是普通 SGD：

\[
\Delta W=-\eta G_p,
\]

则：

\[
\Delta WQ^\top=0.
\]

但 canonical M2 使用 AdamW：

```text
gradient projection
→ gradient clipping
→ AdamW(beta1=.9, beta2=.95, weight_decay=.01)
→ actual parameter update
```

AdamW 的 element-wise moment preconditioner 与右侧 subspace projector 一般不可交换；decoupled weight decay 也不受 raw-gradient projector 约束。因此：

\[
G_pQ^\top=0
\]

并不能自动推出：

\[
\Delta W_{actual}Q^\top=0.
\]

M2-R0 在任何新 certificate 研究之前先审计这一点。

## 2. M2-R0 — Protected Update Invariant Audit

状态：**FROZEN / UNRUN**

M2-R0 不是 continual-learning formal，也不会修改历史 M2 decision。它使用 exact M1 checkpoint 和 pinned WikiText B-train 仅产生真实梯度，固定 certificate，不做 certificate update、不做 growth、不读取旧 A replay。

每个有非零实际事务的 Cell/step 直接测量：

\[
\rho=\frac{\|\Delta WQ^\top\|_F}{\|\Delta W\|_F+10^{-12}}.
\]

`||ΔW|| <= 1e-12` 的 zero/tiny transaction 保留在 raw audit table，但不进入 rho 分布，也不能用于虚增 coverage。

冻结五个 arms：

| arm | optimizer | weight decay | grad projection | actual-update projection | 作用 |
|---|---|---:|---|---|---|
| canonical current | AdamW | .01 | yes | no | exact M2 mechanics |
| AdamW no decay | AdamW | 0 | yes | no | 隔离 adaptive preconditioner |
| SGD no decay | SGD | 0 | yes | no | 代数 reference |
| SGD with decay | SGD | .01 | yes | no | 在无 adaptive preconditioner 时隔离 decay |
| AdamW final-update projection | AdamW | .01 | yes | yes | 对完整 realized delta 做修复 |

final-update repair 定义为：

\[
U_{raw}=W_{after\ AdamW}-W_{before},
\]

\[
U=U_{raw}(I-Q^\top Q),\qquad W\leftarrow W_{before}+U.
\]

这保证约束作用在**实际 parameter transaction**上，而不是只作用在 raw gradient 上。

M2-R0 只允许产生：

```text
INCONCLUSIVE_REFERENCE_FAILURE
CURRENT_UPDATE_INVARIANT_HOLDS
WEIGHT_DECAY_BREAKS_UPDATE_INVARIANT
ADAMW_PRECONDITIONER_BREAKS_UPDATE_INVARIANT
BOTH_PRECONDITIONER_AND_WEIGHT_DECAY_BREAK_UPDATE_INVARIANT
MIXED_OR_INTERACTION_UPDATE_INVARIANT_VIOLATION
```

判定先要求 `SGD no decay` 与 `AdamW final-update projection` 两个 reference 通过严格 max/p95 gate；之后分别用 `AdamW no decay` 和 `SGD with decay` 识别 preconditioner 与 decay 的独立破坏作用。

即使发现并修复 optimizer invariant，M2-R0 也不能声称 continual learning 成功；它只关闭 implementation-level ambiguity。

## 3. M2-R1 — Functional Certificate Reconstruction

状态：**PLANNED / BLOCKED ON M2-R0**

R0 结束后，不立即运行新的 continual-language formal。R1 回到 exact final M1 representation，重新问：**certificate 到底应该表示什么？**

当前 M1 certificate 有三个已知弱点：

1. 每次 update 只记录 routed top-1 Cell，而 forward/gradient 中 `active_cells=2`；
2. 每 50 step 对历史 `cell_input` 只取一个 mean vector，mean 无法表示高方差或 superposed directions；
3. M1 certificate 是 shared representation 仍在移动时累计的，不等价于 final-M1 coordinate system 下的稳定 old-function basis。

R1 不改变 8-Cell topology，不做 growth。所有候选 certificate 必须在同一 final-M1 coordinate system 上构造并比较。

候选族：

1. **Historical baseline**：top-1 mean-vector basis；
2. **All-active weighted activation subspace**：按实际执行概率收集所有 active Cell 的输入，用 streaming covariance / randomized SVD 得到主要方向；
3. **Importance-weighted activation subspace**：不仅保存 direction，也保存旧方向 importance，允许 scaled trust region；
4. **Functional Jacobian / Fisher / Gauss–Newton sketch**：直接近似旧输出对 Cell update 的 functional sensitivity。

对 Cell：

\[
\delta z\approx J_i(x)p_i\Delta W_i h.
\]

定义旧功能损伤：

\[
D_i(\Delta W)=\mathbb E_{old}\|J_i p_i\Delta W_i h\|^2.
\]

R1 优先研究可持久化低秩/Kronecker 近似：

\[
F_i\approx A_i\otimes B_i,
\]

\[
A_i=\mathbb E[p_i^2hh^T],\qquad B_i=\mathbb E[J_i^TJ_i],
\]

于是：

\[
D_i(\Delta W)\approx\operatorname{tr}(B_i\Delta W A_i\Delta W^T).
\]

R1 的核心 gate 不是“rank 多大”，而是 candidate certificate 能否预测/约束 **held-out old-function drift**。必须满足 sketch/evaluation separation、all-active coverage、fixed final-M1 coordinates、明确 storage budget 和 capacity curve；最终机制不得依赖 raw old replay buffer。

## 4. M2-R2 — Fixed-Topology Replay-Free Continual Language

状态：**BLOCKED ON R0 + R1**

R2 才重新做真正的 Native continual-learning formal。

保持原 M2 边界：

```text
exact M1 start
8 Cells fixed
active Cells = 2
shared Transformer frozen
router + route keys frozen
growth disabled
B → C → D
learner raw replay = 0
```

只允许改变两个已经由 R0/R1 单独验证过的部件：

1. actual-update constrained optimizer；
2. selected functional certificate。

推荐写事务形式为 functional trust region：

\[
\min_{\Delta W}\quad g_{new}^T\Delta W+\frac{1}{2\eta}\|\Delta W\|^2
\]

subject to：

\[
D_{old}(\Delta W)\le\epsilon.
\]

R2 使用全新 untouched formal seeds；历史 M2/M3/M3R/M3L-2 seeds 不可复用为 untouched evidence。

end-to-end gate 至少保留：

```text
A absolute regression <= 20%
mean forgetting <= 15%
B/C/D phase gain >= registered plasticity floor
plasticity >= 80% matched control
shared/router frozen
zero learner replay
fixed 8-Cell topology
```

只有 R2 正式 PASS，才允许宣布：

\[
\boxed{\text{Native CLM basic replay-free continual-write primitive supported}}
\]

## 5. Growth 何时重新开放

在 R2 之前，任何新的 growth/mitosis formal 都 blocked。

未来 growth 应从安全写集合定义：

\[
\mathcal S_i(\epsilon)=\{\Delta W:D_i(\Delta W)\le\epsilon\}.
\]

若已有 Cell 能在安全集合内获得足够 new-domain gain，则继续 reuse/write；只有当所有可写 Cells 都无法安全吸收知识时，才有数学理由：

```text
safe-write infeasible
→ capacity allocation / mitosis
```

因此未来 mitosis 是 **safe-write infeasibility 的结果**，不再是独立 heuristic milestone。

## 6. 研究治理规则

1. **M2 未通过前，不创建新的高阶 Native CLM milestone。**
2. R0/R1 都是 diagnostics/mechanism validations，不允许被描述为 continual-learning success。
3. 每个 intervention 必须报告它关闭了原 M2 retention gap 的多少，而不是只报告局部指标改善。
4. 不再以“Cell 更像 Cell / router 更聪明 / growth 更稳定”替代端到端 retention + plasticity gate。
5. M3–M3W-0 继续保留为 failure-decomposition evidence，但不代表已跨越 M2。
6. 30M scale-up 和 M4 ontology 继续 blocked，直到 M2-R2 获得正式正结果。

## 7. 当前执行顺序

```text
M1                                 🟢 PASS
  ↓
M2 original                         🔴 NOT SUPPORTED
  ↓
M2-R0 actual-update invariant       🔵 FROZEN / UNRUN
  ↓
M2-R1 functional certificate        ⚪ BLOCKED
  ↓
M2-R2 fixed-topology continual CL   ⚪ BLOCKED
  ↓ only if PASS
growth / mitosis reopened           ⚪ BLOCKED
```

路线的核心不再是“怎样让 Cell 继续增长”，而是先建立：

\[
\boxed{\text{每一次真实参数事务都能被证明处于旧功能允许的安全区域内。}}
\]
