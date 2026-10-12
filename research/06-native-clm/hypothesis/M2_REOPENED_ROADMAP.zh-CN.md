# Native CLM M2 重开路线图 — Safe Functional Write Primitive

状态：**M2 REOPENED / M2-R0b FROZEN UNRUN**

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

M3、M3R、Address Diagnostic、M3L、M3L-1、M3L-2、M3W-0 统一解释为 **M2 failure-decomposition series**，而不是已经跨过 M2 后的更高 continual-learning milestones。它们仍然是有效的机制证据，但当前 active question 回到最底层的不变量：

\[
\boxed{\text{新知识写入时，实际参数事务是否真的位于旧功能允许的安全区域？}}
\]

## 1. 当前 M2 protected write 的实现假设

对 Cell operator：

\[
y=Wh
\]

M2 certificate 保存 basis `Q`，并在 optimizer step 前执行：

\[
G_p=G(I-Q^\top Q).
\]

若实际更新是普通 SGD：

\[
\Delta W=-\eta G_p,
\]

则在精确算术中应有：

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

## 2. M2-R0 — Protected Update Invariant Audit

状态：**PUBLISHED / `INCONCLUSIVE_REFERENCE_FAILURE`**

发布 commit：`c1f9e132b026efe24a0238e6ea333bdd2ae5fbdb`。

M2-R0 不是 continual-learning formal，也没有修改历史 M2 decision。它使用 exact M1 checkpoint 和 pinned WikiText B-train 仅产生真实梯度，固定 certificate，不做 certificate update、不做 growth、不读取旧 A replay。

每个有非零实际事务的 Cell/step 直接测量：

\[
\rho=\frac{\|\Delta WQ^\top\|_F}{\|\Delta W\|_F+10^{-12}}.
\]

冻结五个 arms 的结果为：

| arm | mean rho | p95 rho | max rho |
|---|---:|---:|---:|
| canonical AdamW + gradient projection | 0.089760 | 0.155416 | 0.240160 |
| AdamW no decay + gradient projection | 0.089761 | 0.155417 | 0.240162 |
| SGD no decay + gradient projection | 0.004237 | 0.015315 | 0.090235 |
| SGD with decay + gradient projection | 0.324492 | 0.425478 | 0.434843 |
| AdamW + final realized-update projection | 1.90e-6 | 5.44e-6 | 1.53e-5 |

这组模式强烈提示：

1. adaptive preconditioning 会让 realized update 离开 certificate nullspace；
2. weight decay 单独也会破坏该约束；
3. 对完整 realized update 再投影，几乎可以把约束恢复到数值零附近。

但 `SGD no decay` 这个代数 reference 也没有通过原先严格的 relative-rho gate。其最小实际 update 约为 `1e-6`，而 absolute residual 约为 `1e-7`。这意味着 R0 不能区分“真正的几何违例”和“float32 parameter transaction 的 add/subtract roundoff floor”。

因此 R0 返回 `INCONCLUSIVE_REFERENCE_FAILURE` 是正确的研究治理行为，而不是把可能的数值误差强行解释成科学结论。

## 3. M2-R0b — Numerical Reference Audit

状态：**FROZEN / UNRUN**

注册文档：`research/stages/06-native-clm/M2_R0B_REGISTRATION.md`。

R0b 不重新运行 continual language，也不消耗新的 formal seeds。它保持：

```text
exact M1 checkpoint
同一个 pinned WikiText-B gradient source
audit seed = 75001
同样五个 optimizer arms
certificate/router/topology frozen
certificate updates = 0
growth = false
learner replay bytes = 0
```

对每个 certificate-ranked Cell/step，R0b 分离七层量：

1. gradient projection + clipping 后的 analytic transaction `-lr * g`；
2. 在 frozen certificate span 上用 fp64 得到的 safe projection；
3. parameter dtype 下的 `fl(W + Δsafe) - W`；
4. optimizer 原始 realized update；
5. 对该 optimizer update 做 matched fp64 safe projection；
6. 同一 update scale 下的 matched-safe parameter commit；
7. arm 可选 final-update projection 后真正持久化的 committed update。

fp64 reference 使用 frozen stored certificate rows 的 QR 正交化。它只修正数值正交误差，不改变 certificate 所表示的 span。

机器数值 floor 在 canonical run 之前冻结为：

\[
\max\left(
\text{empirical matched-safe float-commit residual},
8\epsilon_{dtype}(\|W\|_F+\|\Delta W_{safe}\|_F),
10^{-30}
\right).
\]

定义：

\[
E=\frac{\|\Delta W_{commit}Q^\top\|_F}{\text{machine-floor envelope}}.
\]

冻结 gate：

```text
minimum audited Cell updates / arm = 128
fp64 ideal projection rho max <= 1e-10
reference excess p95 <= 2
reference excess max <= 4
material structural excess p95 >= 16
material committed rho p95 >= 1e-4
```

reference 仍然是：

- `SGD no decay`
- `AdamW final-update projection`

只有两个 reference 都通过，R0b 才允许关闭 numerical/mechanics ambiguity。

注册的 primary classifications：

```text
INCONCLUSIVE_SGD_NUMERICAL_REFERENCE_FAILURE
INCONCLUSIVE_FINAL_PROJECTION_NUMERICAL_REFERENCE_FAILURE
R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF
```

只有最后一个分类可以解除 M2-R1 的 blocker。它只表示 measurement/update-mechanics 层闭合，不表示 continual learning 成功。

## 4. M2-R1 — Functional Certificate Reconstruction

状态：**PLANNED / BLOCKED ON M2-R0b**

R0b 闭合后，不立即运行新的 continual-language formal。R1 回到 exact final M1 representation，重新问：**certificate 到底应该表示什么？**

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

## 5. M2-R2 — Fixed-Topology Replay-Free Continual Language

状态：**BLOCKED ON R0b + R1**

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

只允许改变两个已经由 R0b/R1 单独验证过的部件：

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

## 6. Growth 何时重新开放

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

## 7. 研究治理规则

```text
M1                                  PASS
  ↓
M2 original                         NOT SUPPORTED
  ↓
M2-R0 actual-update invariant       INCONCLUSIVE_REFERENCE_FAILURE
  ↓
M2-R0b numerical reference          FROZEN / UNRUN
  ↓
M2-R1 functional certificate        BLOCKED
  ↓
M2-R2 fixed-topology continual CL   BLOCKED
  ↓ only if PASS
growth / mitosis reopened           BLOCKED
```

1. **M2 未通过前，不创建新的高阶 Native CLM milestone。**
2. R0b/R1 都是 diagnostics/mechanism validations，不允许被描述为 continual-learning success。
3. 每个 intervention 必须报告它关闭了原 M2 retention gap 的多少，而不是只报告局部指标改善。
