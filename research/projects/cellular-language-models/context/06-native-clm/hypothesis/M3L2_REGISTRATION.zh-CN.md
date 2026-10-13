# Native CLM v0 M3L-2 — Online Historical Address-State Integration

状态：**FROZEN / UNRUN**

M3L-2 是 M3R → Address Diagnostic → M3L → M3L-1 机制选择链之后的新 formal continual-language 实验。

## 注册问题

从 canonical M1 checkpoint 精确起步，persistent rank-32 historical query address state 加 lineage-local affine parent/child gate，能否把 checkpoint-level addressability 转化为真实的 replay-free continual-language retention，并优于 matched M3R lineage-cosine algorithm？

这是新的 formal experiment，不会事后改变 M3、M3R、M3L 或 M3L-1 的既有结论。

## 上游证据

```text
M3R       NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
Address   QUERY_GEOMETRY_SEPARABLE
M3L       QUERY_SKETCH_GATE_NOT_FEASIBLE
M3L-1     LOW_RANK_CAPACITY_SUFFICIENT
           minimum passing low-rank Gaussian state = 32
```

M3L-1 publish commit：

```text
5ace6faf344b1b805752a33ffb861aeaf34dad6e
```

## 因果对照

```text
GPU0 / control
  exact M3R immutable-root lineage routing
  parent vs child = frozen cosine-key comparison

GPU1 / treatment
  same M1 checkpoint
  same protected Cell writes
  same growth pressure controller
  same immutable root router
  + persistent rank-32 historical query state
  + lineage-local affine parent/child gate
```

真正新增的 scientific variable 只有 lineage-local address mechanism。M3 growth thresholds 与 B → C → D schedule 不变。

## Address state

每个 active lineage leaf 保存一个针对 normalized frozen-router query 的 bounded Gaussian second-order state：

\[
A_i=(n_i,\mu_i,U_i,\lambda_i,\sigma^2_{i,\mathrm{res}}),\qquad \operatorname{rank}(U_i)\le 32.
\]

注册的 persistent budget：

```text
rank                         32
router query width           384
maximum bytes / Cell         52,360
regularization               1e-4
target historical FPR        0.10
```

不保存 raw historical token/query replay。

当前 stream 的 sufficient statistics 只是 ephemeral float32 first/second moments。每个训练 batch、每个 leaf 最多接收 256 个按确定顺序截取的 routed queries。每 50 step growth check 时，未分裂 leaf 把当前 sufficient statistics 合并进 historical sketch，并重新截断到 rank 32。

## Function-preserving mitosis

如果冻结的 M3 pressure rule 选择一个 leaf parent：

1. 冻结其 pre-window historical sketch；
2. current-window sketch 成为 child 的初始 address state；
3. child operator 精确复制 parent operator；
4. 根据 parent-history 与 current-window moments 构造 affine Gaussian-LDA edge gate；
5. root-level Top-K 与 root probability mass 保持不变。

Local decision：

\[
w^Tq+b>\tau.
\]

由于 birth 时 parent/child operator 完全相同，安装 gate 时 birth probe 的 logits 必须保持到数值容差范围内。

## 显式 bootstrap 边界

Canonical M1 checkpoint 产生时还不存在 query-address sidecar。因此 M3L-2 明确注册一次 **pre-continual bootstrap**，而不是假装这些 state 原本就在 M1 checkpoint 中。

```text
TinyStories train @ f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
10,000 documents
160 batches
sampling seed 74001
        ↓
为原始 8 个 roots 构建 rank-32 sidecars
        ↓
释放 one-shot bootstrap handle
        ↓
B → C → D 开始
```

Bootstrap 限制：

- 使用 TinyStories `train`，与 A retention 的 `validation` split 分离；
- optimizer steps = 0；
- Native CLM parameter updates = 0；
- certificate updates = 0；
- growth = 0；
- raw bootstrap queries 被丢弃；
- B 开始后 learner 无法再次访问 bootstrap data。

因此 M3L-2 的严谨口径是 **zero learner replay after continual start**，而不是“bootstrap-free M1 state”。

## Formal seeds

```text
development  74101 / 74102 / 74103
formal       74211 / 74212 / 74213
```

Formal seeds 在 canonical Kaggle formal runner 被明确执行之前保持 untouched。

M2/M3/M3R 已消耗的 formal seeds 全部列入 forbidden set。

## 主要 formal gates

Positive decision 要求所有 formal seed 的全部 registered gates 同时通过，包括：

```text
control A regression              >= 30%
treatment A regression            <= 20%
A retention advantage             >= 10 percentage points
treatment mean forgetting         <= 15%
B/C/D phase gain                  >= 5% each
treatment/control plasticity      >= 0.80
active fraction vs dense          <= 0.30
spawned children                  1..8
child reuse fraction              >= 0.75
address-state rank                <= 32
address bytes / Cell              <= 52,360
```

同时还要求 zero replay、exact M1 identity、matched seed/data、immutable root read function、function-preserving birth、address-state checkpoint round-trip，以及每个 spawned child 对应一个 affine gate。

## Canonical execution surface

```text
research/notebooks/06-native-clm/
  native-clm-v0-m3l2-online-address-state-kaggle.ipynb

scripts/research/
  prepare_native_clm_v0_m3l2_data.py
  run_native_clm_v0_m3l2.py
  publish_native_clm_v0_m3l2.py
```

Publication 采用 HF-first：6 个 final checkpoints 和 formal decision 必须先上传到 `archelabsxyz/native-clm-v0`，随后才允许 push lightweight Git evidence。

## Decision boundary

在 M3L-2 得到有效 formal decision 前，M4 继续 blocked。Checkpoint-only 的 M3L-1 positive capacity result 不足以授权 30M scale-up。
