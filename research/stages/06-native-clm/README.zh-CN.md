[English](README.md) | 中文

# Stage 06 — Native CLM

状态：**ACTIVE**

Stage 06 把 Constructive CLM 已支持的机制带入真正的 token-predictive model。

## 固定路线图

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🟢 COMPLETE
  M2  fixed-topology replay-free continual language      🔴 NOT SUPPORTED
      certificate protection 有稳定因果收益              🟡 PARTIAL EVIDENCE
  M3  growth-restored continual language                 🔵 ACTIVE
  M4  Cell ontology / specialization analysis            ⚪ PLANNED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

M3 关闭前不升级 30M。当前关键问题已经变成：动态 topology 是否能够恢复 protected continual capacity。

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

## M0 — Architecture + execution — 🟢

M0 已完成 sparse routing、Cell-local gradient、certificate projection、dynamic spawn、optimizer enrollment、动态 checkpoint round-trip 和 generation。

## M1 — 真实 next-token 训练 — 🟢

M1 成功训练 12.15M Native CLM：

```text
validation loss       5.723429 -> 0.788535
perplexity             305.9523 -> 2.2002
active Cell fraction   2/8 = 0.25
```

详见 [M1_CLOSURE.md](M1_CLOSURE.md)。

## M2 — 固定 topology 持续语言学习 — 🔴 NOT SUPPORTED

M2 从完全相同的 M1 checkpoint 出发，只写 Cell operator，shared substrate 和 learned router 冻结，训练流为：

```text
B WikiText-2 raw
  ↓
C cleaned Python CodeParrot
  ↓
D Databricks Dolly
```

A/TinyStories 仅用于 retention evaluation，learner replay 为 0。

三个 formal seeds `73211 / 73212 / 73213` 全部得到：

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

但 protection 有稳定因果收益：

```text
protected mean forgetting     ~0.2115
unsafe mean forgetting        ~0.2790
retention advantage           ~0.0675
protected/unsafe plasticity    ~0.964
```

唯一失败 gate 是 absolute A retention：protected TinyStories regression 仍约 43.9%，高于预注册 <=20% 上限。

详见 [M2_CLOSURE.md](M2_CLOSURE.md)。M2 formal seeds 已消耗，禁止再次作为 untouched evidence。

## M3 — Growth-Restored Continual Language — 🔵 ACTIVE

M3 不修改 M2 gate，也不在 M2 seeds 上调参。新的问题是：

> 当 protected reusable capacity 不足时，自主、context-addressed Cell growth 能否比完全相同的 fixed-topology protected control 更好地保留最老知识，同时维持新域 plasticity？

由于上次 Kaggle session 终止后 M2 原始本地 data manifest 丢失，M3 会创建新的 exact Hub-revision-pinned snapshot，并在**同一个 snapshot、同一个 seed**下直接并发比较：

```text
GPU0  fixed_protected
      永远 8 Cells

GPU1  growth_protected
      从 8 Cells 开始
      最多增长到 16 Cells
```

两臂都保持：

- exact canonical M1 checkpoint；
- identical B -> C -> D data / seed schedule；
- learner replay = 0；
- certificate-projected Cell-local gradients；
- shared Transformer、query projection、norm、原始 8 个 route keys 全部冻结；
- 每 token 只执行 2 Cells。

### Autonomous growth signal

每 50 learner steps，growth arm 只能观察当前训练可见量：

```text
window train loss
Cell route hits
certificate rank
projected/raw Cell-gradient ratio
frozen-router query vectors
```

Growth controller 看不到 domain ID、phase name、evaluation metric、hidden novelty label，也不能访问历史训练样本。

当 protected-write pressure 持续存在时，选择：

```text
route_hits * (1 - projected/raw gradient ratio)
```

最大的 parent，并出生 child：

```text
W_child        = W_parent 完整克隆
route_key      = 当前冲突 context 的 mean query
certificate    = 空 / rank 0
parent_id      = lineage pointer
```

完整克隆 parent operator 是为了尽量避免 birth 时发生函数跳变；随后 child 提供新的 writable directions，并必须证明 post-birth reuse。

### Registered gates

三个 untouched formal seeds 都必须独立通过：

- fixed control 始终 8 Cells，且 A regression >=30%，真实暴露 fixed-capacity limit；
- growth 只能由 learner-visible signal 驱动；
- 生出 1–8 个 children，final Cells <=16；
- 至少 75% children 在出生后获得 >=512 routed token hits；
- active Cell compute <= dense-all-Cell 的 30%；
- B/C/D 每个 phase gain >=5%；
- growth A regression <=20%；
- 相对 matched fixed control，A retention 至少改善 10 个百分点；
- growth mean forgetting <=15%；
- growth mean plasticity >= fixed control 的 80%。

Development seeds：

```text
73301 / 73302 / 73303
```

Untouched formal seeds：

```text
73411 / 73412 / 73413
```

Positive：

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_SUPPORTED
```

Negative：

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

冻结 protocol：[`../../validations/native-clm-v0-m3-growth-restored-continual-language/protocol.json`](../../validations/native-clm-v0-m3-growth-restored-continual-language/protocol.json)

Canonical Kaggle notebook：[`../../notebooks/06-native-clm/native-clm-v0-m3-growth-restored-continual-language-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m3-growth-restored-continual-language-kaggle.ipynb)

Notebook 会在正式训练前先验证两张 GPU 和 Hugging Face model repo 的 write permission；formal 完成后 6 个 fixed/growth end-state `.pt` 上传到 `archelabsxyz/native-clm-v0`，Git 只保存 lightweight evidence。

## 推进规则

如果 M3 SUPPORTED，保持相同模型规模进入 M4 Cell ontology / specialization analysis，然后再考虑 30M scaling reproduction。如果 M3 NOT_SUPPORTED，必须保留负结果并诊断 growth addressing / trigger / certificate lifecycle，禁止在 formal seeds 上 post-hoc 调参。
