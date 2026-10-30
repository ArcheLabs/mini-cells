# EPCLM-002-A0：持久状态回忆

**状态：** 协议已冻结，正式运行尚未开始（`FROZEN_UNRUN`）
**范围：** 随机 Key-Value 事实的跨进程持久化与未来查询；不训练自然语言模型，也不声称替代 Transformer。

## 仓库检查与实现选择

当前 checkout 为 `codex/hybrid-clm-release-preparation-v0.1`，检查时 HEAD 为 `7ba121f`。仓库中已有以下相邻研究资产：

- `history-compression-001` 研究历史校准提示预算，不是 Episode KV 回忆；
- Native CLM M3L-1/M3L-2 已研究持久查询地址草图，但服务于语言流持续学习，不是 A0 的随机映射任务；
- Constructive CLM-005 含学习型路由、写入和增长控制，但并未实现本实验的 Writer/Reader 隔离；
- TextNCA、CLM-005、Core Validation 和历史压缩目录为参考实现与报告格式提供了现有约定。

本仓库没有找到可直接复用的 EPCLM-002-A0 数据生成器或独立 Writer/Reader 运行器。已有知识库库存说明，CLM X3/X4 专名设计文档的精确仓库搜索未定位；这不等于其他分支、旧会话或外部存储中不存在。实现不假定其内容，也不把它们写成现有代码能力。

实验实现放在 `scripts/research/epclm_002_a0/`，沿用 `research/validations/*/protocol.json`、`research/catalog.yaml` 和 `tests/research/` 结构。没有修改 MiniCells 核心模型、既有实验结果或既有协议。当前工作区在本任务开始前已有知识库文件的未提交变更；本实现没有改动这些文件。

## 协议

机器可读预注册见 [`protocol.json`](protocol.json)。每个 Episode 独立重采样 Key-Value 映射。Writer 输入仅包含事实与干扰事件；Writer 子进程退出后，控制器删除 Writer 输入，再独立采样查询并启动 Reader。Reader 的用户数据文件白名单仅允许序列化状态、查询、必要模型 checkpoint 和自身结果文件。Reader 进程内的审计钩子拒绝白名单以外的文件打开、目录枚举、子进程和网络操作，并在数据目录下保留一个显式拒绝路径。状态文件的 SHA-256 在读取前后复核。

任务初始配置采用 16-bit Key/Value、事实数量 `[1, 2, 4, 8, 16, 32]`、干扰数量 `[0, 16, 64]`，查询在写入结束后针对本 Episode 所有最终事实生成。覆盖事件作为单独条件，记录新值准确率、旧值残留率和未覆盖事实准确率。

## 对照和主张边界

- `full-context oracle`：仅由评估器计算的上限；不会交给 Reader。
- `no-memory`：只看 Query 的固定猜测，16-bit Value 的理论准确率为 `1/65536`。
- `direct-address`：序列化的显式 Key-Value 字典，是工程参考上限。即使完全通过，也不能作为学习型 Cell 记忆的证据。
- `flat-recurrent`：单一 GRU 状态向量，状态预算与 Cell 基线相同（8,192 bytes）。
- `learned-cells`：学习型事件编码器和 Query-Key-Value 读出，每个写入事件映射到一个潜在 Cell；仅训练 Episode 使用由最新匹配事实位置导出的辅助槽位标签，Reader 推理时只接收状态和查询。每次查询都会评分固定的 128 个 Cell。它可以用于 H2，但因全量扫描不能支持 H3。Cell 状态和有效位共 8,320 logical bytes；平坦 GRU 状态也固定为 8,320 bytes。
- 学习型稀疏路由：A0 v1 不实现。不得把 `learned-cells` 的全量扫描结果称为 H3 成功。

H1、H2、H3 分开报告。H1 参考实现通过只能支持跨进程序列化和上下文移除协议；H2 必须由学习型模型在三个正式模型种子上分别通过全部门槛；H3 还要求读取至少 95% 准确率、候选数不超过 Cell 总数的 25%，且 `full_scan=false`。

## 运行

仓库虚拟环境须安装 `torch`。先执行低成本开发运行：

```bash
.venv/bin/python -m scripts.research.epclm_002_a0.validate
.venv/bin/python -m scripts.research.epclm_002_a0.train \
  --model learned-cells --seed 98100 --steps 500 --batch-size 8 \
  --output artifacts/experiments/epclm-002-a0-persistent-state-recall/dev-learned-cells.pt
.venv/bin/python -m scripts.research.epclm_002_a0.runner \
  --backend direct-address --episodes 16 --seed 80100 --facts 8 --distractors 0 \
  --output artifacts/experiments/epclm-002-a0-persistent-state-recall/pilot-direct
.venv/bin/python -m scripts.research.epclm_002_a0.runner \
  --backend learned-cells \
  --checkpoint artifacts/experiments/epclm-002-a0-persistent-state-recall/dev-learned-cells.pt \
  --episodes 16 --seed 80100 --facts 8 --distractors 0 \
  --output artifacts/experiments/epclm-002-a0-persistent-state-recall/pilot-learned-cells
.venv/bin/python -m scripts.research.epclm_002_a0.validate \
  artifacts/experiments/epclm-002-a0-persistent-state-recall/pilot-learned-cells/result.json
```

正式训练固定为 500 步、批大小 8，在固定验证集上每 50 步选择最佳 checkpoint。正式评估需对每个事实/干扰条件执行三个正式模型种子，每个组合至少 1,000 个 Episode，并使用协议中与模型种子一一对应的独立评估种子；正式模型训练调用必须带 `--formal`。开发种子不能替代正式种子，测试集不能参与训练、模型选择或门槛调整。正式矩阵命令如下：

```bash
for seed in 98211 98212 98213; do
  .venv/bin/python -m scripts.research.epclm_002_a0.train \
    --model learned-cells --seed "$seed" --formal --steps 500 --batch-size 8 \
    --output "artifacts/experiments/epclm-002-a0-persistent-state-recall/checkpoints/learned-cells-seed-${seed}.pt"
done
.venv/bin/python -m scripts.research.epclm_002_a0.run_matrix \
  --backend learned-cells \
  --checkpoint-dir artifacts/experiments/epclm-002-a0-persistent-state-recall/checkpoints \
  --episodes-per-condition 1000 --formal \
  --output artifacts/experiments/epclm-002-a0-persistent-state-recall/formal-learned-cells
```

`direct-address` 矩阵使用 `run_matrix --backend direct-address --episodes-per-condition 1000 --formal`，只用于隔离与工程参考检查。正式汇总不会从未完成的条件运行中生成。

`direct-address` 单条件试运行示例：

```bash
.venv/bin/python -m scripts.research.epclm_002_a0.runner \
  --backend direct-address --episodes 16 --seed 80100 --facts 8 --distractors 64 --overwrites 2 \
  --output artifacts/experiments/epclm-002-a0-persistent-state-recall/pilot-direct-8f-64d
```

每次条件运行写出 `result.json` 和 `episodes.jsonl`；正式矩阵另写 `decision.json`、`summary.csv` 和 `summary.md`。结果包含协议哈希、实现文件哈希、代码 HEAD/分支、训练与评估种子、状态哈希、隔离审计、逻辑/序列化状态字节、Writer/Reader 耗时、Query 精确率、覆盖和未覆盖事实指标、读取候选数与 `full_scan`。

## 验收与当前状态

自动化检查位于 `scripts/research/epclm_002_a0/validate.py`，协议回归和最小跨进程测试位于 `tests/research/04-continual-learning-core/test_epclm_002_a0.py`。自动检查拒绝协议漂移、Writer 未退出、原始输入未删除、Reader 文件白名单错误、状态哈希变化、错误协议哈希和将直接寻址结果误标为学习证据。

已保存开发期试运行报告于 [`artifacts/experiments/epclm-002-a0-persistent-state-recall/development-pilot/README.md`](../../../artifacts/experiments/epclm-002-a0-persistent-state-recall/development-pilot/README.md)：Direct Address 为 128/128，学习型 Cells 在 100-step 和 500-step 开发模型上分别为 0/64、0/32。它们只验证管线与显示当前候选需要进一步研究，不是正式决策。正式三个训练种子、全部条件和每条件 1,000 Episode 的结果尚未运行；正式 H2 仍是 `INCONCLUSIVE`，当前 v1 learned-cells 明确全量扫描，所以该实现不能通过 H3 的稀疏读取门槛。
