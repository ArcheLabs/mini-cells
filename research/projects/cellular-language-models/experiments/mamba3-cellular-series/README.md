# Mamba-3 Cellular（M3C）

**状态：** PROPOSED_NOT_REGISTERED  
**研究阶段：** 架构提案与实验设计  
**训练授权：** 未授权（0 个正式训练 Token）  
**建立日期：** 2026-11-05

M3C 研究一种以 **Mamba-3 为层级主干、以无持久状态的平面 Cell Pool 为可选计算扩展** 的序列模型。

核心结构：

1. 保留 Mamba-3 的原生 Layer Stack、递归状态、SISO/MIMO 计算路径和官方数值语义。
2. 在 Mamba-3 Block 的层间边界设置可选 Cell 调用点；每个 Token 在每一层都可以跳过，或调用一个平面 Cell Pool 中的计算单元。
3. Cell 调用包含对有界 Workspace 的 Attention 式读取；同一 Mamba 层内的所有 Cell 共享该层 Workspace，不为每个 Cell 复制 KV。
4. Cell 不拥有跨 Token 持久状态，不建立固定 Cell-to-Cell 拓扑，也不取代 Mamba-3 的层级主干。
5. Cell 输出通过受控残差回到主干，供后续 Mamba-3 层继续计算。

## 核心文档

- [M3C 系列实验设计与提案](DESIGN.zh-CN.md)
- [参考资料与研究边界](REFERENCES.md)

## 实验路线

| 编号 | 研究内容 | 状态 |
|---|---|---|
| M3C-001 | 官方 Mamba-3 SISO/MIMO 基线与运行时复现 | 未开始 |
| M3C-002 | 有界 Shared Workspace Attention | 未开始 |
| M3C-003 | 固定调用的 Stateless Cell Adapter | 未开始 |
| M3C-004 | 跨 Mamba 层的动态 Cell / Skip 路由 | 未开始 |
| M3C-005 | 同层多次 Cell 调用与自适应计算深度 | 未开始 |
| M3C-006 | Mamba-3 Chunk State Memory 与 State-Memory Attention | 未开始 |

## 与历史研究的关系

M3C 是新的实验系列，不修改或取代以下历史记录：

- [CLM Workspace Controller-001 / v0.4](../clm-workspace-controller-001/README.md)：Shared Workspace、Stateless Cells、Controller 的历史设计；其 Smoke 与 Pilot 状态必须按原报告理解。
- [EPCLM-001](../epclm-001-incremental-execution/README.md)：增量与事件驱动执行研究。
- [EPCLM-002-A0](../epclm-002-a0-persistent-state-recall/README.md)：持久状态回忆机制研究。
- [EPCLM Core-001](../epclm-core-001/README.md)：事件驱动持久 Cell 计算研究。

历史实验的负结果、未完成门禁和科学边界均保留，不作为 M3C 已成立的证据。

## 重要边界

- 本目录是系列级研究提案，不是冻结协议，也不是 GPU 运行授权。
- Mamba-3 的官方代码版本、模型配置、数据身份、硬件兼容性和精确训练预算必须在 M3C-001 Gate 0 中锁定。
- Mamba-2 的 SSD/DART 推导不能未经证明直接移植到 Mamba-3；M3C-006 必须先推导 Mamba-3 自身的 Chunk State Memory。
- 任何效率结论必须报告真实 Decode 延迟、Prefill 吞吐、显存、Workspace Cache、Recurrent State、Kernel/Dispatch 开销及能耗；理论 FLOPs 或逻辑稀疏率不能替代实测。
