# EPCLM-006 仓库审计

基线为远端 main 99aff2e83a9b08d76f860d1d322dff8fd68a7f5f，创建分支前工作区干净。M2 E3 是 8 层交替 Mamba-1/SWA，dim 256，state 20，W32，FFN 848，vocab 8192，历史静态参数 10,135,808。E3 的 full forward 从初始状态开始，cached 入口显式携带各 Session 独立的卷积/递归张量和 SWA KV。M2 已有 T4 工程结果，但本实验仍需运行自身 GPU 门禁。

冻结 M3 提供 NeuralMemory、Cell、Fast Weights、Momentum、在线更新与 Fork。M4 和 L1-C 的恢复代码可作为分离状态和跨进程的参考。L1-C 工程证据来自单独分支，本实验不依赖其未合并的代码或质量结果。E3 与 Cell 共享静态权重版本，Canonical 编码按独立片段运行，不使用调用方的 Runtime。保留 M2 原 Trainer、Sampler、Ledger 和原子模型 Checkpoint。

M3 的历史 Cell Checkpoint 使用通用 torch 反序列化。新 Portable Artifact 使用同类张量映射，导入时限定 weights_only，额外检查白名单结构、形状、dtype、兼容身份与摘要。本机只有 Quadro K620，正式 T4 门禁由 Notebook 执行。
