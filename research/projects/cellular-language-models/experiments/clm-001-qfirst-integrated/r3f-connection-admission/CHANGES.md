# 修改与复用清单

全部新实现位于独立 `r3f-connection-admission`，原 R3e 科学文件不变。

| 文件 | 原因 |
|---|---|
| `stable_connection.py` | 原始分数、稳定基线与 Margin 硬动作分离；训练采样继续委托 R3e |
| `diagnostics.py` | Calibration-only 原/新规则尺度扫描、零差异检查、一次性阈值校准 |
| `evaluation.py` | 同状态 raw KL/硬动作与独立闭环路径分开；非有限、插值与空更新守卫 |
| `policy_proposal.py` | 复用原 Q Collector；11档全部扫描、真实 Logit 插值验证、明确拒绝原因与分阶段成本 |
| `adaptation.py` | 配对 RNG、显式三种接受状态、EVALUATED 事务恢复、每提案 Trace |
| `training.py` | 显式复用原 Trainer，无新 CE/Cell 优化器实现 |
| `jobs.py` | 从已验证 job 骨架派生 train-only 入口，续接试运行，不在训练阶段读 Dev |
| `data.py` | 复用冻结契约，仅加载当前阶段允许的 Token 数组 |
| `storage.py` | 复用 Store/HF，增加固定原 R3e 身份的只读恢复与预算核验 |
| `runner.py` | 只读→单Seed→有条件完整矩阵→终点评价；聚合准入/适应/质量及新旧对照 |
| `publication.py` | 复用 Git认证，限定 R3f 聚合文件，结果提交以当前 main 为父 |
| `checks.py` | 新规格所需路由、空更新、有限值、三门槛、双架构受控适应与四中断点测试 |
| `config/frozen.json` / `releases.lock` | 新策略、回溯、门槛、原始来源和真实试运行门禁锁定 |
| Notebook / `scripts/run_r3f.py` | 固定 SHA 一次启动；Notebook 不复制训练循环 |
| R3f CI / `shared/scripts/ci/r3e_scope.py` | 新单元测试；隔离改动避免无关历史回归 |
