# CLM003 阶段 0

本阶段只交付协议与 CPU 机制工程，不是 A/B/C/D/E 的完整训练实现。

已实现：使用真实原生 CLM 类的合成 CPU 夹具；A/B/C 共享合法随机 Connection；保持 Cell 与公共参数初值；隔离初始化随机流；C 事件仅匹配 A 的实际提交；配对训练 Seed 作为分析单位；JSON Manifest、元数据白名单和现有 GitHub 回传组件；独立 CPU Notebook。

未实现或未冻结：真实父 Checkpoint 与优化器/游标恢复、训练 Seed 和预算、实际 A 更新算法版本、独立数据角色身份、统计门槛与多重比较、MoE/Dense 实体模型与匹配容差、非劣界限及硬件测量方案。`config.json` 保留十项明确阻断，不擅自填写科学参数。随机重连幅度未匹配，不声称已排除幅度混淆。

既有 FAST002 的终点评价是历史 Confirm，不能静默改称独立 Development。新 Confirmation 未建立，因此不能声称确认性实验就绪。父模型已适应历史路由的事实必须继续披露。

Kaggle 导入 `notebooks/kaggle-phase0-cpu.ipynb`，选择 CPU、Internet 并勾选已有 GITHUB_TOKEN，Run All 即完成合成预检及 GitHub 回传。无需 HF_TOKEN、KAGGLE_API_TOKEN 或历史 Output。权重不进入 Git；`--publish-existing` 只重试结果回传；`--run-science` 明确拒绝。

本次实施不使用 GPU、不执行 Optimizer 更新、不读取模型质量、不改写历史实验。不对 MoE/Dense 的性能或架构可行性作结论。

## 当前最小 A/B/C 筛查入口

新增 `screen.py` 和 `notebooks/kaggle-abc-screening.ipynb`。筛查的唯一配置源为 `SCREENING_MANIFEST.json`（旁置 `.sha256` 仅校验字节），原 `config.json` 只属于已完成的阶段 0，不再决定筛查。

共同父状态是原 FAST001 `2026200004-A-H` 的 A500 Checkpoint，SHA 为 `2616ba9244ddda49c7fab07edc1af0d00aa09b8ad54865477fc847ea0faed62b`；其 Cells 已适应过历史 Connection，不是全新随机模型。三 Seed 为 2026300001、2026300002、2026300003，九条路径各 500 步，共 4,500 主更新、2,304,000 目标 Token。A 学 Connection 与 Cells，B 冻结同一随机 Connection 并训练 Cells，C 在每 50 步固定随机重连并训练 Cells，与 A 是否接受无关。重连幅度不匹配，只记录实际路径扰动。

筛查确认池为新的确定性生成数据，原文、规范化、65-token 窗口、输入和目标均与已知 FAST001/FAST002 Train/Confirm 及 R1 Admission 无精确交集；两份数组的 SHA 已在任何模型读取前冻结在 Manifest。此证据不覆盖全局历史语义独立性，也不构成独立确认性复现。

主要指标为等权两任务 Token-mean NLL；实用改善门槛为 0.01 nats。三 Seed 的 A−B 与 A−C 都须方向一致且均值≤−0.01，才提出另行实施 5 个新 Seed 与全新确认池的确认实验。若两均值均为负但未满足方向/幅度，报告不足并单独决定是否扩展；任一均值非负则停止昂贵基准。仅报告描述性配对 Seed t 区间，不将其解释为校正后的确认性检验。程序不自动扩展或运行 MoE/Dense。

Kaggle 使用 GPU、Internet、GITHUB_TOKEN、KAGGLE_API_TOKEN 后 Run All。历史 Output 自动按固定版本 116 和固定文件哈希读取，无需 Input 挂载。完整报告、九组结果、候选治理明细、失败记录和成本自动回传 GitHub；权重和数组保留 Output。完整结果回传失败时再次 Run All 只重试回传。跨会话必须先恢复本次 Output；不宣称已实现自动跨会话续跑。

新增改动相关测试及短程合成 A/B/C 端到端试验已通过；未重复旧阶段 0 门禁。实际九路径筛查尚未运行，不能提供真实 NLL 或成功结论。运行后查看 `SCREENING_RESULTS.json`、`COST_LEDGER.json`、`jobs/*.json`。
