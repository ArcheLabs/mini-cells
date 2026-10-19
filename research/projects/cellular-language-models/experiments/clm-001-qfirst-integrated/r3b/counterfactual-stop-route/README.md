# CLM-001-R3b-CF-001

独立、只读的 Step 500 Greedy 路由反事实诊断。历史训练实现、权重和102项诊断不变。

## Kaggle

在原私有 `bingjiangli/minicells` 导入本目录 **notebook.ipynb**，打开 Internet、GPU 和已有 `KAGGLE_API_TOKEN`、`GITHUB_TOKEN`，然后 **Save & Run All**。
Notebook 自动执行全部阶段，无需填写 SHA 或切换模式。Kaggle API 凭据仅用于读取私有历史 Output；不是训练认证。
源代码与输入缓存放 `/kaggle/temp`，新私有结果放 `/kaggle/working/minicells-r3b-cf001`。不下载到用户电脑，不写HF。

执行顺序：

1. 从已保存历史Output按公开task key和SHA回读原59项；缺失即停止，绝不重算旧102项。
2. 复用原核验器校验57条Checkpoint和资产，加载QC Step500。
3. Calibration一次性固定MC8和状态数；Development不参与预算选择。
4. Native Greedy和NOOP严格对照，再核对历史128个Story的指标和窗口身份。
5. CF2提前STOP、CF3重复约束、CF1双后缀价值审计，Story配对统计。
6. 原子保存私有chunk，公开聚合与图表回传GitHub。再次运行自动回读Saved Output，幂等续接。

默认段长用于已完成任务后的打包，不杀进程、不丢弃任务；它不是运行时间保证。Saved Output持久化须在另一次Session回读后才能算已验证。

## 工程验证

CPU自动测试仅使用合成模型/合成Saved Output、临时Git远端；Kaggle中不重复跑测试套件。
`SPEC.md`保存原文，`FREEZE.md`明确实施冻结与统计限制，`config/frozen-cf-v1.json`固定身份和参数。

工程交付不等于真实诊断完成。本地无法验证私有Kaggle Output；真实旧59回读、真实基线、GPU耗时、CF结果与新Output跨Session验收仍待Notebook执行。
