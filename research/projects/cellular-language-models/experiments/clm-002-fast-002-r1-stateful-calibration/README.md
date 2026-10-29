# FAST-002 R1：真实状态与完整尺度的工程修订

本目录独立保存工程修订。FAST-001、FAST-002、R3f-v2 的配置、数据、失败记录和结果均不覆盖。

当前交付 Tasks 1–5：历史只读审计、Admission 注册、真实 pre-proposal 暂停与恢复、R3e 候选优化、R3f-v2 指标的22尺度扫描、成本与回传。**不启动 Gate1 的真实重放/校准，更不启动 Gate3。** CLI 明确拒绝 `--run-diagnostic`，直到独立审阅和新的执行授权到位。

## 已实现的关键语义

- 精确复用 FAST-001 的主更新方法，在第50次主更新完成、候选搜索尚未开始的位置暂停。该时点 engine.step=49、stream.step=50，有一个已完成主更新但尚未逻辑提交的事务；绝不伪造为普通 Step50 完成状态。
- 保留模型、Cell优化器、Connection优化器生命周期、梯度、真实第50批、所有 RNG、数据游标、模块模式和哈希链边界。恢复不重做第50次优化。B01 的Cell冻结；B11真实更新50次Cell。
- 仍为12状态×6动作、8次均值损失AdamW；CPU测试对照原始R3e方法验证候选参数与损失序列完全一致。
- 固定0和2^-0…2^-20共22个尺度，全部评价后选最大合格正尺度；包括1/4096，禁止早停、零更新冒充成功。官方模型从不加载诊断候选。
- 使用 R3f-v2 原始双精度 raw-softmax KL、同状态硬动作变化和独立闭环路径变化；插值容差固定rtol1e-5/atol1e-7，不按结果调整。
- Admission使用独立种子271027004、范围[3000,4000)、新角色及样本ID，Story/Arithmetic各16窗口，共32固定输入；与旧Train、旧Confirm及FAST002新Confirm的完整文本、规范化文本、窗口、输入和目标哈希排除。
- Confirm只在隔离的注册哈希审计器中读取身份，不进入运行时数据重建、候选、尺度或质量评价。原next-token损失和所有参数冻结规则不变。
- 10个真实A父状态的云端恢复复用前次Kaggle恢复组件。CPU Notebook仅作身份/优化器读取，不执行前向、反向或训练。

## 必须交独立审阅的两个问题

1. 新方案只要求非零参数变化；R3f-v2的有效更新还要求非空硬路径变化。两套结果并列保存，不悄悄放宽/加严。正式采用哪一套必须审阅冻结。
2. 20上下文各重放50次已用完1,000次上限；若还要求每条真实GPU轨迹独立重放第二遍，将超出上限。本轮CPU夹具证明严格重复一致性及精确恢复；它不构成历史GPU位级重放认证。需要明确真实重复验证的预算及判据，不能伪造PASS。

## 如何运行当前交付

`notebooks/kaggle-r1-cpu-gate0.ipynb` 是 **CPU-only** 的只读 Gate0 工程入口；Internet、原 `GITHUB_TOKEN` 和 `KAGGLE_API_TOKEN` 即可，无需挂载Input、版本号或Seed输入。它恢复所需A父状态、重建并校验原Train字节、核验Admission、回传报告，然后停止，等待审阅与 Gate1 明确授权。

完整源码CI和CPU测试先在本地/GitHub完成，Kaggle不跑旧测试套件。`--publish-existing`仅回传已有报告。权重、原始文本、Token数组和私有状态不进入结果分支；签名URL不落盘。

历史GPU资产在本机未加载时，审计明确显示PENDING。工程或CPU测试通过不代表 Gate1/Gate3 通过，更不代表联合训练的科学收益已获证实。

## Runtime evidence fix001

The CPU Notebook explicitly includes `verification/CPU_VALIDATION.json`, which is consumed by report finalization. It repairs a prior sparse checkout before asset access, reuses the existing asset cache, and writes to `fast-r1-cpu-output-evidence-fix001` to preserve the failed output. Scientific runtime source/config/data and Gate1/Gate3 restrictions are unchanged. The previous published parent audit already verified all10 real A Checkpoints; that failure occurred afterward during report assembly.
