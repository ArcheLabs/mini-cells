# CF-001 实施冻结补充 v1

用户已明确授权立即实施代码、CPU 测试和 Notebook；SPEC 的草案头部不再构成本轮工程阻断。
本提交没有运行 Kaggle、GPU、旧 59 项回读或新的真实诊断。

- 训练、引擎、权重身份以 frozen-cf-v1.json 为准；不训练，不读取 Confirmation。
- D4/D8/D12 均为零起始决策深度；保持冻结 Registry.plan，冲突即失败。
- 历史逐 Token 状态没有保存，不能声称核对过历史指纹。核对历史逐 Story NLL、窗口身份、终止统计、转移矩阵；新 NOOP 与原始执行器逐动作、状态严格一致。
- NLL 预声明容差 atol=1e-6、rtol=1e-5；动作和状态复算严格相同。跨 CUDA 环境不能达到此条件时失败，不自动放宽。
- MC 固定 8 次。95% Student-t(7) 近似区间（临界值2.364624251），不是分布无关保证；跨零为 UNRESOLVED，不计为判断正确。两个后缀策略独立汇报。
- 状态数只由 Calibration 的完整重放+双后缀耗时选择：512/1024/2048 中预测 CF1 不超过3600秒的最大值，均超出则512并明确报告预测超时。这是工作量估计，不终止计算。Development 结果不能改变样本数或MC次数。
- 每个任务结束后安全保存；默认6900秒段长仅在下一个任务开始前打包返回，不发信号，不杀进程。单项任务可能超出估计。下一次相同 Notebook 自动恢复。
- 联合分层：depth×previous(0/3/5/CTX/OTHER)×CTX次数(0/1/2+)×原路径forced_end。先每层1个，然后哈希顺序均分剩余名额。层内固定哈希排列；pi=n_h/N_h，权重N_h/n_h。
- CF1 使用设计加权Hájek比例、Story cluster bootstrap。只覆盖抽中 Story，披露未抽中数；不宣称 Token 独立或未加权自然频率。分层bootstrap近似、MC近似与有限状态抽样均为限制。
- CF2/3 以全部128 Story等权配对，触发子集单列。2000次Story bootstrap；各指标分别对CF2三策略、CF3三策略做Holm校正。所有p值仅探索性，不是多Seed确认。
- 新私有原始结果只保存至私有Kaggle Output。Save不是本地文件存在；下一Session按SHA回读后才标为cross_session_verified。无逐任务HF写入。
- 原始输入全量runner.verify与57条索引；每任务参数版本/地址/shape/dtype、Registry与精确RNG检查，结束阶段完整模型SHA与原文件SHA再核对。不在每状态重复搬运整个模型计算SHA。
- GitHub仅聚合与图表。写权限预检复用原认证组件，明确分开网络fetch、分支准备、dry-run耗时；不把日志空档都归为权限检查。

仍需真实 Kaggle 验证：旧59回读、真实Baseline parity、GPU耗时/显存、CF1–CF3、Saved Output跨Session回读。工程测试不等于完成科学验收。
