# M3C-002B — Gate002 实验设计与预注册候选

当前阶段只形成研究协议，不扩展模型/训练工程、不重跑已通过门禁、不启动训练或开启正式测试。

- [完整预注册候选](PREREGISTRATION.zh-CN.md)：任务与泄漏控制、三臂配对比较、统计单位、两线预算、推进标准和Kaggle运行身份。
- [冻结候选清单](freeze-candidate.json)：明确候选值与尚未决定项，training_authorized=false。不是启动器可用配置。
- [工程证据快照索引](evidence-baseline.json)：从002A原结果提取，记录原分支/Commit/文件哈希及缺失Script Version ID；不改历史文件。

本次明确的关键设计：

1. MQAR/重复检索中，前一次答案不回填后续输入；按Episode聚类，避免泄漏和伪独立样本。
2. 合成Query NLL与自然语言NLL/PPL分开；完整Gate002需要两条线的训练/评估预算，不把Query PPL当成语言PPL。
3. Base/W16/W64在同一线内共享数据顺序及公共初始化；现有三臂不能分离额外参数量贡献。
4. 至少3个正式Seed来自原系列要求；确切数量、目标量及统计精度需要开发方差/真实训练成本支持。
5. 最小实用收益与成本容忍度尚无依据；明确保留空值，不编造推进门槛。
6. 原002A metadata中的Script Version ID确为null。新正式运行需要真实平台身份；不拿Git SHA、时间戳或猜测版本号补历史字段。

状态为 PREREGISTRATION_CANDIDATE_NOT_FROZEN，不称“已预注册/已冻结”。原设计只建议Smoke/Pilot范围且明确非授权，现有推理Benchmark无法替代训练成本测量。本稿不能据此生成正式预算。

完成正式冻结还需要：选定模型/任务资产，确定最小有意义效应和资源取舍，获得受限开发校准的方差/训练P95证据，再确定Seed、R/L两线目标数与样本量，封存协议/数据/身份并另行记录正式授权。
