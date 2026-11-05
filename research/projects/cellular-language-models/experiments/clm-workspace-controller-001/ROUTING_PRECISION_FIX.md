# Pilot Router 精度修复 v2

适用：双 T4、Seed17、六模型各 2M Pilot。保持模型参数、Top-2 定义、损失、初始化、目标数、原数值门限及 120 分钟总预算。

## 失败证据

结果分支 `kaggle/clm-v04-pilot-2m-seed17-c114ac0a548b-e4e99a54`，发布提交 `c83d53e83445917467a23f5b5a02914e4c207439`：

- Controller 等价性、精度速度检查通过；T0-P/T0-F/C0 GPU 一致性通过。
- C1/C2/C3 在长度 255/273 的部分执行方式下未通过一致性检查。
- 变化路径均在近似平分位置，输出最大误差约 0.244–0.257，大于 FP16 绝对门限 0.05。没有非有限数值证据。
- 原运行 `PILOT_BLOCKED`，`pilot_training_started=false`；不得解释成架构科学结论。原结果不修改。

## 本轮执行协议

1. Router 输入投影、State Router、Logit 相加、Top-2 与权重 Softmax 全部禁用 Autocast，使用 FP32。梯度继续传播；不增加 tie-breaking 偏置、量化或强制固定路由。专家继承外部 AMP。
2. FP32 Router 不能消除上游 AMP 扰动。若原六模型 GPU 一致性检查仍失败，允许**一次、训练开始前**统一切换 FP32；全部六模型重新执行相同因果、完整/缓存/逐 Token、逐层 KV 和 FP32 参考检查，门限不变。FP32 不通过则停止。
3. 原 FP16 失败证据与 FP32 检查均记录在 `engineering-gates.json/attempts`，有效精度写入协议、审计、冻结身份与 Worker 配置。两个 Worker、所有模型使用同一策略。不得跨精度/源码版本复用优化器 Checkpoint。
4. FP32 回退复用本次已经实际测得的 FP32 五组训练基准；双 Worker 吞吐仍按最终精度重新实测，再冻结预测预算。精度变更不授权缩减 Targets 或延长时间；所有预检时间仍计入现有预算。
5. Controller 回归单独比较旧的两次 GRU 与修复后一次 GRU。两侧诊断实例采用相同 FP32 Router，以隔离本次精度变化；冻结的历史源码不修改，不声称原 FP16 路径与新数值路径逐位等价。

本地只执行必要的 CPU 回归；真实 T4 一致性、吞吐和 Pilot 完成情况由 Notebook 产生证据。H1–H4 保持未评估，直至实际训练评估完成。

## 本地验证

三个必要回归通过（1.68 秒）：FP16 舍入会合并的 Top-2 边界在 FP32 Router 下正确区分、专家仍采用 CPU FP16 Autocast 且 Router/Controller 梯度有限；失败检查记录保留、仅一次 FP32 回退且第二次失败仍停止；原死 GRU 移除的预测与梯度等价性。修改的 Python 和 Notebook 代码通过语法检查。未执行真实 GPU 训练或跨 Kaggle 恢复测试；此处通过不代表 T4 门禁已通过。
