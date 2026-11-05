# EPCLM Stateful Computation & Routing Experiment 001 — v0.2 修订条款

状态：训练前协议修订稿。与 v0.1 配套阅读，冲突时以本条款为准。未修改仓库，未启动训练。

## 已确定的设计

保留 T0/M0/M1/M2，不增加第五组。M0 是整个递归动态计算方案的可行性基线，不单独证明动态选择优于固定路由。

Router 训练采用 Categorical(softmax(score / 1.0)) 采样，推理 argmax；每有效 Token 两步，每步仅执行选中 CellCore。禁止含混的 ST。记录 Router 梯度、评分开销和实际 CellCore 调用次数。

采用 EMA baseline 的 Policy Gradient；最小化 `(L.detach()-b_old)*log p(action)`，同时正常反向传播语言 CE。Baseline 不依赖当前动作，必须先用旧值计算目标，再更新 EMA。损失归因范围、动作聚合、EMA 初值/系数、目标权重和 overflow 回滚方式仍须冻结，不能由实现任意选择。

## Workspace、TBPTT 和更新单元

最终 Token 表示才写入共享 KV，因此 M0/M1/M2 都存在 Token 递归依赖。参考 full-sequence forward 是同一递归定义的 scan，不是普通并行 Self-Attention。

每个有效 Token 一次历史 read、两步 CellCore、一次最终 KV append；首 Token 空历史返回零上下文。文档边界同时 reset KV 和 State。

完整文档或冻结完整文档组作为 optimizer update 单元。Chunk 反向后同时 detach KV 和 State，保留数值并累积参数梯度；组结束才更新参数，下一组使用全新缓存。不得在更新参数后继续使用旧参数产生的缓存。LR 按已提交有效预测目标数调度。

## State 顺序与消融

保存 h_before；计算 u=F(h_before,e_i,s_old)；更新 h_after=h_before+W_C*u；再计算 s_new=U(s_old,h_before,u)。非原地写回选中 Cell，未选 Cell 不变。第二步重复选择同 Cell 时读取刚更新的状态。

分别定义 CellCore-State-Read-Off、Router-State-Read-Off、StateUpdate-Recurrence-Off，以及同时关闭三者的 All-State-Read-Off。另保留 Write-Off 和 Reset-at-Chunk。

实施前必须冻结每种关闭方式：Router 推荐令 W_S*s 项为零；CellCore 和 Update 的状态替代值需固定为零或对应 Cell 的初始值之一，不能混用。All-State-Read-Off 不是 Write-Off。M1 仅是 Router 无直接 State 输入，仍可能受隐藏表示间接影响。

Shuffle 必须冻结置换、执行时点和读写索引语义，不允许每次读取重新随机打乱。

## 文档数据与公平评估

建立 document_index.jsonl，记录 doc_id、split、Token 范围、有效目标数、文档哈希、EOS 策略；另存 Chunk、文档组、前缀和 loss mask 清单。没有可靠边界则阻断，不将随机 125 Token 片段称为连续文档。

所有模型及消融从相同文档起点重新执行，相同前缀、Chunk、窗口、计分位置；消融从文档起点启用，不复用正常路径缓存。Padding 和不计分前缀不计预算，EOS 统一处理。

完整文档更新与精确 1M/500K/250K 预算存在兼容问题：建议三个数字作为上限，使用其下最大完整文档组前缀，冻结实际目标数且四模型相同。若采用其他规则须先明确；不得最后截断文档仍称完整文档更新。最低有效预算也须明确实际下限。

## 筛选与统计

Δ10=(NLL_M1-NLL_M0)/NLL_M0；Δ21=(NLL_M2-NLL_M1)/NLL_M1。负值分别表示 M1、M2 改善，不是后一个模型改善。

初步改善：相对 NLL 至少降低 1%，文档配对 Bootstrap 95% 区间支持改善。消融：相对正常 NLL 至少恶化 1%，区间支持退化。

冻结文档重采样 Seed、次数、区间算法及聚合方式。建议每次重采样重新计算 token-weighted NLL 比值，避免文档平均与 Token 平均混淆。区间只反映当前训练实例的文档样本不确定性，不包含 Seed 方差，也不默认完成多重比较校正。

未达改善门槛的结果不能全部标为 NO_CLEAR_DIFFERENCE：明确反向结果标为 OBSERVED_DEGRADATION；缺失则 NOT_RUN/INCOMPLETE。参数和优化路径的混杂仍须报告，不能声称 State 能力不可被无状态计算替代。

## 时间与 Warm-up

180 分钟是总上限，允许提前退出。唯一计时起点覆盖依赖、准备、审计、训练和发布；保持累计 10/15/25/105/150/165/180 分钟截止。165 分钟后不再启动训练或评估。

Warm-up 包含真实文档流、跨 Chunk KV/State、反向和 optimizer step，并覆盖较长 KV 与完整文档组。含初始化及保存的训练估算不超过 80 分钟窗口的 80%。选择最高可行共同预算后，重置模型、优化器、Scaler、baseline、全部 RNG 和数据游标。

## 开训前剩余冻结项

1. Policy Gradient 的截断 credit assignment、EMA 和归一化；不将截断代理梯度宣称为全文档目标无偏估计。
2. 完整文档组划分、实际有效目标预算、Token LR schedule。
3. State/Router/CellCore 维度、State 初值及更新器、所有消融替代值。
4. 文档资产和评估清单、Bootstrap 配置。
5. 统一精度、梯度裁剪、非有限数检查和溢出重试的状态一致性。

本次只修订协议。真实审计值、数据哈希和 GPU 速度须实际测量；未运行结果不得写 PASS。短距离可解性检查通过后，才可探索长程任务。
