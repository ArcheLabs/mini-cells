MiniCells / CLM-001-R3b：Step 500 Greedy 路由失稳反事实诊断实施规范
实验代号： CLM-001-R3b-CF-001
文档版本： v1.0-draft（待下一次实施前冻结）
日期： 2026-10-19
适用仓库： ArcheLabs/mini-cells
文档用途： 后续提交 Codex 执行；本次仅定义实验，不修改仓库、不运行诊断。
科学级别： 单训练 Seed、Development 集、只读、探索性反事实研究；不是 Confirmation，不得做正式显著性确认。
0. 一页摘要
0.1 核心问题
为什么 QC-R3b 在 Step 500、相同冻结 Development 样本上，Greedy 路由的平均非 STOP 动作数达到 14.738647/16，强制结束率达到 0.741943（74.19%），而同一模型的 Sampled/Uniform 路由强制结束率约为 13.10%/13.11%？
已有转移矩阵显示：Cell0→CTX=23403、Cell0→Cell0=22679、CTX→Cell0=15868、Cell3→CTX=9950；冻结 Calibration Probe 上，Q(STOP)-Q(best Cell) 的平均值约为 +0.06810（Q 越低越好）。这些是诊断线索，不是根因已被证明。
0.2 三项实验及判别目标
代号	研究对象	要验证的命题	主要结果
CF1	Greedy 实际访问状态的 STOP 价值校准	Router 是否在应停止的状态预测继续计算更有利？	预测 Q 差与实际反事实成本差的符号、偏差、错选率，按深度分层
CF2	Depth 4/8/12 强制提前 STOP	过长的真实执行路径中，继续计算是否造成质量/效用损失？	配对 ΔNLL、Δ成本、Δ效用、STOP/强制结束率
CF3	CTX 至多一次、禁止连续相同 Cell、二者组合	重复 CTX 和相同 Cell 是否是可避免的低效循环？	配对 ΔNLL/效用、重复转移、平均路径长度及终止情况


三项均 仅修改推理时的动作选择/诊断读取方式，不更新参数。严禁把新策略调优结果当作原先 QC Greedy 的性能。
0.3 本次不做的事
不重新训练、不更新 Optimizer、不改变模型参数、不生成新 Cell、不修改 R3b 原科研协议、不改变 Probe/Tokenizer/数据划分、不访问 Confirmation、不新建研究结论的“独立训练种子”、不将私有逐样本诊断公开至 GitHub、不以结果好坏来事后修改干预深度或阈值。
1. 输入身份与先决条件
1.1 锁定不可变训练输入
必须根据训练 Manifest、checkpoint-index.json 和实际文件 SHA256 读取，而不是依赖“最新 checkpoint”。
字段	冻结值
Kaggle Kernel	bingjiangli/minicells
训练 Output 版本	Version 65（文件清单及 57/57 Checkpoint 曾核验）
用户提供的页面参数	scriptVersionId=359735394（尚未证明与 Version 65 一一映射；不直接当 CLI 版本号）
原始训练源码	a7792ef30582dfcaf324cca7e5245c8430df48b2
内部 training run ID	r3b-a7792ef30582-ea3ea51c
原公开训练结果目录	33734fdd1558（并非内部 run ID）
冻结分析引擎源码	ed2accbc90f541325ca2f6c53229e9c81475768d
原始分析协议 SHA256	8f9bd833ab371ce8c624693222825269889da1fde1860f8be7a73f2bb2eb90d0
QC-R3b Step 500 post_event	57b9daaaa54b0a67c6bc43c2f216ac375d0bb127ffbda132ac00f4cd42caffbb
Q-R3b Step 500 post_event	7cd30b9107ecd1af5ab3215233716bdf71536e224e31ea76ab001198812ffcdf
QS-R3b Step 500 post_event	41dc44eb1e2a90faacd214d5ae169845228b5dfec5cd10afac56c2bedcf73610
模型更新步数	500，各组已完成；本轮必须为 0


备用的 Hugging Face 训练 revision 为 828585f1ff93420aa4485c725211ed28f70d15f3（仓库 libingjiang/minicells-research-checkpoints），但本轮以已校验的 Kaggle 历史 Output 为优先输入。切换备份源必须逐文件证明内容身份相同，不能静默替换。
必须调用现有 runner.verify() 等完整性验证，检查 57 条原历史 Checkpoint 记录、continuation.json、runtime-config.json、experiment.lock.json、Tokenizer、资产、冻结 Probe、event revision、训练协议和运行身份；同时验证目标 QC Step 500 Checkpoint 哈希。实验过程仅读取所需模型状态，不能复制覆盖旧 Snapshot。
1.2 先保全上次分析产物（独立存储验收）
前次诊断任务覆盖 102/102，但公开账本仍显示 43 项 HF_VERIFIED、59 项 COMPUTED_LOCAL；output_saved_version=OUTPUT_VERSION_PENDING_SAVE。这说明完整的跨 Session 私有结果回读尚未由公开账本证明。
下一次实施时应优先进行只读保全审计：查找这 59 项已计算的私有任务 JSON 是否进入某一已保存的 Kaggle Output；逐项核对 task key 和 result SHA256；生成独立保全审计结果。若已持久化则升级状态；若未持久化，不要伪报通过，也不要因为本次新的只读反事实研究而重跑旧 102 项。报告阻塞原因，由用户决定是否先补救旧数据；不应破坏旧研究结果。
原来的 Confirmation SEALED、formal_confirmation=false、scientific_decision=false 必须保持不变。
1.3 评估与行动语义
- 数据：原冻结 Development 128 个 Story 窗口 × 64 Token = 8192 条 Token 路径；Story 是主要的配对统计单位。Calibration 仅用于预检和数值检查，不得以 Confirmation 做选择。
- budget=16；minimum_actions=1；lambda_cost=0.01；动作成本单位为非 STOP 执行次数 /16；包括 Cell/CTX 的实际执行，不包括 STOP。
- 现有 Registry 中，6 个 Accepted Cell 加 1 个 Reserved Cell、CTX、STOP。真实可执行集合以该 Checkpoint 的 Registry.legal()、激活状态为准，不可擅自认为保留 Cell 已激活。
- R2 q_connection 返回相对于 STOP 的差值：q=raw - raw[..., STOP]。故 STOP 是数值零点（容忍数值误差），Q 数值越低代表预测成本越低。
- Greedy 在合法动作间 argmin Q，合法平手偏向 STOP；当前硬路由输出不带可微混合权重。
- CTX 可在多个深度出现；同一个 Cell 也可连续选择；执行的状态更新为 h + alpha × update。CTX 的重复调用不能先验认为无用，因为隐藏状态会变化。
- 训练使用采样策略，Step 500 的 epsilon 约 0.15；本轮不更改训练的任何 ε 或参数。若进行 Sampled 后缀反事实，须锁定原配置及随机数流。
- 检查 registry.plan 是否存在强制动作计划：若该 Checkpoint 对目标路径强制指定动作，必须披露具体作用范围；不可静默覆盖它再自称原始 Greedy。对无法满足“策略干预不覆盖已冻结强制规则”的情形应停止相应实验并报告协议冲突。
1.4 基线必须可复现
在不执行任何干预的情况下，新实现必须在相同身份的 Development 集上重现：
QC-R3b Step 500 Greedy	预期基线
NLL	4.699343863874674
平均非 STOP 执行动作数	14.7386474609375
actual_stop_rate	0.258056640625
forced_end_rate	0.741943359375
context_call_fraction	0.27624048567571374
self_loop_rate	0.20667809892755915
Token 路径数	8192


基线的 NLL、动作和状态指纹应尽可能与冻结分析一致；若环境只支持浮点容差，须在运行前声明误差阈值，并确保路由动作、终止位置、窗口身份严格一致。无干预基线不通过则不得公布任何反事实改善结论。
2. 总体研究设计
2.1 假设（先登记，后计算）
- H-CF1（STOP 价值错排）：在 QC Greedy 实际访问、且 STOP 合法的状态上，预测 Q(STOP)-Q(chosen)>0 的一部分状态，其真实成本感知反事实差却为负，即其实应当 STOP。
- H-CF2（过度继续执行）：强制在某个冻结深度 STOP，能够使同一批 Token 的配对成本感知效用改善，而不仅是降低计算开销；还应分别检验对 NLL 的影响。
- H-CF3（重复选择的非必要性）：禁止第二次 CTX 和/或连续相同 Cell，可降低冗余动作，同时保持或改善任务质量；如果仅改善成本却恶化 NLL，需用固定 λ 判断净收益。
- H-CF4（分布与策略放大，解释性）：Greedy 的高强制结束率对应于反复进入低价值的深层状态；固定 Calibration Probe 的边际均值不能代替 Greedy 实际状态分布。本项由 CF1–CF3 的分层记录共同回答，不另开第四个干预实验。
本轮不宣称必然存在某个唯一根因。 若证据相互冲突，状态为 INCONCLUSIVE，不得事后重新选阈值制造结论。
2.2 数据划分与复用
使用原来的冻结 Development 数据和全部 128 个窗口，不生成新数据。所有策略共享相同的窗口、Token 顺序、目标 Token、Tokenizer、模型、评估随机性规范和统计口径。
- 先在 Calibration 上执行少量无标签影响的代码一致性和资源 Pilot；不得用 Calibration 的成败挑选对 Development 最有利的干预深度。
- CF2 深度集合固定为 {4,8,12}；CF3 限制集合固定为 CTX-once、no-immediate-cell-repeat、both。不得追加其他深度/限制到“预注册主分析”；探索性追加另行标识。
- 所有结果保留“全体 Token 的意向处理效果（ITT）”和“真正触发干预的 Token 子集效果”。后一种存在选择条件，应标记为辅助描述，不能替代 ITT。
- NLL 使用原任务逐 Token 的 next-token CE；计算成本按实际路径长度和原协议 λ/B 计算。聚合时先对 Token 形成 Story 级均值，再按 128 个 Story 做等权平均；保留与历史工具对齐的原始平均，若二者定义存在差异要显式列出。
2.3 数据结果身份
新研究必须有自己的独立身份，例如：
experiment_id = CLM-001-R3b-CF-001
input_training_run_id = r3b-a7792ef30582-ea3ea51c
input_checkpoint = QC-R3b / step=500 / post_event / SHA256=57b9...caffbb
analysis_mode = READ_ONLY_COUNTERFACTUAL
new_analysis_protocol_sha256 = <冻结的新规范 hash>
new_orchestrator_source_sha = <实现代码 Git SHA>
analysis_mode、新协议和输出身份不能覆盖原始 R3b 的 analysis source SHA 或原任务 task keys。
3. CF1：Greedy 实际状态的 STOP 价值校准审计
3.1 科学问题
在真实 QC Step 500 Greedy 轨迹中，Router 没有主动 STOP 的那些决策点，继续执行是否真的比立即 STOP 更有利？如果不是，Q 的错误发生在哪些深度、前一动作、CTX 使用状态及强制结束路径上？
3.2 数据采集：首先跑出原始 Greedy 路径
逐 Token 执行冻结的原始 Greedy，流式记录以下必要字段：
story_id, token_position, depth_zero_based, state_fingerprint, previous_action, ctx_seen, n_ctx_so_far, consecutive_same_cell_count, legal_actions, q_scores, chosen_action, actual_stop, forced_end, action_prefix_fingerprint, target_identity。
另记录 Q(STOP)、Q(best Cell)、Q(CTX)、Q(chosen)、STOP 是否合法，以及最佳 Cell 的 ID。注意“最好 Cell”不一定等于实际执行动作（可能是 CTX 或 STOP）。
不要将所有 hidden、memory 大张量常驻 RAM。仅保留可复算动作前缀和必要的哈希；只读重放时重建原状态，验证重建指纹，再调用当前冻结 Oracle。对外公开的汇总不得包含完整 Token 文本/原始 hidden。
3.3 Q 预测差与反事实真值
针对每个被审计、且 STOP 合法的状态 s：
PredGap_STOP_CHOSEN(s) = Q(s, STOP) - Q(s, chosen)
TrueGap_STOP_CHOSEN(s) = J(s, STOP) - J(s, chosen -> suffix_policy)
其中 J 为原始成本感知损失：
\[
J(s,a;\pi)=\mathbb E[\mathrm{NLL}(\text{最终状态})+0.01\cdot N_{\mathrm{nonSTOP}}/16\mid s,\text{首动作 }a,\text{后缀策略 }\pi]\,.
\]
计算约定： 比较同一 s 和同一目标 Token 的增量剩余路径成本；之前已发生的成本两侧相同，应抵消。J(s,STOP) 立即使用当前隐藏状态的预测结果，不再执行 Cell/CTX。真实差 TrueGap < 0 代表立即 STOP 更好。
同一状态应分别计算两种后缀真值并严格分开：
1. Greedy-consistent：备选第一动作后，使用冻结 Greedy 重新规划剩余动作，回答“在当前部署策略下”的决策问题。后缀是确定性的，主要不确定性来自不同 Story/状态而非 MC 随机性。
2. Frozen-sampled-oracle：沿用训练原有 epsilon_boltzmann_frozen_q 后缀策略，使用预生成、配对的 common random numbers (CRN) 采样，回答“当前原 Q 训练目标”的校准问题。建议每状态起始 8 个配对后缀样本，若不确定性过大可扩展到事先冻结的 16/32；预算和扩展规则必须预先冻结。
二者不应被混为同一个“真实 Q”。Greedy 与 Frozen-Sampled 的差异本身是重要结果，可以定位训练—推理策略失配。
3.4 审计规模与抽样
建议按预先确定的 deterministic hash 从所有实际 Greedy 非终止决策状态中分层抽取约 2048 个反事实状态（数目可在实施前根据只读 Pilot 一次性确定并冻结）。分层必须覆盖：
- depth=1..15，特别是 4/8/12 及 13–15；
- previous_action 属于 Cell0/Cell3/Cell5/CTX/其他；
- ctx_seen 和 n_ctx_so_far；
- 来自最后 forced_end=true 与 actual_stop=true 的路径；
- 各 Story，而非反复抽同一 Story 的大量状态。
在 Stage 1 应先对所有 Greedy 状态做成本低廉的 Q 预测分布普查；高成本反事实只对固定抽样集执行。必须保存抽样种子、抽样哈希、抽样概率/权重、每层总体与实际命中数。某层数据不足则全部纳入并报告，不通过换层补数伪造覆盖。
抽样估计总体错选率时使用预先定义的设计权重；不得把按深度/终止结果过采样的样本直接当作全路径自然发生率。主统计单位为 Story，状态相关性按 Story 分簇处理。
3.5 主指标
- pred_stop_advantage = Q(chosen)-Q(STOP) 与 true_stop_advantage = J(chosen)-J(STOP)；
- STOP_misranking_rate：预测继续较优、而配对反事实指示 STOP 较优的比例；另区分 MC 不可判定状态；
- STOP 相对值偏差/MAE、按深度的方向一致率、预测差与真值差散点及分位数；
- forced_end 与 actual_stop 路径的错排率差异；
- Greedy-consistent 与 Frozen-sampled-oracle 两套估计的差异；
- CTX-vs-STOP、CTX-vs-bestCell 的预测/真值差（预算允许时在相同抽样状态记录）；
- 非法 STOP 状态不进入 STOP 错排率分母，而应单独统计（Depth 0）。
若真值区间跨 0，应标记 UNRESOLVED，不能强制分类为“错误”或“正确”。
4. CF2：Depth 4 / 8 / 12 强制停止
4.1 科学问题
如果 Greedy 路径中过多的 Cell/CTX 调用实际上没有改善最终预测，提前 STOP 是否能提高成本感知效用，并在多大程度上改善或损害 NLL？
4.2 干预定义
Depth 是零起始的决策索引：depth=0 表示第一轮动作选择。对 d∈{4,8,12}：
- 当 Token 在该决策点仍 alive 且 STOP 合法时，强制 action=STOP，完成本 Token 的推理；
- 若 Token 在此之前已主动 STOP，不作任何改变；
- 在干预点之前完全使用原始 Greedy，不重置隐藏状态和 CTX；
- STOP 不增加非 STOP 调用成本，不额外调用 Cell/CTX；
- 深度 4/8/12 均满足当前 minimum_actions=1，但执行器仍须动态校验合法性；
- 严禁将“触及预算自然强制结束”伪装成主动 STOP。
实验组必须包含 BASE_GREEDY、FORCE_STOP_D4、FORCE_STOP_D8、FORCE_STOP_D12，以及代码无变化时的 NOOP_CONTROL。
4.3 测量指标
对每个 Story、Token 都计算：
ΔNLL(d)     = NLL(force_stop_d) - NLL(BASE_GREEDY)
ΔCost(d)    = Cost(force_stop_d) - Cost(BASE_GREEDY)
ΔUtility(d) = ΔNLL(d) + 0.01 * ΔCost(d)
Cost = 非STOP动作数 / 16，与原 Frozen protocol 一致。所有差异先按相同 Story 配对，再做分簇估计。
另外分别记录：touched_token_fraction、actual_stop_rate、forced_end_rate、平均路径长度、CTX 调用次数、按深度/Story 的 NLL 恶化比例、触发前 Cell 频数。成功降低强制终止率本身只证明达到了干预预期，不证明改善质量。
4.4 预注册解释
- 若 ΔNLL < 0 且 ΔUtility < 0：支持原 Greedy 确实存在有害的过度计算；
- 若 ΔNLL > 0 且 ΔUtility < 0：只支持原 λ 下的成本/质量折中更好，不能宣称新策略质量更高；
- 若 ΔUtility >= 0：该深度强制 STOP 没有证实净收益，不能仅凭路径缩短称修复成功；
- 若仅极少 Token 触发：报告触发率，不能将无效干预视为阴性机制证据；
- 若 D4/D8/D12 的结果不同：表明额外计算收益与深度有关，不应合并成一个统一最优 STOP 阈值。
注意固定深度干预是一种行为反事实，并不直接等于证明 Q 的错误来源；需要同 CF1 的状态级校准结果交叉验证。
5. CF3：限制重复 CTX 与重复 Cell
5.1 科学问题
现有 Greedy 路由允许多次调用 CTX，以及同一 Cell 连续执行。这些重复操作究竟带来必要增益，还是诱发了低效循环？
5.2 四组固定策略（2×2）
策略 ID	CTX 最多调用一次	禁止连续相同 Cell
BASE_GREEDY	否	否
CTX_ONCE	是	否
NO_ADJACENT_SAME_CELL	否	是
CTX_ONCE_AND_NO_REPEAT	是	是


在每一个真实推理决策点：先按原 Registry.legal() 取得合法动作，再根据路径历史对本次合法集合施加掩码，最后在剩余合法动作中根据原始、未经改写的 Q 值选择最低者；维持原有 STOP 平手优先规则。
- CTX_ONCE：此 Token 的前缀一旦执行过 CTX，以后深度将 CTX 标记不可选；不禁止首次 CTX。
- NO_ADJACENT_SAME_CELL：若上一非 STOP 动作就是 Cell i，下一决策点不允许再选择同一 Cell i；CTX 后再选择 i 属于非相邻，不予禁止；仅依据当时确实激活的 Cell ID 实施。
- 组合组：二者同时生效，不额外设置路径长度阈值、不改变其它约束。
- 在约束后确保至少一个动作合法；原则上既有 6 个 Cell 及 CTX/STOP 足以满足，但必须检测并对不满足情形 FAIL_CLOSED，不可自动放宽条件或插入非法 STOP。
- 不改写 Model.execute()、Registry.legal() 的全局语义，不改变 Q logits，不在模型状态里永久记忆掩码；掩码应存在于本次只读 rollout 的 Token 局部路径状态中。
- 保留 STOP 最少动作数要求、冻结 plan/replay 约束与相同终止逻辑；禁止通过重新初始化 memory、移除 Cell 更新等方式实现“禁止重复”。
5.3 主指标与辅助指标
同 CF2 做配对 ΔNLL/ΔCost/ΔUtility，再报告：
- CTX_calls_per_token、CTX_ge2_rate；
- Cell_i→Cell_i 自循环次数及占比；
- Cell0→CTX→Cell0 三元模式次数/Token（只作为序列模式计数，不能推断没有信息增益）；
- 完整转移矩阵差异、平均路径长度、实际 STOP/强制结束率；
- touched_token_fraction（至少有一次由掩码改变原选择）；
- 每个被触发 Token 的首次触发深度及因此替代的动作；
- 两项约束的交互效应：ΔUtility(both) - ΔUtility(CTX_once) - ΔUtility(no_repeat)（相对同一原基线）；
- 结果按实际被禁止动作类型、深度和最终终止状态分层。
5.4 解释限制
- CTX_ONCE 如果降低计算成本但显著损伤质量，说明多次 CTX 在当前模型下可能有实际作用，不可称为“冗余循环”。
- NO_ADJACENT_SAME_CELL 若只降低自循环，模型可能转为 CellA→CellB→CellA，故必须同时检查完整路径长度、二步循环和效用。
- 二者组合若显著优于单独限制，需检验路径与动作替代的交互，不要单独归因 CTX 或某个 Cell。
- 固定模型参数下的限制效果仅支持部署策略反事实，不证明训练时错误来自哪一种损失项。
6. 统计计划（不得结果驱动地改变）
6.1 主要比较
对 CF2 三个深度和 CF3 三种约束，均相对于各自完全相同的 BASE_GREEDY，主要终点为：
1. 成本感知效用 ΔUtility；
2. 独立报告质量 ΔNLL；
3. 独立报告非 STOP 计算成本 ΔCost。
三者不应合并成一个不透明“性能”数字。每个指标记录值、Story 级标准差、95% CI、实际样本量与配对键。
使用 128 个 Story Cluster 的配对 Bootstrap（建议 2000 次，独立冻结 Seed）。不能将 8192 个 Token 当成 8192 个独立训练样本，也不能把多种策略当作独立训练 Seed。
对 CF2 三次比较与 CF3 三次比较，另做预注册的多重比较校正（例如 Holm），同时报告未校正探索性 CI 和校正后判定；不要事后只强调表现最好的那组。研究主结论保持 EXPLORATORY_SINGLE_SEED，不写正式确认的 p 值断言。
6.2 CF1 的配对/分层原则
CF1 的状态通常在同一 Token、同一 Story 内重复出现，需至少按 Story 聚类；若采用分层过采样，按预注册设计权重恢复总体描述。MC 候选后缀应使用相同随机数流，且冻结原始权重，记录 MC 标准误和“不足以识别差异”的比率。
不要把状态级反事实改进率直接外推到全体真实路径；反事实改变后缀状态分布后，轨迹结构可能随之改变。CF1 是局部价值诊断，CF2/CF3 是真实策略级再执行，二者互相补充。
6.3 指标口径及单位
- NLL：原 next-token 平均 CE，越小越好。
- Cost：原始非 STOP 动作数/16；不把排队时间当作模型执行成本。
- UtilityLoss：NLL + 0.01*Cost，越小越好；不修改 λ、不选一个更有利的 λ 再宣称主结果。
- actual_stop_rate：在预算内确实选择 STOP 的路径比例。
- forced_end_rate：因达到动作预算仍未 STOP 的路径比例。
- 路径长度：实际执行的非 STOP Cell/CTX 次数；Forced STOP 不应加进计数。
- “路由正确率”必须写清它比较的是 STOP vs chosen、CTX vs best Cell 还是 Cell pair，不能拿固定 Probe 的 92% LCG 局部方向一致率作同义指标。
7. 工程执行接口与安全隔离
7.1 推荐新增目录（仅为未来实现位置）
research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/
  counterfactual-stop-route/
    SPEC.md
    config/frozen-cf-v1.json
    notebook.ipynb
    implementation/
      cli.py
      input_validator.py
      baseline_replay.py
      visited_state_audit.py
      forced_stop.py
      route_constraints.py
      paired_statistics.py
      publication.py
    reports/<new-cf-run-id>/
      experiment-manifest.json
      checkpoint-validation.json
      baseline-parity.json
      cf1-stop-value-calibration.json
      cf2-forced-stop.json
      cf3-repeat-constraint.json
      resource-ledger.json
      findings.md
      coverage-ledger.json
      publication.json
实际包名与入口由 Codex 对齐当前仓库已有模块，不能为方便而复制、改写旧的 clm001_r3 训练器。可复用 Model、Registry、Snapshot、训练身份验证器、原 eval/rollout 的逻辑和冻结评估资产。
7.2 只读运行接口示意
以下是期望契约，不表示仓库已经实现这些命令：
python -m ...counterfactual.cli --mode verify-input --kaggle-version 65
python -m ...counterfactual.cli --mode baseline --arm QC-R3b --step 500
python -m ...counterfactual.cli --mode cf1 --arm QC-R3b --step 500
python -m ...counterfactual.cli --mode cf2 --arm QC-R3b --step 500
python -m ...counterfactual.cli --mode cf3 --arm QC-R3b --step 500
python -m ...counterfactual.cli --mode aggregate
任何命令不得调用 Trainer.update()、optimizer.step()、backward()，不改写模型参数、Optimizer、RNG 的原训练持久化状态。新 Monte Carlo RNG 必须独立 Seed；每次任务前后验证模型/Registry/原始 Checkpoint 的身份指纹一致。
7.3 Kaggle 持久化和恢复
- 新的诊断先原子写入 /kaggle/working 下独立目录；每个任务写结果及 SHA256；可按 State-Chunk/Story-Chunk 续接。
- 若需要跨 Session，必须以已经保存并经过下一 Session 回读校验的 Kaggle Output 或受控外部持久化为准，不能把 Session 内文件存在视为 durable。
- 所有已完成子任务有不可变身份：Checkpoint SHA、Story/Token/深度采样清单 SHA、策略配置 SHA、MC RNG 规则、分析引擎 SHA；相同身份重启时幂等复用。
- 本轮不得采用“每个小任务一个 HF commit”的写入方式；HF 如需备份，仅可选用低频批量提交，避免之前的每小时 128 commit 限流。
- GitHub 只发布聚合、图表、代码、协议、覆盖和性能账本；原始 Story/Token/Hidden/逐状态真值数据按原项目的私有边界保存。
- 任何上传凭证用 Kaggle Secrets，不能进入 stdout、异常栈、JSON、HTML 或 Git。
7.4 GPU 预算
CF2 与 CF3 主要是前向推理，可先执行；CF1 的反事实后缀评估可能更昂贵。应先用 Calendar-independent 的小样本只读 Pilot 记录 seconds/state、GPU 显存与 128 Story 回放成本；一次性冻结正式 CF1 的状态样本数和 MC 预算。正式启动后不因结果好坏选择性增加某些状态样本。
8. 测试、门禁与故障处理
8.1 必要 CPU 单测
1. 基线完全相同：无干预情况下动作、STOP、CTX、长度、NLL 与旧执行器契约一致；测试涉及 replay=None 和未改变原始 plan 的场景。
2. STOP 合法性：最低动作数之前不可 STOP；强制 STOP 必须在合法深度，不能将强制预算结束登记成主动 STOP。
3. Mask 语义：CTX 仅在第二次起禁用；连续同 Cell 仅在直接相邻时禁用；其它动作仍合法；不存在非法动作兜底。
4. 相同 Q 与状态：施加限制不重新初始化模型/Memory，不改 logits，不修改原 Registry.legal()，不同策略间没有状态污染。
5. 固定反事实：对给定状态与第一动作，STOP 立即计算的 Loss/Cost 与 Oracle 基准一致；使用 CRN 的配对后缀可重复。
6. RNG 隔离：新 MC 不消耗原训练 RNG，重复运行与批次切分不改变结果。
7. 状态复算：通过动作前缀重新构建 hidden/CTX 标志/上一动作，指纹可校验；一旦不匹配停止分析。
8. 统计正确：严格按 Story 配对；不因 Token 数量误算独立样本；正确处理未触发的 Token、MC 未判定、NaN/Infinity。
9. 协议不变：目标训练 SHA、Checkpoint SHA、原 102 项 Analysis ID、Confirmation 相关文件没有改动。
10. 恢复幂等：中断后重启可回读 verified chunk，不重复累积、不覆盖已发布结果。
8.2 Kaggle 启动前门禁
INPUT_IDENTITY_PASS、CHECKPOINT_HASH_PASS、ASSET_PROBE_PASS、READ_ONLY_PASS、BASELINE_PARITY_PASS、OUTPUT_STORAGE_PLAN_PASS 均通过，才允许进入正式 CF1–CF3。
旧分析 59 项的跨 Session 验收应单独报告 OLD_ANALYSIS_PERSISTENCE_VERIFIED 或 NOT_VERIFIED；如果存在数据丢失风险，先保全、不要继续产生新的私有结果。若不能读取原始 59 项，不应伪造其已恢复。
8.3 异常处理
任一哈希不符、动作非法、RNG/模型状态被修改、统计样本身份错乱或保存无法确认时，标记 BLOCKED/INCOMPLETE，写入具体错误和影响范围。不允许以 0、默认值或跳过缺失故事来伪造完成。不得自动重新训练补齐。
9. 结果解释矩阵
观察到的组合	可以支持的解释	不能直接宣称
CF1 大量 STOP 真值优于继续，但 Q 仍选继续；CF2 提前 STOP 改善 NLL+成本	STOP 对当前 Greedy 路径的价值存在错排，过度继续执行对收益有害	唯一原因就是 LCG；所有训练状态都错排
CF1 Greedy-consistent 错排明显，但 Frozen-sampled-oracle 不明显	训练后缀策略与 Greedy 部署分布不匹配是强候选机制	完全证明分布失配是因果根因
CF2 仅降低成本，NLL 变差，但 λ=0.01 下效用改善	存在可改善的成本/质量折中	语言模型预测质量提高
CF3 CTX_ONCE 优于基线且重复 CTX 明显下降	重复 CTX 在被测分布/当前模型下可能过度	所有重复 CTX 永远无用
CF3 禁连续 Cell 优于基线，但仍出现多步循环	相邻重复是一部分问题	所有循环已修复
CF3 组合优于两个单项	CTX 与 Cell 重复存在交互/替代可能	可将收益完全归因于其中一项
CF1 Q 校准整体尚可、CF2/CF3 都不改善	简单早停/局部禁止重复不足以修复；需进一步看路径信用、收益估计或表示能力	Cell 完全没有能力、MiniCells 架构被证伪
Baseline parity 或存储验证失败	工程问题使科研结论暂不可信	任意机制假设被证伪


无论结果如何，MULTISEED_NOT_TESTED、CAUSAL_FEEDBACK_NOT_TESTED、CONFIRMATION_SEALED 均保持原状。
10. 交付物与验收条件
10.1 必须交付
- 冻结新协议 SPEC.md + 配置和哈希；
- 新 Notebook 与只读执行器代码、独立 CF run ID；
- 完整性验证（57/57 历史索引 + 被分析的 QC Step 500 原始权重 SHA）；
- 原版 Greedy 基线一致性结果与逐指标误差；
- CF1 状态分层/抽样清单、Q 预测与两种后缀真值、配对统计和未判定样本；
- CF2 三种提前 STOP 的完整配对结果；
- CF3 三种限制及交互分析；
- 资源账本（GPU / wall / I/O / 保存用时），明确 MEASURED/DERIVED/NOT_MEASURED；
- 新 Output 的 Saved Version 和独立 Session readback 回执；
- 不泄露私有原始数据的 GitHub 汇总与 Markdown 科学报告；
- CPU 单测、短 GPU 预检、CI 结果及发布 Commit；
- 旧分析 59 项持久化审计报告（不得与本轮 CF 诊断覆盖率混合）。
10.2 完成标准
CF1、CF2、CF3 均完成固定样本与统计；不存在不可解释的丢失 Token/Story；Checkpoint 原哈希与新评估前后模型指纹一致；无训练更新；新私有诊断数据保存并通过跨 Session 回读；原 R3b 源码、权重、Confirmation 与正式结论均未修改。
科学结果可以是“支持”“不支持”或“不确定”。不得把低强制终止率、短路径、低 GPU 耗时直接当作根因被修复。 只有质量与成本感知收益经配对对照证实，才可写成“推理决策改善”。
11. 供下次提交 Codex 的完整执行指令
请阅读 CLM-001-R3b-CF-001 实验规范，并检查当前仓库及已持久化 Kaggle Output。先完成旧 R3b 59 项原始诊断结果的只读跨 Session 保全审计，准确披露是否验证成功；再按新冻结的实验身份实现 CF1/CF2/CF3 的只读反事实分析。所有输入锁定 QC-R3b Step 500 post_event Checkpoint SHA256 57b9daaaa54b0a67c6bc43c2f216ac375d0bb127ffbda132ac00f4cd42caffbb 与原冻结 Development/Calibration 资产。先确保无干预基线与原结果一致，再执行预注册干预、配对 Story Bootstrap、只读聚合、跨 Session 保存和公开汇总。不得重训、修改训练源码和旧权重，不得开放 Confirmation、泄露私有结果或声称单 Seed 的探索性结果为正式验证。遇到验证失败时停止受影响分析，报告准确状态。请在能实际运行 Kaggle 的环境内执行；如果不能运行，只完成代码/测试/Notebook 和一条明确的启动命令，不能假装生成了真实 GPU 结果。

12. 已核验代码与证据位置
- 原训练与分析结果（commit f65a7401）：research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/artifacts/a7792ef30582dfcaf324cca7e5245c8430df48b2/536fad45745b/
- metrics-summary.json：Step 500 的 Greedy 路径、CTX、自循环及 NLL；
- routing-longitudinal-summary.json：冻结 Probe 的 STOP/CTX Q gap、路由方向一致率；
- statistical-contrast.json：R3b 的单 Seed Development 比较；
- kaggle-output-input-reference.json、checkpoint-validation.json：Version 65 输入身份及 57 条 Checkpoint 验证；
- experiment-manifest.json、recovery-ledger.json：102 项完成、43 已跨 Session 验证、59 项尚待持久化核验；
- 冻结分析引擎 ed2accbc90f541325ca2f6c53229e9c81475768d 下：
  - r3/implementation/clm001_r3/rollout.py：硬路由、STOP/forced_end 的真实执行语义；
  - r3/implementation/clm001_r3/competitive_credit.py：仅比较 Cell 的 LCG 机制；
  - r2/implementation/clm001_r2/sampling.py：Greedy 最低 Q 与 STOP 平手优先；
  - r2/implementation/clm001_r2/oracle.py：相对 STOP 的标签和采样后缀；
  - implementation/clm001/oracle.py：成本感知后缀真值的计算；
  - r3b/implementation/clm001_r3b/instrumentation.py：冻结 Probe Q gap、STOP/CTX action 对照。
文档状态：仅研究实施规范，尚未实施或得到任何 CF1/CF2/CF3 新实验结果。