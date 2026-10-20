MiniCells — CLM-001-R3c-MOE-Bridge-001
固定四阶段、逐阶段可微门控与 Cell 联合训练：科研实验方案及 Codex 实施规范
版本： v0.1 — Draft / 待预注册冻结
编制日期： 2026-10-20
目标仓库： ArcheLabs/mini-cells
研究类型： 新的独立机制实验；不是 R3/R3b 续训，也不是已封存 Confirmation 的重分析
当前指令： 本文档仅供审阅和下一阶段实施；未授权本轮真实训练、部署或修改现有仓库。Codex 实施时应先完成 Phase 0–2，经过单独授权后才能启动正式 GPU 训练。
0. 一页决策摘要
0.1 核心科学问题
在不引入动态 STOP 和 CTX 路由竞争的条件下，MiniCells 能否像稀疏 MoE 一样，在每个固定计算阶段对当前状态独立选择 Cell，并通过最终语言模型交叉熵（CE）梯度同时训练 Router 与 Cell，从而获得：
1. 稳定的 Router–Cell 联合训练；
2. 对输入/状态有条件的 Cell 能力差异，且 Router 能利用差异；
3. 与相同执行预算、相同 Cell 容量的固定/随机路由相比，具有可重复的质量收益。
这是一项隔离机制实验，不预设当前 R3b 的 Cell 已经专业化，也不要求本轮证明动态计算深度、持续学习或“优于工业级 MoE”。
0.2 核心设计
- 固定 4 个 Stage，每个 Stage 都重新计算 Router；不预先决定完整路径。
- 每个 Stage 选择 Top-2 Cell，每个 Token 每阶段恰好执行两个 Cell，并用可微的门控权重加权组合输出。
- 默认跨 Stage 共享原有 6 个可训练 Cell，Router 使用共享参数及 Stage Embedding；Stage 专属 Expert 属于后续独立扩展，不在首轮混用。
- 取消动态 STOP 选择；全部完成四个 Stage 后输出。
- 不把 CTX 放进动作候选池。为避免剥夺语言模型的上下文输入能力，所有组在四个 Stage 前执行相同的一次固定 Context Read；此处理与路由无关，成本全部计入 FLOPs。
- 核心损失为同一个最终 CE；不使用原始 Q、LCG、Oracle 后缀标签训练主模型。
- 配置与报告新建实验身份，不可写入或覆盖 R3b、CF-001 原产物。
0.3 最重要的 Go / No-Go
只有在可微门控组相对于至少两个同预算基线产生可重复增益，同时有可独立验证的 Cell 条件差异与 Router 利用证据时，才允许研究动态 STOP 的回归。
若仅 NLL 降低但不能证实 Cell 的互补性，结论只能是“联合训练改善了质量”，不能宣称“Cell 自组织专业化已被证明”。
1. 研究背景与已有证据
1.1 已验证的事实
- R3b：Q/QC/QS 三组训练各 500 Step；57 条 Checkpoint 记录通过核验；102 项诊断已完成。
- Q Step 500 Greedy NLL 约 4.68499、平均路径长 2.98511、强制终止率 2.185%。
- QC Step 500 Greedy NLL 约 4.69934、平均路径长 14.73865、强制终止率 74.194%。
- QC 在冻结特定 Cell 对、Token-ID 奇偶分组下未发现达到原预定判据的条件交叉专业化（不是全部可能专业化不存在的证明）。
- CF-001：QC Step 500 在 Greedy 实际访问状态上存在明显的 STOP/继续执行排序错误；强制第 4 个动作后 STOP 的 NLL 约 4.68901；禁相邻重复 Cell 后 NLL 约 4.68914；简单限制 CTX 一次则 NLL 变差。以上为单 Seed、Development 探索性数据。
- 现有 R3 实现中，实际执行的 Cell 接收 CE 梯度，但硬选择不提供普通可微门控梯度；Router 主要依赖反事实 Q 和 Cell-only LCG。现有 detach_state=True 与最终 CE 无法直接经硬索引训练路由是两件不同的事实，不能混为“Cell 不受 CE 梯度”。
- 旧 R3b 新诊断的 59 项结果已有 Kaggle Version 69 跨 Session 验证；CF-001 的 2,306 个新私有结果块在所见公开报告中仍为 COMPUTED_AWAITING_SAVE / cross_session_verified_chunks=0。启动新阶段之前应重新读取最新持久化状态，不能把这里的历史状态当作最新状态。
1.2 研究解释与未知问题
- 推测： Q/LCG 不一致、样本分布偏移、STOP/CTX 与 Cell 同头竞争、跨深度重用同一 Cell，可能共同导致 QC 失稳。
- 尚未证明： 可微门控一定促进专业化；固定 4 Stage 一定优于 R3b Q；训练时间足以形成互补 Cell；复合顺序协作能带来超额收益。
- 本轮仅检验逐 Stage 局部路由与可微门控训练是否建立最小可重复的联合学习机制。
1.3 只读代码依据
- R3 model.py：Cell 状态更新 h + alpha * Cell(norm(h))，Context 为独立算子。
- R3 rollout.py：硬路由每步选动作，Controller 输入状态/上一动作/深度；detach_state=True。
- R3 training.py：CE 更新实际执行算子；Q 与 LCG 另行更新 Connection。
- R3 competitive_credit.py：LCG 候选限于 Cell，代理梯度不直接训练候选 Cell。
- R3b config/proposed.json 的 engine.model：宽度 128、6 个 accepted Cell、1 个 reserve slot、Cell hidden 248、词表 4096、序列长 64、alpha 0.125；原始 budget 16、minimum_actions 1。本轮阶段数改为 4，但不篡改原配置。
源代码固定参考： ed2accbc90f541325ca2f6c53229e9c81475768d（R3b 冻结测量引擎依赖的 R3 代码快照）。新实现必须保存其实际 Git SHA，不能把这个历史 SHA 当作新实现 SHA。
2. 预注册研究假设与可证伪要求
ID	假设	主要测试	如不成立，应得出的结论
H-F1	固定 4 Stage + 可微 Top-2 门控能稳定端到端训练	CE finite、非零 Gate 任务梯度、多 Seed、路由使用分布	本训练构造尚不能稳定工作，先查梯度/数值/塌缩
H-F2	Gate 的 CE 梯度确实提供质量收益	学习门控 vs stop-gradient 门控，同预算	可微门控未显示独立的 CE 训练贡献
H-F3	学习门控优于简单路由基线	学习门控 vs 固定均衡 / 随机均衡路由	没有证明“学会选择”带来价值
H-F4	Cell 形成可重复、可利用的条件差异	独立状态分组、反事实替换/消融、无标签推理 Gate	无证据证明有用互补，不可只凭参数距离宣称专业化
H-F5	不同 Stage 的路由决策确实依赖当前状态	Stage 间条件分布、固定路径/置换干预	尚未证明逐 Stage 自主决策的实际优势
H-F6	质量收益不是单纯来自更多 FLOPs	相同激活 Cell 数和真实推理 FLOPs 的对照；计算-质量 Pareto	不能声称比其他方法更高效


H-F1–F6 之间不能相互代替。 特别是 H-F2 成立也不能自动推出 H-F4、H-F5。
3. 网络与训练方程（工程契约）
3.1 前向路径
固定采用原始 Embedding、Position、Memory、Init、最终 Head、六个可训练 Cell 的可复用部件。新增的固定 Context Read 为所有组共有的计算：
\[
 h_0=\operatorname{Init}(\operatorname{Embed}(x)+\operatorname{Position}(x)),\quad
 \tilde h_0=h_0+\alpha\operatorname{Context}(\operatorname{Norm}(h_0),M(x))
\]
在 4 个 Stage 中，每一步都根据更新后的当前表示重新路由：
\[
 z_{t}=R_\theta(\tilde h_t,e_t),\qquad
 S_t=\operatorname{Top2}(z_t),\qquad t=0,1,2,3.
\]
\[
 g_{t,i}=\frac{\exp(z_{t,i}/\tau)}{\sum_{j\in S_t}\exp(z_{t,j}/\tau)}, \quad i\in S_t,
\]
\[
 \tilde h_{t+1}=\tilde h_t+\alpha\sum_{i\in S_t}g_{t,i}E_i(\operatorname{Norm}(\tilde h_t)).
\]
\[
 \mathcal L_{\mathrm{task}}=\operatorname{CE}(\operatorname{Head}(\tilde h_4),y).
\]
记号 e_t 是可学习 Stage Embedding；Router 参数在四个 Stage 共享，但每次根据不同隐藏状态、Stage Embedding 独立重新打分。不使用旧的 previous-action edge bias，因为一阶段可选两个 Cell，旧单一 previous ID 不再具有明确定义。将其移除是所有实验组共有的固定结构改动，严禁仅在学习门控组移除。
3.2 梯度语义
- Top-K 成员身份为离散操作；不能声称对“未选中 Expert 被选与否”具有普通梯度。
- 选中 Top-2 后，g_{t,i} 对其 logits 可微；最后 CE 梯度应能流入 Gate，且能流入实际执行的两个 Cell。
- 不能先 Top-1，再在唯一成员上做归一化 Softmax；这样门控权重恒等于 1，主任务 Gate 梯度退化为零。
- 不能把 g.detach() 或 h.detach() 错放在主训练路径上；只有专设 Gate-StopGrad 对照可以 detach 用于组合的门控权重。
- 不能先对所有 6 个 Cell 计算后再屏蔽；前向必须仅实际执行选中两个 Cell，否则“稀疏”计算收益不成立。
- 不使用 Straight-Through、REINFORCE、Q、LCG 于核心主实验，避免同时测试多个梯度估计器。若以后引入，另立实验身份。
3.3 负载均衡
设每 Stage 被实际 Top-2 选择的频率为 f_{t,i}（按选中次数归一化，和为 1），完整 Softmax 的平均概率为 pbar_{t,i}，可定义类似 Switch 的辅助均衡项：
\[
 \mathcal L_{balance}=N\sum_{t=0}^{3}\frac14\sum_i \operatorname{stopgrad}(f_{t,i})\bar p_{t,i}.
\]
N=6 为当前已激活 Cell 数。f 使用不可微的选择统计，pbar 的梯度有效。该项为软约束，不保证严格均衡。不得让 pbar 代替实际 Top-2 选中门控作为最终推理输出。
\[
 \mathcal L=\mathcal L_{task}+\lambda_{balance}\mathcal L_{balance}.
\]
lambda_balance 必须在仅使用 Train/Calibration 的短 Pilot 中按照预先固定的选择规则冻结（例如避免任意 Cell 长期零曝光的最小有效系数），不能在看过 Development 结果后多次调节。另设 无 balance 消融组，以辨别 CE Gate 梯度与均衡机制的作用。
3.4 固定条件与明确差异
冻结条件： 四个 Stage、Top-2、六个可训练 Cell、共享 Cell 参数、相同一次固定 Context Read、原始词表及数据切分、宽度/Cell hidden/head、同种优化器与训练 token 数、共同初始种子配对规则、统一评价协议。
有意改变： 原始 budget=16 的动态多步执行替换为固定 4 Stage；Cell 更新为两个 Expert 的门控加权组合；Q/LCG/STOP/CTX 动作及原 previous-edge 特征退出实验。这是独立研究架构，不能和原 Q-R3b 报告直接作“只改了一个变量”的因果比较。
明确暂不研究： Stage 专属 Expert、动态长度/STOP、可选 CTX、长程 Q/LCG、多路径 Beam Search、容量扩张、JAM/PVM 部署。
4. 最小对照矩阵：每组要回答一个问题
首轮必须至少包含 A、B、C、D；E 为强烈建议的第五组。所有组均 4 Stage × 2 实际执行 Cell / Token / Stage，即每个 Token 8 次 Cell 调用及 1 次固定 Context 调用。所有组应使用相同的模型核心与 Router 网络参数容量；即使某组 Gate 不接受任务梯度，也必须保留同一 Gate 计算以准确计算 FLOPs。
Arm	Cell 对的选择	两个 Cell 的加权	Gate CE 梯度	Balance	关键作用
A — FIXED-BALANCED	预冻结的均衡双 Cell 调度，按 Stage/Token 的确定性映射	0.5 / 0.5	无	无	简单固定路由基线
B — RANDOM-BALANCED	独立随机流选择两个不同 Cell；每 Stage/批次近似均衡	0.5 / 0.5	无	无	随机路由基线，控制固定配对偏差
C — GATED-TOP2（主实验）	当前 State 上独立 Top-2	选中 logits 的 Softmax	有	有，冻结系数	可微门控与均衡的完整联合训练
D — GATE-STOPGRAD	与 C 同构的 Top-2	前向权重相同定义，传给 State 时 detach	无	有，冻结系数	测试 LM CE→Gate 梯度的独立作用
E — GATED-NO-BALANCE	与 C 相同	与 C 相同	有	无	测试 Balance 是否阻止坍缩或损害 CE


说明：
1. D 组并非完全冻结 Gate。 它可以接受 Balance Loss 的梯度，只是不接受 CE 梯度。实验日志必须分别报告 gate_task_grad_norm 与 gate_balance_grad_norm。
2. A/B 必须保留一致的门控网络前向诊断（只读、可选择不参与计算图）；训练参数数量与实际参与训练参数量分别报告，不得用名义参数数相等代替 FLOPs 相等。
3. B 组随机化要使用与 dropout、数据读取无关的固定 RNG 流，验证成对 Seed 重放；不能因改变路由而改变数据顺序。
4. C/D 的 Top-2 选择可能随训练发展而不同，不能预期完整路径相同；对照针对训练规则，而非每个 Token 的相同具体路由。
5. 可增加 Top-1 Q、固定单 Cell、标准层专属 Sparse MoE 作为第二批次参考，但它们的执行 FLOPs 或容量不同，不得纳入第一批次的直接主因果比较。
4.1 是否需要 Stage 独立 Expert？
不在第一批次修改。选取六个跨 Stage 共享 Cell，是为了尽可能保留 MiniCells 的核心模块与参数容量，并隔离“逐阶段门控梯度”这一个研究点。
若 C 相对 A/B 没有收益，可在第二批次建立 stage_specific_experts 组，与同类型共享 Cell Gate 组比较，单独研究“层专属 Expert 参数”是否必要。此组参数量会增加，必须另行做容量匹配，不能混称一个变量。
5. 训练流程与授权门禁
Phase 0 — 产物保护与基线确认（不启动训练）
1. 检查 R3b、CF-001 的最新 GitHub 聚合及 Kaggle 保存状态；CF-001 新结果必须完成跨独立 Session 保存回读，或显式记录 BLOCKED_PERSISTENCE 并暂停依赖其私有数据的后续工作。
2. 独立新路径，例如：
   research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/
3. 新建 SPEC.md、config/proposed.json、实现包、CPU tests、smoke notebook、结果 schema；不得修改旧 r3、r3b、counterfactual-stop-route 目录的不可变内容。
4. 将 R3b 原始配置/数据/Tokenizer/已有开发报告的 SHA、Git SHA、数据切分、旧 train/development usage 写入溯源文件。旧的 Development 已参与研究设计，不能作为未触及的 Confirmation 使用。
5. 禁止将原 HF 私有 Checkpoint 或私有 Token/数据意外推送到公开 GitHub。
Phase 1 — CPU 合成正确性与梯度不变量
- 使用两类可区分合成任务、2–6 个 Cell 的短测试，验证 Gate CE 梯度实际可传播、Cell 更新、负载均衡与稀疏执行。
- 强制 C/D 在同一权重和同一输入的第一步前向完全一致，但反向 gate_task_grad_norm(C)>0、gate_task_grad_norm(D)=0，同时 C/D 都可从 Balance 接受梯度。
- 反事实修改 Gate logits 能改变输出；若所有 Expert 输出完全一致，这个梯度可能为零，应有明确的“无法区分同一输出”的诊断测试。
- A/B 的两个 Expert 在每个 Token/Stage 只计算一次；任何误运行全部 6 Cell 则 CPU test FAIL。
- Stage 1、2、3、4 的输入分别来自上一 Stage 的结果；禁止一次性在最初状态选好全路径。
- 对相同 Seed、Data Order、RNG state 重放的参数、动作计数、loss 与哈希进行确定性测试。
- 断言没有动态 STOP、没有路由 CTX、没有旧 Q/LCG 训练项；Context 固定一次，所有组相同。
- 状态残差、hidden shape、Token 对齐、语言模型 causal mask、CUDA/CPU 数值有限性必须通过。
Phase 1 Go： 全部 CPU 功能和差异梯度测试通过，才允许启动 GPU Pilot。
Phase 2 — 最小 GPU Pilot（仅授权的短运行）
- 只使用 Train 做梯度更新；Calibration 用于数值、吞吐与负载系数选择，Development 不参与任何配置筛选。
- 预先批准极小的 step/token 上限（建议先 10–50 Step smoke；再根据测量确定是否需要额外 100 Step Pilot），不因结果好坏临时延长。
- 检查 CE 是否 finite、Gate CE 梯度非零、Cell 梯度按真实选择传播、实际 FLOPs、显存、吞吐、是否有死 Cell。
- 固定 lambda_balance 的候选集与冻结规则：一旦确定，写入带 SHA 的 Final Protocol；不得随 Development 结果回调。
- 评估单个 Seed、单个 Arm 的时长，确定正式训练的 GPU 预算，而不是假设符合“两小时”窗口。
Phase 2 Go： 除数值与成本通过外，数据划分、Checkpoint 恢复和结果存储必须可跨 Session 验证。
Phase 3 — 正式 Development 研究（单独授权）
- 默认预先计划 至少三个独立 Seed，A/B/C/D/E 同一 Seed 区块配对。三个 Seed 只能建立初步可重复性证据，不应声称高统计功效。
- 首轮以原 R3b 量级 500 Step、每 Step 512 训练 Token、总量 256,000 Token/Arm/Seed 为可预算起点（必须结合 GPU Pilot 冻结，且所有组统一）。这并不保证足以产生专家分化。
- 固定检查点建议 0, 50, 100, 200, 350, 500；需在正式开始之前冻结快照粒度。不得因某一组开始显示优势而独立追加其训练步数。
- Train、Calibration、Development、Confirmation 的边界严格保留；Calibration 可以用于 Pilot 但不能宣称 Confirmation。
- 允许预先定义的更长训练预算作为第二阶段独立扩展实验，不是遇到负结果后临时续训某一组。
Phase 4 — 机制诊断和独立验证
- 完成下文第 6 节的 Cell 差异、Gate 使用、因果消融及质量-计算比较。
- 冻结所有候选、阈值、种子、权重选择规则后才授权一次性 Confirmation。Confirmation 不得用于寻找效果最好的 balance 权重、Stage 数或 Cell 组。
- 若最终仍是单 Seed / Development，则所有文件必须显式 EXPLORATORY_ONLY。
6. 评价指标：不能只看 NLL
6.1 系统质量与效率
必须报告：
- development_mean_nll / PPL、Train/Calibration loss、相同窗口的 paired delta；
- 实测每 Token 推理 Cell 次数（目标均为 8）、Context 次数（目标均为 1）、真实 GPU kernel/硬件相关 FLOPs 或可复算算子 FLOPs、Token/s、峰值显存；
- 训练时每 Token 激活 FLOPs、优化器成本与总训练 GPU 时间；
- 每个 Stage 的实际 Gate entropy、Top-2 使用率、各 Cell 曝光、训练梯度范数、两个 Gate 权重分布；
- 每个 Seed 的最终值与轨迹，不允许只公布 Seeds 的平均值。
主要质量估计应采用相同开发窗口的 Story 聚类配对 Bootstrap；对于多 Seed，应分别给出 Seed 内 Story CI 与 Seed 间变异，且以 Seed 作为训练复现单位，不得将 8,192 个 Token 假装成 8,192 次独立训练。
6.2 Cell 条件差异（互补性）
目标不是证明每个 Cell 参数不同，而是证明其在特定条件下有可利用的收益差。
在冻结的独立状态集合 S 上，使用相同后续策略及计算预算，为每个合法 Cell 计算反事实价值 J_i(s)；对所有对照组使用完全一样的选择/估计协议。
- best_fixed_cell_cost = min_i E[J_i(s)]；
- statewise_oracle_cost = E[min_i J_i(s)]；
- oracle_potential_gap = best_fixed_cell_cost - statewise_oracle_cost 仅作为潜在收益上界诊断，不作为已达到收益；
- 防止逐状态挑最小估计值的乐观偏差：选择和验证必须使用独立反事实样本 / 独立 Evaluation 分块。
- 预先冻结与标签无关、推理时可用的分组（如当前隐藏状态特征聚类、上下文结构类别等），Discovery 用 Calibration；Development 只用于独立检验；不得用 Target 标签事后定义组。
- 检查 Cell 对是否存在组间反向条件优势、各 Cell 消融的分组损失、跨 Seed 在允许 Cell ID 置换后的功能稳定性。
- 语义专业化不是必要条件，真正必要的是可重复的条件收益及其可用性。
重要： 在多步状态空间里 J_i(s) 必须说明“仅替换当前一步，后续由谁路由”。固定后缀与重规划后缀是不同 estimand，要分别标记，不得混合。
6.3 Router 真实利用互补能力
- 在推理时不使用正确 Target，比较学习路由与统一随机、固定均衡、参数冻结、Route Permutation / Cell Swap 的同预算 NLL。
- 对各 Stage 统计真实 Gate Top-2 与独立估计低损失 Cell 的 ranking_agreement / regret / selected_advantage；不要求每个状态都能可靠排序，不可识别状态应单独报告。
- 提供 Gate–Cell 协同扰动：保持 Gate 不变交换两个 Cell 输出，或在保持参数不变时置换 Gate 目标 ID，比较配对质量损失。
- 将路由网络的独立贡献与Cell 能力的变化分开：冻结 Gate/冻结 Cell 的正交消融可在另立小实验中进行，但不可对事后最佳 Arm 只做有利消融。
6.4 联合训练动力学与真正“自强化”
记录每个训练阶段、每个预冻结组 z、每个 Cell i：
\[
 n_{i,z}(t)=\text{该类状态真实分配给 Cell }i\text{ 的训练曝光量}
\]
\[
 J_{i,z}(t)=\text{在固定探测分布上同预算的 Cell 反事实损失}
\]
\[
 p_{i,z}(t)=\text{Router 对该类状态的选择频率/概率}
\]
分析 exposure(t) → ability(t+Δ) → routing(t+2Δ) 的时间关系，但 相关性不足以宣称因果自强化。
若要验证因果反馈，必须增加单独预注册的曝光干预子实验：对匹配状态/Cell 进行小幅随机曝光偏置，保持总训练样本与计算量，之后撤除干预观察 Cell 固定探测损失与 Gate 的响应，至少多 Seed 复现。此干预不是本轮主训练所允许的未申明额外训练信号。
7. 预注册统计与“何时宣称成功”
7.1 主要对照，按优先顺序
1. C vs A： 学习门控是否优于固定均衡选择？
2. C vs B： 学习门控是否优于随机均衡选择？
3. C vs D： 最终 CE→Gate 梯度是否带来独立贡献？
4. C vs E： Balance 是否对学习门控稳定性有贡献？
主要终点：Development 的 paired mean NLL，全部策略均有相同固定 Stage/Top-2 激活次数，且 FLOPs 记录确认预算可比。以一组预先冻结的多重比较方法校正前三项主对照；第 4 项定位辅助机制。
在开始正式训练之前，必须预先写定：
- 最小有意义收益阈值 min_practical_delta_nll；
- NLL 非劣容差（如使用计算效率主张）；
- Seeds 数量、Bootstrap 单位、校正方法、置信区间口径；
- 最小每组 Cell 曝光门槛及其统计定义；
- 哪些次要证据足以支持“有条件互补”，而不是仅支持“门控有效”。
不得在看过 Development 结果后才选择阈值。 建议根据 Train/Calibration 的 Pilot 与计算预算决定阈值并加 SHA 锁；本文不凭空指定统计上看似精确、实际上无校准依据的数值。
7.2 结果判定表
结果	科学结论	下一步
C 稳定优于 A/B/D，且 Cell 条件差异和路由利用证据成立	机制原理验证（Development 层级）	多 Seed 延伸、封存 Confirmation，然后研究动态 STOP
C 优于 A/B，但不优于 D	不能证明 CE Gate 梯度产生了独立收益	研究 Balance 与分配效应
C 优于 A/B/D，但 Cell 条件差异不成立	门控改善质量，但“专业化”未证实	扩展独立功能诊断，不自动解封 STOP
C 有条件差异，但 NLL 未优于 A/B	Cell 可能分化，但 Router 没有带来系统效用	聚焦 Gate 路由与梯度传递
训练中单 Cell 吸走几乎所有流量	需检查 Balance / 温度 / 初始化；不算稳定专业化	只按预注册规则诊断，不事后无限调参
C 与 A/B/D 全部无差别	此配置下没有门控收益证据	先执行结构/梯度正对照，不直接扩大训练
所有组无法基本收敛	问题可能属于共用 Cell / Context / 数据 / 优化器	工程诊断、合成任务正对照


No-Go 仅针对当前具体的训练构造，不自动否定 MiniCells 一般架构。连续多 Seed、多个已预注册预算仍无法过最小门槛时，不应投入更大规模直到提出新的可证伪机制变化。
8. 代码工程任务与预期目录
新目录建议：
research/projects/cellular-language-models/experiments/
  clm-001-qfirst-integrated/
    r3c/
      moe-bridge/
        SPEC.md
        README.md
        config/
          proposed.json
          frozen.json                 # 通过门禁后才生成
          releases.lock              # 新协议/输入/源码 SHA
        implementation/
          minicells_r3c_moe_bridge/
            model.py
            gate.py
            runner.py
            training.py
            evaluation.py
            instrumentation.py
            provenance.py
        notebooks/
          kaggle-smoke.ipynb
          kaggle-development.ipynb
        tests/
          test_forward_parity.py
          test_gate_gradients.py
          test_sparse_dispatch.py
          test_stage_independence.py
          test_rng_and_resume.py
          test_storage_and_manifest.py
          test_statistics.py
        reports/
          <run-id>/
            manifest.json
            coverage-ledger.json
            parity-ledger.json
            metrics-summary.json
            gate-gradient-summary.json
            routing-summary.json
            specialization-summary.json
            performance-summary.json
            SCIENTIFIC-FINDINGS.md
以上为建议路径，不是声称仓库已经存在这些文件。若 Codex 发现项目已有更合理的同功能公共模块，可复用，但不得修改旧冻结协议或引入有副作用的反向兼容更改。
8.1 Codex 实施优先级
1. 核实源仓库、私有数据和原科研结果保全状态；给出准确 Git SHA 与差异清单。
2. 新建新身份、新目录、新 proposed.json，实现固定 Context + 4 Stage Top-2 Gate。
3. A/B/C/D/E 五种 Arm 的差异只通过严格的配置开关生效；除指定差异外模型相同。
4. CPU 测试验证前向、稀疏分派、Gate 与 Cell 梯度、RNG、数据切分、数值有限性、状态一致性。
5. GPU Smoke：仅做被授权的短时测量，生成资源报表与 Balance Pilot 建议；未授权不得自动提交 500-Step 真训练。
6. 生成新 frozen.json 与 Hash 证明，等待正式 GPU 训练授权。
7. 正式训练后进行新 Session readback，并发布不包含私有原始模型和数据的可复算摘要。
8.2 关键实现伪代码
# 结构示意；不代表已可直接复制到生产工程。
h = model.init(embeddings)
h = h + alpha * fixed_context(model.norm(h), static_memory)  # 所有 Arm 一样

for stage in range(4):
    logits = gate(h, stage_embedding[stage])             # 每个阶段重新路由
    indices = torch.topk(logits, k=2, dim=-1).indices
    selected_logits = logits.gather(-1, indices)
    weights = torch.softmax(selected_logits / temperature, dim=-1)

    if arm == "GATE_STOPGRAD":
        weights = weights.detach()  # 仅隔断 Gate 的任务梯度
    elif arm in ("FIXED_BALANCED", "RANDOM_BALANCED"):
        indices = controlled_dispatch(arm, stage, rng)
        weights = torch.full_like(weights, 0.5)

    updates = sparse_dispatch_only_selected_cells(model.cells, model.norm(h), indices)
    h = h + alpha * (weights.unsqueeze(-1) * updates).sum(dim=-2)

loss = next_token_cross_entropy(model.head(h), targets)
if arm in ("GATED_TOP2", "GATE_STOPGRAD"):
    loss += frozen_balance_coef * balance_loss(gate_stats)
# GATED_NO_BALANCE 只优化 CE
loss.backward()
optimizer.step()
注意：这个伪代码不包含 Memory/Token batch 的所有维度与张量 broadcast 细节，Codex 必须依据真实训练形状实现；Balance 应基于明确的 full-softmax logits 与 hard Top-2 分配统计；如 D 的 CE Gate 梯度测试检出非零，视为设计失败。
9. 训练与测量日志的最低内容
对每一次训练更新，至少保存：
- experiment_id, run_id, arm, seed, step, source_sha, config_sha256, data_sha256, tokenizer_sha256, optimizer_sha；
- Train CE、Balance Loss、Gate task/balance grad norms、Cell grad norms、clip ratios、是否 NaN/Inf；
- 每 Stage/Cell 选择次数、平均 Gate 概率、Top-2 overlap、前两个权重、熵、死 Cell 率；
- 实际执行的 Cell 次数、固定 Context 次数、算子 FLOPs、CUDA time、显存；
- 检查点每次保存的原子写入和独立回读哈希；
- confirmation = SEALED、scientific_decision = false（直至按正式实验协议授权）。
结果采用分块、哈希索引、重试幂等设计；所有未测量值使用 NOT_MEASURED，不能填 0。训练日志不包含密钥；GitHub 仅发布可公开摘要和证明，不推送私有模型或完整样本级标签。
10. 风险清单与防止伪结论
风险	预防及判定
Top-1 softmax=1 导致 Gate 零梯度	主实验固定 Top-2；单独梯度单测
软权重实际上全部 Expert 都执行	稀疏分派 Call Counter + FLOPs 审计
固定 Context 缺失导致语言能力退化	所有 Arm 共同一次 Context；固定无 CTX 变体另立研究
Stage 1 一次性选定完整路由	每 Stage 使用当时更新后的 h 重新计算 logits，并测输入扰动
Load balance 对 CE 有利或有害难归因	C/E 对照；D 组仅断开 CE Gate 梯度
共享 Cell 的跨 Stage 梯度干扰	保存逐 Stage 梯度诊断；Stage 专属 Cell 后续单独消融
两个 Cell 输出过于相似导致 Gate 梯度很小	记录 `
专业化指标采用目标信息进行路由	Oracle 只作只读诊断；实际 Gate 禁用 Target、标签与后验组
逐状态 Oracle 选择高估潜力	Discovery/Validation 独立反事实采样，不用同批样本选择兼估
GPU FLOPs 不匹配	实际记录每组 Cell + Context + Gate 总 FLOPs/Token，比较质量-计算曲线
多 Seed 的 Token 误作独立训练重复	Seed 为模型级重复，Story 为样本级嵌套单位
R3b Development 已反复使用	必须保留封存的独立 Confirmation，避免数据泄漏
结果只在某个随机路径上好看	固定基线、每个 Seed 的全部轨迹、预注册阈值、无事后 cherry-pick
Notebook 任务被中断	Kaggle Save & Run All + 独立 Session readback，断点幂等
将本轮误称已修复 R3b 硬 Greedy	本轮固定 Stage Top-2 门控，不涉及原始硬 Greedy/STOP 的直接修复


11. 验收定义（Acceptance Criteria）
11.1 Engineering PASS
- 原冻结源码、历史 Checkpoint、旧科研产物未被修改。
- 对同一个 Token，每 Stage 的实际 Expert Call 恰好为 2（不计一个固定 Context），总计 8。
- Gate 每个 Stage 重新读取 h_t，而非复用第一阶段路径。
- C 的 CE Gate 梯度确实非零；D 的 CE Gate 梯度为零，Balance 可更新 Gate；所有组 Cell CE 梯度按实际选中情况正确传播。
- 经过验证的固定 Seed 可复现，Checkpoint 断点恢复与不间断训练结果在声明的数值容差内一致。
- 真实 GPU Pilot 的 FLOPs、时间、显存已测量；训练资源/输入/输出有哈希可追溯。
- 隐私与 Confirmation 门禁通过；不因工程 PASS 自动进入正式训练。
11.2 Scientific PASS（只对已完成研究阶段判定）
- 最少多 Seed 的主要对照 C vs A/B/D 有预注册的统计与实践效益支持；
- 反事实 Cell 条件差异在独立验证样本上复现，且 Gate 使用这种差异带来系统收益；
- 收益在真实推理计算预算下成立，不依赖 Oracle 使用推理时未知的目标标签；
- 没有以“不同 Cell 参数”“Gate 不均匀”“平均 NLL 下降”单独替代互补与协作证明；
- 需独立 Confirmation 才能升级至正式科学结论。
11.3 No-Go
出现以下任一情况，必须暂停当前实验配置而非扩大训练：
- C 无可验证 Gate CE 梯度；
- Top-2 分派不是实际稀疏执行；
- 训练不可重复、数据或科研结果保存不可验证；
- 同预算 A/B/C/D 的训练条件有未记录差异；
- 只读评价发现 Confirmation 泄漏、标签参与路由、评估样本与训练样本混用。
若模型正常训练但没有质量/互补收益，属于科学负结果，应完整发布而不是工程失败。
12. Codex 工作指令（可直接粘贴）
依据 CLM-001-R3c-MOE-Bridge-001 规范，先只实施 Phase 0、Phase 1 和被明确授权的 Phase 2 GPU Smoke 前置工程。实现固定 4 Stage、每 Stage 根据当前状态独立 Top-2、可微 Gate、跨 Stage 共享 6 个 Cell、所有组共有一次固定 Context Read。构建 A/B/C/D/E 对照，严格隔离 Gate CE 梯度和 Balance 梯度，完成 CPU 回归、稀疏执行、RNG、Checkpoint、分组统计与隐私测试。不要修改任何 R3/R3b/CF-001 冻结产物；不要训练 500 Step；不要开放 Confirmation。提交源码 SHA、门禁测试报告、真实短 Pilot 预算与冻结协议候选，等待人工批准正式多 Seed 训练。若当前 CF-001 Saved Output 未完成跨 Session 回读，先将其列为独立保全阻塞项，不得伪称通过。

13. 当前阶段的科学边界
本轮若成功，能够支持：局部逐 Stage 可微门控可以联合训练 MiniCells Cell，并在固定计算预算下形成和利用有条件价值。
本轮不能单独证明：动态 STOP 有效、长期持续学习、具备 MoE 规模化能力、优于成熟工业 MoE、严格意义上的因果自强化。那些问题分别需要后续动态调度、持续学习、规模与曝光干预实验。
本轮最重要的设计哲学：先让局部专家选择与最终 LM Loss 建立可微且可测的关系，再考虑让模型自己决定要执行多少阶段。