# M3C-002B-CAL-001 — R线 Development Calibration

用户已批准独立开发校准；正式训练与测试仍未授权。不修改Reference、Workspace架构、正式预注册候选或历史结果，不进入M3C-003。

## 本次固定开发预算

| 项目 | 值 |
|---|---:|
| 模型臂 | Base、W16、W64 |
| 开发模型Seed | 27101（不使用18001） |
| 每臂查询目标 | 32,768 |
| 三臂总查询目标 | 98,304 |
| 每臂Episode / 每Episode查询 | 4096 / 8 |
| Batch / 更新次数 | 8 Episode / 512 |
| 每臂实际处理输入Token（含Padding） | 655,360 |
| 总输入Token（含Padding） | 1,966,080 |
| 开发验证Episode | 384，六条件各64 |
| 正式训练 / 正式测试 / LM PPL | 0 / 禁止 / 不运行 |

32,768选取原系列32K–100K Smoke建议的低端并对齐Batch，作为本次独立、已批准的开发上限，不作为正式预算。一个Seed只观察可学习性与开发Episode波动，不能估计训练Seed方差，也不能替代预注册三Seed确认。

配置沿用工程测量规模：SISO、FP32、TF32关闭，dim64/2层/state16/head16/chunk16/vocab256；Reference固定15ba9983123703d9f4f7d6f8709152d0a07edbd9。公共权重逐Tensor哈希核对，Adapter独立Seed流沿用原实现。AdamW lr=.001、betas(.9,.95)、wd=.01、clip1、常数LR；这些是冻结的开发配方，不声称调优最优。无Dropout、AMP、Overflow重试或跨Episode State；完整160 Token序列内保留梯度，Episode间重置所有State/KV。FP32非有限时停止并保留最后良好Checkpoint，不裁剪/更新非有限梯度。

## 数据与评价

仅实现train-dev和validation-dev，代码拒绝formal/test。符号空间互斥：PAD/BOS/EOS/FACT/SEP/QUERY/ANSWER/HOLE=0..7，Key8..71，Value72..135，Filler136..199；200..255未用但仍属于完整argmax词表。每Episode8个随机无重复绑定，独立随机一一映射，数据Seed2710101。不能从Key ID推导Value。

六条件为MQAR/Repeated × Filler Gap0/32/64，输入统一160 Token并Padding（Padding也记入处理成本）。MQAR查询8个Key各一次；Repeated查询随机4个Key各两次。事实段FACT/key/value/SEP，查询段QUERY/key/ANSWER/HOLE；仅ANSWER位置监督Value，后续输入不回填金标准或模型答案。实际Key距离和原始绑定是否落在W16/W64按位置逐查询记录。Window之外的Mamba表示仍可保留压缩历史，不能当作无记忆。

完整数据Oracle、Token/label哈希、序列及绑定映射跨split去重、答案不回填检查均在训练前完成。开发数据本地生成，完整资产留Kaggle Output，无真实语料或正式测试集。三个臂完全同数据、同顺序、同公共初值和有效查询目标数，Loss为Group的CE总和/有效目标数。

逐更新记录训练Query EM与Loss，在0/256/512更新记录同一开发集。Loss下降可能只是学到Value先验，必须结合检索EM和去绑定负控制解读，不能独立当成记忆能力。保存最终逐查询预测与Episode/间隔/重复次数/实际距离，用Episode平均EM、Query NLL（不是语言PPL）、逐条件和首次/重复EM回答学习方向。去除事实Value的负控制用于识别可能的任务捷径，不新增训练臂。配对开发区间仅对共同Episode重采样1000次，Seed2710102；这不是正式显著性检验，也不包含训练Seed方差。

## 运行与真实Kaggle身份

沿用bingjiangli/minicells，T4、Internet、原GITHUB_TOKEN和KAGGLE_API_TOKEN。运行 notebook.ipynb，使用Save Version → Save & Run All，不能用未保存的交互草稿冒充Saved Version。

入口先尝试从当前运行真实提供的字段或Session返回取得ID。公共Kaggle SDK可能根本不暴露scriptVersionId，它的current_version_number与metadata.id不能当作Script Version ID；不解析/公开凭据全量内容。平台分支尚未在本机实测，依用户授权允许真实缺项待补录。

用户已明确批准：本次开发校准可先训练，缺失ID时记录ID_PENDING，保存版本后补齐。协议中的allow_deferred_script_version_id_development_only=true只适用于本次Development Calibration，正式训练仍为false、正式Targets仍为0。未提供ID不再归因为用户没有保存版本；真实冲突或协议标记不一致仍停止。

当前运行真实提供且可核验ID时继续使用；否则execution-identity保存script_version_id=null、ID_PENDING、独立Session标识、当前启动脚本hash及缺项原因，决不使用序号/时间戳/Git SHA代替。训练和评估完成后状态为DEVELOPMENT_CALIBRATION_COMPLETE_ID_PENDING；这表示开发计算完成，平台身份待补录，不表示正式放行。多个待补录Session分别记录，不按null值合并。

保存后把带`?scriptVersionId=真实ID`的Kaggle版本链接用于补录。入口支持`--saved-version-url`，`--output`可直接指向含decision.json的已有运行目录；只追加saved-version-supplement.json并复用现有发布器，不加载优化器或训练。例：

```text
python run.py --output /kaggle/working/<已有运行目录> --candidate /kaggle/working/mamba3-minimal-reference --saved-version-url 'https://www.kaggle.com/code/bingjiangli/minicells?scriptVersionId=<真实ID>'
```

补充记录绑定原run_id、Source与decision文件hash，保留原始ID_PENDING报告，不能用版本序号或其他Notebook的URL补位。仅解析用户提供的保存版本链接不等于独立平台验证，补充记录明确标为USER_SUPPLIED_PLATFORM_SAVED_VERSION_URL / independent_platform_verification=NOT_PERFORMED，不伪造已验证状态；既有补充记录不允许被不同记录静默覆盖。收到实际版本链接后也可在仓库追加对应补充记录，无需重新训练。

单次总上限7200秒：前置/训练/中间评估/必要保存至5400秒、终评/推理至6000秒、发布600秒、余量600秒（Notebook平台输出预留60秒来自余量）。超时保持INCOMPLETE，不减目标数或伪造完成。正式训练不会自动启动。

## Checkpoint与成本

复用原子保存函数、现有45秒/最多三版本的有界恢复和不可变对象传输，不新增跨Kaggle恢复验证。每64次及最终更新保存完整模型/AdamW/计数/评估记录。Checkpoint内含source+protocol+data身份；自身是唯一恢复权威，最终报告hash只用于结果核验，避免checkpoint和旁路hash文件不一致导致无法恢复。相同源码已完成模型恢复后不再更新参数；跨源码或协议不兼容Checkpoint拒绝复用。权重和原始npz只留Output，Git发布小报告/曲线/逐查询CSV。

训练计时包含forward/backward/finite checks/clip/optimizer，但不包含数据传输、准备、验证、保存、逐步日志指标提取和发布；总Session时间另列。记录P50/P95更新耗时、有效目标/s、含Padding输入Token及训练峰值显存。全部训练退出后，逐模型测量训练后Full/Decode一致性与Decode延迟（context96、16 Token、Batch1、3warmup+5采样）；选择mqar-gap64的同一Episode，避免以大量Padding作为主要Decode输入。该context不同于原工程Benchmark的128，不直接对比绝对延迟。

结果状态DEVELOPMENT_CALIBRATION_COMPLETE（或带ID_PENDING后缀）只表示固定开发预算的计算完成；待补录状态仍缺少平台身份。无论哪一臂领先，都不能宣布H-R1/H-R2成立；不按结果自动追加预算或选择正式δ_R。后续使用开发成本/方差支持正式预算选型，实用收益阈值仍需研究决策。

发布失败保留Output；支持 --publish-only，同一已完成模型不重复训练。本地只运行三个有价值的合成回归检查，不做完整本地训练、CPU Benchmark或完整历史测试矩阵。
