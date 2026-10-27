# FAST-002 最小定向修复

旧结果不改写。新增的 [历史审计](registration/HISTORICAL_AUDIT.json) 交叉核对50组公开记录：25,000逻辑更新、12,800,000主目标、15,000次Cell优化器调用、1,600次被拒候选的优化器调用、200提案、1,000尺度检查、正式Connection提交0。

阶段 A 分化通过仍保留；阶段 B 应解释为 Connection 处理未交付、协同价值尚未得到检验。GPU累计同步训练墙钟约8,775.42秒（各任务相加），不是会话时长或GPU kernel计时。原始逐事件Journal未在公开包中，审计不冒称独立验证所有历史优化器执行。

## Kaggle

导入本目录 `notebooks/kaggle-clm002-fast002.ipynb`，开启 Internet 和一张 GPU，授权已有 `GITHUB_TOKEN`、`KAGGLE_API_TOKEN`，然后 Run All。无需挂载 Input，也无需 HF Token、版本号、Seed、哈希或模式。程序复用 R3h/CF-001 的版本化 Output 恢复组件，在已保存的云端 Output 中匹配固定 Run ID，只下载所需30个 Checkpoint。最近12个版本都没有完整匹配时停止，不补训。私有 Output 的读取需要 Kaggle API 授权；已有本地完整缓存则不再访问 API。

先固定数据和30个旧Checkpoint身份，再做20个仅Train的候选校准；校准失败停止并回传证据。全部通过后，才使用新Confirm重新评价旧A/B00/B10，并最多重跑20个B01/B11任务。主训练上限10,000逻辑更新；校准最多160次、复验最多1,600次候选优化器调用另计。单GPU顺序执行。

`/kaggle/working/clm002-fast002-output` 保存报告、Journal与新Checkpoint；`clm002-fast002-assets` 保存生成数组。失败也应保留 Save Version。网络回传失败后 CLI `--publish-existing --output <已完成run目录> --assets <assets目录>` 仅重发结果，不重训。

报告：HISTORICAL_AUDIT、CHECKPOINT_REUSE_PROOF、CALIBRATION、FRESH_CONFIRM_PROOF、BUDGET、FINAL_ANALYSIS、RUN_MANIFEST及每组曲线。GitHub回传使用已有组件并核验远端SHA；权重/Token数组不进Git。每组预算标注语言目标、实际GPU使用、候选调用/拒绝/正式提交和同步墙钟秒数；未知kernel时间保留null。

详细冻结规则见 [协议](PROTOCOL.zh-CN.md)。这是定向修复，不是独立重复实验；不改质量阈值、不解冻输出层、不扩大模型、不追溯旧14项数据边界。

运行时导入细节见 [操作修订001](OPERATIONS_AMENDMENT_001.md)。本修订不改变科学配置、数据身份、Seed、尺度和训练预算。
