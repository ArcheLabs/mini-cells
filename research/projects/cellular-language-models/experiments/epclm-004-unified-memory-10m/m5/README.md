# EPCLM-004 M5 — Innovation Store

状态：实现与 CPU 开发检查完成；真实 Gate 0 / T4 验收待运行。H5 `NOT_EVALUATED`。

直接打开本目录 `notebook.ipynb`，在原 Kaggle Notebook 上选择 Tesla T4、开启 Internet，沿用 `GITHUB_TOKEN` 和 `KAGGLE_API_TOKEN`，Save & Run All。无需选择阶段。入口自动执行 Gate 0→7；失败即停止，保存报告并发布元数据。单次执行预算 5400 秒，另预留发布时间；Notebook 总上限 7200 秒。

Gate 0 固定下载 `bingjiangli/minicells` **Version 187**，逐文件匹配冻结 M4 Hash，归档实际 Model/Optimizer/Cell 到 GitHub Release，并下载回验。187 是版本序号，Script Version ID 保持未知，不复用 M3 的 ID。既有归档匹配时直接复用，不训练 M4。

五臂为 E3、E6、E7-Random/FIFO/Innovation。每臂 FP32、2 次真实外层更新、1024 Input Tokens、2 个 Query Targets，训练输入顺序相同。E3 9,939,200 参数；其余四臂 10,021,120 参数。公共 Backbone 初始化完全相同；E6/E7 完整静态参数初始化相同。

每个 Stream 只有一个全局有界 Store，由各 SWA 层共享。容量 64 训练；容量 64/128 的评估使用同一权重。Predictor 为无参数 EMA，创新度只决定写入；Recall 由 Query 相似度决定。State Summary 来源为前一 Chunk 的真实 SSM State。Local KV、SSM、Predictor、Store 独立持久化。

`configs/protocol.json` 为执行配置；四个 YAML 为薄选择说明。`SPEC.zh-CN.md` 保留原任务。`freeze/` 记录 M4 保全合同，不能误作 M5 科学预注册。`audit/` 为本地开发证据；实际运行报告生成在 Kaggle Output，权重不提交 Git。

不启动 M6，不进行 20M/100M，不访问独立 Test。开发集诊断不构成科学结论。实现细节与限制见报告及 implementation_deviations.md。
