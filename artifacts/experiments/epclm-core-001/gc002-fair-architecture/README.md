# GC002 Phase 0/1 准备结果

**ENGINEERING_PREPARATION_ONLY_NOT_FROZEN — NOT AUTHORIZED TO TRAIN**。

本分支只保存已执行的 CPU 合成工程检查和历史报告审计，基于当前 main 创建；不修改原实验结果、协议或训练实现。代码与 Notebook 在 `codex/epclm-gc002-fair-architecture`。本次执行的精确源码提交与逐文件哈希记录在每份 JSON 的 identity 中；报告来自源码提交 `330980f`，后续 `ba35017` 仅澄清未归属 host/driver 开销时的 GPU total-memory 字段，未改变本次 CPU 结果。

- [Baseline Audit](preparation/audit/baseline-audit.json)：四个新模型 250K Token、977 Updates，历史最终分段恢复字段与墙钟口径澄清。
- [参数匹配候选](preparation/audit/parameter-match-report.json)：967,808 参数，相对 966,784 多 0.106%；尚未冻结。
- [KV 正确性](preparation/profiling/kv-cache-equivalence.json)：两种尺寸；32/128/385 Token、滑动窗口、分块、EOS、快照恢复，FP32 最大误差小于 1e-5。
- [实际缓存存储](preparation/profiling/memory-profile.json)：原尺寸紧凑 GQA 为 262,144 Bytes，PM 候选为 524,288 Bytes；不是总 runtime 内存。
- [阶段门禁](preparation/reports/research-decision.json)。

原始语料/Tokenizer 字节、历史实际消费顺序、GPU profile、完整训练 FLOPs、正式 Seed 与预算未全部核验或冻结。不能将这些准备结果标记为 Phase 0/1 全面 PASS 或 Phase 2 PROTOCOL_FROZEN。CPU 未做性能 Benchmark；没有训练、没有 Optimizer Update、没有 GPU 正式实验。

`protocol/draft-content.sha256` 只校验草案内容，不表示协议冻结；未生成 `protocol.sha256`。H1–H5 NOT_EVALUATED，Human Review PENDING，独立 Test 未访问。
