# EPCLM-004 M4 运行报告

状态：`M4_ENGINEERING_COMPLETE`。H1–H6：`NOT_EVALUATED`。

CPU 融合门禁：`PASS`；真实 T4 Smoke：`PASS`。
原始 CPU 指标在 audit/preflight.json；真实 GPU 训练及资源在 gpu/smoke.json，未运行时不得补填。
分支消融只检查本 E5 的关闭行为，不宣称数值退化为已冻结的 E3/E4。
完整高阶 Memory 更新规则复用 M3；Mamba-1 与 SWA 复用 M2。持久 Cell、SSM、KV 分别统计。
这不是质量、长程互补性或显著性结论。未启动 M5/M6、20M/100M。

```json
{}
```
