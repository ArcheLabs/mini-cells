# M3C-002B-CAL-001 开发校准

状态：DEVELOPMENT_CALIBRATION_COMPLETE_ID_PENDING

R线MQAR/Repeated Retrieval，单开发Seed27101；正式训练/测试/语言PPL未运行，科学假设NOT_EVALUATED。
单Seed只提供Episode抽样波动，不能估计跨训练Seed方差。
若状态带ID_PENDING，仅表示开发训练和评估完成，真实Script Version ID待保存后补录；不放行正式训练。

| Arm | Targets | Initial Query NLL | Final Query NLL | Dev EM | Repeat first/later EM | Train s | Targets/s | Update P95 s | Decode ms/token |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| M3-Base | 32768 | 5.5551 | 4.1758 | 0.0156 | 0.0143/0.0143 | 18.73 | 1749.53 | 0.0372 | 2.7467811249977103 |
| M3-WA-W16 | 32768 | 5.5549 | 4.1754 | 0.0156 | 0.0143/0.0143 | 20.08 | 1631.61 | 0.0419 | 3.970911687503076 |
| M3-WA-W64 | 32768 | 5.5551 | 4.1757 | 0.0156 | 0.0143/0.0143 | 19.79 | 1655.45 | 0.0410 | 4.034489062497926 |

训练时间/TPS仅训练计算与优化器更新，环境准备、传输、评估、保存及发布另计；输入Token含Padding单独记账。
Gap和实际Key可见性条件见各臂result.json；逐查询预测见predictions.csv；去除事实Value的负控制用于排查任务捷径，不作为额外训练组。
置信区间仅开发Episode配对抽样，不能用于宣称H-R1/H-R2成立。不得据开发结果自动扩大训练预算或进入M3C-003。
